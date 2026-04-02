"""
Módulo de zonas de entrada y salida.

Lógica multicapa:
1. Ancla de valoración: valor razonable como suelo de las entradas
2. Ancla técnica: soportes/resistencias como referencias de precio
3. Ajuste por extensión: si el precio está sobreextendido, los rangos se abren hacia abajo
4. Ajuste por tendencia: bajista → entradas más bajas; alcista → se acepta pagar más
5. Ajuste por perfil de riesgo: conservador pide mayor descuento
6. Stop técnico: ATR × factor + soporte más cercano
7. Objetivos de salida: valor base y optimista con ajuste por horizonte

El resultado es orientativo y probabilístico, NO una predicción ni una recomendación de inversión.
"""
import logging

from ..models.inputs import RiskProfile, TimeHorizon
from ..models.outputs import (
    EntryExitPlan,
    PriceZone,
    ValuationAnalysis,
    TechnicalAnalysis,
    FundamentalAnalysis,
    Trend,
    ValuationLevel,
)

logger = logging.getLogger(__name__)


def analyze_entry_exit(
    fundamental: FundamentalAnalysis,
    valuation: ValuationAnalysis,
    technical: TechnicalAnalysis,
    risk_profile: RiskProfile,
    time_horizon: TimeHorizon,
) -> EntryExitPlan:
    """Genera el plan orientativo de entrada/salida."""
    current = valuation.current_price
    fv_base = valuation.fair_value_base
    fv_conservative = valuation.fair_value_conservative
    fv_optimistic = valuation.fair_value_optimistic

    sr = technical.support_resistance
    vol = technical.volatility
    trend_m = technical.trend_medium
    trend_l = technical.trend_long
    is_extended = technical.is_extended

    # ── Factores de ajuste ────────────────────────────────────────────────────
    profile_factor = _profile_discount_factor(risk_profile)  # qué % de descuento exige el perfil
    horizon_factor = _horizon_factor(time_horizon)
    trend_factor = _trend_factor(trend_m, trend_l)  # 1 = neutral, <1 = bajista (más barato), >1 = alcista

    # ── Anclas de precio ──────────────────────────────────────────────────────
    # Soporte más cercano por debajo del precio
    support_close = sr.support_1 or (current * 0.92)
    support_far = sr.support_2 or (current * 0.85)
    resistance_1 = sr.resistance_1 or (fv_base or current * 1.15)

    # ATR como medida de volatilidad para calcular stops
    atr = None
    if vol.atr_14:
        atr = vol.atr_14
    elif vol.atr_pct and vol.atr_pct > 0:
        atr = current * vol.atr_pct / 100
    atr = atr or current * 0.02  # default: 2% si no hay ATR

    # ── Zonas de entrada ─────────────────────────────────────────────────────
    # Conservadora: cerca del valor conservador de valoración o soporte lejano
    # Razonable: entre valor base y soporte cercano
    # Agresiva: si la tendencia es fuerte, se permite entrar más cerca del precio actual

    if fv_base and fv_conservative:
        # Con valoración disponible
        entry_c_low = min(fv_conservative, support_far) * (1 - profile_factor * 0.5)
        entry_c_high = min(fv_conservative, support_close)

        entry_r_low = min(fv_base * 0.93, support_close) * (1 - profile_factor * 0.2)
        entry_r_high = fv_base * (1 - profile_factor * 0.1)

        entry_a_low = fv_base * (1 - profile_factor * 0.05)
        entry_a_high = fv_base * (1 + 0.05)  # acepta hasta un 5% sobre el base
    else:
        # Sin valoración confiable: anclar solo en técnico
        entry_c_low = support_far * 0.97
        entry_c_high = support_far
        entry_r_low = support_close * 0.98
        entry_r_high = support_close
        entry_a_low = current * (1 - atr / current * 2)
        entry_a_high = current * (1 - atr / current)

    # Ajuste por extensión: si el precio está sobreextendido, bajar todas las entradas
    if is_extended:
        pullback_target = current * (1 - min(atr / current * 3, 0.12))
        entry_c_low = min(entry_c_low, pullback_target * 0.95)
        entry_c_high = min(entry_c_high, pullback_target)
        entry_r_low = min(entry_r_low, pullback_target)
        entry_r_high = min(entry_r_high, current * 0.97)
        entry_a_low = min(entry_a_low, current * 0.98)
        entry_a_high = current * 0.99

    # Ajuste por tendencia
    entry_c_low *= trend_factor * 0.97 + (1 - trend_factor) * 1.0
    entry_c_high *= trend_factor * 0.98 + (1 - trend_factor) * 1.0

    # Evitar que las entradas superen el precio actual en > 2% (no es stop de compra con momentum)
    max_entry = current * 1.02
    entry_a_high = min(entry_a_high, max_entry)
    entry_r_high = min(entry_r_high, max_entry)
    entry_c_high = min(entry_c_high, max_entry)

    # Ordenar lógicamente: c_low <= c_high <= r_low (aprox) <= r_high <= a_low <= a_high
    entry_c_low, entry_c_high = sorted([entry_c_low, entry_c_high])
    entry_r_low, entry_r_high = sorted([entry_r_low, entry_r_high])
    entry_a_low, entry_a_high = sorted([entry_a_low, entry_a_high])

    # ── Stop técnico ──────────────────────────────────────────────────────────
    stop_base = support_close - atr * _stop_atr_factor(risk_profile)
    stop_technical = max(stop_base, current * 0.70)  # nunca más del -30% desde precio actual

    # ── Objetivos de salida ───────────────────────────────────────────────────
    if fv_base and fv_optimistic:
        target_partial = fv_base * (1 + 0.03 * horizon_factor)
        target_full = fv_optimistic * (1 - 0.05 * (1 - horizon_factor))
    else:
        # Sin valoración: usar resistencias como objetivos
        target_partial = resistance_1
        target_full = resistance_1 * 1.10 if resistance_1 else current * 1.25

    # Resistencia técnica puede ser objetivo intermedio
    if sr.resistance_1 and sr.resistance_1 < target_partial:
        target_partial_note = f"Objetivo parcial en {target_partial:.2f}, vigilar resistencia técnica en {sr.resistance_1:.2f}"
    else:
        target_partial_note = f"Objetivo parcial en {target_partial:.2f} (valoración base)"

    target_full_note = (
        f"Revisión/salida en {target_full:.2f} (valoración optimista) o si los fundamentales se deterioran"
    )

    # ── Ratio R/R ─────────────────────────────────────────────────────────────
    rr_conservative = _risk_reward(entry_c_high, stop_technical, target_partial)
    rr_reasonable = _risk_reward(entry_r_high, stop_technical, target_partial)

    # ── Acción inmediata recomendada ──────────────────────────────────────────
    immediate_action, conditions_entry, conditions_exit = _immediate_action(
        current, entry_c_high, entry_r_high, entry_a_high,
        is_extended, trend_m, trend_l,
        fundamental, valuation,
    )

    # ── Invalidaciones ───────────────────────────────────────────────────────
    inv_fundamental = _invalidation_fundamental(fundamental, valuation)
    inv_structural = _invalidation_structural(fundamental)

    return EntryExitPlan(
        current_price=current,
        entry_conservative=PriceZone(
            low=_r(entry_c_low),
            high=_r(entry_c_high),
            rationale=f"Zona de precio próxima a valoración conservadora ({fv_conservative:.2f} si disponible) y soporte técnico alejado",
        ),
        entry_reasonable=PriceZone(
            low=_r(entry_r_low),
            high=_r(entry_r_high),
            rationale="Equilibrio entre descuento respecto al valor base y proximidad a soporte técnico",
        ),
        entry_aggressive=PriceZone(
            low=_r(entry_a_low),
            high=_r(entry_a_high),
            rationale="Acepta pagar próximo al precio actual; solo válido con tendencia alcista y fundamentales sólidos",
        ),
        stop_technical=_r(stop_technical),
        stop_technical_rationale=(
            f"Stop por debajo del soporte más cercano ({support_close:.2f}) menos {_stop_atr_factor(risk_profile):.1f}×ATR ({atr:.2f})"
        ),
        invalidation_fundamental=inv_fundamental,
        invalidation_structural=inv_structural,
        target_partial=_r(target_partial),
        target_partial_rationale=target_partial_note,
        target_full_or_review=_r(target_full),
        target_full_rationale=target_full_note,
        risk_reward_conservative=rr_conservative,
        risk_reward_reasonable=rr_reasonable,
        immediate_action=immediate_action,
        conditions_for_entry=conditions_entry,
        conditions_for_exit=conditions_exit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lógica de decisión inmediata
# ─────────────────────────────────────────────────────────────────────────────

def _immediate_action(
    current: float,
    entry_c: float,
    entry_r: float,
    entry_a: float,
    is_extended: bool,
    trend_m: Trend,
    trend_l: Trend,
    fundamental: FundamentalAnalysis,
    valuation: ValuationAnalysis,
) -> tuple[str, list[str], list[str]]:
    conditions_entry = []
    conditions_exit = []

    is_bearish = trend_m in (Trend.bearish, Trend.strong_bearish)
    is_bullish = trend_m in (Trend.bullish, Trend.strong_bullish)
    is_overvalued = valuation.valuation_level == ValuationLevel.overvalued
    is_undervalued = valuation.valuation_level in (ValuationLevel.undervalued, ValuationLevel.fair)
    fund_ok = fundamental.overall_fundamental_score >= 6.0
    fund_bad = fundamental.overall_fundamental_score < 4.0

    # Caso 1: precio extendido — esperar retroceso
    if is_extended and not (is_undervalued and fund_ok):
        action = (
            "ESPERAR RETROCESO. El precio está sobreextendido técnicamente. "
            "Esperar corrección hacia zonas de entrada razonable antes de construir posición."
        )
        conditions_entry = [
            f"Precio retrocede a zona razonable ({entry_r:.2f}–{entry_r*1.03:.2f})",
            "RSI baja de 60 tras el retroceso",
            "Volumen decrece durante el pullback (saludable)",
        ]

    # Caso 2: valoración exigente pero tendencia fuerte y buenos fundamentales
    elif is_overvalued and fund_ok and is_bullish:
        action = (
            "VIGILAR PERO NO ENTRAR. La valoración es exigente. "
            "La tendencia es favorable, pero el margen de seguridad es escaso. "
            "Entrada solo en retrocesos significativos o con catalizador de crecimiento nuevo."
        )
        conditions_entry = [
            f"Corrección al menos al nivel de entrada razonable ({entry_r:.2f})",
            "Actualización de resultados que justifique la valoración actual",
        ]

    # Caso 3: value trap potencial — barata pero deterioro
    elif is_undervalued and fund_bad:
        action = (
            "PRECAUCIÓN: posible value trap. "
            "El precio parece barato, pero los fundamentales muestran debilidad. "
            "No entrar hasta ver estabilización o mejora de métricas operativas."
        )
        conditions_entry = [
            "Al menos un trimestre de estabilización de márgenes o ingresos",
            "Señal técnica de reversión con volumen (mínimo decreciente que aguanta)",
        ]

    # Caso 4: tendencia bajista fuerte
    elif is_bearish:
        action = (
            "NO ENTRAR EN TENDENCIA BAJISTA. "
            "Esperar señales de reversión técnica (SMA50 empieza a girar arriba, mínimos crecientes). "
            "Las entradas conservadoras pueden funcionar si el fundamental es sólido, en escalonado."
        )
        conditions_entry = [
            "Precio cierra por encima de SMA50 con volumen",
            "RSI recupera zona de 40–50 desde sobreventa",
            f"Precio aguanta por encima del soporte clave ({valuation.current_price * 0.90:.2f})",
        ]

    # Caso 5: situación atractiva — tendencia favorable, valoración justa y fundamental bueno
    elif is_undervalued and fund_ok and (is_bullish or trend_l in (Trend.bullish, Trend.strong_bullish)):
        action = (
            f"SITUACIÓN INTERESANTE. Precio actual ({current:.2f}) está dentro o cerca de la zona de entrada razonable. "
            "Considerar entrada escalonada (ej. 50% ahora, 50% en retroceso). "
            "El riesgo/beneficio parece favorable, pero siempre con stop definido."
        )
        conditions_entry = [
            "Confirmar que el precio aguanta por encima del soporte más cercano",
            "RSI no en sobrecompra extrema (>75)",
            "No hay evento de riesgo macro inmediato (resultados, tipos, etc.)",
        ]

    else:
        action = (
            "SEGUIMIENTO ACTIVO RECOMENDADO. "
            "Las condiciones no son inequívocamente favorables ni desfavorables. "
            f"Considerar entrada escalonada si el precio cae a la zona razonable ({entry_r:.2f}–{entry_r*1.03:.2f})."
        )
        conditions_entry = [
            f"Precio retrocede a zona de entrada razonable ({entry_r:.2f})",
            "Confirmación con un cierre en positivo con volumen normal",
        ]

    # Condiciones de salida común
    conditions_exit = [
        "El crecimiento de ingresos cae dos trimestres consecutivos por debajo de lo esperado",
        "El margen operativo se deteriora más de 3 puntos porcentuales sin explicación temporal",
        "La deuda neta/EBITDA supera los niveles de alerta señalados",
        "El precio rompe a la baja el stop técnico definido con cierre semanal",
        "La tesis de inversión cambia materialmente (cambio de negocio, pérdida de ventaja competitiva)",
    ]

    return action, conditions_entry, conditions_exit


# ─────────────────────────────────────────────────────────────────────────────
# Invalidaciones
# ─────────────────────────────────────────────────────────────────────────────

def _invalidation_fundamental(
    fundamental: FundamentalAnalysis,
    valuation: ValuationAnalysis,
) -> str:
    alerts = []
    debt = fundamental.debt
    margins = fundamental.margins
    growth = fundamental.growth

    if debt.net_debt_ebitda is not None:
        threshold = 4.0 if debt.net_debt_ebitda < 3 else debt.net_debt_ebitda * 1.3
        alerts.append(f"ND/EBITDA supera {threshold:.1f}x")

    if margins.operating_margin is not None:
        deterioration = max(margins.operating_margin - 0.05, 0)
        alerts.append(f"Margen operativo cae por debajo de {deterioration:.0%} de forma sostenida")

    if growth.revenue_growth_1y is not None and growth.revenue_growth_1y < 0:
        alerts.append("Segunda caída consecutiva de ingresos")
    else:
        alerts.append("Dos trimestres seguidos con crecimiento de ingresos negativo inesperado")

    alerts.append("Pérdida material de cuota de mercado o entrada de competidor disruptivo")

    return "Invalidación fundamental si: " + " | ".join(alerts)


def _invalidation_structural(fundamental: FundamentalAnalysis) -> str:
    return (
        "Deterioro estructural si: el modelo de negocio pierde relevancia (disrución tecnológica, "
        "regulatoria o de preferencias del consumidor), si la dirección muestra signos de destrucción "
        "de valor recurrente (adquisiciones sobrepagas, dilución agresiva) o si el FCF se vuelve "
        "estructuralmente negativo durante más de 2 años sin explicación de inversión justificada."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _profile_discount_factor(profile: RiskProfile) -> float:
    """Qué % de descuento mínimo exige el perfil sobre el valor razonable."""
    return {"conservative": 0.20, "balanced": 0.10, "aggressive": 0.03}[profile.value]


def _horizon_factor(horizon: TimeHorizon) -> float:
    """Factor que ajusta los objetivos según horizonte: más largo → objetivos más ambiciosos."""
    return {"short": 0.5, "medium": 1.0, "long": 1.5}[horizon.value]


def _trend_factor(trend_m: Trend, trend_l: Trend) -> float:
    """Ajusta ligeramente los rangos de entrada según tendencia."""
    score = 0
    if trend_m in (Trend.bullish, Trend.strong_bullish):
        score += 1
    elif trend_m in (Trend.bearish, Trend.strong_bearish):
        score -= 1
    if trend_l in (Trend.bullish, Trend.strong_bullish):
        score += 0.5
    elif trend_l in (Trend.bearish, Trend.strong_bearish):
        score -= 0.5
    return 1.0 + score * 0.02  # rango: 0.97 a 1.03


def _stop_atr_factor(profile: RiskProfile) -> float:
    """Cuántos ATRs bajo el soporte se coloca el stop."""
    return {"conservative": 1.0, "balanced": 1.5, "aggressive": 2.0}[profile.value]


def _risk_reward(entry: float, stop: float, target: float) -> float | None:
    """R/R = (target - entry) / (entry - stop)."""
    if entry <= 0 or stop <= 0 or stop >= entry:
        return None
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return None
    return round(reward / risk, 2)


def _r(val: float | None, decimals: int = 2) -> float:
    if val is None:
        return 0.0
    return round(val, decimals)
