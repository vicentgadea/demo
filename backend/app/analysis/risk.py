"""
Módulo de análisis de riesgo.
Distingue tres tipos:
  1. Stop táctico (técnico): precio de invalidación de la configuración chartista
  2. Invalidación de tesis: cambio fundamental que cambia la narrativa
  3. Deterioro estructural: el negocio cambia de forma permanente

También calcula métricas de riesgo cuantitativo:
- Distancia al soporte más cercano
- Volatilidad y ATR
- Riesgo de compresión de múltiplos
- Orientación sobre tamaño de posición
"""
import logging

from ..models.inputs import RiskProfile
from ..models.outputs import (
    RiskAnalysis,
    RiskFactor,
    FundamentalAnalysis,
    ValuationAnalysis,
    TechnicalAnalysis,
    ValuationLevel,
    Trend,
)

logger = logging.getLogger(__name__)


def analyze_risk(
    fundamental: FundamentalAnalysis,
    valuation: ValuationAnalysis,
    technical: TechnicalAnalysis,
    risk_profile: RiskProfile,
) -> RiskAnalysis:
    """Genera el análisis de riesgo completo."""
    current = valuation.current_price
    factors: list[RiskFactor] = []

    # ── Riesgo de deuda ───────────────────────────────────────────────────────
    factors.extend(_debt_risks(fundamental))

    # ── Riesgo de valoración ──────────────────────────────────────────────────
    factors.extend(_valuation_risks(valuation))

    # ── Riesgo de crecimiento ─────────────────────────────────────────────────
    factors.extend(_growth_risks(fundamental))

    # ── Riesgo técnico ────────────────────────────────────────────────────────
    factors.extend(_technical_risks(technical))

    # ── Riesgo de márgenes ────────────────────────────────────────────────────
    factors.extend(_margin_risks(fundamental))

    # ── Distancia al soporte ──────────────────────────────────────────────────
    downside_to_support = None
    if technical.support_resistance.support_1:
        downside_to_support = (
            (current - technical.support_resistance.support_1) / current * 100
        )

    # ── Distancia al valor conservador ───────────────────────────────────────
    downside_fv = None
    if valuation.fair_value_conservative and valuation.fair_value_conservative < current:
        downside_fv = (
            (current - valuation.fair_value_conservative) / current * 100
        )

    # ── Riesgo de compresión de múltiplos ────────────────────────────────────
    multiple_compression = _multiple_compression_risk(valuation, fundamental)

    # ── Nivel general de riesgo ───────────────────────────────────────────────
    overall_risk = _overall_risk(factors, fundamental, technical, valuation)

    # ── Orientación de tamaño de posición ────────────────────────────────────
    position_note = _position_size_note(
        risk_profile, overall_risk,
        technical.volatility.historical_volatility_30d,
        technical.volatility.atr_pct,
    )

    return RiskAnalysis(
        overall_risk_level=overall_risk,
        downside_to_support=round(downside_to_support, 1) if downside_to_support else None,
        downside_to_fair_value_low=round(downside_fv, 1) if downside_fv else None,
        multiple_compression_risk=multiple_compression,
        risk_factors=factors,
        position_size_note=position_note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Factores de riesgo específicos
# ─────────────────────────────────────────────────────────────────────────────

def _debt_risks(f: FundamentalAnalysis) -> list[RiskFactor]:
    risks = []
    debt = f.debt

    if debt.net_debt_ebitda is not None:
        if debt.net_debt_ebitda > 4:
            risks.append(RiskFactor(
                name="Deuda elevada",
                severity="high",
                description=(
                    f"Deuda neta de {debt.net_debt_ebitda:.1f}x EBITDA. Empresas con ratio >4x "
                    f"son vulnerables a subidas de tipos, recesión o rebaja de rating."
                ),
            ))
        elif debt.net_debt_ebitda > 2.5:
            risks.append(RiskFactor(
                name="Deuda moderada",
                severity="medium",
                description=f"Deuda neta de {debt.net_debt_ebitda:.1f}x EBITDA. Manejable pero reduce flexibilidad.",
            ))

    if debt.interest_coverage is not None and debt.interest_coverage < 3:
        risks.append(RiskFactor(
            name="Cobertura de intereses ajustada",
            severity="high" if debt.interest_coverage < 1.5 else "medium",
            description=f"Cobertura de {debt.interest_coverage:.1f}x. Por debajo de 3x hay riesgo ante caída del EBIT.",
        ))

    if debt.current_ratio is not None and debt.current_ratio < 1.0:
        risks.append(RiskFactor(
            name="Liquidez corriente insuficiente",
            severity="medium",
            description=f"Current ratio de {debt.current_ratio:.2f}x — pasivo corriente supera al activo corriente.",
        ))

    return risks


def _valuation_risks(v: ValuationAnalysis) -> list[RiskFactor]:
    risks = []

    if v.valuation_level == ValuationLevel.overvalued:
        discount = v.discount_to_base or 0
        risks.append(RiskFactor(
            name="Valoración excesiva",
            severity="high",
            description=(
                f"El precio cotiza con una prima del {abs(discount):.1f}% sobre el valor base estimado. "
                f"Alta exposición a compresión de múltiplos si el crecimiento decepciona."
            ),
        ))
    elif v.valuation_level == ValuationLevel.demanding:
        risks.append(RiskFactor(
            name="Valoración exigente",
            severity="medium",
            description="Prima moderada sobre el valor razonable. El mercado descuenta crecimiento que debe materializarse.",
        ))

    return risks


def _growth_risks(f: FundamentalAnalysis) -> list[RiskFactor]:
    risks = []
    growth = f.growth

    if growth.revenue_growth_1y is not None and growth.revenue_growth_1y < -5:
        risks.append(RiskFactor(
            name="Desaceleración de ingresos",
            severity="high",
            description=f"Ingresos cayendo al {growth.revenue_growth_1y:.1f}% anual. Puede indicar pérdida de cuota o demanda débil.",
        ))
    elif growth.revenue_growth_1y is not None and growth.revenue_growth_1y < 0:
        risks.append(RiskFactor(
            name="Crecimiento negativo",
            severity="medium",
            description=f"Ingresos en ligero declive ({growth.revenue_growth_1y:.1f}%). Vigilar si es temporal.",
        ))

    if growth.eps_growth_1y is not None and growth.eps_growth_1y < -10:
        risks.append(RiskFactor(
            name="Caída de beneficios",
            severity="high",
            description=f"EPS cayendo al {growth.eps_growth_1y:.1f}% anual.",
        ))

    return risks


def _technical_risks(t: TechnicalAnalysis) -> list[RiskFactor]:
    risks = []

    if t.trend_medium in (Trend.bearish, Trend.strong_bearish):
        risks.append(RiskFactor(
            name="Tendencia bajista de medio plazo",
            severity="medium",
            description=(
                "El precio está por debajo de sus medias móviles relevantes y muestra "
                "estructura de máximos y mínimos decrecientes. El viento técnico sopla en contra."
            ),
        ))

    if t.is_extended:
        risks.append(RiskFactor(
            name="Precio sobreextendido",
            severity="medium",
            description=t.extension_note or "El precio está alejado de sus medias — riesgo de corrección técnica.",
        ))

    if t.momentum.rsi_14 and t.momentum.rsi_14 > 75:
        risks.append(RiskFactor(
            name="RSI en sobrecompra extrema",
            severity="low",
            description=f"RSI de {t.momentum.rsi_14:.0f} indica sobrecompra de corto plazo.",
        ))

    if t.volatility.historical_volatility_30d and t.volatility.historical_volatility_30d > 50:
        risks.append(RiskFactor(
            name="Volatilidad muy elevada",
            severity="medium",
            description=f"Volatilidad histórica anualizada del {t.volatility.historical_volatility_30d:.0f}% — stop amplio, posición reducida.",
        ))

    return risks


def _margin_risks(f: FundamentalAnalysis) -> list[RiskFactor]:
    risks = []
    margins = f.margins

    if margins.operating_margin_trend == "deteriorating":
        risks.append(RiskFactor(
            name="Compresión de márgenes operativos",
            severity="medium",
            description="Tendencia de deterioro en márgenes operativos en los últimos años. Puede reflejar presión de costes o pérdida de pricing power.",
        ))

    if margins.operating_margin is not None and margins.operating_margin < 0.05:
        risks.append(RiskFactor(
            name="Margen operativo muy bajo",
            severity="high" if margins.operating_margin < 0 else "medium",
            description=f"Margen operativo de solo {margins.operating_margin:.1%} — la empresa tiene poco colchón ante shocks.",
        ))

    return risks


# ─────────────────────────────────────────────────────────────────────────────
# Riesgo de compresión de múltiplos
# ─────────────────────────────────────────────────────────────────────────────

def _multiple_compression_risk(v: ValuationAnalysis, f: FundamentalAnalysis) -> str:
    if v.valuation_level == ValuationLevel.overvalued:
        return "high"
    if v.valuation_level == ValuationLevel.demanding:
        return "medium"
    if f.growth.revenue_cagr_3y is not None and f.growth.revenue_cagr_3y < 0:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Nivel de riesgo global
# ─────────────────────────────────────────────────────────────────────────────

def _overall_risk(
    factors: list[RiskFactor],
    f: FundamentalAnalysis,
    t: TechnicalAnalysis,
    v: ValuationAnalysis,
) -> str:
    critical_count = sum(1 for r in factors if r.severity == "critical")
    high_count = sum(1 for r in factors if r.severity == "high")
    medium_count = sum(1 for r in factors if r.severity == "medium")

    if critical_count > 0 or high_count >= 3:
        return "high"
    if high_count >= 2 or (high_count == 1 and medium_count >= 2):
        return "elevated"
    if high_count == 1 or medium_count >= 3:
        return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Orientación de tamaño de posición
# ─────────────────────────────────────────────────────────────────────────────

def _position_size_note(
    profile: RiskProfile,
    overall_risk: str,
    vol_30d: float | None,
    atr_pct: float | None,
) -> str:
    base_sizes = {
        "conservative": {"low": "3-5%", "moderate": "2-3%", "elevated": "1-2%", "high": "<1% o no invertir"},
        "balanced": {"low": "5-8%", "moderate": "3-5%", "elevated": "2-3%", "high": "1-2%"},
        "aggressive": {"low": "8-12%", "moderate": "5-8%", "elevated": "3-5%", "high": "2-3%"},
    }
    size_suggestion = base_sizes.get(profile.value, base_sizes["balanced"]).get(overall_risk, "3-5%")

    vol_note = ""
    if vol_30d and vol_30d > 40:
        vol_note = f" Con volatilidad anualizada del {vol_30d:.0f}%, considera reducir el tamaño a la mitad del sugerido."
    elif atr_pct and atr_pct > 3:
        vol_note = f" ATR diario del {atr_pct:.1f}% — activo volátil, gestiona el tamaño con criterio."

    return (
        f"Para perfil {profile.value} con riesgo {overall_risk}: "
        f"posición orientativa del {size_suggestion} de la cartera.{vol_note} "
        f"NOTA: esta orientación no constituye asesoramiento financiero. "
        f"Adapta el tamaño a tu situación personal, diversificación y tolerancia real al riesgo."
    )
