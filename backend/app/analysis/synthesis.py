"""
Motor de síntesis.
Combina los cuatro módulos (fundamental, valoración, técnico, riesgo)
con pesos que dependen del perfil del usuario y el estilo de análisis.
Produce la ficha final con tesis, fortalezas, debilidades, riesgos y conclusión.
"""
import logging

from ..models.inputs import AnalysisStyle, RiskProfile, TimeHorizon
from ..models.outputs import (
    AnalysisSynthesis,
    FundamentalAnalysis,
    ValuationAnalysis,
    TechnicalAnalysis,
    RiskAnalysis,
    ValuationLevel,
    Trend,
)

logger = logging.getLogger(__name__)

# Pesos por estilo de análisis
WEIGHTS: dict[str, dict[str, float]] = {
    "fundamental": {"fundamental": 0.55, "valuation": 0.35, "technical": 0.10},
    "technical":   {"fundamental": 0.15, "valuation": 0.20, "technical": 0.65},
    "mixed":       {"fundamental": 0.40, "valuation": 0.30, "technical": 0.30},
}


def analyze_synthesis(
    fundamental: FundamentalAnalysis,
    valuation: ValuationAnalysis,
    technical: TechnicalAnalysis,
    risk: RiskAnalysis,
    risk_profile: RiskProfile,
    analysis_style: AnalysisStyle,
    time_horizon: TimeHorizon,
    company_name: str,
) -> AnalysisSynthesis:
    """Genera la síntesis final del análisis."""

    weights = WEIGHTS.get(analysis_style.value, WEIGHTS["mixed"])

    # ── Score técnico como número 0-10 ────────────────────────────────────────
    tech_score = _trend_to_score(technical.trend_medium, technical.trend_long)

    # ── Score de valoración 0-10 ──────────────────────────────────────────────
    val_score = _valuation_to_score(valuation)

    # ── Score compuesto ───────────────────────────────────────────────────────
    raw_score = (
        fundamental.overall_fundamental_score * weights["fundamental"]
        + val_score * weights["valuation"]
        + tech_score * weights["technical"]
    )

    # Penalización por riesgo
    risk_penalty = _risk_penalty(risk)
    overall_score = max(0, min(10, raw_score - risk_penalty))

    breakdown = {
        "fundamental": round(fundamental.overall_fundamental_score, 2),
        "fundamental_weight": weights["fundamental"],
        "valuation": round(val_score, 2),
        "valuation_weight": weights["valuation"],
        "technical": round(tech_score, 2),
        "technical_weight": weights["technical"],
        "risk_penalty": round(risk_penalty, 2),
        "overall": round(overall_score, 2),
    }

    # ── Tesis ──────────────────────────────────────────────────────────────────
    thesis = _build_thesis(company_name, fundamental, valuation, technical, overall_score)

    # ── Fortalezas ────────────────────────────────────────────────────────────
    strengths = _extract_strengths(fundamental, valuation, technical)

    # ── Debilidades ───────────────────────────────────────────────────────────
    weaknesses = _extract_weaknesses(fundamental, valuation, technical)

    # ── Riesgos clave ─────────────────────────────────────────────────────────
    key_risks = [f.name + ": " + f.description[:120] for f in risk.risk_factors[:5]]

    # ── ¿Merece seguimiento? ──────────────────────────────────────────────────
    worth_following = overall_score >= 5.5
    worth_reason = _worth_following_reason(overall_score, fundamental, valuation)

    # ── ¿Precio atractivo ahora? ──────────────────────────────────────────────
    current_attractive, attractiveness_cond = _price_attractiveness(valuation, technical, fundamental)

    weighting_note = (
        f"Ponderación para perfil '{risk_profile.value}' y estilo '{analysis_style.value}': "
        f"Fundamental {weights['fundamental']:.0%}, Valoración {weights['valuation']:.0%}, "
        f"Técnico {weights['technical']:.0%}. "
        f"Penalización por riesgo: -{risk_penalty:.1f} puntos."
    )

    return AnalysisSynthesis(
        overall_score=round(overall_score, 2),
        score_breakdown=breakdown,
        investment_thesis=thesis,
        key_strengths=strengths,
        key_weaknesses=weaknesses,
        key_risks=key_risks,
        worth_following=worth_following,
        worth_following_reason=worth_reason,
        current_price_attractive=current_attractive,
        attractiveness_conditions=attractiveness_cond,
        weighting_note=weighting_note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trend_to_score(trend_medium: Trend, trend_long: Trend) -> float:
    score_map = {
        Trend.strong_bullish: 9.0,
        Trend.bullish: 7.0,
        Trend.neutral: 5.0,
        Trend.bearish: 3.0,
        Trend.strong_bearish: 1.5,
    }
    s = score_map.get(trend_medium, 5.0) * 0.6 + score_map.get(trend_long, 5.0) * 0.4
    return round(s, 1)


def _valuation_to_score(v: ValuationAnalysis) -> float:
    level_score = {
        ValuationLevel.undervalued: 9.0,
        ValuationLevel.fair: 7.0,
        ValuationLevel.demanding: 4.5,
        ValuationLevel.overvalued: 2.0,
        ValuationLevel.insufficient_data: 5.0,
    }
    base = level_score.get(v.valuation_level, 5.0)
    # Ajuste fino por % de descuento/prima
    if v.discount_to_base is not None:
        discount = v.discount_to_base
        if discount > 30:
            base = min(base + 1, 10)
        elif discount < -30:
            base = max(base - 1, 0)
    return round(base, 1)


def _risk_penalty(risk: RiskAnalysis) -> float:
    return {
        "low": 0.0,
        "moderate": 0.3,
        "elevated": 0.8,
        "high": 1.5,
    }.get(risk.overall_risk_level, 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Narrativa
# ─────────────────────────────────────────────────────────────────────────────

def _build_thesis(
    name: str,
    f: FundamentalAnalysis,
    v: ValuationAnalysis,
    t: TechnicalAnalysis,
    score: float,
) -> str:
    quality_note = f.quality_score.label
    val_note = v.valuation_level.value.replace("_", " ")
    trend_note = t.trend_medium.value.replace("_", " ")

    thesis = (
        f"{name} presenta una puntuación global de {score:.1f}/10. "
        f"Desde el punto de vista fundamental: {quality_note}. "
        f"La valoración actual es '{val_note}'"
    )

    if v.discount_to_base is not None:
        if v.discount_to_base > 0:
            thesis += f" (descuento del {v.discount_to_base:.1f}% vs. valor base)"
        else:
            thesis += f" (prima del {abs(v.discount_to_base):.1f}% sobre el valor base)"

    thesis += f". La tendencia técnica de medio plazo es '{trend_note}'."

    if score >= 7:
        thesis += " La combinación de factores es favorable para inversión a medio/largo plazo con una gestión de riesgo adecuada."
    elif score >= 5.5:
        thesis += " El perfil es moderado: hay aspectos positivos pero también áreas de incertidumbre que merecen seguimiento."
    else:
        thesis += " El perfil actual presenta más señales de cautela que de oportunidad. Esperar mejora de condiciones antes de actuar."

    return thesis


def _extract_strengths(
    f: FundamentalAnalysis,
    v: ValuationAnalysis,
    t: TechnicalAnalysis,
) -> list[str]:
    strengths = []

    # Fundamentales
    strengths.extend(f.quality_score.factors[:2])
    strengths.extend(f.financial_strength_score.factors[:2])
    if f.overall_fundamental_score >= 7:
        strengths.append(f"Puntuación fundamental alta ({f.overall_fundamental_score:.1f}/10)")

    # Valoración
    if v.valuation_level == ValuationLevel.undervalued:
        strengths.append(f"Empresa cotizando con descuento respecto al valor estimado ({v.discount_to_base:.1f}%)")
    elif v.valuation_level == ValuationLevel.fair:
        strengths.append("Valoración razonable sin prima excesiva")

    # Técnico
    if t.trend_medium in (Trend.bullish, Trend.strong_bullish):
        strengths.append("Tendencia técnica de medio plazo favorable")
    if t.trend_long in (Trend.bullish, Trend.strong_bullish):
        strengths.append("Estructura técnica de largo plazo alcista")

    return strengths[:8]  # máximo 8


def _extract_weaknesses(
    f: FundamentalAnalysis,
    v: ValuationAnalysis,
    t: TechnicalAnalysis,
) -> list[str]:
    weaknesses = []

    weaknesses.extend(f.quality_score.warnings[:2])
    weaknesses.extend(f.financial_strength_score.warnings[:2])
    weaknesses.extend(f.growth_score.warnings[:2])

    if v.valuation_level == ValuationLevel.overvalued:
        weaknesses.append("Precio significativamente por encima del valor razonable estimado")
    elif v.valuation_level == ValuationLevel.demanding:
        weaknesses.append("Valoración exigente — margen de seguridad reducido")

    if t.trend_medium in (Trend.bearish, Trend.strong_bearish):
        weaknesses.append("Tendencia técnica bajista — el precio cotiza por debajo de sus medias")
    if t.is_extended:
        weaknesses.append("Precio sobreextendido respecto a medias móviles")

    return weaknesses[:8]


def _worth_following_reason(
    score: float,
    f: FundamentalAnalysis,
    v: ValuationAnalysis,
) -> str:
    if score >= 7:
        return f"Score global de {score:.1f}/10. Empresa de calidad con condiciones favorables."
    elif score >= 5.5:
        return f"Score de {score:.1f}/10. Merece seguimiento pero no urgencia de acción."
    elif score >= 4:
        return f"Score de {score:.1f}/10. Seguimiento solo si hay mejora de fundamentos o valoración."
    else:
        return f"Score de {score:.1f}/10. Demasiadas señales negativas simultáneas para dedicarle recursos."


def _price_attractiveness(
    v: ValuationAnalysis,
    t: TechnicalAnalysis,
    f: FundamentalAnalysis,
) -> tuple[bool | None, str]:
    if v.valuation_level == ValuationLevel.insufficient_data:
        return None, "No hay suficientes datos de valoración para emitir opinión sobre el precio."

    is_cheap = v.valuation_level in (ValuationLevel.undervalued, ValuationLevel.fair)
    tech_ok = t.trend_medium not in (Trend.bearish, Trend.strong_bearish)
    not_extended = not t.is_extended
    fund_ok = f.overall_fundamental_score >= 6.0

    if is_cheap and fund_ok and tech_ok and not_extended:
        return True, "El precio está en zona atractiva: descuento respecto al valor razonable, fundamentales sólidos y tendencia no adversa."
    elif is_cheap and fund_ok and (not tech_ok or t.is_extended):
        return False, (
            "La valoración sugiere precio razonable, pero el contexto técnico no acompaña todavía. "
            "El precio podría serlo más adelante si hay retroceso o corrección técnica."
        )
    elif not is_cheap and fund_ok:
        return False, (
            f"La empresa es de calidad, pero cotiza con prima ('{v.valuation_level.value}'). "
            f"El precio sería más atractivo cerca de {v.fair_value_base:.2f} o por debajo."
            if v.fair_value_base else
            "La empresa es de calidad pero el precio refleja ya altas expectativas."
        )
    else:
        return False, "La combinación actual de valoración y fundamentales no presenta una oportunidad clara."
