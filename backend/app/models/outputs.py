"""
Modelos de salida. Representan el resultado completo del análisis.
Toda puntuación incluye su explicación para evitar cajas negras.
"""
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums de estado
# ─────────────────────────────────────────────────────────────────────────────

class Trend(str, Enum):
    strong_bullish = "strong_bullish"
    bullish = "bullish"
    neutral = "neutral"
    bearish = "bearish"
    strong_bearish = "strong_bearish"


class ValuationLevel(str, Enum):
    undervalued = "undervalued"
    fair = "fair"
    demanding = "demanding"
    overvalued = "overvalued"
    insufficient_data = "insufficient_data"


class SignalQuality(str, Enum):
    strong = "strong"
    moderate = "moderate"
    weak = "weak"
    conflicting = "conflicting"


# ─────────────────────────────────────────────────────────────────────────────
# Bloques de análisis fundamental
# ─────────────────────────────────────────────────────────────────────────────

class GrowthMetrics(BaseModel):
    revenue_growth_1y: float | None = Field(None, description="Crecimiento ingresos 1 año")
    revenue_cagr_3y: float | None = Field(None, description="CAGR ingresos 3 años")
    eps_growth_1y: float | None = Field(None, description="Crecimiento EPS 1 año")
    eps_cagr_3y: float | None = Field(None, description="CAGR EPS 3 años")
    fcf_growth_1y: float | None = None


class MarginMetrics(BaseModel):
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    ebitda_margin: float | None = None
    gross_margin_trend: str | None = Field(None, description="improving / stable / deteriorating")
    operating_margin_trend: str | None = None


class EfficiencyMetrics(BaseModel):
    roe: float | None = None
    roic: float | None = None
    roa: float | None = None
    asset_turnover: float | None = None


class DebtMetrics(BaseModel):
    net_debt_ebitda: float | None = None
    interest_coverage: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    cash_and_equivalents: float | None = None


class CashFlowMetrics(BaseModel):
    free_cash_flow: float | None = None
    fcf_yield: float | None = None
    fcf_consistency: str | None = Field(None, description="consistent / variable / negative")
    operating_cash_flow: float | None = None
    capex_to_revenue: float | None = None


class ShareholderMetrics(BaseModel):
    shares_outstanding_change_1y: float | None = Field(
        None, description="Positivo = dilución, negativo = recompra"
    )
    buyback_yield: float | None = None
    dividend_yield: float | None = None


class ScoredSection(BaseModel):
    """Sección con puntuación (0–10) y justificación."""
    score: float = Field(..., ge=0, le=10)
    label: str  # e.g. "Bueno (7.2/10)"
    factors: list[str] = Field(default_factory=list, description="Factores que impulsan la puntuación")
    warnings: list[str] = Field(default_factory=list, description="Factores negativos o alertas")


class FundamentalAnalysis(BaseModel):
    growth: GrowthMetrics
    margins: MarginMetrics
    efficiency: EfficiencyMetrics
    debt: DebtMetrics
    cash_flow: CashFlowMetrics
    shareholders: ShareholderMetrics

    quality_score: ScoredSection
    financial_strength_score: ScoredSection
    growth_score: ScoredSection
    efficiency_score: ScoredSection

    overall_fundamental_score: float = Field(..., ge=0, le=10)
    narrative: str = Field(..., description="Explicación en lenguaje natural")
    data_gaps: list[str] = Field(default_factory=list, description="Datos que faltaron")


# ─────────────────────────────────────────────────────────────────────────────
# Bloque de valoración
# ─────────────────────────────────────────────────────────────────────────────

class ValuationMultiple(BaseModel):
    name: str
    current: float | None = None
    historical_avg: float | None = None
    sector_avg: float | None = None
    implied_fair_price: float | None = None
    interpretation: str | None = None


class DCFResult(BaseModel):
    fair_value: float | None = None
    growth_rate_used: float
    discount_rate_used: float
    terminal_growth_used: float
    projection_years: int
    sensitivity_low: float | None = None   # con growth -2pp
    sensitivity_high: float | None = None  # con growth +2pp
    assumptions_note: str


class ValuationAnalysis(BaseModel):
    current_price: float
    multiples: list[ValuationMultiple]
    dcf: DCFResult | None

    fair_value_conservative: float | None = None
    fair_value_base: float | None = None
    fair_value_optimistic: float | None = None

    discount_to_base: float | None = Field(None, description="Descuento(+) o prima(-) vs. precio actual")
    valuation_level: ValuationLevel
    dominant_method: str = Field(..., description="Qué método pesa más y por qué")
    narrative: str
    data_gaps: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Bloque técnico
# ─────────────────────────────────────────────────────────────────────────────

class MovingAverages(BaseModel):
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    ema_20: float | None = None
    price_vs_sma50_pct: float | None = None
    price_vs_sma200_pct: None | float = None


class MomentumIndicators(BaseModel):
    rsi_14: float | None = None
    rsi_interpretation: str | None = None  # overbought / oversold / neutral
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    macd_cross: str | None = None  # bullish_cross / bearish_cross / none


class VolatilityMetrics(BaseModel):
    atr_14: float | None = None
    atr_pct: float | None = Field(None, description="ATR como % del precio")
    historical_volatility_30d: float | None = None
    beta: float | None = None


class SupportResistance(BaseModel):
    support_1: float | None = None
    support_2: float | None = None
    resistance_1: float | None = None
    resistance_2: float | None = None
    method: str = "swing_highs_lows"


class TechnicalAnalysis(BaseModel):
    trend_short: Trend
    trend_medium: Trend
    trend_long: Trend

    moving_averages: MovingAverages
    momentum: MomentumIndicators
    volatility: VolatilityMetrics
    support_resistance: SupportResistance

    is_extended: bool = Field(..., description="True si el precio está sobreextendido vs. medias")
    extension_note: str | None = None
    volume_trend: str | None = None  # increasing / decreasing / neutral
    relative_strength_vs_spy: float | None = Field(None, description="RS vs. SPY últimos 3 meses")

    signal_quality: SignalQuality
    narrative: str
    data_gaps: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Bloque entrada/salida
# ─────────────────────────────────────────────────────────────────────────────

class PriceZone(BaseModel):
    low: float
    high: float
    rationale: str


class EntryExitPlan(BaseModel):
    current_price: float

    entry_conservative: PriceZone
    entry_reasonable: PriceZone
    entry_aggressive: PriceZone

    stop_technical: float = Field(..., description="Stop táctico basado en soporte + ATR")
    stop_technical_rationale: str

    invalidation_fundamental: str = Field(
        ..., description="Qué cambio en fundamentales invalida la tesis"
    )
    invalidation_structural: str = Field(
        ..., description="Señal de deterioro estructural del negocio"
    )

    target_partial: float = Field(..., description="Objetivo para reducción parcial")
    target_partial_rationale: str
    target_full_or_review: float = Field(..., description="Objetivo de salida completa o revisión obligatoria")
    target_full_rationale: str

    risk_reward_conservative: float | None = Field(None, description="R/R desde entrada conservadora")
    risk_reward_reasonable: float | None = Field(None, description="R/R desde entrada razonable")

    immediate_action: str = Field(
        ...,
        description="Qué hacer ahora: esperar retroceso / entrada escalonada / vigilar ruptura / no invertir",
    )
    conditions_for_entry: list[str] = Field(
        default_factory=list,
        description="Condiciones concretas que deben darse para entrar",
    )
    conditions_for_exit: list[str] = Field(
        default_factory=list,
        description="Condiciones concretas para salir antes de los objetivos",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bloque de riesgo
# ─────────────────────────────────────────────────────────────────────────────

class RiskFactor(BaseModel):
    name: str
    severity: str  # low / medium / high / critical
    description: str


class RiskAnalysis(BaseModel):
    overall_risk_level: str  # low / moderate / elevated / high
    downside_to_support: float | None = Field(None, description="% bajada al soporte más cercano")
    downside_to_fair_value_low: float | None = None
    multiple_compression_risk: str | None = None  # low / medium / high

    risk_factors: list[RiskFactor]
    position_size_note: str = Field(
        ...,
        description="Orientación sobre tamaño de posición según volatilidad y perfil",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Síntesis final
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisSynthesis(BaseModel):
    overall_score: float = Field(..., ge=0, le=10)
    score_breakdown: dict[str, float] = Field(
        ..., description="Desglose de scores por módulo con sus pesos"
    )

    investment_thesis: str
    key_strengths: list[str]
    key_weaknesses: list[str]
    key_risks: list[str]

    worth_following: bool
    worth_following_reason: str

    current_price_attractive: bool | None = Field(
        None, description="None si no hay suficiente info para opinar"
    )
    attractiveness_conditions: str

    weighting_note: str = Field(
        ..., description="Explicación de cómo se ponderaron los módulos según perfil"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Respuesta completa
# ─────────────────────────────────────────────────────────────────────────────

class CompanyInfo(BaseModel):
    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str | None = None
    market_cap: float | None = None
    current_price: float | None = None
    price_52w_high: float | None = None
    price_52w_low: float | None = None
    description: str | None = None


class AnalysisResponse(BaseModel):
    """Respuesta completa de análisis. Todos los campos son trazables a sus fuentes."""

    request_id: str
    timestamp: str
    ticker: str
    company_info: CompanyInfo

    fundamental: FundamentalAnalysis
    valuation: ValuationAnalysis
    technical: TechnicalAnalysis
    entry_exit: EntryExitPlan
    risk: RiskAnalysis
    synthesis: AnalysisSynthesis

    # Metadatos de transparencia
    data_provider: str
    analysis_params: dict[str, Any]
    disclaimers: list[str] = Field(
        default_factory=lambda: [
            "Este análisis es una herramienta de apoyo a la decisión, no asesoramiento financiero.",
            "Los niveles de entrada/salida son probabilísticos, no predicciones.",
            "Los supuestos DCF son estimaciones con incertidumbre inherente.",
            "El análisis técnico describe el pasado, no garantiza el futuro.",
            "Consulta a un asesor financiero antes de tomar decisiones de inversión.",
        ]
    )
