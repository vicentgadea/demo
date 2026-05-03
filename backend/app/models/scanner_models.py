"""
Pydantic models for the Stock & ETF Entry Signal Scanner.
Covers: market context, sector analysis, per-ticker scanner results,
combined scoring, and the full scanner response.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────


class MarketRegime(str, Enum):
    alcista = "MERCADO_ALCISTA"
    bajista = "MERCADO_BAJISTA"
    lateral = "MERCADO_LATERAL"


class MarketStrength(int, Enum):
    weak = 1
    moderate = 2
    strong = 3


class SectorStatus(str, Enum):
    lider = "SECTOR_LIDER"
    neutro = "SECTOR_NEUTRO"
    rezagado = "SECTOR_REZAGADO"


class EntrySignal(str, Enum):
    fuerte = "ENTRADA_FUERTE"
    recomendada = "ENTRADA_RECOMENDADA"
    posible = "POSIBLE_ENTRADA"
    esperar = "ESPERAR"
    no_entrar = "NO_ENTRAR"


class TechnicalSignalLevel(str, Enum):
    muy_fuerte = "MUY_FUERTE"
    fuerte = "FUERTE"
    moderada = "MODERADA"
    debil = "DEBIL"


class FundamentalLevel(str, Enum):
    excelente = "EXCELENTE"
    solido = "SÓLIDO"
    moderado = "MODERADO"
    debil = "DÉBIL"


class PortfolioType(str, Enum):
    swing = "swing"      # Portfolio 1 DEGIRO — swing/growth
    etf = "etf"          # Portfolio 2 Trade Republic — Bogleheads ETF


# ── Market Context (Level 1) ──────────────────────────────────────────────────


class MarketContext(BaseModel):
    regime: MarketRegime
    strength: MarketStrength
    price: float | None = None
    sma50: float | None = None
    sma200: float | None = None
    price_vs_sma200_pct: float | None = None
    golden_cross: bool = False
    death_cross: bool = False
    sma50_slope_10d: float | None = None
    description: str = ""


# ── Sector Context (Level 2) ──────────────────────────────────────────────────


class SectorContext(BaseModel):
    sector_name: str
    sector_etf: str | None = None
    status: SectorStatus
    rs_20d: float | None = Field(None, description="Relative strength vs SPY 20d")
    rs_50d: float | None = Field(None, description="Relative strength vs SPY 50d")
    rs_200d: float | None = Field(None, description="Relative strength vs SPY 200d")
    description: str = ""


# ── Technical Sub-scores ──────────────────────────────────────────────────────


class MAScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    sma20: float | None = None
    sma50: float | None = None
    sma200: float | None = None
    golden_cross: bool = False
    death_cross: bool = False
    aligned_bullish: bool = False


class MACDScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    macd_line: float | None = None
    signal_line: float | None = None
    histogram: float | None = None
    bullish_cross: bool = False
    bullish_divergence: bool = False


class RSIScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    rsi: float | None = None


class StochasticScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    pct_k: float | None = None
    pct_d: float | None = None
    bullish_cross_oversold: bool = False


class VolumeScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    vol_ratio_20d: float | None = None
    obv_trend: str | None = None


class SRScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    support_1: float | None = None
    resistance_1: float | None = None
    near_support: bool = False
    breakout: bool = False


class FibonacciScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    fib_236: float | None = None
    fib_382: float | None = None
    fib_500: float | None = None
    fib_618: float | None = None
    in_fib_zone: bool = False
    fib_zone_label: str | None = None


class KoncordeScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    obv_trend: str | None = None
    institutional_accumulation: bool = False
    distribution_signal: bool = False


class IchimokuScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    price_above_cloud: bool | None = None
    tk_cross_bullish: bool = False
    kumo_breakout: bool = False
    enabled: bool = False


class ChartPatternScore(BaseModel):
    raw_points: int = 0
    details: list[str] = Field(default_factory=list)
    double_bottom: bool = False
    inv_head_shoulders: bool = False
    bull_flag: bool = False


class TechnicalScoreDetail(BaseModel):
    total_raw_points: int = 0
    max_possible_points: int = 0
    normalized_score: float = Field(0.0, ge=0, le=100)
    signal_level: TechnicalSignalLevel = TechnicalSignalLevel.debil
    ma: MAScore = Field(default_factory=MAScore)
    macd: MACDScore = Field(default_factory=MACDScore)
    rsi: RSIScore = Field(default_factory=RSIScore)
    stochastic: StochasticScore = Field(default_factory=StochasticScore)
    volume: VolumeScore = Field(default_factory=VolumeScore)
    support_resistance: SRScore = Field(default_factory=SRScore)
    fibonacci: FibonacciScore = Field(default_factory=FibonacciScore)
    koncorde: KoncordeScore = Field(default_factory=KoncordeScore)
    ichimoku: IchimokuScore = Field(default_factory=IchimokuScore)
    patterns: ChartPatternScore = Field(default_factory=ChartPatternScore)
    alerts: list[str] = Field(default_factory=list)


# ── Fundamental Score ─────────────────────────────────────────────────────────


class FundamentalScoreDetail(BaseModel):
    total_raw_points: int = 0
    max_possible_points: int = 0
    normalized_score: float = Field(0.0, ge=0, le=100)
    level: FundamentalLevel = FundamentalLevel.debil
    details: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    per: float | None = None
    peg: float | None = None
    ps_ratio: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    revenue_growth_yoy: float | None = None
    eps_growth_yoy: float | None = None
    roe: float | None = None
    debt_ebitda: float | None = None
    fcf_positive: bool | None = None
    analyst_buy_pct: float | None = None
    eps_surprise_pct: float | None = None


# ── Per-ticker Scanner Result ─────────────────────────────────────────────────


class TickerScanResult(BaseModel):
    ticker: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    current_price: float | None = None
    price_change_pct_1d: float | None = None
    market_cap: float | None = None

    market_context: MarketContext | None = None
    sector_context: SectorContext | None = None
    technical_score: TechnicalScoreDetail = Field(default_factory=TechnicalScoreDetail)
    fundamental_score: FundamentalScoreDetail = Field(default_factory=FundamentalScoreDetail)

    combined_score: float = Field(0.0, ge=0, le=100)
    entry_signal: EntrySignal = EntrySignal.esperar
    confidence: float = Field(0.0, ge=0, le=1, description="0–1 confidence")
    signal_justification: str = ""
    key_alerts: list[str] = Field(default_factory=list)

    error: str | None = None
    data_quality: str = "ok"


# ── Scanner Request / Response ────────────────────────────────────────────────


class ScannerRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=50)
    portfolio_type: PortfolioType = PortfolioType.swing
    enable_ichimoku: bool = False
    override_sector_filter: bool = False
    market_ticker: str = "SPY"
    min_combined_score: float = Field(0.0, ge=0, le=100, description="Filtro mínimo de puntuación")
    only_entry_signals: bool = False


class ScannerResponse(BaseModel):
    request_id: str
    timestamp: str
    market_context: MarketContext
    results: list[TickerScanResult]
    scan_duration_s: float
    tickers_processed: int
    tickers_with_errors: int
    top_opportunities: list[str] = Field(default_factory=list, description="Top 5 tickers by score")
