"""
Scanner scoring engine — TuForoDeBolsa methodology.

Technical scoring (0-100) with detailed point breakdown per indicator family.
Fundamental scoring (0-100) based on key valuation and quality metrics.
Combined scoring weighted by portfolio type.
Final EntrySignal generation with confidence and justification.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..models.scanner_models import (
    ChartPatternScore,
    EntrySignal,
    FibonacciScore,
    FundamentalLevel,
    FundamentalScoreDetail,
    IchimokuScore,
    KoncordeScore,
    MAScore,
    MACDScore,
    MarketContext,
    MarketRegime,
    MarketStrength,
    PortfolioType,
    RSIScore,
    SRScore,
    SectorContext,
    SectorStatus,
    StochasticScore,
    TechnicalScoreDetail,
    TechnicalSignalLevel,
    VolumeScore,
)

logger = logging.getLogger(__name__)

# Maximum raw points per indicator family (for normalisation denominator)
MAX_POINTS_MA = 7       # +3 aligned + +2 golden cross + +2 bounce
MAX_POINTS_MACD = 7     # +2 cross + +1 hist + +3 divergence + +1 positive
MAX_POINTS_RSI = 5      # +2 oversold bounce + +1 momentum + +2 divergence
MAX_POINTS_STOCH = 3    # +2 cross oversold + +1 rising
MAX_POINTS_VOLUME = 6   # +3 breakout + +2 bullish high vol + +1 obv rising
MAX_POINTS_SR = 5       # +2 near support + +3 breakout
MAX_POINTS_FIB = 3      # +3 fib zone + rsi + macd combo
MAX_POINTS_KONCORDE = 3 # +3 accumulation
MAX_POINTS_ICHIMOKU = 7 # +2 above cloud + +2 tk cross + +3 kumo breakout
MAX_POINTS_PATTERNS = 9 # +3 each pattern


def score_moving_averages(ma: dict) -> MAScore:
    result = MAScore(
        sma20=ma.get("sma20"),
        sma50=ma.get("sma50"),
        sma200=ma.get("sma200"),
        golden_cross=ma.get("golden_cross", False),
        death_cross=ma.get("death_cross", False),
        aligned_bullish=ma.get("aligned_bullish", False),
    )
    pts = 0
    details: list[str] = []

    if ma.get("aligned_bullish"):
        pts += 3
        details.append("+3: Alineación alcista perfecta (precio > SMA20 > SMA50 > SMA200)")
    elif ma.get("sma200") and ma.get("current", 0) > ma.get("sma200", 0):
        pts += 1
        details.append("+1: Precio por encima de SMA200")

    if ma.get("golden_cross"):
        pts += 2
        details.append("+2: Golden Cross (SMA50 cruza SMA200 al alza)")
    elif ma.get("death_cross"):
        pts -= 2
        details.append("-2: Death Cross (SMA50 cruza SMA200 a la baja)")

    if ma.get("bounce_sma20"):
        pts += 1
        details.append("+1: Rebote en SMA20 tras pullback")
    if ma.get("bounce_sma50"):
        pts += 2
        details.append("+2: Rebote en SMA50 tras pullback")

    if ma.get("below_sma200"):
        pts -= 2
        details.append("-2: Precio por debajo de SMA200 — tendencia bajista")

    result.raw_points = pts
    result.details = details
    return result


def score_macd(macd: dict) -> MACDScore:
    result = MACDScore(
        macd_line=macd.get("macd_line"),
        signal_line=macd.get("signal_line"),
        histogram=macd.get("histogram"),
        bullish_cross=macd.get("bullish_cross", False),
        bullish_divergence=macd.get("bullish_divergence", False),
    )
    if not macd.get("available"):
        return result

    pts = 0
    details: list[str] = []

    if macd.get("bullish_cross"):
        pts += 2
        details.append("+2: Cruce alcista MACD sobre Signal line")
    elif macd.get("bearish_cross"):
        pts -= 1
        details.append("-1: Cruce bajista MACD bajo Signal line")

    if macd.get("hist_growing"):
        pts += 1
        details.append("+1: Histograma MACD creciente y positivo")

    if macd.get("bullish_divergence"):
        pts += 3
        details.append("+3: Divergencia alcista MACD detectada")

    if macd.get("macd_positive"):
        pts += 1
        details.append("+1: MACD en zona positiva")

    result.raw_points = pts
    result.details = details
    return result


def score_rsi(rsi: dict) -> RSIScore:
    result = RSIScore(rsi=rsi.get("rsi"))
    if not rsi.get("available"):
        return result

    pts = 0
    details: list[str] = []
    alerts: list[str] = []

    if rsi.get("oversold_bounce"):
        pts += 2
        details.append("+2: RSI < 30 (sobreventa) con rebote")
    elif rsi.get("oversold"):
        pts += 1
        details.append("+1: RSI en zona de sobreventa (<30)")

    if rsi.get("momentum_good"):
        pts += 1
        details.append("+1: RSI entre 50-60 y subiendo — momento favorable")

    if rsi.get("bullish_divergence"):
        pts += 2
        details.append("+2: Divergencia alcista RSI detectada")

    if rsi.get("overbought"):
        pts -= 1
        details.append("-1: RSI > 70 (sobrecompra) — precaución en entradas")
        alerts.append(f"RSI sobrecomprado ({rsi.get('rsi', '?')})")

    result.raw_points = pts
    result.details = details
    return result


def score_stochastic(stoch: dict) -> StochasticScore:
    result = StochasticScore(
        pct_k=stoch.get("pct_k"),
        pct_d=stoch.get("pct_d"),
        bullish_cross_oversold=stoch.get("bullish_cross_oversold", False),
    )
    if not stoch.get("available"):
        return result

    pts = 0
    details: list[str] = []

    if stoch.get("bullish_cross_oversold"):
        pts += 2
        details.append("+2: Cruce alcista %K sobre %D en zona de sobreventa (<30)")

    if stoch.get("both_rising_from_low"):
        pts += 1
        details.append("+1: %K y %D subiendo desde zona baja — momentum positivo")

    if stoch.get("overbought"):
        pts -= 1
        details.append("-1: Estocástico sobrecomprado (>80) — precaución")

    result.raw_points = pts
    result.details = details
    return result


def score_volume(vol: dict, ma: dict) -> VolumeScore:
    result = VolumeScore(
        vol_ratio_20d=vol.get("vol_ratio_20d"),
        obv_trend=vol.get("obv_trend"),
    )
    if not vol.get("available"):
        return result

    pts = 0
    details: list[str] = []

    if vol.get("breakout_confirmed"):
        pts += 3
        details.append("+3: Rotura de resistencia con volumen > 150% de la media")
    elif vol.get("bullish_high_vol"):
        pts += 2
        details.append("+2: Día alcista fuerte con volumen alto — acumulación")

    if vol.get("obv_trend") == "rising":
        pts += 1
        details.append("+1: OBV en tendencia alcista — presión compradora")
    elif vol.get("obv_trend") == "falling":
        pts -= 1
        details.append("-1: OBV cayendo — presión vendedora")

    if vol.get("declining_vol_in_uptrend"):
        pts -= 1
        details.append("-1: Tendencia alcista con volumen decreciente — debilidad")

    vol_ratio = vol.get("vol_ratio_20d", 1.0) or 1.0
    if vol_ratio < 0.5:
        pts -= 1
        details.append("-1: Volumen anémico — movimientos poco fiables")

    result.raw_points = pts
    result.details = details
    return result


def score_support_resistance(sr: dict) -> SRScore:
    result = SRScore(
        support_1=sr.get("support_1"),
        resistance_1=sr.get("resistance_1"),
        near_support=sr.get("near_support", False),
        breakout=sr.get("breakout", False),
    )
    pts = 0
    details: list[str] = []

    if sr.get("breakout"):
        pts += 3
        details.append("+3: Precio rompiendo resistencia clave")
    elif sr.get("near_support"):
        pts += 2
        details.append("+2: Precio rebotando en soporte clave confirmado")

    dist = sr.get("distance_to_resistance_pct")
    if dist is not None and dist > 20:
        pts -= 1
        details.append(f"-1: Precio muy alejado de resistencia más cercana (+{dist:.1f}%) — riesgo elevado")

    result.raw_points = pts
    result.details = details
    return result


def score_fibonacci(fib: dict, rsi: dict, macd: dict) -> FibonacciScore:
    result = FibonacciScore(
        fib_236=fib.get("fib_236"),
        fib_382=fib.get("fib_382"),
        fib_500=fib.get("fib_500"),
        fib_618=fib.get("fib_618"),
        in_fib_zone=fib.get("in_fib_zone", False),
        fib_zone_label=fib.get("fib_zone_label"),
    )
    if not fib.get("available"):
        return result

    pts = 0
    details: list[str] = []

    if fib.get("in_fib_zone"):
        zone = fib.get("fib_zone_label", "?")
        # High-quality signal: Fib zone + RSI < 50 + MACD recovering
        rsi_ok = rsi.get("available") and (rsi.get("rsi", 100) or 100) < 50
        macd_ok = macd.get("available") and (
            macd.get("bullish_cross") or macd.get("hist_growing")
        )
        if rsi_ok and macd_ok:
            pts += 3
            details.append(f"+3: En zona Fibonacci {zone} + RSI < 50 + MACD recuperando — señal de alta calidad")
        else:
            pts += 1
            details.append(f"+1: Precio en zona Fibonacci {zone}")

    result.raw_points = pts
    result.details = details
    return result


def score_koncorde(konc: dict) -> KoncordeScore:
    result = KoncordeScore(
        obv_trend=konc.get("obv_trend"),
        institutional_accumulation=konc.get("institutional_accumulation", False),
        distribution_signal=konc.get("distribution_signal", False),
    )
    if not konc.get("available"):
        return result

    pts = 0
    details: list[str] = []

    if konc.get("institutional_accumulation"):
        pts += 3
        details.append("+3: Koncorde: acumulación institucional detectada (OBV sube mientras precio consolida)")

    if konc.get("distribution_signal"):
        pts -= 2
        details.append("-2: Koncorde: señal de distribución (OBV diverge negativamente con precio)")

    result.raw_points = pts
    result.details = details
    return result


def score_ichimoku(ichi: dict, enabled: bool) -> IchimokuScore:
    result = IchimokuScore(enabled=enabled)
    if not enabled or not ichi.get("available"):
        return result

    pts = 0
    details: list[str] = []

    if ichi.get("price_above_cloud"):
        pts += 2
        details.append("+2: Precio sobre la nube Ichimoku — tendencia alcista confirmada")

    if ichi.get("tk_cross_bullish"):
        pts += 2
        details.append("+2: Cruce TK alcista (Tenkan sobre Kijun)")

    if ichi.get("kumo_breakout"):
        pts += 3
        details.append("+3: Kumo Breakout — precio rompe la nube al alza")

    result.raw_points = pts
    result.details = details
    return result


def score_chart_patterns(patterns: dict) -> ChartPatternScore:
    result = ChartPatternScore(
        double_bottom=patterns.get("double_bottom", False),
        inv_head_shoulders=patterns.get("inv_head_shoulders", False),
        bull_flag=patterns.get("bull_flag", False),
    )
    pts = 0
    details: list[str] = []

    if patterns.get("double_bottom"):
        pts += 3
        details.append("+3: Patrón Doble Suelo detectado — señal alcista")

    if patterns.get("inv_head_shoulders"):
        pts += 3
        details.append("+3: Hombro-Cabeza-Hombro Invertido detectado — patrón de suelo")

    if patterns.get("bull_flag"):
        pts += 2
        details.append("+2: Bandera Alcista / Pennant detectada — señal de continuación")

    result.raw_points = pts
    result.details = details
    return result


def build_technical_score(
    ma: dict,
    macd: dict,
    rsi: dict,
    stoch: dict,
    vol: dict,
    sr: dict,
    fib: dict,
    konc: dict,
    ichi: dict,
    patterns: dict,
    enable_ichimoku: bool = False,
) -> TechnicalScoreDetail:
    ma_score = score_moving_averages(ma)
    macd_score = score_macd(macd)
    rsi_score = score_rsi(rsi)
    stoch_score = score_stochastic(stoch)
    vol_score = score_volume(vol, ma)
    sr_score = score_support_resistance(sr)
    fib_score = score_fibonacci(fib, rsi, macd)
    konc_score = score_koncorde(konc)
    ichi_score = score_ichimoku(ichi, enable_ichimoku)
    pat_score = score_chart_patterns(patterns)

    total_raw = (
        ma_score.raw_points + macd_score.raw_points + rsi_score.raw_points
        + stoch_score.raw_points + vol_score.raw_points + sr_score.raw_points
        + fib_score.raw_points + konc_score.raw_points + ichi_score.raw_points
        + pat_score.raw_points
    )

    # Max possible depends on whether Ichimoku is enabled
    max_pts = (
        MAX_POINTS_MA + MAX_POINTS_MACD + MAX_POINTS_RSI + MAX_POINTS_STOCH
        + MAX_POINTS_VOLUME + MAX_POINTS_SR + MAX_POINTS_FIB + MAX_POINTS_KONCORDE
        + MAX_POINTS_PATTERNS
    )
    if enable_ichimoku:
        max_pts += MAX_POINTS_ICHIMOKU

    # Normalise: shift raw to 0-based before normalising
    # Minimum possible is heavily negative; use empirical floor of -12
    min_pts = -12
    adjusted = total_raw - min_pts
    adjusted_max = max_pts - min_pts
    normalized = max(0.0, min(100.0, adjusted / adjusted_max * 100))

    if normalized >= 76:
        signal_level = TechnicalSignalLevel.muy_fuerte
    elif normalized >= 56:
        signal_level = TechnicalSignalLevel.fuerte
    elif normalized >= 31:
        signal_level = TechnicalSignalLevel.moderada
    else:
        signal_level = TechnicalSignalLevel.debil

    # Collect global alerts
    alerts: list[str] = []
    if rsi.get("overbought"):
        alerts.append(f"RSI sobrecomprado ({rsi.get('rsi', '?')})")
    if stoch.get("overbought"):
        alerts.append("Estocástico sobrecomprado")
    if vol.get("declining_vol_in_uptrend"):
        alerts.append("Volumen decreciente en tendencia alcista")
    if konc.get("distribution_signal"):
        alerts.append("Señal de distribución institucional (Koncorde)")
    if ma.get("death_cross"):
        alerts.append("Death Cross reciente")
    if ma.get("below_sma200"):
        alerts.append("Precio por debajo de SMA200")

    return TechnicalScoreDetail(
        total_raw_points=total_raw,
        max_possible_points=max_pts,
        normalized_score=round(normalized, 1),
        signal_level=signal_level,
        ma=ma_score,
        macd=macd_score,
        rsi=rsi_score,
        stochastic=stoch_score,
        volume=vol_score,
        support_resistance=sr_score,
        fibonacci=fib_score,
        koncorde=konc_score,
        ichimoku=ichi_score,
        patterns=pat_score,
        alerts=alerts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental scoring
# ─────────────────────────────────────────────────────────────────────────────

def build_fundamental_score(fundamentals: Any) -> FundamentalScoreDetail:
    """
    Score fundamentals based on the mega-prompt table.
    `fundamentals` is a yfinance ticker.info dict or our internal model.
    """
    info: dict = {}
    if isinstance(fundamentals, dict):
        info = fundamentals
    elif hasattr(fundamentals, "__dict__"):
        info = fundamentals.__dict__

    pts = 0
    details: list[str] = []
    alerts: list[str] = []

    # Helper
    def get(key: str, default=None):
        return info.get(key, default)

    per = get("trailingPE") or get("per") or get("pe_ratio")
    peg = get("pegRatio") or get("peg")
    ps = get("priceToSalesTrailing12Months") or get("ps_ratio")
    pb = get("priceToBook") or get("pb_ratio")
    ev_ebitda = get("enterpriseToEbitda") or get("ev_ebitda")
    rev_growth = get("revenueGrowth") or get("revenue_growth_1y")
    eps_growth = get("earningsGrowth") or get("eps_growth_yoy") or get("eps_growth_1y")
    roe = get("returnOnEquity") or get("roe")
    debt_eq = get("debtToEquity") or get("debt_to_equity")
    fcf = get("freeCashflow") or get("free_cash_flow")
    fcf_yield = get("fcfYield") or get("fcf_yield")
    eps_surprise = get("earningsSurprise") or get("eps_surprise")
    analyst_buy = get("recommendationMean")  # yfinance: 1=Strong Buy, 5=Sell
    num_analyst_opinions = get("numberOfAnalystOpinions") or 0

    # --- PER ---
    if per is not None and per > 0:
        # Without sector avg, use generic thresholds
        if per < 15:
            pts += 2
            details.append(f"+2: PER bajo ({per:.1f}x) — valoración atractiva")
        elif per > 40:
            pts -= 2
            details.append(f"-2: PER elevado ({per:.1f}x) — exigente")
            alerts.append(f"PER alto: {per:.1f}x")

    # --- PEG ---
    if peg is not None and peg > 0:
        if peg < 1:
            pts += 3
            details.append(f"+3: PEG < 1 ({peg:.2f}) — infravalorado vs crecimiento")
        elif peg < 1.5:
            pts += 1
            details.append(f"+1: PEG entre 1-1.5 ({peg:.2f}) — razonable")
        elif peg > 3:
            pts -= 1
            details.append(f"-1: PEG alto ({peg:.2f}) — crecimiento caro")

    # --- Revenue growth ---
    if rev_growth is not None:
        if rev_growth > 0.20:
            pts += 3
            details.append(f"+3: Crecimiento de ingresos > 20% ({rev_growth*100:.1f}%)")
        elif rev_growth > 0.10:
            pts += 2
            details.append(f"+2: Crecimiento de ingresos > 10% ({rev_growth*100:.1f}%)")
        elif rev_growth > 0:
            pts += 1
            details.append(f"+1: Crecimiento de ingresos positivo ({rev_growth*100:.1f}%)")
        elif rev_growth < 0:
            pts -= 2
            details.append(f"-2: Ingresos en declive ({rev_growth*100:.1f}%)")
            alerts.append("Ingresos decrecientes")

    # --- EPS growth ---
    if eps_growth is not None:
        if eps_growth > 0.20:
            pts += 3
            details.append(f"+3: Crecimiento EPS > 20% ({eps_growth*100:.1f}%)")
        elif eps_growth > 0.10:
            pts += 2
            details.append(f"+2: Crecimiento EPS > 10% ({eps_growth*100:.1f}%)")
        elif eps_growth < 0:
            pts -= 1
            details.append(f"-1: EPS en declive ({eps_growth*100:.1f}%)")

    # --- ROE ---
    if roe is not None and roe > 0:
        if roe > 0.20:
            pts += 2
            details.append(f"+2: ROE > 20% ({roe*100:.1f}%) — alta rentabilidad")
        elif roe > 0.10:
            pts += 1
            details.append(f"+1: ROE positivo ({roe*100:.1f}%)")

    # --- Debt/EBITDA proxy (Debt/Equity as fallback) ---
    debt_ebitda: float | None = None
    ebitda = get("ebitda")
    total_debt = get("totalDebt")
    if total_debt and ebitda and ebitda > 0:
        debt_ebitda = total_debt / ebitda
    elif debt_eq is not None:
        # rough approximation: debt_eq / 10
        debt_ebitda = debt_eq / 100 * 3 if debt_eq else None

    if debt_ebitda is not None:
        if debt_ebitda < 1.5:
            pts += 2
            details.append(f"+2: Deuda/EBITDA bajo ({debt_ebitda:.1f}x) — balance saneado")
        elif debt_ebitda > 4:
            pts -= 3
            details.append(f"-3: Deuda/EBITDA elevado ({debt_ebitda:.1f}x) — riesgo financiero")
            alerts.append(f"Deuda alta: {debt_ebitda:.1f}x EBITDA")
        elif debt_ebitda > 3:
            pts -= 1
            details.append(f"-1: Deuda/EBITDA moderada-alta ({debt_ebitda:.1f}x)")

    # --- Free Cash Flow ---
    if fcf is not None:
        if fcf > 0:
            pts += 2
            details.append("+2: Free Cash Flow positivo")
        else:
            pts -= 1
            details.append("-1: Free Cash Flow negativo")
            alerts.append("FCF negativo")

    # --- EPS surprise ---
    if eps_surprise is not None and eps_surprise > 0.10:
        pts += 2
        details.append(f"+2: Sorpresa positiva de BPA > 10% ({eps_surprise*100:.1f}%)")

    # --- Analyst consensus (yfinance recommendationMean: 1=Strong Buy, 5=Sell) ---
    if analyst_buy is not None:
        if analyst_buy <= 1.8:
            pts += 2
            details.append(f"+2: Consenso analistas muy positivo ({analyst_buy:.1f}/5)")
        elif analyst_buy >= 4.0:
            pts -= 3
            details.append(f"-3: Consenso analistas negativo ({analyst_buy:.1f}/5 → mayoría Sell)")
            alerts.append("Consenso analistas negativo")

    # Normalise
    min_raw = -10
    max_raw = 20
    adjusted = pts - min_raw
    adjusted_max = max_raw - min_raw
    normalized = max(0.0, min(100.0, adjusted / adjusted_max * 100))

    if normalized >= 76:
        level = FundamentalLevel.excelente
    elif normalized >= 56:
        level = FundamentalLevel.solido
    elif normalized >= 31:
        level = FundamentalLevel.moderado
    else:
        level = FundamentalLevel.debil

    return FundamentalScoreDetail(
        total_raw_points=pts,
        max_possible_points=max_raw,
        normalized_score=round(normalized, 1),
        level=level,
        details=details,
        alerts=alerts,
        per=round(float(per), 2) if per else None,
        peg=round(float(peg), 2) if peg else None,
        ps_ratio=round(float(ps), 2) if ps else None,
        pb_ratio=round(float(pb), 2) if pb else None,
        ev_ebitda=round(float(ev_ebitda), 2) if ev_ebitda else None,
        revenue_growth_yoy=round(float(rev_growth), 4) if rev_growth is not None else None,
        eps_growth_yoy=round(float(eps_growth), 4) if eps_growth is not None else None,
        roe=round(float(roe), 4) if roe is not None else None,
        debt_ebitda=round(float(debt_ebitda), 2) if debt_ebitda else None,
        fcf_positive=fcf > 0 if fcf is not None else None,
        analyst_buy_pct=round(float(analyst_buy), 2) if analyst_buy else None,
        eps_surprise_pct=round(float(eps_surprise), 4) if eps_surprise else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Combined scoring & final signal
# ─────────────────────────────────────────────────────────────────────────────

def compute_combined_score(
    tech_score: float,
    fund_score: float,
    market: MarketContext,
    sector: SectorContext,
    portfolio_type: PortfolioType,
    override_sector_filter: bool = False,
) -> tuple[float, float, EntrySignal, str]:
    """
    Returns (combined_score, confidence, entry_signal, justification).

    Portfolio weights:
      swing:  Technical 60% + Fundamental 40%
      etf:    Fundamental 70% + Technical 30%

    Market context multiplier:
      ALCISTA_strong:  1.15
      ALCISTA_moderate: 1.08
      ALCISTA_weak:    1.02
      LATERAL:         1.00
      BAJISTA_weak:    0.85
      BAJISTA_moderate: 0.75
      BAJISTA_strong:  0.60

    Sector filter:
      REZAGADO: −10 points penalty (unless override)
    """
    if portfolio_type == PortfolioType.etf:
        tech_w, fund_w = 0.30, 0.70
    else:
        tech_w, fund_w = 0.60, 0.40

    base_score = tech_score * tech_w + fund_score * fund_w

    # Market multiplier
    mult_map = {
        (MarketRegime.alcista, MarketStrength.strong): 1.15,
        (MarketRegime.alcista, MarketStrength.moderate): 1.08,
        (MarketRegime.alcista, MarketStrength.weak): 1.02,
        (MarketRegime.lateral, MarketStrength.strong): 1.00,
        (MarketRegime.lateral, MarketStrength.moderate): 1.00,
        (MarketRegime.lateral, MarketStrength.weak): 0.95,
        (MarketRegime.bajista, MarketStrength.weak): 0.85,
        (MarketRegime.bajista, MarketStrength.moderate): 0.75,
        (MarketRegime.bajista, MarketStrength.strong): 0.60,
    }
    mult = mult_map.get((market.regime, market.strength), 1.0)
    adjusted = base_score * mult

    # Sector adjustment
    sector_penalty = 0.0
    if sector.status == SectorStatus.rezagado and not override_sector_filter:
        sector_penalty = 10.0
        adjusted = max(0.0, adjusted - sector_penalty)
    elif sector.status == SectorStatus.lider:
        adjusted = min(100.0, adjusted + 3.0)

    combined = max(0.0, min(100.0, adjusted))

    # Confidence: based on market strength, data availability
    data_factor = 1.0 if fund_score > 0 else 0.7
    market_factor = {
        MarketRegime.alcista: 0.9,
        MarketRegime.lateral: 0.75,
        MarketRegime.bajista: 0.6,
    }.get(market.regime, 0.75)
    confidence = min(1.0, data_factor * market_factor)

    # Entry signal thresholds
    if combined >= 81:
        signal = EntrySignal.fuerte
    elif combined >= 66:
        signal = EntrySignal.recomendada
    elif combined >= 51:
        signal = EntrySignal.posible
    elif combined >= 35:
        signal = EntrySignal.esperar
    else:
        signal = EntrySignal.no_entrar

    # Justification
    weight_label = "60% técnico / 40% fundamental" if portfolio_type == PortfolioType.swing else "30% técnico / 70% fundamental"
    mult_pct = (mult - 1) * 100
    mult_str = f"{mult_pct:+.0f}%" if mult != 1.0 else "neutro"

    just_parts = [
        f"Score técnico: {tech_score:.0f}/100 | Score fundamental: {fund_score:.0f}/100",
        f"Ponderación: {weight_label}",
        f"Mercado: {market.regime.value} ({market.strength.name}) → multiplicador {mult_str}",
        f"Sector: {sector.status.value}",
    ]
    if sector_penalty > 0:
        just_parts.append(f"Penalización sector rezagado: −{sector_penalty:.0f} puntos")
    just_parts.append(f"Score combinado: {combined:.1f}/100 → {signal.value}")

    return round(combined, 1), round(confidence, 2), signal, " | ".join(just_parts)
