"""
Market Level Analysis — Level 1 of Market → Sector → Value methodology.

Determines overall market health using SPY (S&P 500 ETF) as reference:
- Price vs SMA200 → trend direction
- Golden/Death Cross (SMA50 vs SMA200)
- SMA50 slope over last 10 sessions
- Output: MarketRegime + strength 1-3
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..models.scanner_models import MarketContext, MarketRegime, MarketStrength

logger = logging.getLogger(__name__)


def analyze_market_context(spy_df: pd.DataFrame | None) -> MarketContext:
    """
    Analyse the overall market using SPY OHLCV data.
    Returns a MarketContext with regime and strength level.
    """
    if spy_df is None or spy_df.empty or len(spy_df) < 50:
        return MarketContext(
            regime=MarketRegime.lateral,
            strength=MarketStrength.weak,
            description="Datos de mercado insuficientes (SPY). Análisis de contexto omitido.",
        )

    close = spy_df["close"] if "close" in spy_df.columns else spy_df.iloc[:, 3]
    close = close.dropna()

    if len(close) < 50:
        return MarketContext(
            regime=MarketRegime.lateral,
            strength=MarketStrength.weak,
            description="Serie histórica de SPY demasiado corta.",
        )

    current = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    price_vs_sma200_pct = ((current - sma200) / sma200 * 100) if sma200 else None

    # Golden / Death Cross detection (SMA50 vs SMA200)
    golden_cross = False
    death_cross = False
    if sma200 is not None and len(close) >= 201:
        prev_sma50 = float(close.rolling(50).mean().iloc[-2])
        prev_sma200 = float(close.rolling(200).mean().iloc[-2])
        if prev_sma50 < prev_sma200 and sma50 >= sma200:
            golden_cross = True
        elif prev_sma50 > prev_sma200 and sma50 <= sma200:
            death_cross = True

    # SMA50 slope over last 10 sessions (linear regression slope as % per day)
    sma50_series = close.rolling(50).mean().dropna()
    sma50_slope_10d: float | None = None
    if len(sma50_series) >= 10:
        segment = sma50_series.iloc[-10:]
        x = np.arange(len(segment))
        if segment.mean() != 0:
            coeffs = np.polyfit(x, segment.values, 1)
            sma50_slope_10d = round(float(coeffs[0] / segment.mean() * 100), 3)

    # Score: regime classification
    score = 0

    # Price vs SMA200
    if sma200 is not None:
        if current > sma200:
            score += 2
        else:
            score -= 2

    # Price vs SMA50
    if current > sma50:
        score += 1
    else:
        score -= 1

    # SMA50 vs SMA200
    if sma200 is not None:
        if sma50 > sma200:
            score += 1
        else:
            score -= 1

    # Golden/Death cross bonus
    if golden_cross:
        score += 2
    if death_cross:
        score -= 2

    # SMA50 slope
    if sma50_slope_10d is not None:
        if sma50_slope_10d > 0.05:
            score += 1
        elif sma50_slope_10d < -0.05:
            score -= 1

    # Determine regime
    if score >= 4:
        regime = MarketRegime.alcista
    elif score <= -3:
        regime = MarketRegime.bajista
    else:
        regime = MarketRegime.lateral

    # Strength 1-3
    abs_score = abs(score)
    if abs_score >= 5:
        strength = MarketStrength.strong
    elif abs_score >= 3:
        strength = MarketStrength.moderate
    else:
        strength = MarketStrength.weak

    # Description
    parts: list[str] = []
    if sma200 is not None:
        direction = "por encima" if current > sma200 else "por debajo"
        pct = abs(price_vs_sma200_pct or 0)
        parts.append(f"SPY {direction} de SMA200 ({pct:.1f}%)")
    if sma50 is not None:
        cross_str = "SMA50 > SMA200" if sma50 > (sma200 or 0) else "SMA50 < SMA200"
        parts.append(cross_str)
    if golden_cross:
        parts.append("Golden Cross reciente")
    if death_cross:
        parts.append("Death Cross reciente")
    if sma50_slope_10d is not None:
        slope_word = "positiva" if sma50_slope_10d > 0 else "negativa"
        parts.append(f"Pendiente SMA50 ({sma50_slope_10d:+.3f}%/día) → {slope_word}")

    description = ". ".join(parts) + "."

    return MarketContext(
        regime=regime,
        strength=strength,
        price=round(current, 2),
        sma50=round(sma50, 2),
        sma200=round(sma200, 2) if sma200 else None,
        price_vs_sma200_pct=round(price_vs_sma200_pct, 2) if price_vs_sma200_pct else None,
        golden_cross=golden_cross,
        death_cross=death_cross,
        sma50_slope_10d=sma50_slope_10d,
        description=description,
    )
