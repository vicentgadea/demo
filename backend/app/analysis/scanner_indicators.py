"""
Extended technical indicators for the TuForoDeBolsa scanner.

Implements every indicator family listed in the MEGA PROMPT spec:
  A) Moving Averages (SMA 20/50/200, EMA 12/26) + Golden/Death Cross
  B) MACD (12,26,9) + divergence detection
  C) RSI 14 + divergence detection
  D) Stochastic (14,3,3)
  E) Volume (ratio vs 20-day avg + OBV trend)
  F) Support & Resistance (swing highs/lows + Fibonacci levels)
  G) Fibonacci retracements (on last 6-month swing)
  H) Koncorde simulation (OBV-based institutional vs retail)
  I) Ichimoku (Tenkan/Kijun/Senkou/Chikou) — optional
  J) Chart patterns: Double Bottom, Inverted H&S, Bull Flag/Pennant

All functions receive a DataFrame with columns: open, high, low, close, volume.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> float | None:
    if len(s) < n:
        return None
    val = float(s.rolling(n).mean().iloc[-1])
    return None if np.isnan(val) else round(val, 4)


def _ema(s: pd.Series, n: int) -> float | None:
    if len(s) < n:
        return None
    val = float(s.ewm(span=n, adjust=False).mean().iloc[-1])
    return None if np.isnan(val) else round(val, 4)


def _slope_pct_per_day(s: pd.Series) -> float:
    """Linear regression slope as % of mean per day."""
    if len(s) < 3 or s.mean() == 0:
        return 0.0
    x = np.arange(len(s), dtype=float)
    coeffs = np.polyfit(x, s.values, 1)
    return float(coeffs[0] / s.mean() * 100)


def _local_minima(s: pd.Series, window: int = 5) -> list[int]:
    """Return indices of local minima within ±window candles."""
    indices = []
    arr = s.values
    for i in range(window, len(arr) - window):
        if arr[i] == min(arr[i - window: i + window + 1]):
            indices.append(i)
    return indices


def _local_maxima(s: pd.Series, window: int = 5) -> list[int]:
    arr = s.values
    indices = []
    for i in range(window, len(arr) - window):
        if arr[i] == max(arr[i - window: i + window + 1]):
            indices.append(i)
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# A) Moving Averages
# ─────────────────────────────────────────────────────────────────────────────

def calc_moving_averages(df: pd.DataFrame) -> dict:
    close = df["close"]
    current = float(close.iloc[-1])

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)

    # Golden / Death Cross  (SMA50 crossing SMA200)
    golden_cross = death_cross = False
    if len(close) >= 201:
        sma50_series = close.rolling(50).mean()
        sma200_series = close.rolling(200).mean()
        prev50 = float(sma50_series.iloc[-2])
        prev200 = float(sma200_series.iloc[-2])
        curr50 = float(sma50_series.iloc[-1])
        curr200 = float(sma200_series.iloc[-1])
        if prev50 < prev200 and curr50 >= curr200:
            golden_cross = True
        elif prev50 > prev200 and curr50 <= curr200:
            death_cross = True

    # Perfect bullish alignment: price > SMA20 > SMA50 > SMA200
    aligned_bullish = (
        sma20 is not None and sma50 is not None and sma200 is not None
        and current > sma20 > sma50 > sma200
    )

    # Pullback bounce: price recently touched SMA20 or SMA50 from above
    bounce_sma20 = bounce_sma50 = False
    if sma20 is not None and len(close) >= 5:
        lows_5 = df["low"].tail(5)
        if (lows_5 <= sma20 * 1.01).any() and current > sma20:
            bounce_sma20 = True
    if sma50 is not None and len(close) >= 5:
        lows_5 = df["low"].tail(5)
        if (lows_5 <= sma50 * 1.01).any() and current > sma50:
            bounce_sma50 = True

    return {
        "current": current,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "ema12": ema12,
        "ema26": ema26,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "aligned_bullish": aligned_bullish,
        "bounce_sma20": bounce_sma20,
        "bounce_sma50": bounce_sma50,
        "below_sma200": sma200 is not None and current < sma200,
    }


# ─────────────────────────────────────────────────────────────────────────────
# B) MACD
# ─────────────────────────────────────────────────────────────────────────────

def calc_macd(df: pd.DataFrame) -> dict:
    close = df["close"]
    if len(close) < 26:
        return {"available": False}

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    curr_hist = float(histogram.iloc[-1])
    prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0
    curr_macd = float(macd_line.iloc[-1])
    curr_signal = float(signal_line.iloc[-1])

    bullish_cross = curr_hist > 0 and prev_hist <= 0
    bearish_cross = curr_hist < 0 and prev_hist >= 0
    hist_growing = curr_hist > prev_hist
    macd_positive = curr_macd > 0

    # Bullish divergence: price makes lower low, MACD makes higher low (last 30 bars)
    bullish_divergence = False
    if len(close) >= 30:
        seg_close = close.tail(30)
        seg_macd = macd_line.tail(30)
        price_min_idx = seg_close.idxmin()
        macd_min_idx = seg_macd.idxmin()
        if price_min_idx != seg_close.index[-1]:
            # price most recent low vs earlier low
            recent_price_min = float(seg_close.iloc[-5:].min())
            earlier_price_min = float(seg_close.iloc[:20].min())
            recent_macd_min = float(seg_macd.iloc[-5:].min())
            earlier_macd_min = float(seg_macd.iloc[:20].min())
            if recent_price_min < earlier_price_min and recent_macd_min > earlier_macd_min:
                bullish_divergence = True

    return {
        "available": True,
        "macd_line": round(curr_macd, 5),
        "signal_line": round(curr_signal, 5),
        "histogram": round(curr_hist, 5),
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "hist_growing": hist_growing and curr_hist > 0,
        "macd_positive": macd_positive,
        "bullish_divergence": bullish_divergence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# C) RSI
# ─────────────────────────────────────────────────────────────────────────────

def calc_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    close = df["close"]
    if len(close) < period + 1:
        return {"available": False}

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1])

    if np.isnan(rsi_val):
        return {"available": False}

    oversold = rsi_val < 30
    overbought = rsi_val > 70
    neutral_rising = 50 <= rsi_val <= 60
    momentum_good = 50 < rsi_val <= 60

    # Bullish divergence: price lower low, RSI higher low (last 30 bars)
    bullish_divergence = False
    if len(rsi) >= 30:
        seg_close = close.tail(30)
        seg_rsi = rsi.tail(30).fillna(50)
        recent_price = float(seg_close.iloc[-5:].min())
        earlier_price = float(seg_close.iloc[:20].min())
        recent_rsi = float(seg_rsi.iloc[-5:].min())
        earlier_rsi = float(seg_rsi.iloc[:20].min())
        if recent_price < earlier_price and recent_rsi > earlier_rsi + 3:
            bullish_divergence = True

    # Oversold bounce: RSI was < 30 in last 5 bars and is now recovering
    oversold_bounce = False
    if len(rsi) >= 5:
        rsi_5 = rsi.tail(5).fillna(50)
        if rsi_5.iloc[:-1].min() < 30 and rsi_val > 30:
            oversold_bounce = True

    return {
        "available": True,
        "rsi": round(rsi_val, 1),
        "oversold": oversold,
        "overbought": overbought,
        "neutral_rising": neutral_rising,
        "momentum_good": momentum_good,
        "bullish_divergence": bullish_divergence,
        "oversold_bounce": oversold_bounce,
    }


# ─────────────────────────────────────────────────────────────────────────────
# D) Stochastic (14,3,3)
# ─────────────────────────────────────────────────────────────────────────────

def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth: int = 3) -> dict:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    if len(close) < k_period + d_period:
        return {"available": False}

    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100 * (close - lowest_low) / denom

    pct_k = raw_k.rolling(smooth).mean()  # smoothed %K
    pct_d = pct_k.rolling(d_period).mean()  # %D signal

    k_val = float(pct_k.iloc[-1])
    d_val = float(pct_d.iloc[-1])

    if np.isnan(k_val) or np.isnan(d_val):
        return {"available": False}

    oversold = k_val < 20 and d_val < 20
    overbought = k_val > 80 and d_val > 80

    # Bullish cross from oversold: %K crosses %D upward when both < 20
    bullish_cross_oversold = False
    if len(pct_k) >= 2 and len(pct_d) >= 2:
        prev_k = float(pct_k.iloc[-2])
        prev_d = float(pct_d.iloc[-2])
        if prev_k < prev_d and k_val > d_val and d_val < 30:
            bullish_cross_oversold = True

    k_rising = len(pct_k) >= 2 and k_val > float(pct_k.iloc[-2])
    d_rising = len(pct_d) >= 2 and d_val > float(pct_d.iloc[-2])
    both_rising_from_low = k_rising and d_rising and k_val < 50

    return {
        "available": True,
        "pct_k": round(k_val, 1),
        "pct_d": round(d_val, 1),
        "oversold": oversold,
        "overbought": overbought,
        "bullish_cross_oversold": bullish_cross_oversold,
        "both_rising_from_low": both_rising_from_low,
    }


# ─────────────────────────────────────────────────────────────────────────────
# E) Volume
# ─────────────────────────────────────────────────────────────────────────────

def calc_volume(df: pd.DataFrame) -> dict:
    if "volume" not in df.columns:
        return {"available": False}

    volume = df["volume"].replace(0, np.nan).dropna()
    close = df["close"]

    if len(volume) < 20:
        return {"available": False}

    avg_vol_20 = float(volume.tail(20).mean())
    current_vol = float(volume.iloc[-1])
    vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # Bullish day with high volume (last candle)
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
    bullish_day = last_close > prev_close
    high_vol_day = vol_ratio > 1.5
    bullish_high_vol = bullish_day and high_vol_day

    # Breakout with volume: price makes 20-day high with vol > 150%
    high_20 = float(df["high"].tail(20).max())
    is_near_high = last_close >= high_20 * 0.995
    breakout_confirmed = is_near_high and vol_ratio > 1.5

    # Trend with declining volume (bearish divergence in up trend)
    declining_vol_in_uptrend = False
    if len(volume) >= 10 and len(close) >= 10:
        price_up = float(close.tail(10).iloc[-1]) > float(close.tail(10).iloc[0])
        vol_slope = _slope_pct_per_day(volume.tail(10))
        if price_up and vol_slope < -2:
            declining_vol_in_uptrend = True

    # OBV (On-Balance Volume)
    obv = _calc_obv(df)
    obv_trend = _obv_trend(obv) if obv is not None else "unknown"

    return {
        "available": True,
        "vol_ratio_20d": round(vol_ratio, 2),
        "high_vol_day": high_vol_day,
        "bullish_high_vol": bullish_high_vol,
        "breakout_confirmed": breakout_confirmed,
        "declining_vol_in_uptrend": declining_vol_in_uptrend,
        "obv_trend": obv_trend,
        "obv": obv,
    }


def _calc_obv(df: pd.DataFrame) -> pd.Series | None:
    if "volume" not in df.columns:
        return None
    close = df["close"]
    volume = df["volume"].fillna(0)
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    return obv


def _obv_trend(obv: pd.Series, window: int = 20) -> str:
    if obv is None or len(obv) < window:
        return "insufficient_data"
    seg = obv.tail(window).dropna()
    if len(seg) < 5:
        return "insufficient_data"
    slope = _slope_pct_per_day(seg)
    if slope > 0.5:
        return "rising"
    elif slope < -0.5:
        return "falling"
    return "flat"


# ─────────────────────────────────────────────────────────────────────────────
# F) Support & Resistance
# ─────────────────────────────────────────────────────────────────────────────

def calc_support_resistance(df: pd.DataFrame) -> dict:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    current = float(close.iloc[-1])

    # 52-week high/low
    period_252 = min(252, len(close))
    high_52w = float(high.tail(period_252).max())
    low_52w = float(low.tail(period_252).min())

    # Swing highs/lows (local extrema in last 60 sessions, window=8)
    look_back = min(60, len(close))
    seg_high = high.tail(look_back)
    seg_low = low.tail(look_back)

    swing_highs = [float(seg_high.iloc[i]) for i in _local_maxima(seg_high, window=8)]
    swing_lows = [float(seg_low.iloc[i]) for i in _local_minima(seg_low, window=8)]

    # Filter: supports below price, resistances above
    supports = sorted([p for p in swing_lows if p < current * 0.998], reverse=True)
    resistances = sorted([p for p in swing_highs if p > current * 1.002])

    # Near support: price within 2% of nearest support
    near_support = bool(supports and abs(current - supports[0]) / current < 0.02)

    # Breakout: price within 0.5% above nearest resistance (just broke out)
    breakout = False
    distance_to_resistance: float | None = None
    if resistances:
        dist = (resistances[0] - current) / current
        distance_to_resistance = round(dist * 100, 2)
        if -0.005 <= dist <= 0.005:
            breakout = True

    return {
        "support_1": supports[0] if supports else None,
        "support_2": supports[1] if len(supports) > 1 else None,
        "resistance_1": resistances[0] if resistances else None,
        "resistance_2": resistances[1] if len(resistances) > 1 else None,
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "near_support": near_support,
        "breakout": breakout,
        "distance_to_resistance_pct": distance_to_resistance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# G) Fibonacci Retracements
# ─────────────────────────────────────────────────────────────────────────────

def calc_fibonacci(df: pd.DataFrame) -> dict:
    """
    Calculate Fibonacci retracement levels on the last ~6-month swing.
    Identifies swing high and swing low over last 130 sessions.
    Buy zones: 38.2%, 50%, 61.8% retracement from high to low.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    current = float(close.iloc[-1])

    period = min(130, len(close))
    seg_high = high.tail(period)
    seg_low = low.tail(period)

    swing_high = float(seg_high.max())
    swing_low = float(seg_low.min())

    if swing_high <= swing_low:
        return {"available": False}

    diff = swing_high - swing_low

    fib_levels = {
        "fib_236": round(swing_high - 0.236 * diff, 4),
        "fib_382": round(swing_high - 0.382 * diff, 4),
        "fib_500": round(swing_high - 0.500 * diff, 4),
        "fib_618": round(swing_high - 0.618 * diff, 4),
        "swing_high": round(swing_high, 4),
        "swing_low": round(swing_low, 4),
    }

    # Check if current price is in a key Fibonacci zone (±1.5%)
    in_fib_zone = False
    fib_zone_label: str | None = None
    for label, level in [("61.8%", fib_levels["fib_618"]),
                          ("50.0%", fib_levels["fib_500"]),
                          ("38.2%", fib_levels["fib_382"])]:
        if abs(current - level) / current < 0.015:
            in_fib_zone = True
            fib_zone_label = label
            break

    fib_levels["available"] = True
    fib_levels["in_fib_zone"] = in_fib_zone
    fib_levels["fib_zone_label"] = fib_zone_label
    return fib_levels


# ─────────────────────────────────────────────────────────────────────────────
# H) Koncorde simulation (OBV-based institutional vs retail)
# ─────────────────────────────────────────────────────────────────────────────

def calc_koncorde(df: pd.DataFrame) -> dict:
    """
    Approximate Blai5 Koncorde using OBV:
    - Green line (Big Players / Manos Fuertes): OBV trend
    - Red line (Bearish pressure): RSI of volume on bearish days
    - Signals: OBV rising while price consolidates → institutional accumulation
    """
    if "volume" not in df.columns or len(df) < 30:
        return {"available": False}

    close = df["close"]
    volume = df["volume"].fillna(0)
    obv = _calc_obv(df)

    if obv is None:
        return {"available": False}

    obv_trend = _obv_trend(obv, window=20)

    # OBV accumulation while price is flat/consolidating
    price_range_pct = 0.0
    institutional_accumulation = False
    if len(close) >= 20:
        seg_close = close.tail(20)
        price_range_pct = float((seg_close.max() - seg_close.min()) / seg_close.mean() * 100)
        obv_slope = _slope_pct_per_day(obv.tail(20).dropna())
        if price_range_pct < 8 and obv_slope > 0.3:
            institutional_accumulation = True

    # OBV diverges negatively: price up but OBV falling → distribution
    distribution_signal = False
    if len(close) >= 20:
        seg_close = close.tail(20)
        price_slope = _slope_pct_per_day(seg_close)
        obv_slope = _slope_pct_per_day(obv.tail(20).dropna())
        if price_slope > 0.1 and obv_slope < -0.1:
            distribution_signal = True

    # OBV normalized (last 50 values)
    obv_normalized: float | None = None
    if len(obv) >= 50:
        seg = obv.tail(50)
        seg_min = float(seg.min())
        seg_max = float(seg.max())
        if seg_max > seg_min:
            obv_normalized = round((float(obv.iloc[-1]) - seg_min) / (seg_max - seg_min) * 100, 1)

    return {
        "available": True,
        "obv_trend": obv_trend,
        "obv_normalized": obv_normalized,
        "institutional_accumulation": institutional_accumulation,
        "distribution_signal": distribution_signal,
        "price_range_pct_20d": round(price_range_pct, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# I) Ichimoku Cloud (optional)
# ─────────────────────────────────────────────────────────────────────────────

def calc_ichimoku(df: pd.DataFrame) -> dict:
    """
    Standard Ichimoku (9,26,52,26):
      Tenkan-sen = (max9 + min9) / 2
      Kijun-sen  = (max26 + min26) / 2
      Senkou A   = (Tenkan + Kijun) / 2  (shifted 26 forward)
      Senkou B   = (max52 + min52) / 2   (shifted 26 forward)
      Chikou     = close shifted 26 back
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    if len(close) < 52 + 26:
        return {"available": False}

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)

    t = float(tenkan.iloc[-1])
    k = float(kijun.iloc[-1])
    sa = float(senkou_a.iloc[-1]) if not np.isnan(senkou_a.iloc[-1]) else None
    sb = float(senkou_b.iloc[-1]) if not np.isnan(senkou_b.iloc[-1]) else None
    curr = float(close.iloc[-1])

    if np.isnan(t) or np.isnan(k):
        return {"available": False}

    # Price above cloud?
    price_above_cloud: bool | None = None
    if sa is not None and sb is not None:
        cloud_top = max(sa, sb)
        cloud_bot = min(sa, sb)
        price_above_cloud = curr > cloud_top
        price_in_cloud = cloud_bot <= curr <= cloud_top

    # TK cross bullish
    tk_cross_bullish = False
    if len(tenkan) >= 2 and len(kijun) >= 2:
        prev_t = float(tenkan.iloc[-2])
        prev_k = float(kijun.iloc[-2])
        if prev_t < prev_k and t >= k:
            tk_cross_bullish = True

    # Kumo breakout: price just crossed above cloud top
    kumo_breakout = False
    if sa is not None and sb is not None:
        cloud_top = max(sa, sb)
        if len(close) >= 2:
            prev_close = float(close.iloc[-2])
            if prev_close <= cloud_top and curr > cloud_top:
                kumo_breakout = True

    return {
        "available": True,
        "tenkan": round(t, 4),
        "kijun": round(k, 4),
        "senkou_a": round(sa, 4) if sa else None,
        "senkou_b": round(sb, 4) if sb else None,
        "price_above_cloud": price_above_cloud,
        "tk_cross_bullish": tk_cross_bullish,
        "kumo_breakout": kumo_breakout,
    }


# ─────────────────────────────────────────────────────────────────────────────
# J) Chart Pattern Detection
# ─────────────────────────────────────────────────────────────────────────────

def calc_chart_patterns(df: pd.DataFrame) -> dict:
    """
    Basic algorithmic pattern detection over last 60 sessions:
      - Double Bottom
      - Inverted Head & Shoulders
      - Bull Flag / Pennant
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    look_back = min(60, len(close))

    seg_close = close.tail(look_back).reset_index(drop=True)
    seg_high = high.tail(look_back).reset_index(drop=True)
    seg_low = low.tail(look_back).reset_index(drop=True)

    double_bottom = _detect_double_bottom(seg_low, seg_close)
    inv_hs = _detect_inv_head_shoulders(seg_low, seg_close)
    bull_flag = _detect_bull_flag(seg_close, seg_high, seg_low)

    return {
        "double_bottom": double_bottom,
        "inv_head_shoulders": inv_hs,
        "bull_flag": bull_flag,
    }


def _detect_double_bottom(seg_low: pd.Series, seg_close: pd.Series) -> bool:
    """
    Two local minima at similar price levels (within 3%), separated by a
    recovery peak of at least 5% above the lows.
    """
    min_indices = _local_minima(seg_low, window=5)
    if len(min_indices) < 2:
        return False

    for i in range(len(min_indices) - 1):
        for j in range(i + 1, len(min_indices)):
            idx1, idx2 = min_indices[i], min_indices[j]
            if idx2 - idx1 < 8:  # minima too close
                continue
            low1 = float(seg_low.iloc[idx1])
            low2 = float(seg_low.iloc[idx2])
            if low1 == 0:
                continue
            # Within 3% of each other
            if abs(low1 - low2) / low1 > 0.03:
                continue
            # Recovery peak between the two lows
            between = seg_close.iloc[idx1:idx2 + 1]
            recovery_high = float(between.max())
            avg_low = (low1 + low2) / 2
            if (recovery_high - avg_low) / avg_low < 0.05:
                continue
            # Current price should be above the second low
            if float(seg_close.iloc[-1]) > low2 * 1.01:
                return True
    return False


def _detect_inv_head_shoulders(seg_low: pd.Series, seg_close: pd.Series) -> bool:
    """
    Inverted Head & Shoulders: left shoulder (higher), head (deepest),
    right shoulder (higher), neckline breakout.
    """
    min_indices = _local_minima(seg_low, window=5)
    if len(min_indices) < 3:
        return False

    for i in range(len(min_indices) - 2):
        ls = min_indices[i]
        head = min_indices[i + 1]
        rs = min_indices[i + 2]

        if head - ls < 5 or rs - head < 5:
            continue

        ls_val = float(seg_low.iloc[ls])
        head_val = float(seg_low.iloc[head])
        rs_val = float(seg_low.iloc[rs])

        if head_val == 0:
            continue

        # Head must be lower than both shoulders
        if not (head_val < ls_val and head_val < rs_val):
            continue

        # Shoulders roughly equal height (within 5%)
        if ls_val == 0:
            continue
        if abs(ls_val - rs_val) / ls_val > 0.05:
            continue

        # Neckline: average of peaks between shoulders
        neckline = float(seg_close.iloc[ls:rs + 1].max())

        # Price should be near or above neckline
        current = float(seg_close.iloc[-1])
        if current >= neckline * 0.98:
            return True

    return False


def _detect_bull_flag(
    seg_close: pd.Series,
    seg_high: pd.Series,
    seg_low: pd.Series,
) -> bool:
    """
    Bull flag: strong upward pole (>8% in 5-10 bars), followed by
    tight consolidation (channel or range <4%).
    """
    n = len(seg_close)
    if n < 20:
        return False

    # Look for a pole in bars 0..40 and flag in the last 10 bars
    for pole_start in range(0, n - 15):
        pole_end = pole_start + 7  # ~7-bar pole
        if pole_end >= n - 5:
            break

        pole_low = float(seg_low.iloc[pole_start])
        pole_high = float(seg_high.iloc[pole_start:pole_end].max())
        if pole_low == 0:
            continue
        pole_gain = (pole_high - pole_low) / pole_low
        if pole_gain < 0.08:  # need at least 8% gain in pole
            continue

        # Flag: consolidation after pole (last part of segment)
        flag_seg = seg_close.iloc[pole_end:]
        if len(flag_seg) < 5:
            continue
        flag_range = (float(flag_seg.max()) - float(flag_seg.min()))
        if float(flag_seg.mean()) == 0:
            continue
        flag_range_pct = flag_range / float(flag_seg.mean())
        if flag_range_pct < 0.04:  # tight consolidation < 4%
            return True

    return False
