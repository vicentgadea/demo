"""
Módulo de análisis técnico.
Calcula indicadores con pandas-ta, los interpreta en contexto
y produce una visión de tendencia, momentum, soporte/resistencia y extensión.

Filosofía: el técnico describe el estado del precio, no predice el futuro.
Las conclusiones son probabilísticas y se complementan con el fundamental.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from ..providers.base import PriceHistory
from ..models.outputs import (
    TechnicalAnalysis,
    MovingAverages,
    MomentumIndicators,
    VolatilityMetrics,
    SupportResistance,
    Trend,
    SignalQuality,
)

logger = logging.getLogger(__name__)


def analyze_technical(
    price_data: PriceHistory,
    benchmark_data: PriceHistory | None = None,
) -> TechnicalAnalysis:
    """Punto de entrada principal del módulo técnico."""
    df = price_data.df.copy()
    gaps: list[str] = []

    if df.empty or len(df) < 20:
        return _insufficient_data(price_data.ticker)

    # Precios de cierre como serie limpia
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df.columns else None

    mas = _calc_moving_averages(close, df)
    momentum = _calc_momentum(df, gaps)
    volatility = _calc_volatility(df, close)
    sr = _calc_support_resistance(high, low, close)

    trend_short = _classify_trend(close, mas, window_days=20)
    trend_medium = _classify_trend(close, mas, window_days=50)
    trend_long = _classify_trend(close, mas, window_days=200)

    is_extended, extension_note = _check_extension(close, mas, momentum)
    volume_trend = _volume_trend(volume) if volume is not None else None
    rs = _relative_strength(close, benchmark_data) if benchmark_data else None
    signal_quality = _assess_signal_quality(trend_short, trend_medium, momentum, is_extended)
    narrative = _build_narrative(
        close, mas, momentum, sr, trend_short, trend_medium, trend_long,
        is_extended, extension_note, volume_trend, rs,
    )

    return TechnicalAnalysis(
        trend_short=trend_short,
        trend_medium=trend_medium,
        trend_long=trend_long,
        moving_averages=mas,
        momentum=momentum,
        volatility=volatility,
        support_resistance=sr,
        is_extended=is_extended,
        extension_note=extension_note,
        volume_trend=volume_trend,
        relative_strength_vs_spy=rs,
        signal_quality=signal_quality,
        narrative=narrative,
        data_gaps=gaps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Indicadores
# ─────────────────────────────────────────────────────────────────────────────

def _calc_moving_averages(close: pd.Series, df: pd.DataFrame) -> MovingAverages:
    current = float(close.iloc[-1])

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    pct_vs_50 = ((current - sma50) / sma50 * 100) if sma50 else None
    pct_vs_200 = ((current - sma200) / sma200 * 100) if sma200 else None

    return MovingAverages(
        sma_20=_r(sma20),
        sma_50=_r(sma50),
        sma_200=_r(sma200),
        ema_20=_r(ema20),
        price_vs_sma50_pct=_r(pct_vs_50, 2),
        price_vs_sma200_pct=_r(pct_vs_200, 2),
    )


def _calc_momentum(df: pd.DataFrame, gaps: list) -> MomentumIndicators:
    close = df["close"]
    rsi_val = None
    rsi_interp = None
    macd_line = macd_signal = macd_hist = None
    macd_cross = None

    if HAS_PANDAS_TA and len(close) >= 14:
        try:
            rsi_series = ta.rsi(close, length=14)
            if rsi_series is not None and not rsi_series.empty:
                rsi_val = float(rsi_series.iloc[-1])
                if rsi_val > 70:
                    rsi_interp = "overbought"
                elif rsi_val < 30:
                    rsi_interp = "oversold"
                else:
                    rsi_interp = "neutral"
        except Exception as e:
            logger.warning(f"RSI calc error: {e}")
            gaps.append("rsi")
    elif len(close) >= 14:
        # RSI manual si no hay pandas_ta
        rsi_val = _rsi_manual(close)
        if rsi_val is not None:
            rsi_interp = "overbought" if rsi_val > 70 else ("oversold" if rsi_val < 30 else "neutral")
    else:
        gaps.append("rsi")

    if HAS_PANDAS_TA and len(close) >= 26:
        try:
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                cols = macd_df.columns.tolist()
                macd_line = float(macd_df[cols[0]].iloc[-1])
                macd_hist = float(macd_df[cols[1]].iloc[-1])
                macd_signal = float(macd_df[cols[2]].iloc[-1])
                # Cruce reciente (últimas 3 velas)
                prev_hist = float(macd_df[cols[1]].iloc[-2]) if len(macd_df) > 1 else 0
                if macd_hist > 0 and prev_hist <= 0:
                    macd_cross = "bullish_cross"
                elif macd_hist < 0 and prev_hist >= 0:
                    macd_cross = "bearish_cross"
                else:
                    macd_cross = "none"
        except Exception as e:
            logger.warning(f"MACD calc error: {e}")
            gaps.append("macd")
    elif len(close) >= 26:
        macd_line, macd_signal, macd_hist = _macd_manual(close)
        macd_cross = "none"
    else:
        gaps.append("macd")

    return MomentumIndicators(
        rsi_14=_r(rsi_val, 1),
        rsi_interpretation=rsi_interp,
        macd_line=_r(macd_line, 4),
        macd_signal=_r(macd_signal, 4),
        macd_histogram=_r(macd_hist, 4),
        macd_cross=macd_cross,
    )


def _calc_volatility(df: pd.DataFrame, close: pd.Series) -> VolatilityMetrics:
    atr = _atr_manual(df, 14)
    current_price = float(close.iloc[-1])
    atr_pct = (atr / current_price * 100) if atr else None

    # Volatilidad histórica 30 días (desviación estándar de retornos log * sqrt(252))
    hist_vol = None
    if len(close) >= 30:
        returns = np.log(close / close.shift(1)).dropna()
        if len(returns) >= 20:
            hist_vol = float(returns.tail(30).std() * np.sqrt(252) * 100)

    return VolatilityMetrics(
        atr_14=_r(atr),
        atr_pct=_r(atr_pct, 2),
        historical_volatility_30d=_r(hist_vol, 1),
        beta=None,  # se puede rellenar desde fundamentales
    )


def _calc_support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> SupportResistance:
    """
    Soporte y resistencia por swing highs/lows: mínimos y máximos locales
    en ventana de 10 días. Toma los 2 niveles más relevantes de cada lado.
    """
    current = float(close.iloc[-1])

    swing_highs = _swing_points(high, window=10, kind="high")
    swing_lows = _swing_points(low, window=10, kind="low")

    resistances = sorted([p for p in swing_highs if p > current * 1.005])
    supports = sorted([p for p in swing_lows if p < current * 0.995], reverse=True)

    # También incluir SMA 50 y 200 como niveles dinámicos
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    if sma50 and sma50 < current * 0.995:
        supports.append(sma50)
        supports = sorted(list(set([round(s, 2) for s in supports])), reverse=True)
    if sma200 and sma200 < current * 0.995:
        supports.append(sma200)
        supports = sorted(list(set([round(s, 2) for s in supports])), reverse=True)

    return SupportResistance(
        support_1=supports[0] if len(supports) > 0 else None,
        support_2=supports[1] if len(supports) > 1 else None,
        resistance_1=resistances[0] if len(resistances) > 0 else None,
        resistance_2=resistances[1] if len(resistances) > 1 else None,
        method="swing_highs_lows_with_sma",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Clasificación de tendencia
# ─────────────────────────────────────────────────────────────────────────────

def _classify_trend(close: pd.Series, mas: MovingAverages, window_days: int) -> Trend:
    """
    Clasifica la tendencia mirando:
    - Precio vs. media móvil relevante
    - Pendiente de la media móvil (regresión lineal)
    - Estructura de máximos y mínimos recientes
    """
    current = float(close.iloc[-1])

    if window_days == 20:
        ma_val = mas.sma_20
    elif window_days == 50:
        ma_val = mas.sma_50
    else:
        ma_val = mas.sma_200

    # Si no tenemos la media, clasificar por pendiente del precio
    if ma_val is None:
        window = min(window_days, len(close))
        if window < 10:
            return Trend.neutral
        segment = close.tail(window)
        slope = _linear_slope_pct(segment)
        if slope > 3:
            return Trend.bullish
        elif slope < -3:
            return Trend.bearish
        return Trend.neutral

    pct_above = (current - ma_val) / ma_val * 100

    # Pendiente de la MA
    n = min(window_days, len(close))
    segment = close.tail(n)
    sma_segment = segment.rolling(min(20, n // 2)).mean().dropna()
    ma_slope = _linear_slope_pct(sma_segment) if len(sma_segment) >= 5 else 0

    # Estructura HH/HL o LH/LL
    highs = close.tail(n).rolling(5).max().dropna()
    lows = close.tail(n).rolling(5).min().dropna()
    structure_score = 0
    if len(highs) >= 3:
        if highs.iloc[-1] > highs.iloc[-3]:
            structure_score += 1
        else:
            structure_score -= 1
    if len(lows) >= 3:
        if lows.iloc[-1] > lows.iloc[-3]:
            structure_score += 1
        else:
            structure_score -= 1

    # Decisión compuesta
    score = 0
    score += 2 if pct_above > 5 else (1 if pct_above > 1 else (-1 if pct_above < -1 else (-2 if pct_above < -5 else 0)))
    score += 1 if ma_slope > 1 else (-1 if ma_slope < -1 else 0)
    score += structure_score

    if score >= 4:
        return Trend.strong_bullish
    elif score >= 2:
        return Trend.bullish
    elif score <= -4:
        return Trend.strong_bearish
    elif score <= -2:
        return Trend.bearish
    return Trend.neutral


def _check_extension(
    close: pd.Series,
    mas: MovingAverages,
    momentum: MomentumIndicators,
) -> tuple[bool, str | None]:
    """Detecta si el precio está sobreextendido respecto a medias y RSI."""
    reasons = []
    current = float(close.iloc[-1])

    if mas.price_vs_sma50_pct is not None and mas.price_vs_sma50_pct > 15:
        reasons.append(f"precio {mas.price_vs_sma50_pct:.1f}% sobre SMA50")
    if mas.price_vs_sma200_pct is not None and mas.price_vs_sma200_pct > 30:
        reasons.append(f"precio {mas.price_vs_sma200_pct:.1f}% sobre SMA200")
    if momentum.rsi_14 is not None and momentum.rsi_14 > 72:
        reasons.append(f"RSI en zona de sobrecompra ({momentum.rsi_14:.0f})")

    if reasons:
        return True, "Precio extendido: " + "; ".join(reasons) + ". Prudente esperar retroceso o consolidación."
    return False, None


def _volume_trend(volume: pd.Series) -> str:
    """Compara volumen medio últimas 2 semanas vs. últimas 8 semanas."""
    if len(volume) < 20:
        return "insufficient_data"
    recent_avg = float(volume.tail(10).mean())
    baseline_avg = float(volume.tail(40).head(30).mean())
    if baseline_avg == 0:
        return "neutral"
    ratio = recent_avg / baseline_avg
    if ratio > 1.30:
        return "increasing"
    elif ratio < 0.75:
        return "decreasing"
    return "neutral"


def _relative_strength(close: pd.Series, benchmark: PriceHistory) -> float | None:
    """Retorno relativo del activo vs. benchmark en los últimos 3 meses."""
    try:
        bench_close = benchmark.df["close"]
        # Alinear por fecha
        merged = pd.concat(
            [close.rename("asset"), bench_close.rename("bench")], axis=1
        ).dropna()
        if len(merged) < 60:
            return None
        window = merged.tail(63)  # ~3 meses hábiles
        asset_ret = (window["asset"].iloc[-1] / window["asset"].iloc[0] - 1) * 100
        bench_ret = (window["bench"].iloc[-1] / window["bench"].iloc[0] - 1) * 100
        return round(asset_ret - bench_ret, 2)
    except Exception:
        return None


def _assess_signal_quality(
    trend_short: Trend,
    trend_medium: Trend,
    momentum: MomentumIndicators,
    is_extended: bool,
) -> SignalQuality:
    """Evalúa la coherencia de las señales técnicas."""
    # Si short y medium apuntan en la misma dirección y no está extendido
    same_direction = (
        (trend_short in (Trend.bullish, Trend.strong_bullish) and trend_medium in (Trend.bullish, Trend.strong_bullish))
        or
        (trend_short in (Trend.bearish, Trend.strong_bearish) and trend_medium in (Trend.bearish, Trend.strong_bearish))
    )

    conflicting = (
        (trend_short in (Trend.bullish, Trend.strong_bullish) and trend_medium in (Trend.bearish, Trend.strong_bearish))
        or
        (trend_short in (Trend.bearish, Trend.strong_bearish) and trend_medium in (Trend.bullish, Trend.strong_bullish))
    )

    if conflicting:
        return SignalQuality.conflicting
    if same_direction and not is_extended:
        if momentum.macd_cross == "bullish_cross" or (momentum.rsi_14 and 40 < momentum.rsi_14 < 65):
            return SignalQuality.strong
        return SignalQuality.moderate
    if same_direction and is_extended:
        return SignalQuality.weak
    return SignalQuality.moderate


def _build_narrative(
    close: pd.Series,
    mas: MovingAverages,
    momentum: MomentumIndicators,
    sr: SupportResistance,
    trend_s: Trend,
    trend_m: Trend,
    trend_l: Trend,
    is_extended: bool,
    extension_note: str | None,
    volume_trend: str | None,
    rs: float | None,
) -> str:
    current = float(close.iloc[-1])
    parts = []

    # Tendencia
    trend_map = {
        Trend.strong_bullish: "tendencia alcista fuerte",
        Trend.bullish: "tendencia alcista",
        Trend.neutral: "movimiento lateral",
        Trend.bearish: "tendencia bajista",
        Trend.strong_bearish: "tendencia bajista fuerte",
    }
    parts.append(
        f"En el corto plazo el precio muestra {trend_map[trend_s]}, "
        f"en el medio plazo {trend_map[trend_m]}, "
        f"y en el largo plazo {trend_map[trend_l]}."
    )

    # Medias móviles
    if mas.sma_50 and mas.sma_200:
        if mas.sma_50 > mas.sma_200:
            parts.append("La SMA50 cotiza por encima de la SMA200 (Golden Cross / estructura alcista de largo plazo).")
        else:
            parts.append("La SMA50 está por debajo de la SMA200 (Death Cross / estructura bajista de largo plazo).")

    # RSI
    if momentum.rsi_14:
        if momentum.rsi_interpretation == "overbought":
            parts.append(f"El RSI ({momentum.rsi_14:.0f}) indica sobrecompra de corto plazo — mayor precaución con nuevas compras.")
        elif momentum.rsi_interpretation == "oversold":
            parts.append(f"El RSI ({momentum.rsi_14:.0f}) indica sobreventa — posible rebote técnico, pero no garantiza reversión de tendencia.")
        else:
            parts.append(f"El RSI ({momentum.rsi_14:.0f}) está en zona neutral, sin lecturas extremas.")

    # Extensión
    if extension_note:
        parts.append(extension_note)

    # Soportes/resistencias
    if sr.support_1:
        parts.append(f"Soporte inmediato en torno a {sr.support_1:.2f}.")
    if sr.resistance_1:
        parts.append(f"Resistencia más próxima en {sr.resistance_1:.2f}.")

    # Volumen
    if volume_trend == "increasing":
        parts.append("El volumen reciente ha aumentado, lo que refuerza el movimiento de precio.")
    elif volume_trend == "decreasing":
        parts.append("El volumen ha disminuido — la tendencia puede estar perdiendo convicción.")

    # Fortaleza relativa
    if rs is not None:
        if rs > 5:
            parts.append(f"El activo muestra fuerza relativa positiva frente al benchmark (+{rs:.1f}pp en 3 meses).")
        elif rs < -5:
            parts.append(f"El activo muestra debilidad relativa frente al benchmark ({rs:.1f}pp en 3 meses).")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Implementaciones manuales de indicadores (fallback sin pandas_ta)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_manual(close: pd.Series, period: int = 14) -> float | None:
    try:
        if len(close) < period + 1:
            return None
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        last_loss = float(avg_loss.iloc[-1])
        last_gain = float(avg_gain.iloc[-1])
        if np.isnan(last_gain) or np.isnan(last_loss):
            return None
        if last_loss == 0:
            return 100.0 if last_gain > 0 else 50.0
        rs = last_gain / last_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi) if not np.isnan(rsi) else None
    except Exception:
        return None


def _macd_manual(close: pd.Series) -> tuple[float | None, float | None, float | None]:
    try:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
    except Exception:
        return None, None, None


def _atr_manual(df: pd.DataFrame, period: int = 14) -> float | None:
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(com=period - 1, min_periods=period).mean()
        return float(atr.iloc[-1])
    except Exception:
        return None


def _swing_points(series: pd.Series, window: int = 10, kind: str = "high") -> list[float]:
    """Detecta swing highs o lows como máximos/mínimos locales."""
    result = []
    arr = series.values
    n = len(arr)
    for i in range(window, n - window):
        if kind == "high":
            if arr[i] == max(arr[i - window: i + window + 1]):
                result.append(float(arr[i]))
        else:
            if arr[i] == min(arr[i - window: i + window + 1]):
                result.append(float(arr[i]))
    # Eliminar duplicados cercanos (dentro del 1%)
    if not result:
        return result
    unique = [result[0]]
    for v in result[1:]:
        if abs(v - unique[-1]) / max(abs(unique[-1]), 1) > 0.01:
            unique.append(v)
    return unique


def _sma(series: pd.Series, period: int) -> float | None:
    if len(series) < period:
        return None
    return float(series.rolling(period).mean().iloc[-1])


def _ema(series: pd.Series, period: int) -> float | None:
    if len(series) < period:
        return None
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def _linear_slope_pct(series: pd.Series) -> float:
    """Pendiente de la regresión lineal como % del valor medio."""
    if len(series) < 3:
        return 0.0
    try:
        x = np.arange(len(series))
        slope = float(np.polyfit(x, series.values.astype(float), 1)[0])
        mean_val = abs(float(series.mean()))
        if mean_val == 0:
            return 0.0
        return (slope / mean_val) * 100 * len(series)
    except Exception:
        return 0.0


def _r(val: float | None, decimals: int = 2) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(val, decimals)


def _insufficient_data(ticker: str) -> TechnicalAnalysis:
    return TechnicalAnalysis(
        trend_short=Trend.neutral,
        trend_medium=Trend.neutral,
        trend_long=Trend.neutral,
        moving_averages=MovingAverages(),
        momentum=MomentumIndicators(),
        volatility=VolatilityMetrics(),
        support_resistance=SupportResistance(),
        is_extended=False,
        volume_trend=None,
        signal_quality=SignalQuality.weak,
        narrative="Datos insuficientes para análisis técnico fiable.",
        data_gaps=["insufficient_price_history"],
    )
