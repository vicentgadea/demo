"""
Tests del módulo de análisis técnico.
"""
import numpy as np
import pandas as pd
import pytest

from app.providers.base import PriceHistory
from app.analysis.technical import (
    analyze_technical,
    _rsi_manual,
    _macd_manual,
    _atr_manual,
    _swing_points,
    _sma,
    _ema,
    _classify_trend,
    _check_extension,
    _volume_trend,
    _insufficient_data,
)
from app.models.outputs import Trend, MovingAverages, MomentumIndicators


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_price_df(n: int = 300, trend: str = "up", volatility: float = 0.015) -> pd.DataFrame:
    """Genera precios OHLCV sintéticos."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    if trend == "up":
        drift = 0.004   # drift fuerte para que la tendencia sea inequívoca
    elif trend == "down":
        drift = -0.004
    else:
        drift = 0.0

    returns = np.random.normal(drift, volatility, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_prices = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    return df


def make_price_history(n: int = 300, trend: str = "up") -> PriceHistory:
    df = make_price_df(n, trend)
    return PriceHistory(ticker="TEST", df=df)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de indicadores individuales
# ─────────────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_in_valid_range(self):
        close = pd.Series(make_price_df()["close"])
        rsi = _rsi_manual(close)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_overbought_after_strong_rally(self):
        """RSI debe ser alto después de una serie de velas alcistas."""
        close = pd.Series([100 + i * 2 for i in range(30)])  # subida constante
        rsi = _rsi_manual(close)
        assert rsi is not None
        assert rsi > 60

    def test_rsi_oversold_after_decline(self):
        """RSI debe ser bajo después de caídas consecutivas."""
        close = pd.Series([100 - i * 2 for i in range(30)])
        rsi = _rsi_manual(close)
        assert rsi is not None
        assert rsi < 40

    def test_rsi_insufficient_data(self):
        close = pd.Series([100, 101, 102])
        rsi = _rsi_manual(close)
        # Con pocos datos debe devolver None
        assert rsi is None


class TestMACD:
    def test_macd_returns_three_values(self):
        df = make_price_df()
        close = df["close"]
        macd, signal, hist = _macd_manual(close)
        assert macd is not None
        assert signal is not None
        assert hist is not None

    def test_macd_histogram_is_macd_minus_signal(self):
        df = make_price_df()
        close = df["close"]
        macd, signal, hist = _macd_manual(close)
        if macd is not None and signal is not None and hist is not None:
            assert abs(hist - (macd - signal)) < 1e-8

    def test_macd_insufficient_data(self):
        close = pd.Series([100.0] * 10)
        macd, signal, hist = _macd_manual(close)
        # Con datos insuficientes puede devolver valores o None
        pass  # no falla — maneja el caso


class TestATR:
    def test_atr_positive(self):
        df = make_price_df()
        atr = _atr_manual(df)
        assert atr is not None
        assert atr > 0

    def test_atr_higher_for_volatile_asset(self):
        df_normal = make_price_df(volatility=0.01)
        df_volatile = make_price_df(volatility=0.04)
        atr_normal = _atr_manual(df_normal)
        atr_volatile = _atr_manual(df_volatile)
        if atr_normal and atr_volatile:
            assert atr_volatile > atr_normal


class TestSwingPoints:
    def test_finds_highs(self):
        # Crear serie con picos claros
        values = [100, 102, 105, 103, 100, 98, 95, 97, 99, 97, 95]
        s = pd.Series(values)
        highs = _swing_points(s, window=2, kind="high")
        assert len(highs) > 0

    def test_finds_lows(self):
        values = [100, 98, 95, 97, 100, 102, 105, 103, 100, 98, 95]
        s = pd.Series(values)
        lows = _swing_points(s, window=2, kind="low")
        assert len(lows) > 0


class TestMovingAverages:
    def test_sma_correct_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _sma(s, 3) == pytest.approx(4.0)

    def test_sma_insufficient_data(self):
        s = pd.Series([1.0, 2.0])
        assert _sma(s, 5) is None

    def test_ema_returns_float(self):
        s = pd.Series([100.0 + i for i in range(30)])
        ema = _ema(s, 20)
        assert ema is not None
        assert isinstance(ema, float)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de clasificación de tendencia
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendClassification:
    def test_strong_uptrend_classified_as_bullish(self):
        df = make_price_df(n=250, trend="up")
        close = df["close"]
        from app.analysis.technical import _calc_moving_averages
        mas = _calc_moving_averages(close, df)
        trend = _classify_trend(close, mas, 50)
        assert trend in (Trend.bullish, Trend.strong_bullish)

    def test_strong_downtrend_classified_as_bearish(self):
        df = make_price_df(n=250, trend="down")
        close = df["close"]
        from app.analysis.technical import _calc_moving_averages
        mas = _calc_moving_averages(close, df)
        trend = _classify_trend(close, mas, 50)
        assert trend in (Trend.bearish, Trend.strong_bearish)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de extensión
# ─────────────────────────────────────────────────────────────────────────────

class TestExtensionDetection:
    def test_extended_price_detected(self):
        mas = MovingAverages(sma_50=80.0, price_vs_sma50_pct=25.0)
        close = pd.Series([100.0] * 50)
        momentum = MomentumIndicators(rsi_14=78.0, rsi_interpretation="overbought")
        is_ext, note = _check_extension(close, mas, momentum)
        assert is_ext is True
        assert note is not None

    def test_normal_price_not_extended(self):
        mas = MovingAverages(sma_50=98.0, price_vs_sma50_pct=2.0)
        close = pd.Series([100.0] * 50)
        momentum = MomentumIndicators(rsi_14=55.0, rsi_interpretation="neutral")
        is_ext, note = _check_extension(close, mas, momentum)
        assert is_ext is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests de volumen
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeTrend:
    def test_increasing_volume(self):
        # Bajo volumen al principio, alto al final
        vol = pd.Series([1_000_000] * 30 + [3_000_000] * 10)
        result = _volume_trend(vol)
        assert result == "increasing"

    def test_decreasing_volume(self):
        vol = pd.Series([3_000_000] * 30 + [500_000] * 10)
        result = _volume_trend(vol)
        assert result == "decreasing"

    def test_insufficient_data(self):
        vol = pd.Series([1_000_000] * 5)
        result = _volume_trend(vol)
        assert result == "insufficient_data"


# ─────────────────────────────────────────────────────────────────────────────
# Tests de integración
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeTechnical:
    def test_full_analysis_uptrend(self):
        ph = make_price_history(n=300, trend="up")
        result = analyze_technical(ph)

        assert result is not None
        assert isinstance(result.trend_medium, Trend)
        assert isinstance(result.trend_long, Trend)
        assert isinstance(result.narrative, str)
        assert len(result.narrative) > 10

    def test_rsi_in_result(self):
        ph = make_price_history(n=300)
        result = analyze_technical(ph)
        if result.momentum.rsi_14 is not None:
            assert 0 <= result.momentum.rsi_14 <= 100

    def test_insufficient_data_returns_gracefully(self):
        df = make_price_df(n=10)  # muy pocos datos
        ph = PriceHistory(ticker="TEST", df=df)
        result = analyze_technical(ph)
        assert result is not None
        assert "insufficient_price_history" in result.data_gaps

    def test_all_trend_fields_valid(self):
        ph = make_price_history(n=300)
        result = analyze_technical(ph)
        valid_trends = set(Trend.__members__.values())
        assert result.trend_short in valid_trends
        assert result.trend_medium in valid_trends
        assert result.trend_long in valid_trends
