"""
Scanner orchestrator — ties together all three analysis levels.

Market → Sector → Value pipeline for each ticker.

Flow per ticker:
  1. Fetch OHLCV (1y, 1d interval)
  2. Fetch fundamentals (yfinance info dict)
  3. Calculate all technical indicators
  4. Score technical
  5. Score fundamental
  6. Apply market context & sector adjustments
  7. Compute combined score & final signal
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from ..models.scanner_models import (
    EntrySignal,
    MarketContext,
    PortfolioType,
    ScannerRequest,
    ScannerResponse,
    SectorContext,
    TickerScanResult,
)
from .market_context import analyze_market_context
from .scanner_indicators import (
    calc_chart_patterns,
    calc_fibonacci,
    calc_ichimoku,
    calc_koncorde,
    calc_macd,
    calc_moving_averages,
    calc_rsi,
    calc_stochastic,
    calc_support_resistance,
    calc_volume,
)
from .scanner_scoring import (
    build_fundamental_score,
    build_technical_score,
    compute_combined_score,
)
from .sector_analysis import analyze_sector, get_sector_etf

logger = logging.getLogger(__name__)

# Sector ETF price cache (avoid re-fetching same ETF for multiple tickers)
_SECTOR_ETF_CACHE: dict[str, pd.DataFrame] = {}


def _fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        return df
    except Exception as e:
        logger.warning(f"OHLCV fetch failed for {ticker}: {e}")
        return None


def _fetch_info(ticker: str) -> dict:
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info or {}
        return info
    except Exception as e:
        logger.warning(f"Info fetch failed for {ticker}: {e}")
        return {}


def _fetch_sector_etf(etf_ticker: str) -> pd.DataFrame | None:
    if etf_ticker in _SECTOR_ETF_CACHE:
        return _SECTOR_ETF_CACHE[etf_ticker]
    df = _fetch_ohlcv(etf_ticker)
    if df is not None:
        _SECTOR_ETF_CACHE[etf_ticker] = df
    return df


def _analyze_ticker(
    ticker: str,
    spy_df: pd.DataFrame | None,
    market_ctx: MarketContext,
    portfolio_type: PortfolioType,
    enable_ichimoku: bool,
    override_sector_filter: bool,
) -> TickerScanResult:
    """Full analysis pipeline for a single ticker."""
    t0 = time.time()
    ticker = ticker.strip().upper()

    try:
        # ── Fetch data ────────────────────────────────────────────────────────
        df = _fetch_ohlcv(ticker)
        info = _fetch_info(ticker)

        if df is None or len(df) < 30:
            return TickerScanResult(
                ticker=ticker,
                company_name=info.get("longName") or info.get("shortName") or ticker,
                error="Datos de precio insuficientes o ticker inválido",
                data_quality="no_data",
                market_context=market_ctx,
            )

        current_price = float(df["close"].iloc[-1])
        prev_price = float(df["close"].iloc[-2]) if len(df) > 1 else current_price
        price_change_1d = (current_price - prev_price) / prev_price * 100

        # ── Technical indicators ──────────────────────────────────────────────
        ma = calc_moving_averages(df)
        macd = calc_macd(df)
        rsi = calc_rsi(df)
        stoch = calc_stochastic(df)
        vol = calc_volume(df)
        sr = calc_support_resistance(df)
        fib = calc_fibonacci(df)
        konc = calc_koncorde(df)
        ichi = calc_ichimoku(df) if enable_ichimoku else {"available": False}
        patterns = calc_chart_patterns(df)

        # ── Technical score ───────────────────────────────────────────────────
        tech_score_obj = build_technical_score(
            ma, macd, rsi, stoch, vol, sr, fib, konc, ichi, patterns, enable_ichimoku
        )

        # ── Sector context ────────────────────────────────────────────────────
        sector_name = info.get("sector")
        etf_ticker = get_sector_etf(sector_name)
        etf_df = _fetch_sector_etf(etf_ticker) if etf_ticker else None
        sector_ctx = analyze_sector(sector_name, etf_df, spy_df)

        # ── Fundamental score ─────────────────────────────────────────────────
        fund_score_obj = build_fundamental_score(info)

        # ── Combined score & signal ───────────────────────────────────────────
        combined, confidence, signal, justification = compute_combined_score(
            tech_score=tech_score_obj.normalized_score,
            fund_score=fund_score_obj.normalized_score,
            market=market_ctx,
            sector=sector_ctx,
            portfolio_type=portfolio_type,
            override_sector_filter=override_sector_filter,
        )

        # ── Key alerts ────────────────────────────────────────────────────────
        all_alerts = tech_score_obj.alerts + fund_score_obj.alerts

        return TickerScanResult(
            ticker=ticker,
            company_name=info.get("longName") or info.get("shortName") or ticker,
            sector=sector_name,
            industry=info.get("industry"),
            current_price=round(current_price, 2),
            price_change_pct_1d=round(price_change_1d, 2),
            market_cap=info.get("marketCap"),
            market_context=market_ctx,
            sector_context=sector_ctx,
            technical_score=tech_score_obj,
            fundamental_score=fund_score_obj,
            combined_score=combined,
            entry_signal=signal,
            confidence=confidence,
            signal_justification=justification,
            key_alerts=all_alerts,
            data_quality="ok",
        )

    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
        return TickerScanResult(
            ticker=ticker,
            error=str(e),
            data_quality="error",
            market_context=market_ctx,
        )


def run_scanner(request: ScannerRequest) -> ScannerResponse:
    """
    Run the full Market → Sector → Value scanner for all requested tickers.
    Uses a thread pool for parallel data fetching.
    """
    import uuid
    from datetime import datetime, timezone

    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{request_id}] Scanner started for {len(request.tickers)} tickers")

    # ── Level 1: Market context ───────────────────────────────────────────────
    spy_df = _fetch_ohlcv(request.market_ticker, period="2y")
    market_ctx = analyze_market_context(spy_df)
    logger.info(f"[{request_id}] Market: {market_ctx.regime.value} (strength={market_ctx.strength})")

    # ── Level 2+3: Per-ticker analysis (parallel) ─────────────────────────────
    results: list[TickerScanResult] = []
    tickers = [t.strip().upper() for t in request.tickers if t.strip()]

    max_workers = min(8, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _analyze_ticker,
                ticker,
                spy_df,
                market_ctx,
                request.portfolio_type,
                request.enable_ichimoku,
                request.override_sector_filter,
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                logger.error(f"Future failed for {ticker}: {e}")
                results.append(TickerScanResult(
                    ticker=ticker,
                    error=f"Timeout o error inesperado: {e}",
                    data_quality="error",
                    market_context=market_ctx,
                ))

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = results
    if request.min_combined_score > 0:
        filtered = [r for r in results if r.combined_score >= request.min_combined_score or r.error]

    if request.only_entry_signals:
        valid_signals = {EntrySignal.fuerte, EntrySignal.recomendada, EntrySignal.posible}
        filtered = [r for r in filtered if r.entry_signal in valid_signals or r.error]

    # Sort by combined score desc
    filtered.sort(key=lambda r: r.combined_score, reverse=True)

    # Top 5 opportunities
    top_5 = [
        r.ticker for r in filtered
        if not r.error and r.entry_signal in {EntrySignal.fuerte, EntrySignal.recomendada, EntrySignal.posible}
    ][:5]

    errors_count = sum(1 for r in results if r.error)
    elapsed = time.time() - start_time

    logger.info(
        f"[{request_id}] Scanner completed in {elapsed:.1f}s. "
        f"Processed: {len(results)}, Errors: {errors_count}, Top: {top_5}"
    )

    return ScannerResponse(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        market_context=market_ctx,
        results=filtered,
        scan_duration_s=round(elapsed, 2),
        tickers_processed=len(results),
        tickers_with_errors=errors_count,
        top_opportunities=top_5,
    )
