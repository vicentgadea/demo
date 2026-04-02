"""
Proveedor de datos basado en yfinance (Yahoo Finance).
Fuente primaria: gratuita, sin API key, cubre mercados globales.
Limitaciones conocidas: datos institucionales limitados, sin estimaciones de analistas detalladas.
"""
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from .base import DataProvider, FundamentalData, FinancialStatements, PriceHistory

logger = logging.getLogger(__name__)


class YFinanceProvider(DataProvider):
    """Implementación de DataProvider usando yfinance."""

    @property
    def name(self) -> str:
        return "yfinance"

    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> PriceHistory:
        logger.info(f"[yfinance] Descargando precios: {ticker} | period={period} interval={interval}")
        try:
            tkr = yf.Ticker(ticker)
            df = tkr.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No se encontraron datos de precio para {ticker}")
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)  # normalizar timezone
            # Renombrar columnas a snake_case estándar
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            currency = tkr.fast_info.get("currency", "USD") if hasattr(tkr, "fast_info") else "USD"
            return PriceHistory(ticker=ticker, df=df, currency=str(currency))
        except Exception as exc:
            logger.error(f"[yfinance] Error obteniendo precios de {ticker}: {exc}")
            raise

    def get_fundamentals(self, ticker: str) -> FundamentalData:
        logger.info(f"[yfinance] Obteniendo fundamentales: {ticker}")
        missing: list[str] = []
        try:
            tkr = yf.Ticker(ticker)
            info: dict = tkr.info or {}
        except Exception as exc:
            logger.error(f"[yfinance] Error obteniendo info de {ticker}: {exc}")
            raise

        def _get(key: str, default=None):
            val = info.get(key, default)
            if val is None or val == "N/A":
                missing.append(key)
                return None
            return val

        # Calcular EV/EBITDA si no viene directo
        ev = _get("enterpriseValue")
        ebitda = _get("ebitda")
        ev_ebitda = None
        if ev and ebitda and ebitda > 0:
            ev_ebitda = ev / ebitda
        elif info.get("enterpriseToEbitda"):
            ev_ebitda = info["enterpriseToEbitda"]

        data = FundamentalData(
            ticker=ticker,
            name=_get("longName") or _get("shortName"),
            sector=_get("sector"),
            industry=_get("industry"),
            country=_get("country"),
            currency=_get("currency"),
            description=_get("longBusinessSummary"),
            current_price=_get("currentPrice") or _get("regularMarketPrice"),
            market_cap=_get("marketCap"),
            price_52w_high=_get("fiftyTwoWeekHigh"),
            price_52w_low=_get("fiftyTwoWeekLow"),
            beta=_get("beta"),
            pe_ratio=_get("trailingPE"),
            forward_pe=_get("forwardPE"),
            peg_ratio=_get("pegRatio"),
            price_to_sales=_get("priceToSalesTrailing12Months"),
            price_to_book=_get("priceToBook"),
            ev_ebitda=ev_ebitda,
            enterprise_value=ev,
            gross_margin=_get("grossMargins"),
            operating_margin=_get("operatingMargins"),
            net_margin=_get("profitMargins"),
            ebitda_margin=self._safe_divide(ebitda, _get("totalRevenue")),
            roe=_get("returnOnEquity"),
            roa=_get("returnOnAssets"),
            revenue_growth=_get("revenueGrowth"),
            earnings_growth=_get("earningsGrowth"),
            total_cash=_get("totalCash"),
            total_debt=_get("totalDebt"),
            debt_to_equity=_get("debtToEquity"),
            current_ratio=_get("currentRatio"),
            book_value_per_share=_get("bookValue"),
            free_cash_flow=_get("freeCashflow"),
            operating_cash_flow=_get("operatingCashflow"),
            dividend_yield=_get("dividendYield"),
            payout_ratio=_get("payoutRatio"),
            shares_outstanding=_get("sharesOutstanding"),
            trailing_eps=_get("trailingEps"),
            forward_eps=_get("forwardEps"),
            missing_fields=list(set(missing)),
        )
        logger.debug(f"[yfinance] Campos faltantes para {ticker}: {data.missing_fields}")
        return data

    def get_financial_statements(self, ticker: str) -> FinancialStatements:
        logger.info(f"[yfinance] Obteniendo estados financieros: {ticker}")
        missing: list[str] = []
        try:
            tkr = yf.Ticker(ticker)

            income = self._clean_df(tkr.financials, "income_statement", missing)
            balance = self._clean_df(tkr.balance_sheet, "balance_sheet", missing)
            cashflow = self._clean_df(tkr.cashflow, "cash_flow", missing)

            return FinancialStatements(
                ticker=ticker,
                income_statement=income,
                balance_sheet=balance,
                cash_flow=cashflow,
                missing_fields=missing,
            )
        except Exception as exc:
            logger.error(f"[yfinance] Error obteniendo estados financieros de {ticker}: {exc}")
            raise

    # ── Helpers privados ─────────────────────────────────────────────────────

    @staticmethod
    def _safe_divide(a, b) -> float | None:
        if a is not None and b is not None and b != 0:
            return a / b
        return None

    @staticmethod
    def _clean_df(df: pd.DataFrame | None, name: str, missing: list) -> pd.DataFrame | None:
        """Transpone y limpia un DataFrame de yfinance, ordenando por fecha ascendente."""
        if df is None or df.empty:
            missing.append(name)
            return None
        df = df.T.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        df.columns = [str(c) for c in df.columns]
        return df
