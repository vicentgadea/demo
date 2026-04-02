"""
Interfaz abstracta de proveedor de datos.
Cualquier fuente nueva (Alpha Vantage, Finnhub, Polygon...) debe implementar
esta clase para ser intercambiable sin tocar el resto de la app.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PriceHistory:
    ticker: str
    df: pd.DataFrame  # columnas: Open, High, Low, Close, Volume; índice: datetime
    currency: str = "USD"


@dataclass
class FundamentalData:
    ticker: str
    # Identificación
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str | None = None
    description: str | None = None

    # Precio
    current_price: float | None = None
    market_cap: float | None = None
    price_52w_high: float | None = None
    price_52w_low: float | None = None
    beta: float | None = None

    # Ratios de valoración
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_sales: float | None = None
    price_to_book: float | None = None
    ev_ebitda: float | None = None
    enterprise_value: float | None = None

    # Rentabilidad
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    ebitda_margin: float | None = None
    roe: float | None = None
    roa: float | None = None

    # Crecimiento (ratios directos si disponibles)
    revenue_growth: float | None = None
    earnings_growth: float | None = None

    # Balance
    total_cash: float | None = None
    total_debt: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    book_value_per_share: float | None = None

    # Cash flow
    free_cash_flow: float | None = None
    operating_cash_flow: float | None = None

    # Dividendo / recompras
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    shares_outstanding: float | None = None

    # EPS
    trailing_eps: float | None = None
    forward_eps: float | None = None

    # Datos faltantes
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class FinancialStatements:
    """Series históricas de estados financieros (últimos 4–5 años)."""
    ticker: str
    income_statement: pd.DataFrame | None = None   # revenue, gross_profit, ebit, net_income, eps
    balance_sheet: pd.DataFrame | None = None      # total_assets, total_debt, cash, equity
    cash_flow: pd.DataFrame | None = None          # operating_cf, capex, free_cf, shares_outstanding
    missing_fields: list[str] = field(default_factory=list)


class DataProvider(ABC):
    """Contrato que debe cumplir cualquier proveedor de datos financieros."""

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> PriceHistory:
        """Historial de precios OHLCV."""

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> FundamentalData:
        """Snapshot de métricas fundamentales actuales."""

    @abstractmethod
    def get_financial_statements(self, ticker: str) -> FinancialStatements:
        """Estados financieros históricos (últimos años)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificativo del proveedor."""
