"""
Fábrica de proveedores de datos.
Añadir un nuevo proveedor: crear la clase en un nuevo archivo,
implementar DataProvider, y registrarla aquí.
"""
from .base import DataProvider, FundamentalData, FinancialStatements, PriceHistory
from .yfinance_provider import YFinanceProvider


def get_provider(name: str = "yfinance") -> DataProvider:
    """Devuelve la instancia del proveedor según nombre."""
    registry: dict[str, type[DataProvider]] = {
        "yfinance": YFinanceProvider,
        # "alpha_vantage": AlphaVantageProvider,  # añadir cuando esté implementado
        # "finnhub": FinnhubProvider,
    }
    if name not in registry:
        raise ValueError(
            f"Proveedor '{name}' no registrado. Opciones: {list(registry.keys())}"
        )
    return registry[name]()


__all__ = [
    "DataProvider",
    "FundamentalData",
    "FinancialStatements",
    "PriceHistory",
    "YFinanceProvider",
    "get_provider",
]
