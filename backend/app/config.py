"""
Configuración centralizada de la aplicación.
Carga variables desde .env mediante pydantic-settings.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Servidor
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Proveedor de datos
    data_provider: str = "yfinance"
    alpha_vantage_api_key: str = ""
    finnhub_api_key: str = ""

    # Caché (segundos)
    cache_ttl_prices: int = 300
    cache_ttl_fundamentals: int = 3600

    # Frontend
    backend_url: str = "http://localhost:8000"

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Instancia singleton de Settings."""
    return Settings()
