"""
Modelos de entrada validados con Pydantic.
Define exactamente qué puede pedir el usuario y con qué restricciones.
"""
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class TimeHorizon(str, Enum):
    short = "short"    # < 3 meses
    medium = "medium"  # 3–18 meses
    long = "long"      # > 18 meses


class RiskProfile(str, Enum):
    conservative = "conservative"
    balanced = "balanced"
    aggressive = "aggressive"


class AnalysisStyle(str, Enum):
    fundamental = "fundamental"
    technical = "technical"
    mixed = "mixed"


class ValuationApproach(str, Enum):
    base = "base"               # Sin margen de seguridad adicional
    conservative = "conservative"  # Margen de seguridad mayor


class AnalysisRequest(BaseModel):
    """Solicitud completa de análisis para un ticker."""

    ticker: str = Field(
        ...,
        description="Símbolo bursátil, e.g. AAPL, MSFT, IBE.MC",
        min_length=1,
        max_length=20,
    )
    market: str | None = Field(
        default=None,
        description="Mercado/bolsa opcional si el ticker es ambiguo, e.g. MC para Madrid",
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.medium,
        description="Horizonte temporal del análisis",
    )
    risk_profile: RiskProfile = Field(
        default=RiskProfile.balanced,
        description="Perfil de riesgo del inversor",
    )
    analysis_style: AnalysisStyle = Field(
        default=AnalysisStyle.mixed,
        description="Qué tipo de análisis priorizar",
    )
    valuation_approach: ValuationApproach = Field(
        default=ValuationApproach.base,
        description="Enfoque de valoración: base o con margen de seguridad exigente",
    )

    # Supuestos DCF editables
    dcf_growth_rate_override: float | None = Field(
        default=None,
        ge=-0.5,
        le=1.0,
        description="Tasa de crecimiento manual para DCF (0.10 = 10%). Si None, se estima del histórico.",
    )
    dcf_discount_rate_override: float | None = Field(
        default=None,
        ge=0.04,
        le=0.30,
        description="Tasa de descuento manual para DCF. Si None, se estima según beta y perfil.",
    )
    dcf_terminal_growth_override: float | None = Field(
        default=None,
        ge=0.0,
        le=0.05,
        description="Crecimiento terminal manual para DCF.",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()
