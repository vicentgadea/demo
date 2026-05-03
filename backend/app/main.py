"""
FastAPI — Punto de entrada de la API de análisis.

Endpoints:
  POST /analyze          — análisis completo (single ticker)
  POST /scanner/scan     — scanner multi-ticker (Market → Sector → Value)
  GET  /company/{ticker} — info básica de empresa
  GET  /health           — health check
"""
import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .models.inputs import AnalysisRequest
from .models.outputs import AnalysisResponse, CompanyInfo
from .models.scanner_models import ScannerRequest, ScannerResponse
from .providers import get_provider
from .analysis import (
    analyze_fundamental,
    analyze_technical,
    analyze_valuation,
    analyze_entry_exit,
    analyze_risk,
    analyze_synthesis,
)
from .analysis.scanner_orchestrator import run_scanner

# ── Logging ───────────────────────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stock Analyzer API",
    description="Herramienta profesional de análisis de empresas cotizadas. No constituye asesoramiento financiero.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción, restringir al origen del frontend
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Análisis completo ─────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def run_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """
    Análisis completo de una empresa cotizada.
    Combina análisis fundamental, técnico, valoración, gestión del riesgo
    y genera un plan orientativo de entrada/salida.
    """
    request_id = str(uuid.uuid4())
    ticker = request.ticker
    logger.info(f"[{request_id}] Análisis solicitado: {ticker} | {request.model_dump()}")

    try:
        provider = get_provider(settings.data_provider)

        # 1. Obtener datos
        logger.info(f"[{request_id}] Obteniendo datos de {ticker}...")
        fundamentals = provider.get_fundamentals(ticker)
        statements = provider.get_financial_statements(ticker)
        price_data = provider.get_price_history(ticker, period="2y", interval="1d")

        # Intentar obtener benchmark (SPY) para fortaleza relativa
        benchmark = None
        try:
            benchmark = provider.get_price_history("SPY", period="2y", interval="1d")
        except Exception:
            logger.warning(f"[{request_id}] No se pudo obtener benchmark SPY")

        # 2. Análisis por módulos
        logger.info(f"[{request_id}] Ejecutando análisis fundamental...")
        fundamental = analyze_fundamental(fundamentals, statements)

        logger.info(f"[{request_id}] Ejecutando análisis técnico...")
        technical = analyze_technical(price_data, benchmark)

        logger.info(f"[{request_id}] Ejecutando análisis de valoración...")
        valuation = analyze_valuation(
            fundamentals, statements,
            valuation_approach=request.valuation_approach,
            dcf_growth_override=request.dcf_growth_rate_override,
            dcf_discount_override=request.dcf_discount_rate_override,
            dcf_terminal_override=request.dcf_terminal_growth_override,
        )

        logger.info(f"[{request_id}] Calculando plan de entrada/salida...")
        entry_exit = analyze_entry_exit(
            fundamental, valuation, technical,
            risk_profile=request.risk_profile,
            time_horizon=request.time_horizon,
        )

        logger.info(f"[{request_id}] Calculando riesgo...")
        risk = analyze_risk(fundamental, valuation, technical, request.risk_profile)

        logger.info(f"[{request_id}] Generando síntesis...")
        synthesis = analyze_synthesis(
            fundamental, valuation, technical, risk,
            risk_profile=request.risk_profile,
            analysis_style=request.analysis_style,
            time_horizon=request.time_horizon,
            company_name=fundamentals.name or ticker,
        )

        company_info = CompanyInfo(
            ticker=ticker,
            name=fundamentals.name,
            sector=fundamentals.sector,
            industry=fundamentals.industry,
            country=fundamentals.country,
            currency=fundamentals.currency,
            market_cap=fundamentals.market_cap,
            current_price=fundamentals.current_price,
            price_52w_high=fundamentals.price_52w_high,
            price_52w_low=fundamentals.price_52w_low,
            description=fundamentals.description,
        )

        response = AnalysisResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            company_info=company_info,
            fundamental=fundamental,
            valuation=valuation,
            technical=technical,
            entry_exit=entry_exit,
            risk=risk,
            synthesis=synthesis,
            data_provider=provider.name,
            analysis_params=request.model_dump(exclude={"ticker"}),
        )

        logger.info(f"[{request_id}] Análisis completado. Score: {synthesis.overall_score}")
        return response

    except ValueError as e:
        logger.warning(f"[{request_id}] Error de validación: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[{request_id}] Error inesperado al analizar {ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener o procesar datos para {ticker}. "
                   f"Verifica que el ticker es válido y hay conexión a internet. Detalle: {str(e)}",
        )


# ── Info de empresa ───────────────────────────────────────────────────────────
@app.get("/company/{ticker}", response_model=CompanyInfo, tags=["Data"])
def get_company_info(ticker: str) -> CompanyInfo:
    """Devuelve información básica de la empresa sin análisis completo."""
    ticker = ticker.strip().upper()
    try:
        provider = get_provider(settings.data_provider)
        f = provider.get_fundamentals(ticker)
        return CompanyInfo(
            ticker=ticker,
            name=f.name,
            sector=f.sector,
            industry=f.industry,
            country=f.country,
            currency=f.currency,
            market_cap=f.market_cap,
            current_price=f.current_price,
            price_52w_high=f.price_52w_high,
            price_52w_low=f.price_52w_low,
            description=f.description,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No se encontró información para '{ticker}': {e}")


# ── Scanner multi-ticker ──────────────────────────────────────────────────────
@app.post("/scanner/scan", response_model=ScannerResponse, tags=["Scanner"])
def scanner_scan(request: ScannerRequest) -> ScannerResponse:
    """
    Scanner multi-ticker siguiendo la metodología Market → Sector → Value
    de TuForoDeBolsa. Analiza hasta 50 tickers en paralelo y devuelve
    señales de entrada con puntuación técnica + fundamental combinada.
    """
    try:
        # Normalise ticker list
        request.tickers = [t.strip().upper() for t in request.tickers if t.strip()]
        if not request.tickers:
            raise HTTPException(status_code=422, detail="La lista de tickers está vacía.")

        logger.info(
            f"Scanner request: {len(request.tickers)} tickers | "
            f"portfolio={request.portfolio_type} | ichimoku={request.enable_ichimoku}"
        )
        return run_scanner(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scanner error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error en el scanner: {str(e)}",
        )
