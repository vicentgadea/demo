"""
Stock Analyzer — Frontend Streamlit
Interfaz profesional de análisis de empresas cotizadas.

Requisitos:
  - Backend FastAPI corriendo en BACKEND_URL (ver .env)
  - pip install -r requirements.txt

Ejecutar: streamlit run frontend/app.py
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

from components.charts import price_chart, score_radar, valuation_waterfall, rsi_gauge
from components.tables import fundamentals_table, valuation_table, scores_table, entry_exit_table
from components.narrative import (
    render_executive_summary,
    render_fundamental_narrative,
    render_valuation_narrative,
    render_technical_narrative,
    render_entry_exit_narrative,
    render_risks,
    render_conclusion,
)

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    h1 { color: #42A5F5; }
    h2 { color: #90CAF9; border-bottom: 1px solid #37474F; padding-bottom: 4px; }
    h3 { color: #B3E5FC; }
    .stMetric label { font-size: 0.75rem; color: #90A4AE; }
    .stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Stock Analyzer")
    st.caption("Herramienta de análisis de empresas cotizadas")
    st.divider()

    ticker_input = st.text_input(
        "Ticker", placeholder="AAPL, MSFT, IBE.MC...", help="Introduce el símbolo bursátil"
    ).strip().upper()

    market_input = st.text_input(
        "Mercado (opcional)", placeholder="MC, L, PA...",
        help="Sufijo de mercado si el ticker es ambiguo"
    )

    st.divider()
    st.subheader("Parámetros de análisis")

    time_horizon = st.selectbox(
        "Horizonte temporal",
        options=["medium", "long", "short"],
        format_func=lambda x: {"short": "Corto plazo (<3m)", "medium": "Medio plazo (3-18m)", "long": "Largo plazo (>18m)"}[x],
        index=0,
    )

    risk_profile = st.selectbox(
        "Perfil de riesgo",
        options=["balanced", "conservative", "aggressive"],
        format_func=lambda x: {"conservative": "Conservador", "balanced": "Equilibrado", "aggressive": "Agresivo"}[x],
        index=0,
    )

    analysis_style = st.selectbox(
        "Estilo de análisis",
        options=["mixed", "fundamental", "technical"],
        format_func=lambda x: {"fundamental": "Fundamental", "technical": "Técnico", "mixed": "Mixto"}[x],
        index=0,
    )

    valuation_approach = st.selectbox(
        "Enfoque de valoración",
        options=["base", "conservative"],
        format_func=lambda x: {"base": "Estándar", "conservative": "Con margen de seguridad (+30%)"}[x],
        index=0,
    )

    with st.expander("Supuestos DCF (avanzado)"):
        dcf_growth = st.number_input(
            "Tasa de crecimiento (%)", min_value=-50.0, max_value=100.0,
            value=0.0, step=0.5,
            help="0 = estimación automática"
        )
        dcf_discount = st.number_input(
            "Tasa de descuento/WACC (%)", min_value=4.0, max_value=30.0,
            value=0.0, step=0.5,
            help="0 = estimación automática"
        )
        dcf_terminal = st.number_input(
            "Crecimiento terminal (%)", min_value=0.0, max_value=5.0,
            value=0.0, step=0.1,
            help="0 = 2.5% por defecto"
        )

    st.divider()
    analyze_btn = st.button("Analizar", type="primary", use_container_width=True, disabled=not ticker_input)

    st.caption(
        "⚠ Esta herramienta es de apoyo a la decisión. No constituye asesoramiento financiero."
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if not ticker_input and not st.session_state.get("last_result"):
    st.title("Análisis profesional de empresas cotizadas")
    st.markdown("""
    Herramienta de análisis fundamental, técnico y de valoración para uso personal.

    **Cómo usar:**
    1. Introduce el ticker en el panel izquierdo (ej. `AAPL`, `MSFT`, `IBE.MC`)
    2. Elige tu horizonte temporal, perfil de riesgo y estilo de análisis
    3. Pulsa **Analizar**

    **Qué obtienes:**
    - Diagnóstico fundamental: calidad del negocio, márgenes, deuda, crecimiento
    - Análisis de valoración: múltiplos + DCF con rangos conservador/base/optimista
    - Análisis técnico: tendencia, soportes, RSI, MACD
    - Plan orientativo de entrada/salida con zonas probabilísticas
    - Gestión del riesgo: stop técnico, invalidación de tesis
    - Conclusión narrativa explicada

    **Importante:** Los niveles de entrada/salida son orientativos y probabilísticos.
    No son predicciones ni recomendaciones de inversión.
    """)
    st.stop()

# ── Obtener datos de precio para el gráfico ────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(ticker: str) -> pd.DataFrame | None:
    """Obtiene historial de precios directamente con yfinance para el gráfico."""
    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)
        df = tkr.history(period="2y", interval="1d", auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                 "Close": "close", "Volume": "volume"})
        return df
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def call_analyze_api(payload: dict) -> dict | None:
    """Llama al backend FastAPI para el análisis completo."""
    try:
        resp = requests.post(f"{BACKEND_URL}/analyze", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"No se pudo conectar al backend en {BACKEND_URL}. ¿Está corriendo?"}
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": f"Error del servidor: {detail}"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}


# ── Disparar análisis ──────────────────────────────────────────────────────────
if analyze_btn and ticker_input:
    payload = {
        "ticker": ticker_input,
        "market": market_input or None,
        "time_horizon": time_horizon,
        "risk_profile": risk_profile,
        "analysis_style": analysis_style,
        "valuation_approach": valuation_approach,
        "dcf_growth_rate_override": dcf_growth / 100 if dcf_growth != 0 else None,
        "dcf_discount_rate_override": dcf_discount / 100 if dcf_discount != 0 else None,
        "dcf_terminal_growth_override": dcf_terminal / 100 if dcf_terminal != 0 else None,
    }

    with st.spinner(f"Analizando {ticker_input}..."):
        start = time.time()
        result = call_analyze_api(payload)
        elapsed = time.time() - start

    if result and "error" not in result:
        st.session_state["last_result"] = result
        st.session_state["last_ticker"] = ticker_input
        st.success(f"Análisis completado en {elapsed:.1f}s")
    elif result:
        st.error(result["error"])
        st.stop()

# ── Mostrar resultados ─────────────────────────────────────────────────────────
result = st.session_state.get("last_result")
if not result:
    st.stop()

company = result.get("company_info", {})
ticker_display = result.get("ticker", "")
company_name = company.get("name") or ticker_display

st.title(f"{company_name} ({ticker_display})")

sector = company.get("sector")
industry = company.get("industry")
country = company.get("country")
if sector or industry:
    st.caption(f"{sector or ''} {'|' if sector and industry else ''} {industry or ''} {'|' if country else ''} {country or ''}")

# ── A. Resumen ejecutivo ───────────────────────────────────────────────────────
st.header("A. Resumen ejecutivo")
render_executive_summary(result)

st.divider()

# ── Gráfico de precios ────────────────────────────────────────────────────────
st.header("Gráfico de precio")
price_df = fetch_price_data(ticker_display)
if price_df is not None:
    ee = result.get("entry_exit", {})
    val = result.get("valuation", {})
    tech = result.get("technical", {})
    mas = tech.get("moving_averages", {})
    sr = tech.get("support_resistance", {})

    fig = price_chart(
        price_df=price_df,
        ticker=ticker_display,
        entry_low=ee.get("entry_reasonable", {}).get("low"),
        entry_high=ee.get("entry_reasonable", {}).get("high"),
        stop=ee.get("stop_technical"),
        target_partial=ee.get("target_partial"),
        target_full=ee.get("target_full_or_review"),
        fair_value_base=val.get("fair_value_base"),
        sma_50=mas.get("sma_50"),
        sma_200=mas.get("sma_200"),
        support_1=sr.get("support_1"),
        resistance_1=sr.get("resistance_1"),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No se pudo cargar el historial de precios para el gráfico.")

st.divider()

# ── Scores radar ──────────────────────────────────────────────────────────────
col_radar, col_val = st.columns([1, 1])

with col_radar:
    fund = result.get("fundamental", {})
    val = result.get("valuation", {})
    tech = result.get("technical", {})
    synthesis = result.get("synthesis", {})

    # Score técnico aproximado
    trend_map = {
        "strong_bullish": 9, "bullish": 7, "neutral": 5, "bearish": 3, "strong_bearish": 1
    }
    tech_score_approx = trend_map.get(tech.get("trend_medium", "neutral"), 5)

    val_level_map = {
        "undervalued": 9, "fair": 7, "demanding": 4.5, "overvalued": 2, "insufficient_data": 5
    }
    val_score_approx = val_level_map.get(val.get("valuation_level", "insufficient_data"), 5)

    fig_radar = score_radar(
        fundamental_score=fund.get("quality_score", {}).get("score", 5),
        financial_strength=fund.get("financial_strength_score", {}).get("score", 5),
        growth_score=fund.get("growth_score", {}).get("score", 5),
        efficiency_score=fund.get("efficiency_score", {}).get("score", 5),
        valuation_score=val_score_approx,
        technical_score=tech_score_approx,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_val:
    fv_c = val.get("fair_value_conservative")
    fv_b = val.get("fair_value_base")
    fv_o = val.get("fair_value_optimistic")
    curr = company.get("current_price") or val.get("current_price")
    if fv_c and fv_b and fv_o and curr:
        fig_wf = valuation_waterfall(fv_c, fv_b, fv_o, curr)
        st.plotly_chart(fig_wf, use_container_width=True)

st.divider()

# ── B. Análisis fundamental ───────────────────────────────────────────────────
st.header("B. Diagnóstico fundamental")

tab_fund1, tab_fund2 = st.tabs(["Tabla de métricas", "Narrativa"])
with tab_fund1:
    fundamentals_table(result)
with tab_fund2:
    render_fundamental_narrative(result)

st.divider()

# ── C. Valoración ─────────────────────────────────────────────────────────────
st.header("C. Diagnóstico de valoración")

tab_val1, tab_val2 = st.tabs(["Tabla de múltiplos", "Narrativa"])
with tab_val1:
    valuation_table(result)
with tab_val2:
    render_valuation_narrative(result)

st.divider()

# ── D. Análisis técnico ───────────────────────────────────────────────────────
st.header("D. Diagnóstico técnico")

col_tech1, col_tech2 = st.columns([2, 1])
with col_tech1:
    render_technical_narrative(result)
with col_tech2:
    rsi_val = result.get("technical", {}).get("momentum", {}).get("rsi_14")
    if rsi_val:
        fig_rsi = rsi_gauge(rsi_val)
        st.plotly_chart(fig_rsi, use_container_width=True)

st.divider()

# ── E. Plan operativo ─────────────────────────────────────────────────────────
st.header("E. Plan operativo orientativo")
st.caption(
    "Los niveles son probabilísticos y orientativos. No son predicciones ni señales de compra/venta."
)

tab_ee1, tab_ee2 = st.tabs(["Tabla de niveles", "Narrativa"])
with tab_ee1:
    entry_exit_table(result)
with tab_ee2:
    render_entry_exit_narrative(result)

st.divider()

# ── F. Riesgos ────────────────────────────────────────────────────────────────
st.header("F. Riesgos principales")
render_risks(result)

st.divider()

# ── G. Conclusión ─────────────────────────────────────────────────────────────
st.header("G. Conclusión para inversor")
render_conclusion(result)

# ── Metadatos ─────────────────────────────────────────────────────────────────
with st.expander("Metadatos del análisis"):
    st.json({
        "request_id": result.get("request_id"),
        "timestamp": result.get("timestamp"),
        "data_provider": result.get("data_provider"),
        "analysis_params": result.get("analysis_params"),
        "data_gaps_fundamental": result.get("fundamental", {}).get("data_gaps", []),
        "data_gaps_valuation": result.get("valuation", {}).get("data_gaps", []),
        "data_gaps_technical": result.get("technical", {}).get("data_gaps", []),
    })
