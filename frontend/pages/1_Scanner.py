"""
Stock & ETF Entry Signal Scanner — TuForoDeBolsa Methodology
Dashboard Streamlit: Market → Sector → Value

Portfolios:
  Portfolio 1 (DEGIRO): Swing trading / growth — stocks
  Portfolio 2 (Trade Republic): Bogleheads ETF — long term
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Scanner — TuForoDeBolsa",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    h1 { color: #42A5F5; font-size: 1.8rem; }
    h2 { color: #90CAF9; border-bottom: 1px solid #37474F; padding-bottom: 4px; }
    h3 { color: #B3E5FC; }
    .stMetric label { font-size: 0.72rem; color: #90A4AE; }
    .signal-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-weight: 600; font-size: 0.8rem;
    }
    .badge-fuerte       { background: #1B5E20; color: #A5D6A7; }
    .badge-recomendada  { background: #2E7D32; color: #C8E6C9; }
    .badge-posible      { background: #F57F17; color: #FFF9C4; }
    .badge-esperar      { background: #37474F; color: #CFD8DC; }
    .badge-no_entrar    { background: #B71C1C; color: #FFCDD2; }
    .score-bar { height: 8px; border-radius: 4px; margin-top: 2px; }
    table { width: 100%; }
    th { font-size: 0.75rem; color: #78909C !important; }
    td { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Default tickers ────────────────────────────────────────────────────────────
DEFAULT_SWING_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "V", "UNH",
    "ASML", "SAP", "NOVO-B.CO", "SAN.MC", "IBE.MC",
]

DEFAULT_ETF_TICKERS = [
    "SPY", "QQQ", "VTI", "VEA", "VWO",
    "IWDA.AS", "VWRL.AS", "EQQQ.AS", "VUSA.AS",
    "XLK", "XLV", "XLF", "XLE",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "ENTRADA_FUERTE":    ("#1B5E20", "#A5D6A7", "🟢"),
    "ENTRADA_RECOMENDADA": ("#2E7D32", "#C8E6C9", "🟩"),
    "POSIBLE_ENTRADA":   ("#E65100", "#FFE0B2", "🟡"),
    "ESPERAR":           ("#37474F", "#CFD8DC", "⏳"),
    "NO_ENTRAR":         ("#7F0000", "#FFCDD2", "🔴"),
}

REGIME_ICON = {
    "MERCADO_ALCISTA": ("🟢", "Alcista"),
    "MERCADO_BAJISTA": ("🔴", "Bajista"),
    "MERCADO_LATERAL": ("🟡", "Lateral"),
}

SECTOR_ICON = {
    "SECTOR_LIDER":    ("🟢", "Líder"),
    "SECTOR_NEUTRO":   ("🟡", "Neutro"),
    "SECTOR_REZAGADO": ("🔴", "Rezagado"),
}

TECH_SIGNAL_COLORS = {
    "MUY_FUERTE": "#1B5E20",
    "FUERTE":     "#2E7D32",
    "MODERADA":   "#E65100",
    "DEBIL":      "#7F0000",
}

FUND_LEVEL_COLORS = {
    "EXCELENTE": "#1B5E20",
    "SÓLIDO":    "#2E7D32",
    "MODERADO":  "#E65100",
    "DÉBIL":     "#7F0000",
}


def score_color(score: float) -> str:
    if score >= 76:
        return "#4CAF50"
    elif score >= 56:
        return "#8BC34A"
    elif score >= 35:
        return "#FF9800"
    return "#F44336"


def fmt_score_bar(score: float, label: str = "") -> str:
    color = score_color(score)
    return f"""
    <div style="font-size:0.75rem; color:#90A4AE;">{label}</div>
    <div style="background:#263238; border-radius:4px; height:6px; width:100%;">
      <div style="background:{color}; border-radius:4px; height:6px; width:{score:.0f}%;"></div>
    </div>
    <div style="font-size:0.72rem; color:{color}; text-align:right;">{score:.0f}/100</div>
    """


def fmt_signal_badge(signal: str) -> str:
    bg, fg, icon = SIGNAL_COLORS.get(signal, ("#37474F", "#CFD8DC", "?"))
    label = signal.replace("_", " ")
    return f'<span style="background:{bg}; color:{fg}; padding:2px 10px; border-radius:10px; font-size:0.78rem; font-weight:600;">{icon} {label}</span>'


def fmt_market_badge(regime: str) -> str:
    icon, label = REGIME_ICON.get(regime, ("❓", regime))
    return f"{icon} **{label}**"


def fmt_sector_badge(status: str) -> str:
    icon, label = SECTOR_ICON.get(status, ("❓", status))
    return f"{icon} {label}"


def call_scanner_api(payload: dict) -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/scanner/scan",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"No se pudo conectar al backend en {BACKEND_URL}."}
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": f"Error del servidor: {detail}"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}


def _render_indicator_pill(label: str, value: float | None, ok: bool | None = None) -> str:
    if value is None:
        return f'<span style="color:#546E7A; font-size:0.75rem;">{label}: N/A</span>'
    color = "#4CAF50" if ok else ("#F44336" if ok is False else "#78909C")
    return (
        f'<span style="background:#1E272C; border:1px solid {color}; '
        f'border-radius:6px; padding:1px 6px; font-size:0.73rem; color:{color}; margin:1px;">'
        f'{label}: {value:.1f}</span>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Scanner")
    st.caption("TuForoDeBolsa — Market → Sector → Value")
    st.page_link("app.py", label="📊 Analyzer (single ticker)", icon="📊")
    st.divider()

    portfolio_type = st.radio(
        "Portfolio",
        options=["swing", "etf"],
        format_func=lambda x: {
            "swing": "📈 Portfolio 1 — Swing/Growth (DEGIRO)",
            "etf": "🌍 Portfolio 2 — Bogleheads ETF (TR)",
        }[x],
        index=0,
    )

    default_tickers = DEFAULT_ETF_TICKERS if portfolio_type == "etf" else DEFAULT_SWING_TICKERS
    tickers_input = st.text_area(
        "Tickers (uno por línea o separados por coma)",
        value="\n".join(default_tickers),
        height=200,
        help="Hasta 50 tickers. Para mercados europeos usar sufijo: IBE.MC, AMS.AS, SAP.DE…",
    )

    st.divider()
    st.subheader("Opciones")

    enable_ichimoku = st.checkbox(
        "Activar Ichimoku",
        value=False,
        help="Añade análisis de nube Ichimoku (más preciso pero más lento)",
    )
    override_sector = st.checkbox(
        "Ignorar filtro de sector",
        value=False,
        help="Analizar tickers aunque su sector esté rezagado",
    )
    min_score = st.slider(
        "Score mínimo para mostrar",
        min_value=0,
        max_value=80,
        value=0,
        step=5,
    )
    only_entries = st.checkbox(
        "Solo mostrar señales de entrada",
        value=False,
        help="Oculta señales ESPERAR y NO_ENTRAR",
    )

    st.divider()
    scan_btn = st.button("🔍 Ejecutar Scanner", type="primary", use_container_width=True)

    st.caption("⚠ No constituye asesoramiento financiero.")


# ── Parse tickers ──────────────────────────────────────────────────────────────
raw_text = tickers_input.replace(",", "\n").replace(";", "\n")
tickers = [t.strip().upper() for t in raw_text.splitlines() if t.strip()]
tickers = list(dict.fromkeys(tickers))[:50]  # deduplicate, max 50

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📡 Stock & ETF Entry Signal Scanner")
st.markdown(
    "**Metodología TuForoDeBolsa** | Análisis Mercado → Sector → Valor | "
    "Indicadores: SMA/EMA, MACD, RSI, Estocástico, Fibonacci, Koncorde (simulado), Soportes/Resistencias"
)

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    portfolio_label = "Swing/Growth (60% técnico)" if portfolio_type == "swing" else "Bogleheads ETF (70% fundamental)"
    st.info(f"**Portfolio:** {portfolio_label}")
with col_info2:
    st.info(f"**Tickers cargados:** {len(tickers)}")
with col_info3:
    opts = []
    if enable_ichimoku:
        opts.append("Ichimoku ON")
    if override_sector:
        opts.append("Filtro sector OFF")
    st.info(f"**Opciones:** {', '.join(opts) if opts else 'Estándar'}")

# ── Run scan ───────────────────────────────────────────────────────────────────
if scan_btn:
    if not tickers:
        st.error("Introduce al menos un ticker.")
        st.stop()

    payload = {
        "tickers": tickers,
        "portfolio_type": portfolio_type,
        "enable_ichimoku": enable_ichimoku,
        "override_sector_filter": override_sector,
        "min_combined_score": min_score,
        "only_entry_signals": only_entries,
        "market_ticker": "SPY",
    }

    with st.spinner(f"Escaneando {len(tickers)} tickers... (puede tardar hasta 2 minutos)"):
        t0 = time.time()
        data = call_scanner_api(payload)
        elapsed = time.time() - t0

    if data and "error" not in data:
        st.session_state["scanner_result"] = data
        st.session_state["scanner_elapsed"] = elapsed
        st.success(
            f"Scan completado en {elapsed:.1f}s | "
            f"Procesados: {data.get('tickers_processed', 0)} | "
            f"Errores: {data.get('tickers_with_errors', 0)}"
        )
    elif data:
        st.error(data["error"])
        st.stop()

# ── Display results ────────────────────────────────────────────────────────────
data = st.session_state.get("scanner_result")
if not data:
    st.markdown("---")
    st.markdown("""
    ### Cómo usar el Scanner

    1. **Selecciona el portfolio** en la barra lateral
    2. **Edita la lista de tickers** (formato: `AAPL`, `IBE.MC`, `ASML.AS`…)
    3. **Activa opciones** avanzadas si lo deseas (Ichimoku, override sector)
    4. **Pulsa "Ejecutar Scanner"**

    El scanner analiza cada ticker en tres niveles:
    - **MERCADO:** Estado del S&P 500 (SMA50/200, Golden Cross, pendiente)
    - **SECTOR:** Fuerza relativa del sector vs SPY (20/50/200 días)
    - **VALOR:** 10 familias de indicadores técnicos + análisis fundamental

    **Señales de salida:**
    - 🟢 ENTRADA FUERTE (score ≥ 81)
    - 🟩 ENTRADA RECOMENDADA (score 66-80)
    - 🟡 POSIBLE ENTRADA (score 51-65)
    - ⏳ ESPERAR (score 35-50)
    - 🔴 NO ENTRAR (score < 35)
    """)
    st.stop()

# ── Market Context Banner ─────────────────────────────────────────────────────
mkt = data.get("market_context", {})
regime = mkt.get("regime", "")
strength = mkt.get("strength", 1)
regime_icon, regime_label = REGIME_ICON.get(regime, ("❓", regime))

strength_stars = "★" * strength + "☆" * (3 - strength)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Mercado (SPY)", f"{regime_icon} {regime_label}", f"Fuerza {strength_stars}")
with col_m2:
    spy_price = mkt.get("price")
    spy_sma200 = mkt.get("sma200")
    delta = None
    if spy_price and spy_sma200:
        delta = f"vs SMA200: {(spy_price/spy_sma200-1)*100:+.1f}%"
    st.metric("SPY Precio", f"${spy_price:.2f}" if spy_price else "N/A", delta)
with col_m3:
    gc = "✅ Golden Cross" if mkt.get("golden_cross") else ("⚠️ Death Cross" if mkt.get("death_cross") else "Sin cruces recientes")
    st.metric("Cruces SMA", gc)
with col_m4:
    slope = mkt.get("sma50_slope_10d")
    st.metric("Pendiente SMA50", f"{slope:+.3f}%/día" if slope is not None else "N/A")
with col_m5:
    top_opps = data.get("top_opportunities", [])
    st.metric("Top Oportunidades", ", ".join(top_opps[:3]) if top_opps else "—")

st.caption(mkt.get("description", ""))
st.divider()

# ── Results table ─────────────────────────────────────────────────────────────
results = data.get("results", [])
if not results:
    st.warning("No se encontraron resultados con los filtros aplicados.")
    st.stop()

st.subheader(f"Resultados del Scan ({len(results)} tickers)")

# Build DataFrame for the overview table
table_rows = []
for r in results:
    if r.get("error"):
        table_rows.append({
            "Ticker": r["ticker"],
            "Empresa": "— Error —",
            "Precio": "—",
            "Δ 1d": "—",
            "Técnico": 0,
            "Fundamental": 0,
            "Combinado": 0,
            "Señal": "ERROR",
            "Mercado": "—",
            "Sector": "—",
            "Alertas": r.get("error", ""),
        })
        continue

    sec_ctx = r.get("sector_context") or {}
    tech = r.get("technical_score") or {}
    fund = r.get("fundamental_score") or {}
    sec_status = sec_ctx.get("status", "")
    sec_icon = SECTOR_ICON.get(sec_status, ("❓", ""))[0]
    mkt_r = r.get("market_context") or {}
    mkt_regime = mkt_r.get("regime", "")
    mkt_icon = REGIME_ICON.get(mkt_regime, ("❓", ""))[0]

    price_val = r.get("current_price")
    price_str = f"${price_val:.2f}" if price_val else "—"
    change = r.get("price_change_pct_1d")
    change_str = f"{change:+.1f}%" if change is not None else "—"

    table_rows.append({
        "Ticker": r["ticker"],
        "Empresa": (r.get("company_name") or "")[:30],
        "Precio": price_str,
        "Δ 1d": change_str,
        "Técnico": round(tech.get("normalized_score", 0), 1),
        "Fundamental": round(fund.get("normalized_score", 0), 1),
        "Combinado": round(r.get("combined_score", 0), 1),
        "Señal": r.get("entry_signal", "ESPERAR"),
        "Mercado": mkt_icon,
        "Sector": f"{sec_icon} {sec_ctx.get('sector_name', '')[:12]}",
        "Alertas": "; ".join(r.get("key_alerts", [])[:2]),
    })

df_table = pd.DataFrame(table_rows)

# Style the dataframe
def _color_signal(val):
    colors = {
        "ENTRADA_FUERTE": "background-color: #1B5E20; color: #A5D6A7;",
        "ENTRADA_RECOMENDADA": "background-color: #2E7D32; color: #C8E6C9;",
        "POSIBLE_ENTRADA": "background-color: #4E342E; color: #FFCCBC;",
        "ESPERAR": "background-color: #1C313A; color: #B0BEC5;",
        "NO_ENTRAR": "background-color: #7F0000; color: #FFCDD2;",
        "ERROR": "background-color: #212121; color: #F44336;",
    }
    return colors.get(val, "")


def _color_score(val):
    try:
        v = float(val)
        if v >= 76:
            return "color: #4CAF50; font-weight: 600;"
        elif v >= 56:
            return "color: #8BC34A;"
        elif v >= 35:
            return "color: #FF9800;"
        return "color: #F44336;"
    except Exception:
        return ""


styled = (
    df_table.style
    .applymap(_color_signal, subset=["Señal"])
    .applymap(_color_score, subset=["Técnico", "Fundamental", "Combinado"])
    .set_properties(**{"font-size": "0.82rem"})
    .hide(axis="index")
)
st.dataframe(styled, use_container_width=True, height=min(600, 50 + len(df_table) * 36))

# ── Signal distribution chart ─────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px

signal_counts = df_table["Señal"].value_counts()
signal_order = ["ENTRADA_FUERTE", "ENTRADA_RECOMENDADA", "POSIBLE_ENTRADA", "ESPERAR", "NO_ENTRAR", "ERROR"]
signal_colors_map = {
    "ENTRADA_FUERTE": "#4CAF50",
    "ENTRADA_RECOMENDADA": "#8BC34A",
    "POSIBLE_ENTRADA": "#FF9800",
    "ESPERAR": "#607D8B",
    "NO_ENTRAR": "#F44336",
    "ERROR": "#424242",
}

col_chart1, col_chart2 = st.columns([1, 2])
with col_chart1:
    st.subheader("Distribución de señales")
    labels = [s for s in signal_order if s in signal_counts.index]
    values = [signal_counts[s] for s in labels]
    colors = [signal_colors_map[s] for s in labels]

    fig_pie = go.Figure(go.Pie(
        labels=[l.replace("_", " ") for l in labels],
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig_pie.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font_color="#CFD8DC", showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10), height=280,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_chart2:
    st.subheader("Scores combinados")
    valid_results = [r for r in results if not r.get("error")]
    if valid_results:
        tickers_sorted = [r["ticker"] for r in valid_results]
        scores_combined = [r.get("combined_score", 0) for r in valid_results]
        scores_tech = [r.get("technical_score", {}).get("normalized_score", 0) for r in valid_results]
        scores_fund = [r.get("fundamental_score", {}).get("normalized_score", 0) for r in valid_results]
        bar_colors = [signal_colors_map.get(r.get("entry_signal", "ESPERAR"), "#607D8B") for r in valid_results]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Técnico",
            x=tickers_sorted,
            y=scores_tech,
            marker_color="#42A5F5",
            opacity=0.6,
        ))
        fig_bar.add_trace(go.Bar(
            name="Fundamental",
            x=tickers_sorted,
            y=scores_fund,
            marker_color="#AB47BC",
            opacity=0.6,
        ))
        fig_bar.add_trace(go.Scatter(
            name="Combinado",
            x=tickers_sorted,
            y=scores_combined,
            mode="markers+lines",
            marker=dict(color=bar_colors, size=9, symbol="diamond"),
            line=dict(color="#FFB300", width=1.5),
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
            font_color="#CFD8DC", barmode="group",
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[0, 105], title="Score (0-100)"),
            margin=dict(l=10, r=10, t=30, b=60),
            height=280,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Detail cards per ticker ────────────────────────────────────────────────────
st.subheader("Detalle por Ticker")

# Filter controls
col_f1, col_f2 = st.columns([1, 3])
with col_f1:
    signal_filter = st.multiselect(
        "Filtrar por señal",
        options=["ENTRADA_FUERTE", "ENTRADA_RECOMENDADA", "POSIBLE_ENTRADA", "ESPERAR", "NO_ENTRAR"],
        default=[],
        format_func=lambda x: x.replace("_", " "),
    )
with col_f2:
    ticker_filter = st.multiselect(
        "Filtrar por ticker",
        options=[r["ticker"] for r in results],
        default=[],
    )

# Apply filters to detail view
display_results = results
if signal_filter:
    display_results = [r for r in display_results if r.get("entry_signal") in signal_filter]
if ticker_filter:
    display_results = [r for r in display_results if r["ticker"] in ticker_filter]

for r in display_results:
    ticker_name = r["ticker"]
    company_name = r.get("company_name") or ticker_name
    signal = r.get("entry_signal", "ESPERAR")
    bg_color, fg_color, sig_icon = SIGNAL_COLORS.get(signal, ("#37474F", "#CFD8DC", "?"))

    with st.expander(
        f"{sig_icon} {ticker_name} — {company_name[:40]} | Score: {r.get('combined_score', 0):.0f}/100",
        expanded=False,
    ):
        if r.get("error"):
            st.error(f"Error al analizar este ticker: {r['error']}")
            continue

        tech = r.get("technical_score") or {}
        fund = r.get("fundamental_score") or {}
        sec_ctx = r.get("sector_context") or {}
        mkt_ctx = r.get("market_context") or {}

        # ── Header row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            price = r.get("current_price")
            chg = r.get("price_change_pct_1d")
            chg_str = f"{chg:+.2f}%" if chg is not None else "—"
            st.metric("Precio", f"${price:.2f}" if price else "—", chg_str)
        with c2:
            st.markdown(fmt_signal_badge(signal), unsafe_allow_html=True)
            st.markdown(f"Confianza: **{r.get('confidence', 0)*100:.0f}%**")
        with c3:
            st.markdown(fmt_score_bar(tech.get("normalized_score", 0), "Score Técnico"), unsafe_allow_html=True)
        with c4:
            st.markdown(fmt_score_bar(fund.get("normalized_score", 0), "Score Fundamental"), unsafe_allow_html=True)

        # ── Justification
        justification = r.get("signal_justification", "")
        if justification:
            st.caption(justification)

        # ── Alerts
        alerts = r.get("key_alerts", [])
        if alerts:
            for a in alerts:
                st.warning(f"⚠ {a}", icon=None)

        tab_tech, tab_fund, tab_ctx = st.tabs(["Indicadores Técnicos", "Fundamentales", "Contexto Mercado/Sector"])

        # ── Technical tab
        with tab_tech:
            tech_level = tech.get("signal_level", "DEBIL")
            tech_color = TECH_SIGNAL_COLORS.get(tech_level, "#607D8B")
            st.markdown(
                f'Señal técnica: <span style="color:{tech_color}; font-weight:600;">{tech_level}</span> '
                f'({tech.get("normalized_score", 0):.1f}/100 | {tech.get("total_raw_points", 0)} puntos brutos)',
                unsafe_allow_html=True,
            )

            # Indicators in columns
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                # Moving Averages
                ma_data = tech.get("ma") or {}
                with st.container():
                    st.markdown("**A) Medias Móviles**")
                    sma20 = ma_data.get("sma20")
                    sma50 = ma_data.get("sma50")
                    sma200 = ma_data.get("sma200")
                    if sma20:
                        st.markdown(
                            _render_indicator_pill("SMA20", sma20)
                            + " " + _render_indicator_pill("SMA50", sma50)
                            + " " + _render_indicator_pill("SMA200", sma200),
                            unsafe_allow_html=True,
                        )
                    if ma_data.get("aligned_bullish"):
                        st.success("Alineación alcista perfecta")
                    if ma_data.get("golden_cross"):
                        st.success("Golden Cross")
                    if ma_data.get("death_cross"):
                        st.error("Death Cross")
                    for d in (ma_data.get("details") or []):
                        color = "#4CAF50" if d.startswith("+") else "#F44336"
                        st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

                # MACD
                macd_data = tech.get("macd") or {}
                st.markdown("**B) MACD (12,26,9)**")
                ml = macd_data.get("macd_line")
                sig = macd_data.get("signal_line")
                hist = macd_data.get("histogram")
                st.markdown(
                    _render_indicator_pill("MACD", ml, ml > 0 if ml is not None else None)
                    + " " + _render_indicator_pill("Signal", sig)
                    + " " + _render_indicator_pill("Hist", hist, hist > 0 if hist is not None else None),
                    unsafe_allow_html=True,
                )
                for d in (macd_data.get("details") or []):
                    color = "#4CAF50" if d.startswith("+") else "#F44336"
                    st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

                # RSI
                rsi_data = tech.get("rsi") or {}
                st.markdown("**C) RSI 14**")
                rsi_val = rsi_data.get("rsi")
                rsi_ok = rsi_val is not None and 40 <= rsi_val <= 65
                st.markdown(
                    _render_indicator_pill("RSI", rsi_val, rsi_ok),
                    unsafe_allow_html=True,
                )
                for d in (rsi_data.get("details") or []):
                    color = "#4CAF50" if d.startswith("+") else "#F44336"
                    st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

                # Stochastic
                stoch_data = tech.get("stochastic") or {}
                st.markdown("**D) Estocástico (14,3,3)**")
                pk = stoch_data.get("pct_k")
                pd_val = stoch_data.get("pct_d")
                stoch_ok = pk is not None and pk < 50
                st.markdown(
                    _render_indicator_pill("%K", pk, stoch_ok)
                    + " " + _render_indicator_pill("%D", pd_val),
                    unsafe_allow_html=True,
                )
                for d in (stoch_data.get("details") or []):
                    color = "#4CAF50" if d.startswith("+") else "#F44336"
                    st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

            with col_t2:
                # Volume
                vol_data = tech.get("volume") or {}
                st.markdown("**E) Volumen**")
                vol_ratio = vol_data.get("vol_ratio_20d")
                obv_trend = vol_data.get("obv_trend")
                vol_ok = vol_ratio is not None and vol_ratio > 1.0
                st.markdown(
                    _render_indicator_pill("Ratio Vol", vol_ratio, vol_ok)
                    + f' <span style="font-size:0.73rem; color:#78909C;"> OBV: {obv_trend or "N/A"}</span>',
                    unsafe_allow_html=True,
                )
                for d in (vol_data.get("details") or []):
                    color = "#4CAF50" if d.startswith("+") else "#F44336"
                    st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

                # S&R
                sr_data = tech.get("support_resistance") or {}
                st.markdown("**F) Soportes & Resistencias**")
                sup1 = sr_data.get("support_1")
                res1 = sr_data.get("resistance_1")
                st.markdown(
                    _render_indicator_pill("Soporte", sup1, sr_data.get("near_support"))
                    + " " + _render_indicator_pill("Resistencia", res1),
                    unsafe_allow_html=True,
                )
                for d in (sr_data.get("details") or []):
                    color = "#4CAF50" if d.startswith("+") else "#F44336"
                    st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

                # Fibonacci
                fib_data = tech.get("fibonacci") or {}
                st.markdown("**G) Fibonacci**")
                fib_618 = fib_data.get("fib_618")
                fib_382 = fib_data.get("fib_382")
                if fib_data.get("in_fib_zone"):
                    zone = fib_data.get("fib_zone_label", "?")
                    st.success(f"En zona Fibonacci {zone}")
                st.markdown(
                    _render_indicator_pill("F61.8%", fib_618)
                    + " " + _render_indicator_pill("F38.2%", fib_382),
                    unsafe_allow_html=True,
                )
                for d in (fib_data.get("details") or []):
                    color = "#4CAF50" if d.startswith("+") else "#F44336"
                    st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

                # Koncorde
                konc_data = tech.get("koncorde") or {}
                st.markdown("**H) Koncorde (simulado)**")
                if konc_data.get("institutional_accumulation"):
                    st.success("Acumulación institucional detectada")
                if konc_data.get("distribution_signal"):
                    st.error("Señal de distribución")
                obv_n = konc_data.get("obv_normalized")
                if obv_n is not None:
                    st.markdown(
                        _render_indicator_pill("OBV norm.", obv_n, obv_n > 50),
                        unsafe_allow_html=True,
                    )
                for d in (konc_data.get("details") or []):
                    color = "#4CAF50" if d.startswith("+") else "#F44336"
                    st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

                # Ichimoku
                ichi_data = tech.get("ichimoku") or {}
                if ichi_data.get("enabled"):
                    st.markdown("**I) Ichimoku**")
                    if ichi_data.get("price_above_cloud"):
                        st.success("Precio sobre la nube")
                    if ichi_data.get("tk_cross_bullish"):
                        st.success("Cruce TK alcista")
                    if ichi_data.get("kumo_breakout"):
                        st.success("Kumo Breakout")
                    for d in (ichi_data.get("details") or []):
                        color = "#4CAF50" if d.startswith("+") else "#F44336"
                        st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

                # Chart patterns
                pat_data = tech.get("patterns") or {}
                st.markdown("**J) Figuras Chartistas**")
                pats_found = []
                if pat_data.get("double_bottom"):
                    pats_found.append("Doble Suelo")
                if pat_data.get("inv_head_shoulders"):
                    pats_found.append("HCH Invertido")
                if pat_data.get("bull_flag"):
                    pats_found.append("Bandera Alcista")
                if pats_found:
                    st.success(", ".join(pats_found))
                else:
                    st.caption("Sin patrones detectados")
                for d in (pat_data.get("details") or []):
                    color = "#4CAF50" if d.startswith("+") else "#F44336"
                    st.markdown(f'<span style="color:{color}; font-size:0.78rem;">{d}</span>', unsafe_allow_html=True)

        # ── Fundamental tab
        with tab_fund:
            fund_level = fund.get("level", "DÉBIL")
            fund_color = FUND_LEVEL_COLORS.get(fund_level, "#607D8B")
            st.markdown(
                f'Fundamentales: <span style="color:{fund_color}; font-weight:600;">{fund_level}</span> '
                f'({fund.get("normalized_score", 0):.1f}/100)',
                unsafe_allow_html=True,
            )

            # Key metrics grid
            col_f1, col_f2, col_f3 = st.columns(3)
            metrics_list = [
                ("PER", fund.get("per"), None, None),
                ("PEG", fund.get("peg"), None, lambda v: v < 1),
                ("P/S", fund.get("ps_ratio"), None, lambda v: v < 2),
                ("P/B", fund.get("pb_ratio"), None, lambda v: v < 1.5),
                ("EV/EBITDA", fund.get("ev_ebitda"), None, lambda v: v < 12),
                ("ROE", fund.get("roe"), "%", lambda v: v > 0.15),
                ("D/EBITDA", fund.get("debt_ebitda"), "x", lambda v: v < 2),
                ("Rev. Growth", fund.get("revenue_growth_yoy"), "%", lambda v: v > 0),
                ("EPS Growth", fund.get("eps_growth_yoy"), "%", lambda v: v > 0),
            ]
            for i, (label, val, unit, check) in enumerate(metrics_list):
                col = [col_f1, col_f2, col_f3][i % 3]
                with col:
                    if val is not None:
                        display_val = f"{val*100:.1f}{unit}" if unit == "%" else (f"{val:.2f}{unit or ''}")
                        ok = check(val) if check else None
                        color = "#4CAF50" if ok else ("#F44336" if ok is False else "#CFD8DC")
                        st.markdown(
                            f'<div style="font-size:0.72rem; color:#78909C;">{label}</div>'
                            f'<div style="font-size:1rem; color:{color}; font-weight:600;">{display_val}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="font-size:0.72rem; color:#78909C;">{label}</div>'
                            f'<div style="font-size:0.85rem; color:#546E7A;">N/A</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown("---")
            st.markdown("**Factores evaluados:**")
            for d in (fund.get("details") or []):
                color = "#4CAF50" if d.startswith("+") else "#F44336"
                st.markdown(f'<span style="color:{color}; font-size:0.8rem;">• {d}</span>', unsafe_allow_html=True)

            for a in (fund.get("alerts") or []):
                st.warning(f"⚠ {a}")

        # ── Context tab
        with tab_ctx:
            col_ctx1, col_ctx2 = st.columns(2)

            with col_ctx1:
                st.markdown("**NIVEL 1 — MERCADO**")
                regime_v = mkt_ctx.get("regime", "")
                ri, rl = REGIME_ICON.get(regime_v, ("?", regime_v))
                st.markdown(f"**Estado:** {ri} {rl}")
                mkt_str = mkt_ctx.get("strength", 1)
                st.markdown(f"**Fuerza:** {'★' * mkt_str + '☆' * (3 - mkt_str)}")
                spy_p = mkt_ctx.get("price")
                spy_s200 = mkt_ctx.get("sma200")
                if spy_p and spy_s200:
                    pct = (spy_p / spy_s200 - 1) * 100
                    st.markdown(f"**SPY vs SMA200:** {pct:+.1f}%")
                if mkt_ctx.get("golden_cross"):
                    st.success("Golden Cross reciente")
                if mkt_ctx.get("death_cross"):
                    st.error("Death Cross reciente")
                st.caption(mkt_ctx.get("description", ""))

            with col_ctx2:
                st.markdown("**NIVEL 2 — SECTOR**")
                sec_s = sec_ctx.get("status", "")
                si, sl = SECTOR_ICON.get(sec_s, ("?", sec_s))
                sec_name = sec_ctx.get("sector_name", "N/A")
                sec_etf = sec_ctx.get("sector_etf", "N/A")
                st.markdown(f"**Sector:** {sec_name}")
                st.markdown(f"**ETF proxy:** {sec_etf}")
                st.markdown(f"**Estado:** {si} {sl}")
                rs20 = sec_ctx.get("rs_20d")
                rs50 = sec_ctx.get("rs_50d")
                rs200 = sec_ctx.get("rs_200d")
                if rs20:
                    ok = rs20 >= 1.0
                    color = "#4CAF50" if ok else "#F44336"
                    st.markdown(f'RS 20d: <span style="color:{color};">{rs20:.2f}</span>', unsafe_allow_html=True)
                if rs50:
                    ok = rs50 >= 1.0
                    color = "#4CAF50" if ok else "#F44336"
                    st.markdown(f'RS 50d: <span style="color:{color};">{rs50:.2f}</span>', unsafe_allow_html=True)
                if rs200:
                    ok = rs200 >= 1.0
                    color = "#4CAF50" if ok else "#F44336"
                    st.markdown(f'RS 200d: <span style="color:{color};">{rs200:.2f}</span>', unsafe_allow_html=True)
                st.caption(sec_ctx.get("description", ""))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠ Herramienta de apoyo a la decisión de inversión. "
    "No constituye asesoramiento financiero. "
    "Los indicadores son probabilísticos y orientativos. "
    "Siempre aplica tu propio juicio y gestión del riesgo."
)
elapsed = st.session_state.get("scanner_elapsed", 0)
if elapsed:
    st.caption(
        f"Último scan: {data.get('timestamp', '')} | "
        f"Duración: {elapsed:.1f}s | "
        f"Request ID: {data.get('request_id', '')} | "
        f"Tickers procesados: {data.get('tickers_processed', 0)}"
    )
