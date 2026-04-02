"""
Componentes de gráficos para el frontend Streamlit.
Usa Plotly para gráficos interactivos.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def price_chart(
    price_df: pd.DataFrame,
    ticker: str,
    entry_low: float | None = None,
    entry_high: float | None = None,
    stop: float | None = None,
    target_partial: float | None = None,
    target_full: float | None = None,
    fair_value_base: float | None = None,
    sma_50: float | None = None,
    sma_200: float | None = None,
    support_1: float | None = None,
    resistance_1: float | None = None,
) -> go.Figure:
    """
    Gráfico de precios con velas japonesas + volumen + niveles clave.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Velas
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df["open"],
            high=price_df["high"],
            low=price_df["low"],
            close=price_df["close"],
            name="Precio",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Medias móviles
    if sma_50:
        fig.add_hline(y=sma_50, line_dash="dot", line_color="#FFA726",
                      annotation_text=f"SMA50 {sma_50:.2f}", row=1, col=1)
    if sma_200:
        fig.add_hline(y=sma_200, line_dash="dash", line_color="#AB47BC",
                      annotation_text=f"SMA200 {sma_200:.2f}", row=1, col=1)

    # Niveles del plan
    if entry_low and entry_high:
        fig.add_hrect(
            y0=entry_low, y1=entry_high,
            fillcolor="rgba(38, 166, 154, 0.15)", line_width=0,
            annotation_text="Zona entrada razonable",
            annotation_position="right", row=1, col=1,
        )
    if stop:
        fig.add_hline(y=stop, line_color="#ef5350", line_dash="dashdot",
                      annotation_text=f"Stop {stop:.2f}", row=1, col=1)
    if target_partial:
        fig.add_hline(y=target_partial, line_color="#42A5F5", line_dash="dot",
                      annotation_text=f"Objetivo parcial {target_partial:.2f}", row=1, col=1)
    if target_full:
        fig.add_hline(y=target_full, line_color="#66BB6A", line_dash="dot",
                      annotation_text=f"Obj. revisión {target_full:.2f}", row=1, col=1)
    if fair_value_base:
        fig.add_hline(y=fair_value_base, line_color="#FFCA28", line_dash="longdash",
                      annotation_text=f"Valor razonable {fair_value_base:.2f}", row=1, col=1)
    if support_1:
        fig.add_hline(y=support_1, line_color="#78909C", line_dash="dot",
                      annotation_text=f"Soporte {support_1:.2f}", row=1, col=1)
    if resistance_1:
        fig.add_hline(y=resistance_1, line_color="#FF7043", line_dash="dot",
                      annotation_text=f"Resistencia {resistance_1:.2f}", row=1, col=1)

    # Volumen
    if "volume" in price_df.columns:
        colors = [
            "#26a69a" if price_df["close"].iloc[i] >= price_df["open"].iloc[i] else "#ef5350"
            for i in range(len(price_df))
        ]
        fig.add_trace(
            go.Bar(
                x=price_df.index,
                y=price_df["volume"],
                name="Volumen",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=f"{ticker} — Precio histórico con niveles clave",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=40, t=60, b=20),
    )
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)

    return fig


def score_radar(
    fundamental_score: float,
    financial_strength: float,
    growth_score: float,
    efficiency_score: float,
    valuation_score: float,
    technical_score: float,
) -> go.Figure:
    """Radar chart con los scores principales del análisis."""
    categories = [
        "Calidad negocio", "Fortaleza financiera", "Crecimiento",
        "Eficiencia", "Valoración", "Técnico",
    ]
    values = [
        fundamental_score, financial_strength, growth_score,
        efficiency_score, valuation_score, technical_score,
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        name="Score",
        line_color="#42A5F5",
        fillcolor="rgba(66, 165, 245, 0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        template="plotly_dark",
        title="Perfil de análisis (0–10)",
        height=380,
        margin=dict(l=40, r=40, t=50, b=20),
    )
    return fig


def valuation_waterfall(
    fv_conservative: float,
    fv_base: float,
    fv_optimistic: float,
    current_price: float,
) -> go.Figure:
    """Gráfico de barras horizontales para comparar rangos de valoración."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Rango valoración",
        x=["Conservador", "Base", "Optimista", "Precio actual"],
        y=[fv_conservative, fv_base, fv_optimistic, current_price],
        marker_color=["#78909C", "#42A5F5", "#66BB6A", "#FFCA28"],
        text=[f"{v:.2f}" for v in [fv_conservative, fv_base, fv_optimistic, current_price]],
        textposition="outside",
    ))

    fig.update_layout(
        title="Rangos de valoración vs. precio actual",
        template="plotly_dark",
        height=350,
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis_title="Precio estimado",
    )
    return fig


def rsi_gauge(rsi_value: float) -> go.Figure:
    """Medidor de RSI."""
    if rsi_value >= 70:
        color = "#ef5350"
    elif rsi_value <= 30:
        color = "#26a69a"
    else:
        color = "#42A5F5"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "RSI (14)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "rgba(38, 166, 154, 0.3)"},
                {"range": [30, 70], "color": "rgba(66, 165, 245, 0.1)"},
                {"range": [70, 100], "color": "rgba(239, 83, 80, 0.3)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": rsi_value,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
