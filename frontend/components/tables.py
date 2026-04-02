"""
Componentes de tablas para mostrar datos estructurados en Streamlit.
"""
import pandas as pd
import streamlit as st


def fundamentals_table(analysis: dict) -> None:
    """Muestra tabla de métricas fundamentales."""
    fundamental = analysis.get("fundamental", {})
    growth = fundamental.get("growth", {})
    margins = fundamental.get("margins", {})
    efficiency = fundamental.get("efficiency", {})
    debt = fundamental.get("debt", {})
    cashflow = fundamental.get("cash_flow", {})

    rows = []

    # Crecimiento
    rows.append({"Categoría": "Crecimiento", "Métrica": "Crecimiento ingresos 1A",
                 "Valor": _fmt_pct(growth.get("revenue_growth_1y")), "Nota": ""})
    rows.append({"Categoría": "Crecimiento", "Métrica": "CAGR ingresos 3A",
                 "Valor": _fmt_pct(growth.get("revenue_cagr_3y")), "Nota": ""})
    rows.append({"Categoría": "Crecimiento", "Métrica": "Crecimiento EPS 1A",
                 "Valor": _fmt_pct(growth.get("eps_growth_1y")), "Nota": ""})
    rows.append({"Categoría": "Crecimiento", "Métrica": "CAGR EPS 3A",
                 "Valor": _fmt_pct(growth.get("eps_cagr_3y")), "Nota": ""})

    # Márgenes
    rows.append({"Categoría": "Márgenes", "Métrica": "Margen bruto",
                 "Valor": _fmt_pct(margins.get("gross_margin")),
                 "Nota": margins.get("gross_margin_trend", "")})
    rows.append({"Categoría": "Márgenes", "Métrica": "Margen operativo",
                 "Valor": _fmt_pct(margins.get("operating_margin")),
                 "Nota": margins.get("operating_margin_trend", "")})
    rows.append({"Categoría": "Márgenes", "Métrica": "Margen neto",
                 "Valor": _fmt_pct(margins.get("net_margin")), "Nota": ""})

    # Eficiencia
    rows.append({"Categoría": "Eficiencia", "Métrica": "ROE",
                 "Valor": _fmt_pct(efficiency.get("roe")), "Nota": ""})
    rows.append({"Categoría": "Eficiencia", "Métrica": "ROA",
                 "Valor": _fmt_pct(efficiency.get("roa")), "Nota": ""})

    # Deuda
    rows.append({"Categoría": "Deuda", "Métrica": "Deuda neta / EBITDA",
                 "Valor": _fmt_x(debt.get("net_debt_ebitda")), "Nota": ""})
    rows.append({"Categoría": "Deuda", "Métrica": "Cobertura intereses",
                 "Valor": _fmt_x(debt.get("interest_coverage")), "Nota": ""})
    rows.append({"Categoría": "Deuda", "Métrica": "Deuda/Equity",
                 "Valor": _fmt_x(debt.get("debt_to_equity")), "Nota": ""})
    rows.append({"Categoría": "Deuda", "Métrica": "Current ratio",
                 "Valor": _fmt_x(debt.get("current_ratio")), "Nota": ""})

    # Cash flow
    rows.append({"Categoría": "Cash Flow", "Métrica": "FCF yield",
                 "Valor": _fmt_pct(cashflow.get("fcf_yield")),
                 "Nota": cashflow.get("fcf_consistency", "")})
    rows.append({"Categoría": "Cash Flow", "Métrica": "Capex/Revenue",
                 "Valor": _fmt_pct(cashflow.get("capex_to_revenue")), "Nota": ""})

    df = pd.DataFrame(rows)
    df = df[df["Valor"] != "—"]  # ocultar filas sin datos

    st.dataframe(
        df.style.apply(_color_rows, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def valuation_table(analysis: dict) -> None:
    """Muestra tabla de múltiplos de valoración."""
    multiples = analysis.get("valuation", {}).get("multiples", [])
    if not multiples:
        st.info("No hay múltiplos disponibles")
        return

    rows = []
    for m in multiples:
        rows.append({
            "Múltiplo": m.get("name", ""),
            "Actual": _fmt_x(m.get("current")),
            "Referencia sector": _fmt_x(m.get("historical_avg") or m.get("sector_avg")),
            "Precio implícito": _fmt_price(m.get("implied_fair_price")),
            "Interpretación": (m.get("interpretation") or "")[:80],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def scores_table(analysis: dict) -> None:
    """Muestra resumen de puntuaciones."""
    fundamental = analysis.get("fundamental", {})
    synthesis = analysis.get("synthesis", {})

    rows = [
        {"Módulo": "Calidad del negocio", "Score": _score_bar(fundamental.get("quality_score", {}).get("score"))},
        {"Módulo": "Fortaleza financiera", "Score": _score_bar(fundamental.get("financial_strength_score", {}).get("score"))},
        {"Módulo": "Crecimiento", "Score": _score_bar(fundamental.get("growth_score", {}).get("score"))},
        {"Módulo": "Eficiencia", "Score": _score_bar(fundamental.get("efficiency_score", {}).get("score"))},
        {"Módulo": "Score global", "Score": _score_bar(synthesis.get("overall_score"))},
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def entry_exit_table(analysis: dict) -> None:
    """Muestra el plan de entrada/salida."""
    ee = analysis.get("entry_exit", {})
    current = ee.get("current_price", 0)

    rows = [
        {"Nivel": "Entrada conservadora",
         "Rango": f"{ee.get('entry_conservative', {}).get('low', 0):.2f} – {ee.get('entry_conservative', {}).get('high', 0):.2f}",
         "vs. Precio actual": _pct_diff(ee.get("entry_conservative", {}).get("high", 0), current),
         "Justificación": ee.get("entry_conservative", {}).get("rationale", "")[:60]},
        {"Nivel": "Entrada razonable",
         "Rango": f"{ee.get('entry_reasonable', {}).get('low', 0):.2f} – {ee.get('entry_reasonable', {}).get('high', 0):.2f}",
         "vs. Precio actual": _pct_diff(ee.get("entry_reasonable", {}).get("high", 0), current),
         "Justificación": ee.get("entry_reasonable", {}).get("rationale", "")[:60]},
        {"Nivel": "Entrada agresiva",
         "Rango": f"{ee.get('entry_aggressive', {}).get('low', 0):.2f} – {ee.get('entry_aggressive', {}).get('high', 0):.2f}",
         "vs. Precio actual": _pct_diff(ee.get("entry_aggressive", {}).get("high", 0), current),
         "Justificación": ee.get("entry_aggressive", {}).get("rationale", "")[:60]},
        {"Nivel": "Stop técnico",
         "Rango": f"{ee.get('stop_technical', 0):.2f}",
         "vs. Precio actual": _pct_diff(ee.get("stop_technical", 0), current),
         "Justificación": ee.get("stop_technical_rationale", "")[:60]},
        {"Nivel": "Objetivo parcial",
         "Rango": f"{ee.get('target_partial', 0):.2f}",
         "vs. Precio actual": _pct_diff(ee.get("target_partial", 0), current),
         "Justificación": ee.get("target_partial_rationale", "")[:60]},
        {"Nivel": "Objetivo revisión",
         "Rango": f"{ee.get('target_full_or_review', 0):.2f}",
         "vs. Precio actual": _pct_diff(ee.get("target_full_or_review", 0), current),
         "Justificación": ee.get("target_full_rationale", "")[:60]},
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de formato
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}%"


def _fmt_x(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}x"


def _fmt_price(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def _score_bar(score) -> str:
    if score is None:
        return "—"
    filled = int(round(score))
    return f"{'█' * filled}{'░' * (10 - filled)} {score:.1f}/10"


def _pct_diff(level: float, current: float) -> str:
    if not level or not current:
        return "—"
    diff = (level - current) / current * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}%"


def _color_rows(row):
    """Aplicar color sutil según categoría."""
    cat = row.get("Categoría", "")
    colors = {
        "Crecimiento": "background-color: rgba(66, 165, 245, 0.05)",
        "Márgenes": "background-color: rgba(38, 166, 154, 0.05)",
        "Eficiencia": "background-color: rgba(171, 71, 188, 0.05)",
        "Deuda": "background-color: rgba(239, 83, 80, 0.05)",
        "Cash Flow": "background-color: rgba(255, 167, 38, 0.05)",
    }
    style = colors.get(cat, "")
    return [style] * len(row)
