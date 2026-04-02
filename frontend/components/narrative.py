"""
Componentes de narrativa y conclusión para el frontend.
Muestra el análisis en lenguaje natural con formato claro y profesional.
"""
import streamlit as st


def render_executive_summary(data: dict) -> None:
    """Panel A: Resumen ejecutivo."""
    company = data.get("company_info", {})
    synthesis = data.get("synthesis", {})
    valuation = data.get("valuation", {})
    risk = data.get("risk", {})
    technical = data.get("technical", {})

    score = synthesis.get("overall_score", 0)
    score_color = _score_color(score)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        price = company.get("current_price") or valuation.get("current_price")
        currency = company.get("currency", "")
        st.metric("Precio actual", f"{price:.2f} {currency}" if price else "—")
    with col2:
        st.metric(
            "Score global",
            f"{score:.1f}/10",
            delta=_score_label(score),
            delta_color="off",
        )
    with col3:
        val_level = valuation.get("valuation_level", "—").replace("_", " ")
        discount = valuation.get("discount_to_base")
        st.metric(
            "Valoración",
            val_level.capitalize(),
            delta=f"{discount:+.1f}% vs. base" if discount is not None else None,
        )
    with col4:
        trend = technical.get("trend_medium", "neutral").replace("_", " ")
        st.metric("Tendencia media", trend.capitalize())
    with col5:
        risk_level = risk.get("overall_risk_level", "—")
        st.metric("Riesgo", risk_level.capitalize())

    # 52W range
    high_52 = company.get("price_52w_high")
    low_52 = company.get("price_52w_low")
    if high_52 and low_52 and price:
        pos_in_range = (price - low_52) / (high_52 - low_52) * 100 if high_52 != low_52 else 50
        st.caption(
            f"Rango 52 semanas: {low_52:.2f} — {high_52:.2f} | "
            f"Posición actual: {pos_in_range:.0f}% del rango"
        )


def render_fundamental_narrative(data: dict) -> None:
    """Panel B: Diagnóstico fundamental."""
    fund = data.get("fundamental", {})
    narrative = fund.get("narrative", "Sin narrativa disponible.")
    quality = fund.get("quality_score", {})
    strength = fund.get("financial_strength_score", {})
    growth_sc = fund.get("growth_score", {})
    efficiency = fund.get("efficiency_score", {})

    st.markdown(f"**{quality.get('label', '')}** | **{strength.get('label', '')}**")
    st.markdown(narrative)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fortalezas:**")
        for f in quality.get("factors", []) + strength.get("factors", []):
            st.markdown(f"  ✓ {f}")
    with col2:
        st.markdown("**Alertas:**")
        for w in quality.get("warnings", []) + strength.get("warnings", []):
            st.markdown(f"  ⚠ {w}")

    if fund.get("data_gaps"):
        st.caption(f"Datos no disponibles: {', '.join(fund['data_gaps'][:5])}")


def render_valuation_narrative(data: dict) -> None:
    """Panel C: Diagnóstico de valoración."""
    val = data.get("valuation", {})
    narrative = val.get("narrative", "Sin narrativa de valoración.")
    dcf = val.get("dcf")

    st.markdown(narrative)

    col1, col2, col3 = st.columns(3)
    with col1:
        fv_c = val.get("fair_value_conservative")
        st.metric("Valor conservador", f"{fv_c:.2f}" if fv_c else "—")
    with col2:
        fv_b = val.get("fair_value_base")
        st.metric("Valor base", f"{fv_b:.2f}" if fv_b else "—")
    with col3:
        fv_o = val.get("fair_value_optimistic")
        st.metric("Valor optimista", f"{fv_o:.2f}" if fv_o else "—")

    if dcf:
        with st.expander("Supuestos DCF"):
            st.markdown(dcf.get("assumptions_note", ""))
            cols = st.columns(3)
            cols[0].metric("Tasa crecimiento", f"{dcf.get('growth_rate_used', 0):.1%}")
            cols[1].metric("WACC", f"{dcf.get('discount_rate_used', 0):.1%}")
            cols[2].metric("Crec. terminal", f"{dcf.get('terminal_growth_used', 0):.1%}")
            if dcf.get("sensitivity_low") and dcf.get("sensitivity_high"):
                st.caption(
                    f"Sensibilidad (±2pp crecimiento): "
                    f"{dcf['sensitivity_low']:.2f} — {dcf['sensitivity_high']:.2f}"
                )


def render_technical_narrative(data: dict) -> None:
    """Panel D: Diagnóstico técnico."""
    tech = data.get("technical", {})
    narrative = tech.get("narrative", "Sin narrativa técnica.")

    st.markdown(narrative)

    momentum = tech.get("momentum", {})
    mas = tech.get("moving_averages", {})
    sr = tech.get("support_resistance", {})
    vol = tech.get("volatility", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        rsi = momentum.get("rsi_14")
        rsi_interp = momentum.get("rsi_interpretation", "")
        st.metric("RSI (14)", f"{rsi:.1f}" if rsi else "—", delta=rsi_interp)
        sma50 = mas.get("sma_50")
        pct50 = mas.get("price_vs_sma50_pct")
        if sma50:
            st.metric("SMA50", f"{sma50:.2f}", delta=f"{pct50:+.1f}% vs precio" if pct50 else None)
    with col2:
        macd_cross = momentum.get("macd_cross", "none")
        st.metric("MACD", momentum.get("macd_histogram") and f"{momentum['macd_histogram']:.4f}" or "—",
                  delta=macd_cross.replace("_", " ") if macd_cross != "none" else None)
        sma200 = mas.get("sma_200")
        pct200 = mas.get("price_vs_sma200_pct")
        if sma200:
            st.metric("SMA200", f"{sma200:.2f}", delta=f"{pct200:+.1f}% vs precio" if pct200 else None)
    with col3:
        atr_pct = vol.get("atr_pct")
        hist_vol = vol.get("historical_volatility_30d")
        st.metric("ATR%", f"{atr_pct:.2f}%" if atr_pct else "—")
        if hist_vol:
            st.metric("Vol. histórica 30d (anual.)", f"{hist_vol:.1f}%")

    if tech.get("is_extended") and tech.get("extension_note"):
        st.warning(f"⚠ {tech['extension_note']}")


def render_entry_exit_narrative(data: dict) -> None:
    """Panel E: Plan operativo orientativo."""
    ee = data.get("entry_exit", {})
    current = ee.get("current_price", 0)

    st.info(f"**Acción inmediata sugerida:** {ee.get('immediate_action', '—')}")

    st.markdown("**Condiciones para entrar:**")
    for c in ee.get("conditions_for_entry", []):
        st.markdown(f"  → {c}")

    st.markdown("**Condiciones para salir antes de objetivos:**")
    for c in ee.get("conditions_for_exit", []):
        st.markdown(f"  → {c}")

    col1, col2 = st.columns(2)
    with col1:
        rr_c = ee.get("risk_reward_conservative")
        rr_r = ee.get("risk_reward_reasonable")
        if rr_c:
            st.metric("R/R (entrada conservadora)", f"{rr_c:.2f}x")
        if rr_r:
            st.metric("R/R (entrada razonable)", f"{rr_r:.2f}x")
    with col2:
        st.markdown(f"**Invalidación fundamental:** {ee.get('invalidation_fundamental', '—')}")
        st.markdown(f"**Deterioro estructural:** {ee.get('invalidation_structural', '—')[:150]}...")


def render_risks(data: dict) -> None:
    """Panel F: Riesgos principales."""
    risk = data.get("risk", {})
    factors = risk.get("risk_factors", [])

    level = risk.get("overall_risk_level", "—")
    col1, col2, col3 = st.columns(3)
    col1.metric("Riesgo global", level.capitalize())
    col2.metric("Bajada a soporte", f"-{risk.get('downside_to_support', 0):.1f}%" if risk.get("downside_to_support") else "—")
    col3.metric("Riesgo compresión múltiplos", (risk.get("multiple_compression_risk") or "—").capitalize())

    if factors:
        severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        for factor in factors:
            icon = severity_icons.get(factor.get("severity", "low"), "⚪")
            with st.expander(f"{icon} {factor.get('name', '')} ({factor.get('severity', '')})"):
                st.markdown(factor.get("description", ""))

    st.caption(risk.get("position_size_note", ""))


def render_conclusion(data: dict) -> None:
    """Panel G: Conclusión para inversor."""
    synthesis = data.get("synthesis", {})
    score = synthesis.get("overall_score", 0)

    st.markdown(f"### Score global: {score:.1f}/10")
    st.markdown(f"**Tesis:** {synthesis.get('investment_thesis', '—')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fortalezas principales:**")
        for s in synthesis.get("key_strengths", [])[:5]:
            st.markdown(f"  ✓ {s}")
    with col2:
        st.markdown("**Debilidades principales:**")
        for w in synthesis.get("key_weaknesses", [])[:5]:
            st.markdown(f"  ✗ {w}")

    worth = synthesis.get("worth_following", False)
    attractive = synthesis.get("current_price_attractive")

    st.markdown(f"**¿Merece seguimiento?** {'Sí' if worth else 'No'} — {synthesis.get('worth_following_reason', '')}")

    if attractive is True:
        st.success(f"**El precio actual parece atractivo.** {synthesis.get('attractiveness_conditions', '')}")
    elif attractive is False:
        st.warning(f"**El precio actual no es especialmente atractivo ahora.** {synthesis.get('attractiveness_conditions', '')}")
    else:
        st.info(f"**Datos insuficientes para opinar sobre el precio.** {synthesis.get('attractiveness_conditions', '')}")

    st.caption(synthesis.get("weighting_note", ""))

    # Disclaimers
    with st.expander("Avisos legales y limitaciones"):
        for d in data.get("disclaimers", []):
            st.markdown(f"• {d}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 7:
        return "green"
    elif score >= 5:
        return "orange"
    return "red"


def _score_label(score: float) -> str:
    if score >= 8:
        return "Excelente"
    elif score >= 6.5:
        return "Bueno"
    elif score >= 5:
        return "Moderado"
    elif score >= 3:
        return "Débil"
    return "Muy débil"
