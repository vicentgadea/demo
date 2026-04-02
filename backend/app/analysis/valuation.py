"""
Módulo de valoración.
Combina múltiplos históricos, comparativa sectorial (cuando hay datos) y DCF simplificado.
Produce un rango de valor razonable (conservador / base / optimista) y lo compara
con el precio actual para determinar si hay descuento, prima o precio justo.

Principios:
- Cada múltiplo produce un precio implícito y se justifica.
- El DCF muestra sus supuestos y sensibilidad explícita.
- La valoración final es una síntesis ponderada, no un número mágico.
"""
import logging
from dataclasses import dataclass

import numpy as np

from ..providers.base import FundamentalData, FinancialStatements
from ..models.inputs import ValuationApproach
from ..models.outputs import (
    ValuationAnalysis,
    ValuationMultiple,
    DCFResult,
    ValuationLevel,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Múltiplos de referencia por sector (benchmark razonable cuando no hay datos reales)
# En producción, esto se puede sustituir por una llamada a API de datos sectoriales.
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_BENCHMARKS: dict[str, dict[str, float]] = {
    "Technology": {"pe": 28, "ev_ebitda": 20, "ps": 7},
    "Consumer Cyclical": {"pe": 22, "ev_ebitda": 14, "ps": 1.5},
    "Consumer Defensive": {"pe": 20, "ev_ebitda": 13, "ps": 1.2},
    "Healthcare": {"pe": 24, "ev_ebitda": 15, "ps": 3},
    "Financial Services": {"pe": 15, "ev_ebitda": None, "ps": 2.5},
    "Industrials": {"pe": 20, "ev_ebitda": 12, "ps": 1.5},
    "Energy": {"pe": 14, "ev_ebitda": 7, "ps": 0.8},
    "Utilities": {"pe": 17, "ev_ebitda": 10, "ps": 2},
    "Real Estate": {"pe": 30, "ev_ebitda": 18, "ps": 5},
    "Communication Services": {"pe": 20, "ev_ebitda": 12, "ps": 3},
    "Basic Materials": {"pe": 15, "ev_ebitda": 8, "ps": 1},
}
DEFAULT_BENCHMARK = {"pe": 20, "ev_ebitda": 12, "ps": 2}


def analyze_valuation(
    fundamentals: FundamentalData,
    statements: FinancialStatements,
    valuation_approach: ValuationApproach = ValuationApproach.base,
    dcf_growth_override: float | None = None,
    dcf_discount_override: float | None = None,
    dcf_terminal_override: float | None = None,
) -> ValuationAnalysis:
    """Punto de entrada principal del módulo de valoración."""
    gaps: list[str] = []
    current_price = fundamentals.current_price
    if not current_price:
        raise ValueError(f"No hay precio actual disponible para {fundamentals.ticker}")

    sector_bench = SECTOR_BENCHMARKS.get(fundamentals.sector or "", DEFAULT_BENCHMARK)
    multiples = _calc_multiples(fundamentals, sector_bench, gaps)
    dcf = _calc_dcf(
        fundamentals, statements,
        dcf_growth_override, dcf_discount_override, dcf_terminal_override,
        valuation_approach,
    )

    # Recoger precios implícitos de cada método
    implied_prices = [m.implied_fair_price for m in multiples if m.implied_fair_price]
    if dcf and dcf.fair_value:
        implied_prices.append(dcf.fair_value)

    fv_conservative, fv_base, fv_optimistic = _build_fair_value_ranges(
        implied_prices, dcf, valuation_approach
    )

    discount = None
    if fv_base:
        discount = (fv_base - current_price) / current_price * 100

    valuation_level = _classify_valuation(discount, current_price, fv_base, fv_conservative, fv_optimistic)
    dominant_method = _dominant_method(multiples, dcf, implied_prices)

    narrative = _build_narrative(
        fundamentals, multiples, dcf,
        fv_conservative, fv_base, fv_optimistic,
        current_price, discount, valuation_level, dominant_method,
    )

    return ValuationAnalysis(
        current_price=current_price,
        multiples=multiples,
        dcf=dcf,
        fair_value_conservative=_r(fv_conservative),
        fair_value_base=_r(fv_base),
        fair_value_optimistic=_r(fv_optimistic),
        discount_to_base=_r(discount, 1),
        valuation_level=valuation_level,
        dominant_method=dominant_method,
        narrative=narrative,
        data_gaps=list(set(gaps)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Múltiplos
# ─────────────────────────────────────────────────────────────────────────────

def _calc_multiples(
    f: FundamentalData,
    sector_bench: dict,
    gaps: list,
) -> list[ValuationMultiple]:
    multiples = []
    current = f.current_price

    # ── PER Trailing ──────────────────────────────────────────────────────────
    pe_hist_avg = sector_bench.get("pe", 20)  # usamos sector como proxy si no hay hist propio
    if f.pe_ratio and f.trailing_eps and f.trailing_eps > 0:
        implied = f.trailing_eps * pe_hist_avg
        if f.pe_ratio > pe_hist_avg * 1.5:
            interp = f"PER actual ({f.pe_ratio:.1f}x) muy por encima del referente sector ({pe_hist_avg}x)"
        elif f.pe_ratio < pe_hist_avg * 0.7:
            interp = f"PER actual ({f.pe_ratio:.1f}x) claramente por debajo del referente sector ({pe_hist_avg}x)"
        else:
            interp = f"PER actual ({f.pe_ratio:.1f}x) próximo al referente sector ({pe_hist_avg}x)"
        multiples.append(ValuationMultiple(
            name="PER Trailing",
            current=f.pe_ratio,
            historical_avg=pe_hist_avg,
            sector_avg=pe_hist_avg,
            implied_fair_price=_r(implied),
            interpretation=interp,
        ))
    else:
        gaps.append("pe_ratio")

    # ── Forward PER ───────────────────────────────────────────────────────────
    if f.forward_pe and f.forward_eps and f.forward_eps > 0:
        implied = f.forward_eps * pe_hist_avg
        multiples.append(ValuationMultiple(
            name="PER Forward",
            current=f.forward_pe,
            historical_avg=pe_hist_avg,
            implied_fair_price=_r(implied),
            interpretation=f"PER forward {f.forward_pe:.1f}x vs. referente {pe_hist_avg}x",
        ))

    # ── EV/EBITDA ─────────────────────────────────────────────────────────────
    ev_bench = sector_bench.get("ev_ebitda")
    if f.ev_ebitda and ev_bench:
        if f.enterprise_value and f.ev_ebitda > 0:
            ebitda_abs = f.enterprise_value / f.ev_ebitda
            shares = f.shares_outstanding or 1
            mktcap_at_fair = ebitda_abs * ev_bench
            # Precio implícito: ajustando deuda
            net_debt = (f.total_debt or 0) - (f.total_cash or 0)
            equity_val = mktcap_at_fair - net_debt
            implied = equity_val / shares if shares > 0 else None
        else:
            implied = None
        multiples.append(ValuationMultiple(
            name="EV/EBITDA",
            current=_r(f.ev_ebitda, 1),
            historical_avg=ev_bench,
            sector_avg=ev_bench,
            implied_fair_price=_r(implied),
            interpretation=f"EV/EBITDA {f.ev_ebitda:.1f}x vs. referente sector {ev_bench}x",
        ))
    elif f.ev_ebitda is None:
        gaps.append("ev_ebitda")

    # ── Price/Sales ───────────────────────────────────────────────────────────
    ps_bench = sector_bench.get("ps", 2.0)
    if f.price_to_sales:
        # Precio implícito: revenue/share * PS_benchmark
        rev_per_share = None
        if f.market_cap and f.shares_outstanding and f.price_to_sales > 0:
            rev_total = f.market_cap / f.price_to_sales
            rev_per_share = rev_total / f.shares_outstanding
        implied = rev_per_share * ps_bench if rev_per_share else None
        multiples.append(ValuationMultiple(
            name="Price/Sales",
            current=_r(f.price_to_sales, 1),
            historical_avg=ps_bench,
            sector_avg=ps_bench,
            implied_fair_price=_r(implied),
            interpretation=f"P/S {f.price_to_sales:.1f}x vs. referente {ps_bench}x",
        ))
    else:
        gaps.append("price_to_sales")

    # ── Price/FCF ─────────────────────────────────────────────────────────────
    if f.free_cash_flow and f.shares_outstanding and f.shares_outstanding > 0:
        fcf_per_share = f.free_cash_flow / f.shares_outstanding
        if fcf_per_share > 0:
            pfcf_current = current / fcf_per_share if current else None
            pfcf_bench = 20  # típico para empresa FCF-generadora de calidad media
            implied = fcf_per_share * pfcf_bench
            multiples.append(ValuationMultiple(
                name="Price/FCF",
                current=_r(pfcf_current, 1),
                historical_avg=pfcf_bench,
                implied_fair_price=_r(implied),
                interpretation=f"P/FCF {pfcf_current:.1f}x — precio implícito con ratio {pfcf_bench}x: {implied:.2f}" if pfcf_current else "P/FCF calculado",
            ))
    else:
        gaps.append("price_to_fcf")

    return multiples


# ─────────────────────────────────────────────────────────────────────────────
# DCF simplificado
# ─────────────────────────────────────────────────────────────────────────────

def _calc_dcf(
    f: FundamentalData,
    s: FinancialStatements,
    growth_override: float | None,
    discount_override: float | None,
    terminal_override: float | None,
    approach: ValuationApproach,
) -> DCFResult | None:
    """
    DCF en dos fases: crecimiento elevado (años 1-5) y madurez (años 6-10),
    con valor terminal por Gordon Growth Model.
    Base: FCF por acción o EPS si no hay FCF.
    """
    if not f.shares_outstanding or f.shares_outstanding == 0:
        return None

    # Determinar flujo base por acción
    base_flow = None
    flow_name = ""
    if f.free_cash_flow and f.free_cash_flow > 0:
        base_flow = f.free_cash_flow / f.shares_outstanding
        flow_name = "FCF por acción"
    elif f.trailing_eps and f.trailing_eps > 0:
        base_flow = f.trailing_eps * 0.7  # ajuste de payout conservador
        flow_name = "EPS ajustado"

    if not base_flow or base_flow <= 0:
        return None

    # Supuestos de crecimiento
    if growth_override is not None:
        g1 = growth_override  # fase 1
    else:
        g1 = _estimate_growth_rate(f, s)
    g2 = g1 * 0.6  # fase 2: desaceleración

    # Tasa de descuento
    if discount_override is not None:
        wacc = discount_override
    else:
        wacc = _estimate_wacc(f)

    # Crecimiento terminal
    if terminal_override is not None:
        g_terminal = terminal_override
    else:
        g_terminal = 0.025  # 2.5% nominal en perpetuidad

    # Margen de seguridad extra si approach es conservative
    safety_margin = 0.30 if approach == ValuationApproach.conservative else 0.0

    # Proyección
    years_phase1 = 5
    years_phase2 = 5
    pv = 0.0
    flow = base_flow

    for year in range(1, years_phase1 + 1):
        flow *= (1 + g1)
        pv += flow / (1 + wacc) ** year

    for year in range(years_phase1 + 1, years_phase1 + years_phase2 + 1):
        flow *= (1 + g2)
        pv += flow / (1 + wacc) ** year

    # Valor terminal
    terminal_flow = flow * (1 + g_terminal)
    terminal_value = terminal_flow / (wacc - g_terminal) if wacc > g_terminal else flow * 15
    pv_terminal = terminal_value / (1 + wacc) ** (years_phase1 + years_phase2)

    fair_value = (pv + pv_terminal) * (1 - safety_margin)

    # Sensibilidad: ±2pp en tasa de crecimiento fase 1
    g1_low = max(g1 - 0.02, -0.05)
    g1_high = g1 + 0.02

    fv_low = _dcf_simple(base_flow, g1_low, g1_low * 0.6, g_terminal, wacc, years_phase1, years_phase2) * (1 - safety_margin)
    fv_high = _dcf_simple(base_flow, g1_high, g1_high * 0.6, g_terminal, wacc, years_phase1, years_phase2) * (1 - safety_margin)

    assumptions = (
        f"Base: {flow_name} = {base_flow:.2f}. "
        f"Crecimiento fase 1 ({years_phase1}a): {g1:.1%}, "
        f"fase 2 ({years_phase2}a): {g2:.1%}. "
        f"Tasa de descuento (WACC estimado): {wacc:.1%}. "
        f"Crecimiento terminal: {g_terminal:.1%}. "
        f"{'Margen de seguridad adicional del 30% aplicado.' if safety_margin else ''}"
    )

    return DCFResult(
        fair_value=_r(fair_value),
        growth_rate_used=round(g1, 4),
        discount_rate_used=round(wacc, 4),
        terminal_growth_used=round(g_terminal, 4),
        projection_years=years_phase1 + years_phase2,
        sensitivity_low=_r(fv_low),
        sensitivity_high=_r(fv_high),
        assumptions_note=assumptions,
    )


def _dcf_simple(
    base: float, g1: float, g2: float, g_t: float,
    wacc: float, y1: int, y2: int,
) -> float:
    pv = 0.0
    flow = base
    for y in range(1, y1 + 1):
        flow *= (1 + g1)
        pv += flow / (1 + wacc) ** y
    for y in range(y1 + 1, y1 + y2 + 1):
        flow *= (1 + g2)
        pv += flow / (1 + wacc) ** y
    terminal = flow * (1 + g_t) / (wacc - g_t) if wacc > g_t else flow * 15
    pv += terminal / (1 + wacc) ** (y1 + y2)
    return pv


def _estimate_growth_rate(f: FundamentalData, s: FinancialStatements) -> float:
    """Estima tasa de crecimiento para el DCF a partir del histórico y forward guidance."""
    rates = []
    if f.revenue_growth:
        rates.append(f.revenue_growth)
    if f.earnings_growth:
        rates.append(f.earnings_growth)

    # Extraer de estados financieros
    if s.income_statement is not None:
        inc = s.income_statement
        for col_name in ["Total Revenue", "Revenue"]:
            for col in inc.columns:
                if col_name.lower() in col.lower():
                    vals = inc[col].dropna().astype(float)
                    if len(vals) >= 3:
                        cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / (len(vals) - 1)) - 1
                        rates.append(cagr)
                    break

    if not rates:
        return 0.07  # default conservador: 7%

    avg = sum(rates) / len(rates)
    # Cap en rango razonable para proyección
    return max(min(avg, 0.35), -0.05)


def _estimate_wacc(f: FundamentalData) -> float:
    """WACC estimado simplificado."""
    # Risk-free: ~4.5% (entorno actual de tipos)
    rf = 0.045
    # Prima de riesgo de mercado: 5.5%
    erp = 0.055
    beta = f.beta if f.beta and 0.3 < f.beta < 3.0 else 1.0
    cost_equity = rf + beta * erp

    # Estimación estructura de capital
    debt = f.total_debt or 0
    mktcap = f.market_cap or 1
    total_capital = mktcap + debt
    weight_equity = mktcap / total_capital
    weight_debt = debt / total_capital
    cost_debt_after_tax = 0.04 * 0.75  # 4% antes de impuestos, tasa efectiva 25%

    wacc = weight_equity * cost_equity + weight_debt * cost_debt_after_tax
    return max(min(wacc, 0.20), 0.07)  # entre 7% y 20%


# ─────────────────────────────────────────────────────────────────────────────
# Rangos de valor razonable
# ─────────────────────────────────────────────────────────────────────────────

def _build_fair_value_ranges(
    prices: list[float],
    dcf: DCFResult | None,
    approach: ValuationApproach,
) -> tuple[float | None, float | None, float | None]:
    """
    Construye rango conservador/base/optimista a partir de los precios implícitos.
    El DCF recibe más peso si hay pocos múltiplos fiables.
    """
    if not prices:
        return None, None, None

    sorted_prices = sorted(prices)
    n = len(sorted_prices)

    # Base: mediana
    base = sorted_prices[n // 2]
    # Conservador: percentil ~25
    conservative = sorted_prices[max(0, n // 4)]
    # Optimista: percentil ~75
    optimistic = sorted_prices[min(n - 1, (3 * n) // 4)]

    # Si tenemos DCF, incluir en la síntesis con peso explícito
    if dcf and dcf.fair_value:
        weight_dcf = 0.40
        weight_multiples = 0.60
        base = base * weight_multiples + dcf.fair_value * weight_dcf
        if dcf.sensitivity_low:
            conservative = conservative * weight_multiples + dcf.sensitivity_low * weight_dcf
        if dcf.sensitivity_high:
            optimistic = optimistic * weight_multiples + dcf.sensitivity_high * weight_dcf

    # Rango mínimo del 10% entre conservador y optimista
    if optimistic <= conservative * 1.05:
        conservative *= 0.90
        optimistic *= 1.10

    return conservative, base, optimistic


def _classify_valuation(
    discount: float | None,
    current: float,
    fv_base: float | None,
    fv_conservative: float | None,
    fv_optimistic: float | None,
) -> ValuationLevel:
    if discount is None or fv_base is None:
        return ValuationLevel.insufficient_data
    if discount > 20:
        return ValuationLevel.undervalued
    elif discount > 5:
        return ValuationLevel.fair
    elif discount > -15:
        return ValuationLevel.demanding
    else:
        return ValuationLevel.overvalued


def _dominant_method(
    multiples: list[ValuationMultiple],
    dcf: DCFResult | None,
    prices: list[float],
) -> str:
    if not prices:
        return "Datos insuficientes para determinar método dominante"
    if dcf and dcf.fair_value and len(multiples) < 2:
        return "DCF (pocos múltiplos disponibles para contraste)"
    if dcf and dcf.fair_value:
        return "Combinación DCF (40%) + múltiplos (60%): P/FCF, EV/EBITDA y PER pesan más en la estimación."
    return "Múltiplos comparativos (PER, EV/EBITDA, P/S, P/FCF)"


def _build_narrative(
    f: FundamentalData,
    multiples: list[ValuationMultiple],
    dcf: DCFResult | None,
    fv_c: float | None,
    fv_b: float | None,
    fv_o: float | None,
    current: float,
    discount: float | None,
    level: ValuationLevel,
    dominant: str,
) -> str:
    parts = []
    name = f.name or f.ticker

    level_map = {
        ValuationLevel.undervalued: "infravalorada",
        ValuationLevel.fair: "razonablemente valorada",
        ValuationLevel.demanding: "a valoración exigente",
        ValuationLevel.overvalued: "sobrevalorada",
        ValuationLevel.insufficient_data: "con datos insuficientes para valorar",
    }

    parts.append(
        f"Con el precio actual de {current:.2f}, "
        f"{name} parece cotizar {level_map[level]}."
    )

    if fv_b:
        parts.append(
            f"El valor razonable estimado (escenario base) se sitúa en torno a {fv_b:.2f}, "
            f"con rango conservador {fv_c:.2f}–{fv_b:.2f} y optimista hasta {fv_o:.2f}."
            if fv_c and fv_o else
            f"Valor razonable base estimado: {fv_b:.2f}."
        )

    if discount is not None:
        if discount > 0:
            parts.append(f"El precio cotiza con un descuento del {discount:.1f}% respecto al valor base.")
        else:
            parts.append(f"El precio cotiza con una prima del {abs(discount):.1f}% sobre el valor base — implica expectativas de crecimiento elevadas.")

    if dcf:
        parts.append(f"El DCF usa: {dcf.assumptions_note}")
        parts.append(
            f"Sensibilidad del DCF: escenario pesimista {dcf.sensitivity_low:.2f}, "
            f"optimista {dcf.sensitivity_high:.2f}."
            if dcf.sensitivity_low and dcf.sensitivity_high else ""
        )

    parts.append(f"Método dominante en la estimación: {dominant}.")

    return " ".join([p for p in parts if p])


def _r(val: float | None, decimals: int = 2) -> float | None:
    if val is None:
        return None
    import math
    if math.isnan(val) or math.isinf(val):
        return None
    return round(val, decimals)
