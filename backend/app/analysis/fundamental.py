"""
Módulo de análisis fundamental.
Calcula métricas, tendencias históricas, puntuaciones y genera narrativa explicada.

Filosofía:
- Ninguna puntuación sale de una caja negra: cada factor contribuyente está documentado.
- Si faltan datos, se avisa y se pondera con menor peso, no se ignora.
- Las tendencias se extraen de los estados financieros históricos, no solo del snapshot.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from ..providers.base import FundamentalData, FinancialStatements
from ..models.outputs import (
    FundamentalAnalysis,
    GrowthMetrics,
    MarginMetrics,
    EfficiencyMetrics,
    DebtMetrics,
    CashFlowMetrics,
    ShareholderMetrics,
    ScoredSection,
)

logger = logging.getLogger(__name__)


def analyze_fundamental(
    fundamentals: FundamentalData,
    statements: FinancialStatements,
) -> FundamentalAnalysis:
    """Punto de entrada principal del módulo fundamental."""
    data_gaps: list[str] = list(fundamentals.missing_fields)

    growth = _build_growth_metrics(fundamentals, statements, data_gaps)
    margins = _build_margin_metrics(fundamentals, statements, data_gaps)
    efficiency = _build_efficiency_metrics(fundamentals, data_gaps)
    debt = _build_debt_metrics(fundamentals, statements, data_gaps)
    cashflow = _build_cashflow_metrics(fundamentals, statements, data_gaps)
    shareholders = _build_shareholder_metrics(fundamentals, statements, data_gaps)

    quality_score = _score_quality(growth, margins, efficiency, cashflow)
    financial_strength_score = _score_financial_strength(debt, cashflow, fundamentals)
    growth_score = _score_growth(growth)
    efficiency_score = _score_efficiency(efficiency, margins)

    overall = _weighted_overall(
        quality_score.score,
        financial_strength_score.score,
        growth_score.score,
        efficiency_score.score,
    )

    narrative = _build_narrative(
        fundamentals, growth, margins, efficiency, debt, cashflow,
        quality_score, financial_strength_score, growth_score, efficiency_score,
    )

    return FundamentalAnalysis(
        growth=growth,
        margins=margins,
        efficiency=efficiency,
        debt=debt,
        cash_flow=cashflow,
        shareholders=shareholders,
        quality_score=quality_score,
        financial_strength_score=financial_strength_score,
        growth_score=growth_score,
        efficiency_score=efficiency_score,
        overall_fundamental_score=round(overall, 2),
        narrative=narrative,
        data_gaps=list(set(data_gaps)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de métricas
# ─────────────────────────────────────────────────────────────────────────────

def _build_growth_metrics(
    f: FundamentalData,
    s: FinancialStatements,
    gaps: list,
) -> GrowthMetrics:
    rev_1y = f.revenue_growth
    eps_1y = f.earnings_growth
    rev_cagr_3y = None
    eps_cagr_3y = None
    fcf_1y = None

    if s.income_statement is not None:
        inc = s.income_statement
        rev_col = _find_col(inc, ["Total Revenue", "Revenue"])
        if rev_col:
            revs = inc[rev_col].dropna()
            if len(revs) >= 2 and rev_1y is None:
                rev_1y = _yoy_growth(revs)
            if len(revs) >= 4:
                rev_cagr_3y = _cagr(revs, 3)

        eps_col = _find_col(inc, ["Basic EPS", "Diluted EPS", "EPS"])
        if eps_col:
            eps_series = inc[eps_col].dropna()
            if len(eps_series) >= 2 and eps_1y is None:
                eps_1y = _yoy_growth(eps_series)
            if len(eps_series) >= 4:
                eps_cagr_3y = _cagr(eps_series, 3)

    if s.cash_flow is not None:
        fcf_col = _find_col(s.cash_flow, ["Free Cash Flow", "FreeCashFlow"])
        if fcf_col:
            fcf_series = s.cash_flow[fcf_col].dropna()
            if len(fcf_series) >= 2:
                fcf_1y = _yoy_growth(fcf_series)

    if rev_1y is None:
        gaps.append("revenue_growth_1y")
    if eps_1y is None:
        gaps.append("eps_growth_1y")

    return GrowthMetrics(
        revenue_growth_1y=_pct(rev_1y),
        revenue_cagr_3y=_pct(rev_cagr_3y),
        eps_growth_1y=_pct(eps_1y),
        eps_cagr_3y=_pct(eps_cagr_3y),
        fcf_growth_1y=_pct(fcf_1y),
    )


def _build_margin_metrics(
    f: FundamentalData,
    s: FinancialStatements,
    gaps: list,
) -> MarginMetrics:
    gross_trend = None
    op_trend = None

    if s.income_statement is not None:
        inc = s.income_statement
        gross_col = _find_col(inc, ["Gross Profit"])
        rev_col = _find_col(inc, ["Total Revenue", "Revenue"])
        if gross_col and rev_col:
            gross_margins = (inc[gross_col] / inc[rev_col]).dropna()
            gross_trend = _trend_label(gross_margins)

        ebit_col = _find_col(inc, ["EBIT", "Operating Income"])
        if ebit_col and rev_col:
            op_margins = (inc[ebit_col] / inc[rev_col]).dropna()
            op_trend = _trend_label(op_margins)

    return MarginMetrics(
        gross_margin=f.gross_margin,
        operating_margin=f.operating_margin,
        net_margin=f.net_margin,
        ebitda_margin=f.ebitda_margin,
        gross_margin_trend=gross_trend,
        operating_margin_trend=op_trend,
    )


def _build_efficiency_metrics(f: FundamentalData, gaps: list) -> EfficiencyMetrics:
    roic = None
    # ROIC aproximado: EBIT*(1-tax) / (equity + net_debt)
    # Si no están todos los datos, dejamos None y lo anotamos
    if f.roe is None:
        gaps.append("roe")
    return EfficiencyMetrics(
        roe=f.roe,
        roic=roic,  # requeriría NOPAT y capital invertido, no siempre disponible
        roa=f.roa,
        asset_turnover=None,
    )


def _build_debt_metrics(
    f: FundamentalData,
    s: FinancialStatements,
    gaps: list,
) -> DebtMetrics:
    net_debt_ebitda = None

    # Calcular EBITDA si tenemos estados
    ebitda_abs = None
    if s.income_statement is not None:
        inc = s.income_statement
        ebitda_col = _find_col(inc, ["EBITDA", "Normalized EBITDA"])
        if ebitda_col:
            vals = inc[ebitda_col].dropna()
            if len(vals) > 0:
                ebitda_abs = float(vals.iloc[-1])

    if ebitda_abs is None and f.ebitda_margin and f.market_cap and f.net_margin:
        # Estimación burda
        pass

    cash = f.total_cash or 0
    debt = f.total_debt or 0
    net_debt = debt - cash
    if ebitda_abs and ebitda_abs > 0:
        net_debt_ebitda = net_debt / ebitda_abs
    elif net_debt < 0:
        net_debt_ebitda = -0.5  # empresa con caja neta positiva

    interest_coverage = None
    if s.income_statement is not None:
        inc = s.income_statement
        ebit_col = _find_col(inc, ["EBIT", "Operating Income"])
        int_col = _find_col(inc, ["Interest Expense", "Net Interest Income"])
        if ebit_col and int_col:
            ebit_vals = inc[ebit_col].dropna()
            int_vals = inc[int_col].dropna().abs()
            if len(ebit_vals) > 0 and len(int_vals) > 0:
                latest_ebit = float(ebit_vals.iloc[-1])
                latest_int = float(int_vals.iloc[-1])
                if latest_int > 0:
                    interest_coverage = latest_ebit / latest_int

    return DebtMetrics(
        net_debt_ebitda=round(net_debt_ebitda, 2) if net_debt_ebitda is not None else None,
        interest_coverage=round(interest_coverage, 2) if interest_coverage else None,
        debt_to_equity=f.debt_to_equity,
        current_ratio=f.current_ratio,
        cash_and_equivalents=f.total_cash,
    )


def _build_cashflow_metrics(
    f: FundamentalData,
    s: FinancialStatements,
    gaps: list,
) -> CashFlowMetrics:
    fcf_yield = None
    fcf_consistency = None
    capex_to_rev = None

    if f.free_cash_flow and f.market_cap and f.market_cap > 0:
        fcf_yield = f.free_cash_flow / f.market_cap

    if s.cash_flow is not None:
        cf = s.cash_flow
        fcf_col = _find_col(cf, ["Free Cash Flow", "FreeCashFlow"])
        if fcf_col:
            fcf_series = cf[fcf_col].dropna()
            if len(fcf_series) >= 3:
                n_positive = (fcf_series > 0).sum()
                ratio = n_positive / len(fcf_series)
                if ratio >= 0.85:
                    fcf_consistency = "consistent"
                elif ratio >= 0.6:
                    fcf_consistency = "variable"
                else:
                    fcf_consistency = "negative"

        capex_col = _find_col(cf, ["Capital Expenditure", "Purchase Of PPE"])
        rev_col = None
        if s.income_statement is not None:
            rev_col_name = _find_col(s.income_statement, ["Total Revenue", "Revenue"])
            if rev_col_name:
                rev_latest = s.income_statement[rev_col_name].dropna()
                if len(rev_latest) > 0 and capex_col:
                    capex_latest = cf[capex_col].dropna()
                    if len(capex_latest) > 0:
                        capex_to_rev = abs(float(capex_latest.iloc[-1])) / float(rev_latest.iloc[-1])

    return CashFlowMetrics(
        free_cash_flow=f.free_cash_flow,
        fcf_yield=_pct(fcf_yield) if fcf_yield is not None else None,
        fcf_consistency=fcf_consistency,
        operating_cash_flow=f.operating_cash_flow,
        capex_to_revenue=_pct(capex_to_rev) if capex_to_rev is not None else None,
    )


def _build_shareholder_metrics(
    f: FundamentalData,
    s: FinancialStatements,
    gaps: list,
) -> ShareholderMetrics:
    shares_change = None

    if s.cash_flow is not None:
        cf = s.cash_flow
        shares_col = _find_col(cf, ["Common Stock", "Issuance Of Common Stock", "Repurchase Of Common Stock"])
        if shares_col and f.shares_outstanding:
            vals = cf[shares_col].dropna()
            if len(vals) >= 2:
                shares_change = float(vals.iloc[-1] - vals.iloc[-2]) / max(abs(float(vals.iloc[-2])), 1)

    buyback_yield = None
    if s.cash_flow is not None:
        bb_col = _find_col(s.cash_flow, ["Repurchase Of Common Stock", "Common Stock Repurchase"])
        if bb_col and f.market_cap and f.market_cap > 0:
            bb_vals = s.cash_flow[bb_col].dropna()
            if len(bb_vals) > 0:
                buyback_yield = abs(float(bb_vals.iloc[-1])) / f.market_cap

    return ShareholderMetrics(
        shares_outstanding_change_1y=_pct(shares_change) if shares_change is not None else None,
        buyback_yield=_pct(buyback_yield) if buyback_yield is not None else None,
        dividend_yield=f.dividend_yield,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Puntuaciones (0–10, con justificación)
# ─────────────────────────────────────────────────────────────────────────────

def _score_quality(
    growth: GrowthMetrics,
    margins: MarginMetrics,
    efficiency: EfficiencyMetrics,
    cashflow: CashFlowMetrics,
) -> ScoredSection:
    factors: list[str] = []
    warnings: list[str] = []
    points = 0.0
    max_pts = 0.0

    # Margen bruto (max 2pts)
    if margins.gross_margin is not None:
        max_pts += 2
        gm = margins.gross_margin
        if gm > 0.60:
            points += 2; factors.append(f"Margen bruto excelente ({gm:.0%})")
        elif gm > 0.40:
            points += 1.5; factors.append(f"Margen bruto sólido ({gm:.0%})")
        elif gm > 0.20:
            points += 1; factors.append(f"Margen bruto moderado ({gm:.0%})")
        else:
            points += 0.5; warnings.append(f"Margen bruto bajo ({gm:.0%})")
        if margins.gross_margin_trend == "improving":
            points += 0.2; factors.append("Tendencia de márgenes brutos mejorando")
        elif margins.gross_margin_trend == "deteriorating":
            points -= 0.3; warnings.append("Tendencia de márgenes brutos deteriorándose")

    # Margen operativo (max 2pts)
    if margins.operating_margin is not None:
        max_pts += 2
        om = margins.operating_margin
        if om > 0.25:
            points += 2; factors.append(f"Margen operativo alto ({om:.0%})")
        elif om > 0.15:
            points += 1.5; factors.append(f"Margen operativo bueno ({om:.0%})")
        elif om > 0.05:
            points += 0.8; factors.append(f"Margen operativo moderado ({om:.0%})")
        else:
            points += 0; warnings.append(f"Margen operativo muy bajo o negativo ({om:.0%})")

    # FCF consistency (max 1.5pts)
    if cashflow.fcf_consistency:
        max_pts += 1.5
        if cashflow.fcf_consistency == "consistent":
            points += 1.5; factors.append("FCF históricamente consistente y positivo")
        elif cashflow.fcf_consistency == "variable":
            points += 0.7; warnings.append("FCF variable, no siempre positivo")
        else:
            points += 0; warnings.append("FCF negativo en la mayoría de períodos")

    # ROE (max 1.5pts)
    if efficiency.roe is not None:
        max_pts += 1.5
        roe = efficiency.roe
        if roe > 0.20:
            points += 1.5; factors.append(f"ROE elevado ({roe:.0%})")
        elif roe > 0.12:
            points += 1; factors.append(f"ROE razonable ({roe:.0%})")
        elif roe > 0.05:
            points += 0.5
        else:
            warnings.append(f"ROE bajo ({roe:.0%})")

    score = (points / max_pts * 10) if max_pts > 0 else 5.0
    score = min(max(score, 0), 10)
    return ScoredSection(
        score=round(score, 1),
        label=_score_label("Calidad del negocio", score),
        factors=factors,
        warnings=warnings,
    )


def _score_financial_strength(
    debt: DebtMetrics,
    cashflow: CashFlowMetrics,
    f: FundamentalData,
) -> ScoredSection:
    factors: list[str] = []
    warnings: list[str] = []
    points = 0.0
    max_pts = 0.0

    # Deuda neta / EBITDA (max 3pts)
    if debt.net_debt_ebitda is not None:
        max_pts += 3
        nd = debt.net_debt_ebitda
        if nd < 0:
            points += 3; factors.append("Caja neta positiva (sin deuda neta)")
        elif nd < 1:
            points += 2.5; factors.append(f"Deuda neta muy baja (ND/EBITDA {nd:.1f}x)")
        elif nd < 2:
            points += 2; factors.append(f"Deuda manejable (ND/EBITDA {nd:.1f}x)")
        elif nd < 3.5:
            points += 1; warnings.append(f"Deuda moderada-alta (ND/EBITDA {nd:.1f}x)")
        else:
            points += 0; warnings.append(f"Deuda elevada (ND/EBITDA {nd:.1f}x) — riesgo de balance")

    # Cobertura de intereses (max 2pts)
    if debt.interest_coverage is not None:
        max_pts += 2
        ic = debt.interest_coverage
        if ic > 10:
            points += 2; factors.append(f"Cobertura de intereses excelente ({ic:.1f}x)")
        elif ic > 5:
            points += 1.5; factors.append(f"Cobertura de intereses sólida ({ic:.1f}x)")
        elif ic > 2:
            points += 0.8; warnings.append(f"Cobertura de intereses ajustada ({ic:.1f}x)")
        else:
            points += 0; warnings.append(f"Cobertura de intereses muy baja ({ic:.1f}x) — riesgo")

    # Current ratio (max 1pt)
    if debt.current_ratio is not None:
        max_pts += 1
        cr = debt.current_ratio
        if cr > 2:
            points += 1; factors.append(f"Liquidez corriente buena ({cr:.1f}x)")
        elif cr > 1.2:
            points += 0.7; factors.append(f"Liquidez corriente aceptable ({cr:.1f}x)")
        else:
            points += 0.2; warnings.append(f"Liquidez corriente ajustada ({cr:.1f}x)")

    score = (points / max_pts * 10) if max_pts > 0 else 5.0
    score = min(max(score, 0), 10)
    return ScoredSection(
        score=round(score, 1),
        label=_score_label("Fortaleza financiera", score),
        factors=factors,
        warnings=warnings,
    )


def _score_growth(growth: GrowthMetrics) -> ScoredSection:
    factors: list[str] = []
    warnings: list[str] = []
    points = 0.0
    max_pts = 0.0

    # Crecimiento ingresos 1y (max 2.5pts)
    if growth.revenue_growth_1y is not None:
        max_pts += 2.5
        rg = growth.revenue_growth_1y / 100  # viene en %
        if rg > 0.20:
            points += 2.5; factors.append(f"Crecimiento de ingresos acelerado (+{rg:.0%})")
        elif rg > 0.10:
            points += 2; factors.append(f"Crecimiento de ingresos sólido (+{rg:.0%})")
        elif rg > 0.03:
            points += 1.3; factors.append(f"Crecimiento de ingresos moderado (+{rg:.0%})")
        elif rg > 0:
            points += 0.7
        else:
            points += 0; warnings.append(f"Ingresos en declive ({rg:.0%})")

    # CAGR ingresos 3y (max 2pts)
    if growth.revenue_cagr_3y is not None:
        max_pts += 2
        cagr = growth.revenue_cagr_3y / 100
        if cagr > 0.15:
            points += 2; factors.append(f"CAGR ingresos 3 años elevado ({cagr:.0%})")
        elif cagr > 0.07:
            points += 1.3; factors.append(f"CAGR ingresos 3 años razonable ({cagr:.0%})")
        elif cagr > 0:
            points += 0.7
        else:
            warnings.append(f"CAGR ingresos 3 años negativo ({cagr:.0%})")

    # Crecimiento EPS (max 2pts)
    if growth.eps_growth_1y is not None:
        max_pts += 2
        eg = growth.eps_growth_1y / 100
        if eg > 0.20:
            points += 2; factors.append(f"Crecimiento EPS fuerte (+{eg:.0%})")
        elif eg > 0.10:
            points += 1.5
        elif eg > 0:
            points += 0.8
        else:
            warnings.append(f"EPS en declive ({eg:.0%})")

    # FCF growth (max 1.5pts)
    if growth.fcf_growth_1y is not None:
        max_pts += 1.5
        fg = growth.fcf_growth_1y / 100
        if fg > 0.15:
            points += 1.5; factors.append(f"FCF creciendo fuerte (+{fg:.0%})")
        elif fg > 0:
            points += 0.8
        else:
            warnings.append(f"FCF decreciendo ({fg:.0%})")

    if max_pts == 0:
        return ScoredSection(
            score=5.0,
            label="Crecimiento: sin datos suficientes",
            factors=[],
            warnings=["No hay suficientes datos de crecimiento disponibles"],
        )

    score = min(max((points / max_pts * 10), 0), 10)
    return ScoredSection(
        score=round(score, 1),
        label=_score_label("Crecimiento", score),
        factors=factors,
        warnings=warnings,
    )


def _score_efficiency(efficiency: EfficiencyMetrics, margins: MarginMetrics) -> ScoredSection:
    factors: list[str] = []
    warnings: list[str] = []
    points = 0.0
    max_pts = 0.0

    if efficiency.roe is not None:
        max_pts += 3
        roe = efficiency.roe
        if roe > 0.25:
            points += 3; factors.append(f"ROE excelente ({roe:.0%})")
        elif roe > 0.15:
            points += 2; factors.append(f"ROE sólido ({roe:.0%})")
        elif roe > 0.08:
            points += 1
        else:
            warnings.append(f"ROE bajo ({roe:.0%})")

    if efficiency.roa is not None:
        max_pts += 2
        roa = efficiency.roa
        if roa > 0.10:
            points += 2; factors.append(f"ROA alto ({roa:.0%})")
        elif roa > 0.05:
            points += 1.3
        elif roa > 0:
            points += 0.7
        else:
            warnings.append(f"ROA negativo ({roa:.0%})")

    if margins.net_margin is not None:
        max_pts += 2
        nm = margins.net_margin
        if nm > 0.20:
            points += 2; factors.append(f"Margen neto alto ({nm:.0%})")
        elif nm > 0.10:
            points += 1.5
        elif nm > 0.03:
            points += 0.8
        else:
            warnings.append(f"Margen neto bajo ({nm:.0%})")

    if max_pts == 0:
        return ScoredSection(score=5.0, label="Eficiencia: sin datos", factors=[], warnings=[])

    score = min(max((points / max_pts * 10), 0), 10)
    return ScoredSection(
        score=round(score, 1),
        label=_score_label("Eficiencia", score),
        factors=factors,
        warnings=warnings,
    )


def _weighted_overall(quality: float, strength: float, growth: float, efficiency: float) -> float:
    """Media ponderada de las cuatro dimensiones."""
    return quality * 0.30 + strength * 0.25 + growth * 0.25 + efficiency * 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Narrativa en lenguaje natural
# ─────────────────────────────────────────────────────────────────────────────

def _build_narrative(
    f: FundamentalData,
    growth: GrowthMetrics,
    margins: MarginMetrics,
    efficiency: EfficiencyMetrics,
    debt: DebtMetrics,
    cashflow: CashFlowMetrics,
    quality: ScoredSection,
    strength: ScoredSection,
    growth_sc: ScoredSection,
    eff: ScoredSection,
) -> str:
    parts = []

    name = f.name or f.ticker
    parts.append(
        f"{name} obtiene una puntuación de calidad de negocio de {quality.score}/10 "
        f"y una fortaleza financiera de {strength.score}/10."
    )

    if margins.operating_margin is not None:
        if margins.operating_margin > 0.15:
            parts.append(
                f"El margen operativo del {margins.operating_margin:.1%} refleja una estructura "
                f"de costes eficiente con capacidad de pricing."
            )
        elif margins.operating_margin < 0.05:
            parts.append(
                f"El margen operativo del {margins.operating_margin:.1%} es bajo, "
                f"lo que deja poco margen ante presiones de costes o caída de ingresos."
            )

    if growth.revenue_growth_1y is not None:
        if growth.revenue_growth_1y > 15:
            parts.append(
                f"El crecimiento de ingresos del {growth.revenue_growth_1y:.1f}% "
                f"en el último año es notable y sugiere demanda robusta."
            )
        elif growth.revenue_growth_1y < 0:
            parts.append(
                f"Los ingresos han caído un {abs(growth.revenue_growth_1y):.1f}% "
                f"en el último año — hay que vigilar si es temporal o estructural."
            )

    if debt.net_debt_ebitda is not None:
        if debt.net_debt_ebitda < 0:
            parts.append("La empresa mantiene caja neta, lo que reduce el riesgo financiero.")
        elif debt.net_debt_ebitda > 3:
            parts.append(
                f"La deuda neta de {debt.net_debt_ebitda:.1f}x EBITDA es elevada "
                f"y puede limitar la flexibilidad financiera."
            )

    if cashflow.fcf_consistency == "consistent":
        parts.append("El free cash flow ha sido consistentemente positivo en los últimos años, lo cual es una señal de calidad.")
    elif cashflow.fcf_consistency == "negative":
        parts.append("El FCF ha sido negativo la mayor parte del tiempo — importante distinguir si es por inversión o por pérdidas operativas.")

    # Fortalezas y debilidades resumidas
    if quality.factors:
        parts.append("Fortalezas clave: " + "; ".join(quality.factors[:3]) + ".")
    if quality.warnings:
        parts.append("Alertas: " + "; ".join(quality.warnings[:3]) + ".")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Busca el primer nombre de columna que coincida (case-insensitive parcial)."""
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in col.lower():
                return col
    return None


def _yoy_growth(series: pd.Series) -> float | None:
    """YoY entre el último y penúltimo valor."""
    s = series.dropna()
    if len(s) < 2:
        return None
    prev, last = float(s.iloc[-2]), float(s.iloc[-1])
    if prev == 0:
        return None
    return (last - prev) / abs(prev)


def _cagr(series: pd.Series, years: int) -> float | None:
    s = series.dropna()
    if len(s) < years + 1:
        return None
    start = float(s.iloc[-(years + 1)])
    end = float(s.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def _trend_label(series: pd.Series) -> str:
    """Determina si una serie tiene tendencia creciente, estable o decreciente."""
    s = series.dropna()
    if len(s) < 3:
        return "insufficient_data"
    try:
        x = np.arange(len(s))
        slope = float(np.polyfit(x, s.values.astype(float), 1)[0])
        mean_val = abs(float(s.mean()))
        if mean_val == 0:
            return "stable"
        relative_slope = slope / mean_val
        if relative_slope > 0.02:
            return "improving"
        elif relative_slope < -0.02:
            return "deteriorating"
        return "stable"
    except Exception:
        return "stable"


def _pct(val: float | None) -> float | None:
    """Convierte ratio a porcentaje redondeado a 2 decimales."""
    if val is None:
        return None
    return round(val * 100, 2)


def _score_label(name: str, score: float) -> str:
    if score >= 8:
        quality = "Excelente"
    elif score >= 6.5:
        quality = "Bueno"
    elif score >= 5:
        quality = "Moderado"
    elif score >= 3:
        quality = "Débil"
    else:
        quality = "Muy débil"
    return f"{name}: {quality} ({score:.1f}/10)"
