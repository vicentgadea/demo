"""
Tests del módulo de análisis fundamental.
Usan datos sintéticos para verificar la lógica de scoring y métricas.
"""
import pandas as pd
import pytest

from app.providers.base import FundamentalData, FinancialStatements
from app.analysis.fundamental import (
    analyze_fundamental,
    _score_quality,
    _score_financial_strength,
    _score_growth,
    _yoy_growth,
    _cagr,
    _trend_label,
)
from app.models.outputs import GrowthMetrics, MarginMetrics, EfficiencyMetrics, CashFlowMetrics, DebtMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_fundamentals(**kwargs) -> FundamentalData:
    defaults = dict(
        ticker="TEST",
        name="Test Company",
        sector="Technology",
        current_price=100.0,
        market_cap=10_000_000_000,
        gross_margin=0.60,
        operating_margin=0.25,
        net_margin=0.20,
        roe=0.25,
        roa=0.12,
        total_cash=1_000_000_000,
        total_debt=500_000_000,
        current_ratio=2.5,
        free_cash_flow=800_000_000,
        shares_outstanding=100_000_000,
        trailing_eps=4.0,
        forward_eps=5.0,
        pe_ratio=25.0,
        forward_pe=20.0,
        beta=1.1,
    )
    defaults.update(kwargs)
    return FundamentalData(**defaults)


def make_statements_with_growth(revenue_growth: float = 0.15) -> FinancialStatements:
    """Crea estados financieros con crecimiento de ingresos dado."""
    years = [2020, 2021, 2022, 2023]
    base_revenue = 5_000_000_000
    revenues = [base_revenue * (1 + revenue_growth) ** i for i in range(len(years))]

    inc = pd.DataFrame({
        "Total Revenue": revenues,
        "Gross Profit": [r * 0.60 for r in revenues],
        "EBIT": [r * 0.25 for r in revenues],
        "Net Income": [r * 0.20 for r in revenues],
        "Basic EPS": [r / 100_000_000 * 0.20 for r in revenues],
    }, index=pd.to_datetime([f"{y}-12-31" for y in years]))

    cf = pd.DataFrame({
        "Free Cash Flow": [r * 0.18 for r in revenues],
        "Capital Expenditure": [r * 0.05 * -1 for r in revenues],
    }, index=pd.to_datetime([f"{y}-12-31" for y in years]))

    return FinancialStatements(
        ticker="TEST",
        income_statement=inc,
        cash_flow=cf,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests de utilidades
# ─────────────────────────────────────────────────────────────────────────────

class TestGrowthUtils:
    def test_yoy_growth_positive(self):
        s = pd.Series([100, 120])
        assert abs(_yoy_growth(s) - 0.20) < 1e-6

    def test_yoy_growth_negative(self):
        s = pd.Series([100, 80])
        assert abs(_yoy_growth(s) - (-0.20)) < 1e-6

    def test_yoy_growth_insufficient_data(self):
        s = pd.Series([100])
        assert _yoy_growth(s) is None

    def test_yoy_growth_zero_base(self):
        s = pd.Series([0, 100])
        assert _yoy_growth(s) is None

    def test_cagr_3y(self):
        s = pd.Series([100, 110, 121, 133.1])  # 10% CAGR exacto
        result = _cagr(s, 3)
        assert result is not None
        assert abs(result - 0.10) < 0.001

    def test_cagr_insufficient_data(self):
        s = pd.Series([100, 110])
        assert _cagr(s, 3) is None


class TestTrendLabel:
    def test_improving_trend(self):
        s = pd.Series([10, 12, 14, 16, 18])
        assert _trend_label(s) == "improving"

    def test_deteriorating_trend(self):
        s = pd.Series([18, 16, 14, 12, 10])
        assert _trend_label(s) == "deteriorating"

    def test_stable_trend(self):
        s = pd.Series([10, 10.1, 9.9, 10.0, 10.1])
        assert _trend_label(s) == "stable"


# ─────────────────────────────────────────────────────────────────────────────
# Tests de scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityScore:
    def test_high_quality_company(self):
        growth = GrowthMetrics(revenue_growth_1y=20.0, eps_growth_1y=25.0)
        margins = MarginMetrics(gross_margin=0.65, operating_margin=0.28, gross_margin_trend="improving")
        efficiency = EfficiencyMetrics(roe=0.28, roa=0.14)
        cashflow = CashFlowMetrics(fcf_consistency="consistent")

        score = _score_quality(growth, margins, efficiency, cashflow)
        assert score.score >= 7.5
        assert len(score.factors) > 0
        assert len(score.warnings) == 0

    def test_low_quality_company(self):
        growth = GrowthMetrics()
        margins = MarginMetrics(gross_margin=0.10, operating_margin=0.02, gross_margin_trend="deteriorating")
        efficiency = EfficiencyMetrics(roe=0.03)
        cashflow = CashFlowMetrics(fcf_consistency="negative")

        score = _score_quality(growth, margins, efficiency, cashflow)
        assert score.score < 5.0
        assert len(score.warnings) > 0

    def test_score_between_0_and_10(self):
        for gm in [0.1, 0.3, 0.5, 0.8]:
            growth = GrowthMetrics()
            margins = MarginMetrics(gross_margin=gm, operating_margin=gm * 0.4)
            efficiency = EfficiencyMetrics(roe=0.15)
            cashflow = CashFlowMetrics(fcf_consistency="consistent")
            score = _score_quality(growth, margins, efficiency, cashflow)
            assert 0 <= score.score <= 10


class TestFinancialStrengthScore:
    def test_net_cash_company(self):
        debt = DebtMetrics(net_debt_ebitda=-0.5, interest_coverage=50, current_ratio=3.0)
        cashflow = CashFlowMetrics()
        f = make_fundamentals()
        score = _score_financial_strength(debt, cashflow, f)
        assert score.score >= 8.0

    def test_high_leverage(self):
        debt = DebtMetrics(net_debt_ebitda=5.0, interest_coverage=1.5, current_ratio=0.9)
        cashflow = CashFlowMetrics()
        f = make_fundamentals()
        score = _score_financial_strength(debt, cashflow, f)
        assert score.score < 4.0
        assert any("alta" in w.lower() or "elevada" in w.lower() for w in score.warnings)


class TestGrowthScore:
    def test_strong_growth(self):
        growth = GrowthMetrics(
            revenue_growth_1y=25.0,
            revenue_cagr_3y=20.0,
            eps_growth_1y=30.0,
            fcf_growth_1y=20.0,
        )
        score = _score_growth(growth)
        assert score.score >= 8.0

    def test_declining_revenue(self):
        growth = GrowthMetrics(revenue_growth_1y=-10.0, eps_growth_1y=-15.0)
        score = _score_growth(growth)
        assert score.score < 4.0
        assert any("declive" in w.lower() or "cayendo" in w.lower() or "negativo" in w.lower()
                   for w in score.warnings)

    def test_no_data_returns_neutral(self):
        growth = GrowthMetrics()
        score = _score_growth(growth)
        assert "sin datos" in score.label.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Tests de integración del módulo completo
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeFundamental:
    def test_complete_analysis_returns_valid_scores(self):
        f = make_fundamentals()
        s = make_statements_with_growth(0.15)

        result = analyze_fundamental(f, s)

        assert 0 <= result.overall_fundamental_score <= 10
        assert 0 <= result.quality_score.score <= 10
        assert 0 <= result.financial_strength_score.score <= 10
        assert 0 <= result.growth_score.score <= 10
        assert 0 <= result.efficiency_score.score <= 10
        assert isinstance(result.narrative, str)
        assert len(result.narrative) > 20

    def test_high_quality_company_scores_well(self):
        f = make_fundamentals(
            gross_margin=0.70, operating_margin=0.30, net_margin=0.25,
            roe=0.30, roa=0.15, total_debt=0, current_ratio=4.0,
        )
        s = make_statements_with_growth(0.20)
        result = analyze_fundamental(f, s)
        assert result.overall_fundamental_score >= 7.0

    def test_distressed_company_scores_poorly(self):
        f = make_fundamentals(
            gross_margin=0.05, operating_margin=-0.05, net_margin=-0.10,
            roe=-0.10, total_debt=10_000_000_000, current_ratio=0.7,
            free_cash_flow=-500_000_000,
        )
        s = make_statements_with_growth(-0.10)
        result = analyze_fundamental(f, s)
        assert result.overall_fundamental_score < 4.0

    def test_missing_data_handled_gracefully(self):
        f = FundamentalData(ticker="NODATA")
        s = FinancialStatements(ticker="NODATA")
        result = analyze_fundamental(f, s)
        assert result is not None
        assert 0 <= result.overall_fundamental_score <= 10
        assert len(result.data_gaps) > 0

    def test_growth_metrics_from_statements(self):
        f = make_fundamentals(revenue_growth=None)  # forzar cálculo desde statements
        s = make_statements_with_growth(0.20)
        result = analyze_fundamental(f, s)
        assert result.growth.revenue_growth_1y is not None
        assert result.growth.revenue_growth_1y > 0
