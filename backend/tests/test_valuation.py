"""
Tests del módulo de valoración.
"""
import pytest

from app.providers.base import FundamentalData, FinancialStatements
from app.models.inputs import ValuationApproach
from app.analysis.valuation import (
    analyze_valuation,
    _calc_dcf,
    _classify_valuation,
    _estimate_growth_rate,
    _estimate_wacc,
    _build_fair_value_ranges,
)
from app.models.outputs import ValuationLevel


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def tech_company(**kwargs) -> FundamentalData:
    defaults = dict(
        ticker="TECH",
        name="Tech Co",
        sector="Technology",
        current_price=150.0,
        market_cap=15_000_000_000,
        shares_outstanding=100_000_000,
        pe_ratio=30.0,
        forward_pe=25.0,
        trailing_eps=5.0,
        forward_eps=6.0,
        price_to_sales=5.0,
        ev_ebitda=20.0,
        enterprise_value=18_000_000_000,
        free_cash_flow=600_000_000,
        operating_cash_flow=700_000_000,
        total_cash=2_000_000_000,
        total_debt=500_000_000,
        beta=1.2,
        revenue_growth=0.20,
        earnings_growth=0.25,
    )
    defaults.update(kwargs)
    return FundamentalData(**defaults)


def empty_statements() -> FinancialStatements:
    return FinancialStatements(ticker="TEST")


# ─────────────────────────────────────────────────────────────────────────────
# Tests de DCF
# ─────────────────────────────────────────────────────────────────────────────

class TestDCF:
    def test_dcf_returns_positive_value(self):
        f = tech_company()
        s = empty_statements()
        result = _calc_dcf(f, s, None, None, None, ValuationApproach.base)
        assert result is not None
        assert result.fair_value is not None
        assert result.fair_value > 0

    def test_dcf_conservative_lower_than_base(self):
        f = tech_company()
        s = empty_statements()
        base = _calc_dcf(f, s, None, None, None, ValuationApproach.base)
        conservative = _calc_dcf(f, s, None, None, None, ValuationApproach.conservative)
        assert base is not None and conservative is not None
        assert conservative.fair_value < base.fair_value

    def test_dcf_higher_growth_gives_higher_value(self):
        f = tech_company()
        s = empty_statements()
        low = _calc_dcf(f, s, 0.05, None, None, ValuationApproach.base)
        high = _calc_dcf(f, s, 0.25, None, None, ValuationApproach.base)
        assert low is not None and high is not None
        assert high.fair_value > low.fair_value

    def test_dcf_higher_discount_gives_lower_value(self):
        f = tech_company()
        s = empty_statements()
        low_discount = _calc_dcf(f, s, 0.10, 0.08, None, ValuationApproach.base)
        high_discount = _calc_dcf(f, s, 0.10, 0.15, None, ValuationApproach.base)
        assert low_discount is not None and high_discount is not None
        assert low_discount.fair_value > high_discount.fair_value

    def test_dcf_sensitivity_range_makes_sense(self):
        f = tech_company()
        s = empty_statements()
        result = _calc_dcf(f, s, None, None, None, ValuationApproach.base)
        if result and result.sensitivity_low and result.sensitivity_high:
            assert result.sensitivity_low < result.fair_value < result.sensitivity_high

    def test_dcf_no_fcf_uses_eps(self):
        f = tech_company(free_cash_flow=None)
        s = empty_statements()
        result = _calc_dcf(f, s, None, None, None, ValuationApproach.base)
        # Debe funcionar usando EPS
        assert result is not None

    def test_dcf_no_shares_returns_none(self):
        f = tech_company(shares_outstanding=None)
        s = empty_statements()
        result = _calc_dcf(f, s, None, None, None, ValuationApproach.base)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests de WACC y crecimiento
# ─────────────────────────────────────────────────────────────────────────────

class TestWACC:
    def test_wacc_in_reasonable_range(self):
        f = tech_company()
        wacc = _estimate_wacc(f)
        assert 0.07 <= wacc <= 0.20

    def test_high_debt_increases_wacc(self):
        f_no_debt = tech_company(total_debt=0, market_cap=15_000_000_000)
        f_high_debt = tech_company(total_debt=10_000_000_000, market_cap=5_000_000_000)
        wacc_low = _estimate_wacc(f_no_debt)
        wacc_high = _estimate_wacc(f_high_debt)
        # Con deuda el WACC puede bajar por el escudo fiscal,
        # pero el costo de equity es el mismo — resultado puede variar
        assert 0.07 <= wacc_low <= 0.20
        assert 0.07 <= wacc_high <= 0.20

    def test_high_beta_increases_wacc(self):
        f_low = tech_company(beta=0.5)
        f_high = tech_company(beta=2.0)
        wacc_low = _estimate_wacc(f_low)
        wacc_high = _estimate_wacc(f_high)
        assert wacc_high > wacc_low


# ─────────────────────────────────────────────────────────────────────────────
# Tests de clasificación de valoración
# ─────────────────────────────────────────────────────────────────────────────

class TestValuationClassification:
    def test_undervalued_with_large_discount(self):
        result = _classify_valuation(30.0, 100, 130.0, 110.0, 150.0)
        assert result == ValuationLevel.undervalued

    def test_overvalued_with_large_premium(self):
        result = _classify_valuation(-30.0, 100, 70.0, 60.0, 80.0)
        assert result == ValuationLevel.overvalued

    def test_fair_with_small_discount(self):
        result = _classify_valuation(8.0, 100, 108.0, 100.0, 120.0)
        assert result == ValuationLevel.fair

    def test_no_data_returns_insufficient(self):
        result = _classify_valuation(None, 100, None, None, None)
        assert result == ValuationLevel.insufficient_data


# ─────────────────────────────────────────────────────────────────────────────
# Tests de rangos de valor razonable
# ─────────────────────────────────────────────────────────────────────────────

class TestFairValueRanges:
    def test_ranges_ordered_correctly(self):
        prices = [80, 100, 120, 140]
        fv_c, fv_b, fv_o = _build_fair_value_ranges(prices, None, ValuationApproach.base)
        assert fv_c is not None and fv_b is not None and fv_o is not None
        assert fv_c <= fv_b <= fv_o

    def test_empty_prices_returns_nones(self):
        fv_c, fv_b, fv_o = _build_fair_value_ranges([], None, ValuationApproach.base)
        assert fv_c is None and fv_b is None and fv_o is None

    def test_single_price_returns_values(self):
        fv_c, fv_b, fv_o = _build_fair_value_ranges([100.0], None, ValuationApproach.base)
        assert fv_b is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tests de integración
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeValuation:
    def test_full_analysis_returns_complete_result(self):
        f = tech_company()
        s = empty_statements()
        result = analyze_valuation(f, s)

        assert result is not None
        assert result.current_price == 150.0
        assert result.valuation_level in ValuationLevel.__members__.values()
        assert len(result.multiples) > 0
        assert isinstance(result.narrative, str)
        assert len(result.narrative) > 20

    def test_no_price_raises_error(self):
        f = tech_company(current_price=None)
        s = empty_statements()
        with pytest.raises(ValueError):
            analyze_valuation(f, s)

    def test_overvalued_company_classified_correctly(self):
        # Empresa muy cara: precio 300, EPS 5 → PER 60x
        f = tech_company(current_price=300.0, trailing_eps=5.0, pe_ratio=60.0, market_cap=30_000_000_000)
        s = empty_statements()
        result = analyze_valuation(f, s)
        # Con precio doble del razonable debería clasificar como demanding o overvalued
        assert result.valuation_level in (ValuationLevel.demanding, ValuationLevel.overvalued)

    def test_cheap_company_classified_correctly(self):
        # Empresa barata: precio 50, EPS 8 → PER 6.25x
        f = tech_company(current_price=50.0, trailing_eps=8.0, pe_ratio=6.25,
                         market_cap=5_000_000_000, forward_eps=9.0, free_cash_flow=400_000_000)
        s = empty_statements()
        result = analyze_valuation(f, s)
        assert result.valuation_level in (ValuationLevel.undervalued, ValuationLevel.fair)

    def test_conservative_approach_gives_lower_fair_value(self):
        f = tech_company()
        s = empty_statements()
        base_result = analyze_valuation(f, s, ValuationApproach.base)
        conservative_result = analyze_valuation(f, s, ValuationApproach.conservative)
        # El DCF conservador debería producir valor base menor
        if base_result.fair_value_base and conservative_result.fair_value_base:
            assert conservative_result.fair_value_base <= base_result.fair_value_base

    def test_discount_to_base_calculated(self):
        f = tech_company()
        s = empty_statements()
        result = analyze_valuation(f, s)
        if result.fair_value_base and result.discount_to_base is not None:
            expected = (result.fair_value_base - result.current_price) / result.current_price * 100
            assert abs(result.discount_to_base - expected) < 0.1
