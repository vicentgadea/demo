"""
Sector Level Analysis — Level 2 of Market → Sector → Value methodology.

Maps each ticker to its GICS sector ETF proxy, then compares its
relative strength vs SPY over 20/50/200 days.

Sector ETF Map (SPDR Select Sector ETFs):
  Technology           → XLK
  Financials           → XLF
  Energy               → XLE
  Health Care          → XLV
  Consumer Discretionary → XLY
  Consumer Staples     → XLP
  Industrials          → XLI
  Materials            → XLB
  Real Estate          → XLRE
  Utilities            → XLU
  Communication Services → XLC
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..models.scanner_models import SectorContext, SectorStatus

logger = logging.getLogger(__name__)

SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
    "Telecommunications": "XLC",
}

# Relative strength thresholds (sector / SPY return ratio)
RS_LEADER_THRESHOLD = 1.05    # sector outperforms SPY by >5%
RS_LAGGARD_THRESHOLD = 0.95   # sector underperforms SPY by >5%


def get_sector_etf(sector_name: str | None) -> str | None:
    """Return the SPDR sector ETF for a given GICS sector name."""
    if not sector_name:
        return None
    return SECTOR_ETF_MAP.get(sector_name)


def _relative_strength(etf_df: pd.DataFrame, spy_df: pd.DataFrame, days: int) -> float | None:
    """
    Relative strength = return(ETF, days) / return(SPY, days).
    Returns None if data is insufficient.
    """
    def _ret(df: pd.DataFrame, n: int) -> float | None:
        close_col = "close" if "close" in df.columns else df.columns[0]
        c = df[close_col].dropna()
        if len(c) < n + 1:
            return None
        start = float(c.iloc[-(n + 1)])
        end = float(c.iloc[-1])
        if start == 0:
            return None
        return (end - start) / start

    etf_ret = _ret(etf_df, days)
    spy_ret = _ret(spy_df, days)

    if etf_ret is None or spy_ret is None:
        return None
    # Avoid division by near-zero
    denominator = abs(spy_ret) if abs(spy_ret) > 0.001 else 0.001
    return round(etf_ret / denominator, 3)


def analyze_sector(
    sector_name: str | None,
    etf_df: pd.DataFrame | None,
    spy_df: pd.DataFrame | None,
) -> SectorContext:
    """
    Analyse sector relative strength and classify as LIDER / NEUTRO / REZAGADO.
    """
    etf_ticker = get_sector_etf(sector_name)

    if etf_df is None or spy_df is None or etf_df.empty or spy_df.empty:
        return SectorContext(
            sector_name=sector_name or "Unknown",
            sector_etf=etf_ticker,
            status=SectorStatus.neutro,
            description="Datos de sector no disponibles. Se asume sector neutro.",
        )

    rs_20 = _relative_strength(etf_df, spy_df, 20)
    rs_50 = _relative_strength(etf_df, spy_df, 50)
    rs_200 = _relative_strength(etf_df, spy_df, 200)

    # Composite RS score weighted: recent more important
    values = [v for v in [rs_20, rs_50, rs_200] if v is not None]
    weights_used = []
    weighted_sum = 0.0
    total_w = 0.0
    for rs, w in zip([rs_20, rs_50, rs_200], [0.5, 0.3, 0.2]):
        if rs is not None:
            weighted_sum += rs * w
            total_w += w

    composite_rs = (weighted_sum / total_w) if total_w > 0 else None

    if composite_rs is None:
        status = SectorStatus.neutro
    elif composite_rs >= RS_LEADER_THRESHOLD:
        status = SectorStatus.lider
    elif composite_rs <= RS_LAGGARD_THRESHOLD:
        status = SectorStatus.rezagado
    else:
        status = SectorStatus.neutro

    # Build description
    parts: list[str] = [f"Sector: {sector_name or 'N/A'} | ETF: {etf_ticker or 'N/A'}"]
    if rs_20 is not None:
        parts.append(f"RS 20d={rs_20:.2f}")
    if rs_50 is not None:
        parts.append(f"RS 50d={rs_50:.2f}")
    if rs_200 is not None:
        parts.append(f"RS 200d={rs_200:.2f}")
    if composite_rs is not None:
        parts.append(f"RS compuesta={composite_rs:.2f}")

    label_map = {
        SectorStatus.lider: "LÍDER (sector supera al mercado)",
        SectorStatus.neutro: "NEUTRO (sector en línea con el mercado)",
        SectorStatus.rezagado: "REZAGADO (sector por debajo del mercado)",
    }
    parts.append(label_map[status])

    return SectorContext(
        sector_name=sector_name or "Unknown",
        sector_etf=etf_ticker,
        status=status,
        rs_20d=rs_20,
        rs_50d=rs_50,
        rs_200d=rs_200,
        description=". ".join(parts) + ".",
    )
