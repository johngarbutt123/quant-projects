# src/quant/data/panels.py
"""
Panels = "shape management" for market data.
- sort/clean indices
- handle duplicates
- resample prices to a chosen frequency
- align columns to a common date index (inner/outer)
- apply a missing-data policy (ffill/bfill/none)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import pandas as pd

Freq = Literal["D", "W", "M"]
AlignMethod = Literal["inner", "outer"]
FillMethod = Optional[Literal["ffill", "bfill", "none"]]


@dataclass(frozen=True)
class MarketPanel:
    """A canonical market panel: aligned prices at a chosen frequency."""

    prices: pd.DataFrame
    frequency: Freq
    align: AlignMethod
    fill: FillMethod


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sort index, drop duplicate timestamps (keep last), drop all-NaN rows."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    out = df.copy()
    out = out.sort_index()

    # Drop duplicate index entries (keep last observation)
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]

    # Drop rows where everything is NaN
    out = out.dropna(how="all")
    return out


def standardize_columns(
    df: pd.DataFrame, columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Optional: enforce a specific column order / subset.
    If columns provided: reindex to those columns (missing become NaN).
    """
    out = df.copy()
    if columns is not None:
        cols = list(columns)
        out = out.reindex(columns=cols)
    return out


def resample_prices(prices: pd.DataFrame, freq: Freq = "M") -> pd.DataFrame:
    """
    Resample prices to period-end.
    - D: no resampling
    - W: weekly, Friday close (last obs in W-FRI bucket)
    - M: month-end, last obs in month
    """
    prices = _clean_index(prices)

    if freq == "D":
        return prices

    rule = {"W": "W-FRI", "M": "M"}[freq]
    out = prices.resample(rule).last()
    out = out.dropna(how="all")
    return out


def align_panel(
    prices: pd.DataFrame,
    method: AlignMethod = "inner",
    fill: FillMethod = "ffill",
) -> pd.DataFrame:
    """
    Align prices across assets.

    method:
      - inner: keep only dates where ALL assets have prices (strict)
      - outer: keep union of dates (lenient), then optionally fill gaps

    fill (only meaningful for outer):
      - ffill / bfill / none
    """
    prices = _clean_index(prices)

    if method == "inner":
        # strict intersection: drop any date with any missing value
        return prices.dropna(how="any")

    # outer: keep union, optionally fill gaps
    out = prices.copy()

    if fill is None or fill == "none":
        return out.dropna(how="all")

    if fill == "ffill":
        out = out.ffill()
    elif fill == "bfill":
        out = out.bfill()
    else:
        raise ValueError(f"Unknown fill method: {fill}")

    # still allow leading rows to be NaN for some assets; drop all-NaN rows only
    out = out.dropna(how="all")
    return out


def trim_date_range(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience slicing by date strings."""
    out = df
    if start is not None:
        out = out.loc[pd.to_datetime(start) :]
    if end is not None:
        out = out.loc[: pd.to_datetime(end)]
    return out


def build_market_panel(
    prices: pd.DataFrame,
    freq: Freq = "M",
    align: AlignMethod = "inner",
    fill: FillMethod = "ffill",
    start: Optional[str] = None,
    end: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
) -> MarketPanel:
    """
    One-stop canonicaliser:
      raw prices -> cleaned -> resampled -> aligned -> trimmed -> column standardised

    This is the function you call from your notebook to create `prices_df`.
    """
    p = prices.copy()
    p = standardize_columns(p, columns=columns)
    p = _clean_index(p)
    p = trim_date_range(p, start=start, end=end)
    p = resample_prices(p, freq=freq)
    p = align_panel(p, method=align, fill=fill)

    return MarketPanel(prices=p, frequency=freq, align=align, fill=fill)
