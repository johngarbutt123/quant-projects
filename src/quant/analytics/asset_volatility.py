# src/quant/analytics/volatility.py

import numpy as np
import pandas as pd


def rolling_vol(
    returns: pd.DataFrame,
    window: int,
    periods_per_year: float,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Rolling annualised volatility.

    returns: DataFrame of simple or log returns (consistent frequency).
    window: lookback window in rows.
    periods_per_year: 12 for monthly, 252 for daily, etc.
    """
    if window <= 1:
        raise ValueError("window must be > 1")

    if min_periods is None:
        min_periods = window

    vol = returns.rolling(window=window, min_periods=min_periods).std()
    return vol * np.sqrt(periods_per_year)


def ewma_vol(
    returns: pd.DataFrame,
    span: int,
    periods_per_year: float,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    EWMA annualised volatility (RiskMetrics style).

    span: pandas ewm span parameter (higher = smoother).
    """
    if span <= 1:
        raise ValueError("span must be > 1")

    vol = returns.ewm(span=span, min_periods=min_periods, adjust=False).std()
    return vol * np.sqrt(periods_per_year)


def vol_floor(vol: pd.DataFrame, floor: float = 1e-6) -> pd.DataFrame:
    """Prevent division-by-zero / crazy leverage."""
    return vol.clip(lower=floor)
