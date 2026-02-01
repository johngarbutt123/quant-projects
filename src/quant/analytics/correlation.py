# src/quant/analytics/correlation.py

import numpy as np
import pandas as pd


def rolling_correlation(
    returns: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Rolling correlation matrices.

    Returns
    -------
    dict:
        key   = end date of window
        value = correlation matrix (DataFrame)
    """
    if window <= 1:
        raise ValueError("window must be > 1")

    if min_periods is None:
        min_periods = window

    corrs: dict[pd.Timestamp, pd.DataFrame] = {}

    for end in range(window, len(returns) + 1):
        slice_ = returns.iloc[end - window : end]
        if slice_.notna().sum().min() < min_periods:
            continue

        corrs[returns.index[end - 1]] = slice_.corr()

    return corrs


def covariance_from_vol_and_corr(
    vol: pd.Series,
    corr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct covariance matrix from vol vector and correlation matrix.

    Σ = D · R · D
    """
    if not corr.index.equals(corr.columns):
        raise ValueError("Correlation matrix must have matching index/columns")

    vol = vol.reindex(corr.index)
    D = np.diag(vol.values)
    cov = D @ corr.values @ D

    return pd.DataFrame(cov, index=corr.index, columns=corr.columns)


def rolling_covariance(
    returns: pd.DataFrame,
    window: int,
    periods_per_year: float,
    min_periods: int | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Rolling annualised covariance matrices.
    """
    if window <= 1:
        raise ValueError("window must be > 1")

    if min_periods is None:
        min_periods = window

    covs: dict[pd.Timestamp, pd.DataFrame] = {}

    for end in range(window, len(returns) + 1):
        slice_ = returns.iloc[end - window : end]
        if slice_.notna().sum().min() < min_periods:
            continue

        cov = slice_.cov() * periods_per_year
        covs[returns.index[end - 1]] = cov

    return covs
