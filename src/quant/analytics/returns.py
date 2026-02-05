import pandas as pd
import numpy as np


def prices_to_returns(
    prices: pd.DataFrame,
    method: str = "log",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Convert price series to returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Price time series.
    method:
        - 'log'    : log(P_t / P_{t-1})
        - 'simple' : P_t / P_{t-1} - 1
    dropna : bool
        Drop initial NaN row.

    Returns
    -------
    pd.DataFrame
    """
    if method == "log":
        prices = prices.where(prices > 0)
        rets = np.log(prices).diff()
        # rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")

    return rets.dropna(how="all") if dropna else rets


def prices_to_diffs(
    prices: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Convert level series to first differences.

    Intended for:
    - yields
    - spreads
    - macro level series
    """
    diffs = prices.diff()
    return diffs.dropna(how="all") if dropna else diffs
