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
    method : {"log", "simple"}
    dropna : bool
        Drop initial NaN row.

    Returns
    -------
    pd.DataFrame
    """
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")

    if dropna:
        rets = rets.dropna(how="all")

    return rets
