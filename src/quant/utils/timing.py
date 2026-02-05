import pandas as pd


def lag_for_trading(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Shift a time series so values known at t are used at t+lag.
    For daily trading, lag=1 is the standard "no look-ahead" convention.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    return df.shift(lag)
