import numpy as np
import pandas as pd


def lag_weights(target_weights: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Lag target weights to represent implementable holdings.
    If lag=1, weights decided at t are held over (t+1) returns.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    return target_weights.shift(lag)


def rebalance_schedule(index: pd.DatetimeIndex, freq: str = "M") -> pd.Series:
    """
    Create a boolean Series indicating rebalance dates.
    freq:
      - 'D' : every date
      - 'W' : weekly (period end)
      - 'M' : month end
      - 'Q' : quarter end
    """
    if freq not in {"D", "W", "M", "Q"}:
        raise ValueError("freq must be one of {'D','W','M','Q'}")

    if freq == "D":
        return pd.Series(True, index=index)

    periods = index.to_period(freq)
    is_end = periods != periods.shift(-1)
    return pd.Series(is_end, index=index)


def apply_rebalance(target_weights: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    Only change weights on rebalance dates; otherwise carry forward last weights.
    """
    w = target_weights.copy()
    w = w.sort_index()
    mask = rebalance_schedule(w.index, freq=freq)

    # Set non-rebalance dates to NaN then ffill to carry holdings
    w.loc[~mask.values, :] = np.nan
    w = w.ffill().fillna(0.0)
    return w


def turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Gross turnover per period:
        TO_t = sum_i |w_t - w_{t-1}|
    """
    w = weights.fillna(0.0)
    dw = w.diff().abs()
    return dw.sum(axis=1).fillna(0.0)


def transaction_costs(
    weights: pd.DataFrame,
    cost_bps: float = 5.0,
) -> pd.Series:
    """
    Simple linear transaction costs model:
        cost_t = (cost_bps / 10_000) * turnover_t
    """
    tc = (cost_bps / 10_000.0) * turnover(weights)
    return tc


def portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
) -> pd.Series:
    """
    Compute portfolio returns from weights and asset returns, optionally net of costs.

    Assumes:
      - weights are holdings weights applied to the SAME row of returns (i.e., already lagged)
      - returns are simple returns
    """
    # Align both
    w, r = weights.align(returns, join="inner", axis=0)
    w, r = w.align(r, join="inner", axis=1)

    w = w.fillna(0.0)
    r = r.fillna(0.0)

    gross = (w * r).sum(axis=1)

    if cost_bps and cost_bps != 0.0:
        costs = transaction_costs(w, cost_bps=cost_bps)
        net = gross - costs.reindex(gross.index).fillna(0.0)
        return net

    return gross


def run_execution(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    lag: int = 1,
    rebalance_freq: str = "M",
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    End-to-end execution wrapper.

    Steps:
      1) Rebalance target weights at chosen frequency
      2) Lag to get implementable holdings
      3) Compute turnover and portfolio returns (net of costs)

    Returns:
      holdings_weights, portfolio_returns, turnover
    """
    # 1) apply rebalance schedule to targets
    targets_reb = apply_rebalance(target_weights, freq=rebalance_freq)

    # 2) lag to implementable holdings
    holdings = lag_weights(targets_reb, lag=lag).fillna(0.0)

    # 3) turnover and portfolio returns
    to = turnover(holdings)
    port_ret = portfolio_returns(holdings, returns, cost_bps=cost_bps)

    return holdings, port_ret, to
