import pandas as pd


def align_weights_and_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align weights and returns on dates + columns.
    Assumes weights are the holdings applied to the SAME row of returns
    (i.e., already lagged / implementable).
    """
    w, r = weights.align(returns, join="inner", axis=0)
    w, r = w.align(r, join="inner", axis=1)
    return w.fillna(0.0), r.fillna(0.0)


def contribution_by_asset(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-asset contribution to portfolio return:
        contrib_{t,i} = w_{t,i} * r_{t,i}
    """
    w, r = align_weights_and_returns(weights, returns)
    return w * r


def portfolio_return_from_contrib(contrib: pd.DataFrame) -> pd.Series:
    """
    Portfolio return series implied by contribution matrix.
    """
    return contrib.sum(axis=1)


def cumulative_contribution(
    contrib: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cumulative contribution by asset, additive in simple return space:
        CumContrib_i(t) = sum_{s<=t} contrib_{s,i}
    This sums to cumulative portfolio return only approximately (because compounding),
    but is standard for attribution charts.
    """
    return contrib.cumsum()


def rolling_contribution(
    contrib: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Rolling window contribution sums.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    return contrib.rolling(window=window).sum()


def group_contribution(
    contrib: pd.DataFrame,
    group_map: dict[str, str],
) -> pd.DataFrame:
    """
    Aggregate contribution by groups (e.g. asset class).

    group_map: dict[asset -> group]
    Assets not found in group_map are assigned to 'UNMAPPED'.
    """
    assets = contrib.columns
    groups = pd.Series({a: group_map.get(a, "UNMAPPED") for a in assets})
    grouped = contrib.groupby(groups, axis=1).sum()
    return grouped


def top_contributors(
    contrib: pd.DataFrame,
    n: int = 10,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """
    Top contributors over a period (by total contribution).
    """
    c = contrib.copy()
    if start is not None:
        c = c.loc[c.index >= pd.to_datetime(start)]
    if end is not None:
        c = c.loc[c.index <= pd.to_datetime(end)]
    totals = c.sum().sort_values(ascending=False)
    return totals.head(n)


# ---------- Active attribution vs benchmark (optional) ----------


def active_contribution_by_asset(
    port_weights: pd.DataFrame,
    bench_weights: pd.DataFrame,
    port_returns: pd.DataFrame,
    bench_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Active contribution by asset.

    Minimal ex-post version:
      active_contrib_{t,i} = (w_p - w_b) * r_i

    If bench_returns is None, uses port_returns for r_i (common approximation when
    asset returns are the same for portfolio and benchmark constituents).
    """
    if bench_returns is None:
        bench_returns = port_returns

    wp, rp = align_weights_and_returns(port_weights, port_returns)
    wb, rb = align_weights_and_returns(bench_weights, bench_returns)

    # Align all together
    wp, wb = wp.align(wb, join="inner", axis=0)
    wp, wb = wp.align(wb, join="inner", axis=1)
    rp, rb = rp.align(rb, join="inner", axis=0)
    rp, rb = rp.align(rb, join="inner", axis=1)
    wp, rp = wp.align(rp, join="inner", axis=0)
    wp, rp = wp.align(rp, join="inner", axis=1)

    active_w = (wp - wb).fillna(0.0)
    r = rp.fillna(0.0)  # common underlying return
    return active_w * r
