import numpy as np
import pandas as pd
from quant.analytics.transforms import clip_signal
from quant.utils.timing import lag_for_trading
# signals.py  →  what you THINK the market will do
# sizing.py   →  how BIG you are willing to bet on that view


def inverse_vol_positions(
    signal: pd.DataFrame,
    asset_vol: pd.DataFrame,
    vol_floor: float = 1e-4,
    gross: float = 1.0,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Build positions proportional to signal / vol (risk parity-ish at asset level),
    then normalise so sum(abs(w)) = gross at each date.

    Parameters
    ----------
    signal : DataFrame
        Signal in [-1, 1] (date × asset).
    asset_vol : DataFrame
        Annualised vol per asset (date × asset), same frequency as signal.
    vol_floor : float
        Minimum vol to avoid infinite leverage.
    gross : float
        Target gross exposure per date (sum abs weights).
    lag : int
        Apply signal with a lag to avoid look-ahead.

    Returns
    -------
    DataFrame of weights (date × asset).
    """
    s = lag_for_trading(signal, lag=lag)

    # Align
    s, v = signal.align(asset_vol, join="inner", axis=None)

    v = v.clip(lower=vol_floor)
    raw = s / v  # higher weight for lower vol assets

    # Normalise to gross exposure per row
    denom = raw.abs().sum(axis=1).replace(0.0, np.nan)
    w = raw.div(denom, axis=0) * gross

    return w.fillna(0.0)


def target_portfolio_vol_scalar(
    weights: pd.DataFrame,
    covs: dict[pd.Timestamp, pd.DataFrame],
    target_vol: float = 0.10,
    max_scalar: float | None = None,
) -> pd.Series:
    """
    Compute a time series of scalars k_t such that:
        scaled_weights_t = k_t * weights_t
    targets portfolio vol approximately equal to target_vol.

    covs: dict keyed by date with annualised covariance matrix (same assets).
    If cov matrix missing for a date, scalar will be NaN.



    5) Potential lookahead: are your covs dated correctly?

    target_portfolio_vol_scalar() uses covs.get(dt) for the same dt as the weights row. That’s only safe if:

    cov at dt was computed using data up to dt-1, and is stamped at dt.

    If your cov dict is keyed by the end date of the window (same day), you might be leaking today’s return into today’s vol target (small but real lookahead).

    Defensive fix: pass lag or use covs.get(dt - 1BDay) style, or build covs explicitly lagged.

    (You may already be doing this upstream — just flagging it because it’s the most common hidden bug in vol targeting.)

    """
    scalars = {}

    for dt, w_row in weights.iterrows():
        cov = covs.get(dt)
        if cov is None:
            scalars[dt] = np.nan
            continue

        # Align assets
        w = w_row.reindex(cov.index).fillna(0.0).values.reshape(-1, 1)
        sigma = cov.values

        port_var = float(w.T @ sigma @ w)
        port_vol = np.sqrt(port_var) if port_var > 0 else np.nan

        if not np.isfinite(port_vol) or port_vol == 0:
            scalars[dt] = np.nan
            continue

        k = target_vol / port_vol

        if max_scalar is not None:
            k = min(k, max_scalar)

        scalars[dt] = k

    return pd.Series(scalars).sort_index()


def apply_scalar(weights: pd.DataFrame, scalar: pd.Series) -> pd.DataFrame:
    """Multiply each row of weights by scalar at that date."""
    scalar = scalar.reindex(weights.index)
    return weights.mul(scalar, axis=0).fillna(0.0)


def make_tradable_signal(
    signal_raw: pd.DataFrame,
    smooth_span: int = 40,
    clip: float | None = 1.0,
    lag: int = 1,
) -> pd.DataFrame:
    # 1) smooth (trading choice)
    s = signal_raw.ewm(span=smooth_span, adjust=False).mean()

    # 2) clip (risk discipline)
    if clip is not None:
        # Defensive bound (future-proofing, not strictly needed today as signal is already in [-1, 1])
        s = clip_signal(s, -clip, clip)

    # 3) Tradable lag (no lookahead) otherwise perfect foresight
    return lag_for_trading(s, lag=lag)
