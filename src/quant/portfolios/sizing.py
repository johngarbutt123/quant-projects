import numpy as np
import pandas as pd


def lag_positions(signal: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Lag signals to positions to avoid look-ahead bias.
    Example: signal at t applied as position at t+1 => lag=1.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    return signal.shift(lag)


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
    s = lag_positions(signal, lag=lag)

    # Align
    s, v = s.align(asset_vol, join="inner", axis=0)
    s, v = s.align(v, join="inner", axis=1)

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
    max_leverage: float | None = None,
) -> pd.Series:
    """
    Compute a time series of scalars k_t such that:
        scaled_weights_t = k_t * weights_t
    targets portfolio vol approximately equal to target_vol.

    covs: dict keyed by date with annualised covariance matrix (same assets).
    If cov matrix missing for a date, scalar will be NaN.
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

        if max_leverage is not None:
            k = min(k, max_leverage)

        scalars[dt] = k

    return pd.Series(scalars).sort_index()


def apply_scalar(weights: pd.DataFrame, scalar: pd.Series) -> pd.DataFrame:
    """Multiply each row of weights by scalar at that date."""
    scalar = scalar.reindex(weights.index)
    return weights.mul(scalar, axis=0).fillna(0.0)
