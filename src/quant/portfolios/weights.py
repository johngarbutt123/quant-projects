import numpy as np
import pandas as pd

from quant.analytics.portfolio_risk import equal_risk_contribution_weights


def normalize_gross(
    weights: pd.DataFrame | pd.Series, gross: float = 1.0
) -> pd.DataFrame | pd.Series:
    """
    Normalize so sum(abs(w)) == gross.
    Works for Series (single date) or DataFrame (time series).
    """
    if isinstance(weights, pd.Series):
        denom = float(weights.abs().sum())
        return weights * (gross / denom) if denom != 0 else weights * 0.0

    denom = weights.abs().sum(axis=1).replace(0.0, np.nan)
    out = weights.div(denom, axis=0) * gross
    return out.fillna(0.0)


def normalize_net(
    weights: pd.DataFrame | pd.Series, net: float = 1.0
) -> pd.DataFrame | pd.Series:
    """
    Normalize so sum(w) == net.
    Useful for long-only, or forcing net exposure in long/short.
    """
    if isinstance(weights, pd.Series):
        denom = float(weights.sum())
        return weights * (net / denom) if denom != 0 else weights * 0.0

    denom = weights.sum(axis=1).replace(0.0, np.nan)
    out = weights.div(denom, axis=0) * net
    return out.fillna(0.0)


def equal_weighted(assets: list[str]) -> pd.Series:
    """Simple equal weights that sum to 1 (long-only)."""
    n = len(assets)
    if n == 0:
        raise ValueError("assets must be non-empty")
    return pd.Series(1.0 / n, index=assets)


def risk_parity_base_weights(
    cov: pd.DataFrame,
) -> pd.Series:
    """
    Long-only risk parity / ERC base weights that sum to 1.
    Wrapper around analytics.portfolio_risk.equal_risk_contribution_weights.
    """
    return equal_risk_contribution_weights(cov, long_only=True)


def build_base_weights_over_time(
    covs: dict[pd.Timestamp, pd.DataFrame],
    method: str = "erc",
) -> pd.DataFrame:
    """
    Build a time series of base (long-only) weights from rolling cov matrices.

    covs: dict[date -> cov matrix]
    method: 'erc' or 'equal'
    """
    if method not in {"erc", "equal"}:
        raise ValueError("method must be 'erc' or 'equal'")

    rows = {}
    for dt, cov in covs.items():
        assets = list(cov.index)
        if method == "equal":
            w = equal_weighted(assets)
        else:
            w = risk_parity_base_weights(cov)
        rows[dt] = w

    base = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return base.fillna(0.0)


def apply_signal_to_base_weights(
    base_weights: pd.DataFrame,
    signal: pd.DataFrame,
    gross: float = 1.0,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Combine long-only base weights with long/short signals.

    Typical use:
      base_weights: ERC weights (sum to 1, all positive)
      signal:       trend signal in [-1, +1] or {-1,0,+1}

    Output:
      signed weights normalized to target gross exposure.

    Notes:
    - Lag is applied to the signal (avoid look-ahead).
    - Final normalization is gross, so long/short gross is controlled.
    """
    s = signal.shift(lag)

    # Align dates/assets
    bw, s = base_weights.align(s, join="inner", axis=0)
    bw, s = bw.align(s, join="inner", axis=1)

    w = bw * s
    w = normalize_gross(w, gross=gross)

    return w


def cap_gross(
    weights: pd.DataFrame,
    max_gross: float,
) -> pd.DataFrame:
    """
    Cap gross exposure by scaling down only when sum(abs(w)) > max_gross.
    Leaves weights unchanged when below the cap.
    """
    if max_gross <= 0:
        raise ValueError("max_gross must be > 0")

    gross = weights.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0)
    return weights.mul(scale, axis=0)


def asset_vol_target_weights(
    signal_lagged: pd.DataFrame,
    vol_lagged: pd.DataFrame,
    target_vol: float,
    *,
    weight_cap: float | None = None,
    max_gross: float | None = None,
) -> pd.DataFrame:
    """
    Vol-target weights from a *lagged* signal and *lagged* annualised volatility.

    Assumes:
      - signal_lagged is already tradable (i.e., based on info up to t-1 for trading at t)
      - vol_lagged is also tradable (e.g., rolling vol shifted by 1)

    Formula:
      w_i,t = signal_i,t-1 * (target_vol / vol_i,t-1)

    Then optionally:
      - cap each asset weight to +/- weight_cap
      - cap portfolio gross exposure to max_gross (scale down only)

    Returns:
      weights DataFrame aligned on common dates/assets and filled with 0.0.
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be > 0")

    # Align dates/assets (inner join)
    s, v = signal_lagged.align(vol_lagged, join="inner", axis=0)
    s, v = s.align(v, join="inner", axis=1)

    # Raw sizing
    w = s * (target_vol / v)
    w = w.replace([np.inf, -np.inf], np.nan)

    # Per-asset cap
    if weight_cap is not None:
        if weight_cap <= 0:
            raise ValueError("weight_cap must be > 0")
        w = w.clip(lower=-weight_cap, upper=weight_cap)

    # Gross cap (scale down only)
    if max_gross is not None:
        w = cap_gross(w, max_gross=max_gross)

    return w.fillna(0.0)


def portfolio_vol_target_weights(
    weights: pd.DataFrame,
    port_vol_tm1: pd.Series,
    target_vol: float,
    scale_cap: float | None = None,
    max_gross: float | None = None,
) -> pd.DataFrame:
    """
    Scale portfolio weights to a target annualised portfolio volatility.

    Parameters
    ----------
    weights : DataFrame (T x N)
        Day-t weights (already tradable for day t).
    port_vol_tm1 : Series (T,)
        Estimated portfolio vol using information up to t-1.
    target_vol : float
        Desired annualised portfolio volatility (e.g., 0.10).
    scale_cap : float | None
        Optional cap on scaling factor to avoid extreme leverage.
    max_gross : float | None
        Optional cap on final gross exposure.

    Returns
    -------
    DataFrame of scaled weights.
    """

    # Gross exposure to detect flat days
    gross = weights.abs().sum(axis=1)
    flat = gross == 0

    # Compute raw scale
    scale = target_vol / port_vol_tm1

    # Numerical clean-up
    scale = scale.replace([np.inf, -np.inf], np.nan)

    if scale_cap is not None:
        scale = scale.clip(upper=scale_cap)

    # Flat days stay flat
    scale.loc[flat] = 0.0

    # Early warm-up / missing data → flat
    scale = scale.fillna(0.0)

    # Apply scaling
    w_out = weights.mul(scale, axis=0)

    # Optional gross cap
    if max_gross is not None:
        w_out = cap_gross(w_out, max_gross=max_gross)

    return w_out
