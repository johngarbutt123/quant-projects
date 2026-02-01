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
