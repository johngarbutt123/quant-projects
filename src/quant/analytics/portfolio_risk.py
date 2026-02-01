# src/quant/analytics/portfolio_risk.py

import numpy as np
import pandas as pd


def _to_series(w, index) -> pd.Series:
    """Coerce weights to a pd.Series aligned to index."""
    if isinstance(w, pd.Series):
        return w.reindex(index).fillna(0.0)
    return pd.Series(w, index=index, dtype=float).fillna(0.0)


def portfolio_variance(weights: pd.Series, cov: pd.DataFrame) -> float:
    """
    Portfolio variance: w' Σ w
    """
    w = _to_series(weights, cov.index).values.reshape(-1, 1)
    sigma = cov.loc[cov.index, cov.index].values
    return float(w.T @ sigma @ w)


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """
    Portfolio volatility: sqrt(w' Σ w)
    """
    var = portfolio_variance(weights, cov)
    return float(np.sqrt(var)) if var >= 0 else np.nan


def marginal_risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """
    Marginal Risk Contribution (MRC):
        MRC_i = (Σ w)_i / σ_p
    """
    w = _to_series(weights, cov.index)
    sigma_w = cov @ w  # vector
    port_vol = portfolio_volatility(w, cov)

    if port_vol == 0 or np.isnan(port_vol):
        return pd.Series(np.nan, index=cov.index)

    return sigma_w / port_vol


def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """
    Component Risk Contribution (CRC):
        RC_i = w_i * MRC_i
    These sum to portfolio volatility.
    """
    w = _to_series(weights, cov.index)
    mrc = marginal_risk_contribution(w, cov)
    return w * mrc


def normalize_gross(weights: pd.Series, gross: float = 1.0) -> pd.Series:
    """
    Normalize weights so sum(abs(w)) = gross.
    Good for long/short strategies.
    """
    w = weights.copy()
    denom = float(np.abs(w).sum())
    if denom == 0:
        return w * 0.0
    return w * (gross / denom)


def normalize_net(weights: pd.Series, net: float = 1.0) -> pd.Series:
    """
    Normalize weights so sum(w) = net.
    Good for long-only portfolios.
    """
    w = weights.copy()
    denom = float(w.sum())
    if denom == 0:
        return w * 0.0
    return w * (net / denom)


def equal_risk_contribution_weights(
    cov: pd.DataFrame,
    long_only: bool = True,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> pd.Series:
    """
    Compute Equal Risk Contribution (ERC) / Risk Parity weights.

    This is a simple iterative solver (good enough for learning + prototyping).
    - long_only=True: weights >= 0 and sum to 1
    - long_only=False: not supported in this minimal solver (yet)

    Returns
    -------
    pd.Series weights indexed by asset
    """
    if not long_only:
        raise NotImplementedError(
            "This minimal ERC solver is long-only only (for now)."
        )

    assets = cov.index
    n = len(assets)

    # Start with equal weights
    w = np.ones(n) / n
    sigma = cov.values

    for _ in range(max_iter):
        # Portfolio vol and contributions
        port_var = float(w.T @ sigma @ w)
        if port_var <= 0:
            break
        port_vol = np.sqrt(port_var)

        sigma_w = sigma @ w
        mrc = sigma_w / port_vol
        rc = w * mrc  # component risk contributions (sum to port_vol)
        target = port_vol / n

        # multiplicative update to push rc toward target
        # (damped to keep stable)
        adj = target / np.maximum(rc, 1e-12)
        w_new = w * adj
        w_new = np.clip(w_new, 1e-12, None)
        w_new = w_new / w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return pd.Series(w, index=assets)


def risk_budget_weights(
    cov: pd.DataFrame,
    budgets: pd.Series,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> pd.Series:
    """
    Risk budgeting generalisation of ERC (long-only):
    budgets sum to 1 and represent desired fraction of total risk.

    Returns weights summing to 1.
    """
    assets = cov.index
    b = budgets.reindex(assets).fillna(0.0).astype(float)
    b_sum = float(b.sum())
    if b_sum <= 0:
        raise ValueError("budgets must sum to > 0")
    b = b / b_sum

    n = len(assets)
    w = np.ones(n) / n
    sigma = cov.values

    for _ in range(max_iter):
        port_var = float(w.T @ sigma @ w)
        if port_var <= 0:
            break
        port_vol = np.sqrt(port_var)

        sigma_w = sigma @ w
        mrc = sigma_w / port_vol
        rc = w * mrc  # sum to port_vol

        target_rc = b.values * port_vol

        adj = target_rc / np.maximum(rc, 1e-12)
        w_new = w * adj
        w_new = np.clip(w_new, 1e-12, None)
        w_new = w_new / w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return pd.Series(w, index=assets)
