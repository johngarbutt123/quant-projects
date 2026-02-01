import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """
    Convert simple returns to an equity curve (cumulative wealth index).
    """
    r = returns.fillna(0.0)
    return start * (1.0 + r).cumprod()


def cagr(returns: pd.Series, periods_per_year: float) -> float:
    """
    Compound annual growth rate from simple returns.
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    ec = equity_curve(r, start=1.0)
    total = float(ec.iloc[-1])

    years = len(r) / periods_per_year
    if years <= 0:
        return np.nan

    return total ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: float) -> float:
    """
    Annualized volatility of simple returns.
    """
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    """
    Sharpe ratio using simple returns.
    rf is annual risk-free rate (constant).
    """
    r = returns.dropna()
    if len(r) < 2:
        return np.nan

    # Convert annual rf to per-period rf (simple approximation)
    rf_period = rf / periods_per_year
    ex = r - rf_period

    vol = ex.std(ddof=1)
    if vol == 0:
        return np.nan

    return float(ex.mean() / vol * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: float, mar: float = 0.0) -> float:
    """
    Sortino ratio.
    mar is annual minimum acceptable return (constant).
    """
    r = returns.dropna()
    if len(r) < 2:
        return np.nan

    mar_period = mar / periods_per_year
    downside = (r - mar_period).clip(upper=0.0)

    dd = downside.std(ddof=1)
    if dd == 0:
        return np.nan

    ex_mean = (r - mar_period).mean()
    return float(ex_mean / dd * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Drawdown series from simple returns.
    """
    ec = equity_curve(returns, start=1.0)
    peak = ec.cummax()
    dd = ec / peak - 1.0
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown (most negative drawdown).
    """
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else np.nan


def calmar(returns: pd.Series, periods_per_year: float) -> float:
    """
    Calmar ratio = CAGR / |MaxDrawdown|
    """
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return np.nan
    return float(cagr(returns, periods_per_year) / abs(mdd))


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    periods_per_year: float,
    rf: float = 0.0,
) -> pd.Series:
    """
    Rolling Sharpe over a fixed window (in rows).
    """
    r = returns.dropna()
    if window <= 1:
        raise ValueError("window must be > 1")

    def _sh(x: pd.Series) -> float:
        return sharpe(x, periods_per_year=periods_per_year, rf=rf)

    return r.rolling(window=window).apply(_sh, raw=False)


def rolling_vol(
    returns: pd.Series,
    window: int,
    periods_per_year: float,
) -> pd.Series:
    """
    Rolling annualized volatility.
    """
    r = returns.dropna()
    if window <= 1:
        raise ValueError("window must be > 1")

    return r.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_cagr(
    returns: pd.Series,
    window: int,
    periods_per_year: float,
) -> pd.Series:
    """
    Rolling CAGR over a window.
    """
    r = returns.dropna()
    if window <= 1:
        raise ValueError("window must be > 1")

    def _c(x: pd.Series) -> float:
        return cagr(x, periods_per_year=periods_per_year)

    return r.rolling(window=window).apply(_c, raw=False)


def performance_summary(
    returns: pd.Series,
    periods_per_year: float,
    rf: float = 0.0,
    mar: float = 0.0,
) -> pd.Series:
    """
    One-line summary stats for a strategy return series.
    """
    out = {
        "CAGR": cagr(returns, periods_per_year),
        "Vol": annualized_vol(returns, periods_per_year),
        "Sharpe": sharpe(returns, periods_per_year, rf=rf),
        "Sortino": sortino(returns, periods_per_year, mar=mar),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar(returns, periods_per_year),
    }
    return pd.Series(out)
