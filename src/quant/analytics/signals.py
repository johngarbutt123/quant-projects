import numpy as np
import pandas as pd

# momentum, carry, value


def momentum_sign(
    prices: pd.DataFrame,
    lookback: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Sign-based time-series momentum signal:
        s_t = sign(P_t - P_{t-L})

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel (date × asset).
    lookback : int
        Lookback in rows (e.g., months if prices are monthly).
    min_periods : int | None
        Minimum observations required. Defaults to lookback.

    Returns
    -------
    pd.DataFrame
        Signal in {-1, 0, +1}.
    """
    if lookback <= 0:
        raise ValueError("lookback must be a positive integer")

    if min_periods is None:
        min_periods = lookback

    # Basic guard: require enough history
    valid = (
        prices.notna().rolling(window=lookback + 1, min_periods=min_periods + 1).sum()
        > min_periods
    )
    sig = np.sign(prices - prices.shift(lookback))
    sig = sig.where(valid)  # keep NaN where insufficient history

    return sig


def composite_trend_signal(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (3, 6, 12),
    weights: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """
    Composite trend signal:
        S_t = average_k s_{k}(t)   (or weighted average)

    Default:
        S_t = (s_3 + s_6 + s_12) / 3

    Parameters
    ----------
    prices : pd.DataFrame
    lookbacks : tuple[int, ...]
    weights : tuple[float, ...] | None
        If provided, must match lookbacks length.

    Returns
    -------
    pd.DataFrame
        Composite signal in [-1, +1].
    """
    if len(lookbacks) == 0:
        raise ValueError("lookbacks must be non-empty")

    if weights is None:
        w = np.ones(len(lookbacks), dtype=float) / len(lookbacks)
    else:
        if len(weights) != len(lookbacks):
            raise ValueError("weights must have the same length as lookbacks")
        w = np.asarray(weights, dtype=float)
        if np.isclose(w.sum(), 0.0):
            raise ValueError("weights sum must be non-zero")
        w = w / w.sum()

    sigs = [momentum_sign(prices, lb) for lb in lookbacks]
    composite = sum(wi * si for wi, si in zip(w, sigs))

    return composite


def clip_signal(
    signal: pd.DataFrame, lo: float = -1.0, hi: float = 1.0
) -> pd.DataFrame:
    """
    Clip continuous signals to a bounded range (useful later).
    """
    return signal.clip(lower=lo, upper=hi)
