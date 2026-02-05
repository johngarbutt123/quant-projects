import pandas as pd


def clip_signal(
    signal: pd.DataFrame,
    lo: float = -1.0,
    hi: float = 1.0,
) -> pd.DataFrame:
    """
    Generic transform to bound continuous signals.

    This is not portfolio logic — it's a reusable analytics helper.
    """
    return signal.clip(lower=lo, upper=hi)
