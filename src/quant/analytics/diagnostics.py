import pandas as pd


def summarize_signal(signal: pd.DataFrame, warmup: int) -> dict[str, object]:
    missing = signal.iloc[warmup:].isna().mean().sort_values(ascending=False)
    abs_mean = signal.abs().mean().sort_values(ascending=False)
    lo, hi = float(signal.min().min()), float(signal.max().max())
    desc = signal.stack().describe()
    return {
        "missing": missing,
        "abs_mean": abs_mean,
        "bounds": (lo, hi),
        "describe": desc,
    }
