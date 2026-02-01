from __future__ import annotations

import pandas as pd
from xbbg import blp


def bbg_bdh(
    tickers: list[str],
    fields: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Raw Bloomberg pull. Returns DataFrame with MultiIndex columns: (ticker, field).
    """
    df = blp.bdh(
        tickers=tickers,
        flds=fields,
        start_date=start_date,
        end_date=end_date,
    ).sort_index()

    # xbbg typically returns MultiIndex columns already; just name them
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        raise ValueError("Expected MultiIndex columns (ticker, field) from xbbg.bdh")

    df.columns = df.columns.set_names(["ticker", "field"])
    return df


def get_bbg_field_panels(
    tickers: list[str],
    fields: list[str],
    start_date: str,
    end_date: str,
    ticker_names: dict[str, str] | None = None,
    field_names: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch Bloomberg historical data and return panels by field.

    Contract (always):
    ------------------
    Returns:
        dict[field] -> DataFrame
            index   = DatetimeIndex
            columns = tickers (or renamed tickers)
            values  = Bloomberg field values

    Example:
        {
            "PX_LAST":  DataFrame(date × ticker),
            "PX_VOLUME": DataFrame(date × ticker),
        }
    """
    if len(fields) == 0:
        raise ValueError("fields must contain at least one Bloomberg field")

    raw = blp.bdh(
        tickers=tickers,
        flds=fields,
        start_date=start_date,
        end_date=end_date,
    ).sort_index()

    # Expect MultiIndex columns: (ticker, field)
    if not isinstance(raw.columns, pd.MultiIndex) or raw.columns.nlevels != 2:
        raise ValueError("Expected MultiIndex columns (ticker, field) from xbbg.bdh")

    raw.columns = raw.columns.set_names(["ticker", "field"])

    # Optional renaming
    if ticker_names:
        raw = raw.rename(columns=ticker_names, level="ticker")
    if field_names:
        raw = raw.rename(columns=field_names, level="field")

    # Final field labels after renaming
    final_fields = raw.columns.get_level_values("field").unique().tolist()

    panels: dict[str, pd.DataFrame] = {}

    for f in final_fields:
        panel = raw.xs(f, axis=1, level="field", drop_level=True)
        panel = panel.dropna(how="all").sort_index()
        panels[f] = panel

    return panels
