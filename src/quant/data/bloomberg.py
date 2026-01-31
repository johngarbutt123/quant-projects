from xbbg import blp
import pandas as pd


def get_bbg_data(
    tickers: list[str],
    fields: list[str],
    start_date: str,
    end_date: str,
    ticker_names: dict[str, str] | None = None,
    field_names: dict[str, str] | None = None,
    flatten: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical data from Bloomberg and optionally return a clean panel.

    Parameters
    ----------
    tickers : list[str]
    fields : list[str]
    start_date, end_date : str
    ticker_names : dict, optional
        Mapping Bloomberg ticker -> friendly name
    field_names : dict, optional
        Mapping Bloomberg field -> friendly name
    flatten : bool
        Flatten MultiIndex columns into single-level names

    Returns
    -------
    pd.DataFrame
    """
    df = blp.bdh(
        tickers=tickers,
        flds=fields,
        start_date=start_date,
        end_date=end_date,
    ).sort_index()

    # ensure we have a proper MultiIndex
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["ticker", "field"])

    # rename ticker level
    if ticker_names is not None:
        df = df.rename(columns=ticker_names, level="ticker")

    # rename field level
    if field_names is not None:
        df = df.rename(columns=field_names, level="field")

    if flatten:
        df.columns = [f"{t}-{f}" if f else t for t, f in df.columns]

    return df
