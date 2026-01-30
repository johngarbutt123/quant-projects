from xbbg import blp
import pandas as pd


def get_bbg_data(
    tickers: list[str],
    fields: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical data from Bloomberg.

    Parameters
    ----------
    tickers : list[str]
        Bloomberg tickers (e.g. 'AAPL US Equity')
    fields : list[str]
        Bloomberg fields (e.g. 'PX_LAST')
    start_date : str
        YYYY-MM-DD
    end_date : str
        YYYY-MM-DD

    Returns
    -------
    pd.DataFrame
    """
    return blp.bdh(
        tickers=tickers,
        flds=fields,
        start_date=start_date,
        end_date=end_date,
    )
