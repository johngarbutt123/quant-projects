from __future__ import annotations

from typing import Literal, TypedDict


AssetType = Literal["price", "yield", "filter"]
AssetClass = Literal["equity", "rates", "credit", "fx", "commodity", "vol", "bond_etf"]


class AssetSpec(TypedDict):
    bbg: str
    type: AssetType
    asset_class: AssetClass


# ---- SINGLE SOURCE OF TRUTH ----
UNIVERSE: dict[str, AssetSpec] = {
    # friendly_name: {bbg, type, asset_class}
    # equities
    "spx": {"bbg": "SPX Index", "type": "price", "asset_class": "equity"},
    "ndx": {"bbg": "NDX Index", "type": "price", "asset_class": "equity"},
    "rty": {"bbg": "RTY Index", "type": "price", "asset_class": "equity"},
    "nikkei": {"bbg": "NKY Index", "type": "price", "asset_class": "equity"},
    "ftse": {"bbg": "UKX Index", "type": "price", "asset_class": "equity"},
    # yields (rates levels)
    "us2y": {"bbg": "USGG2YR Index", "type": "yield", "asset_class": "rates"},
    "us10y": {"bbg": "USGG10YR Index", "type": "yield", "asset_class": "rates"},
    "uk10y": {"bbg": "GUKG10 Index", "type": "yield", "asset_class": "rates"},
    "bund10y": {"bbg": "DE10Y Index", "type": "yield", "asset_class": "rates"},
    # credit (ETFs => tradable prices)
    "ig_credit": {"bbg": "LQD US Equity", "type": "price", "asset_class": "credit"},
    "hy_credit": {"bbg": "HYG US Equity", "type": "price", "asset_class": "credit"},
    # volatility (filter/regime variable)
    "vix": {"bbg": "VIX Index", "type": "filter", "asset_class": "vol"},
    # FX
    "dxy": {"bbg": "DXY Index", "type": "price", "asset_class": "fx"},
    "eurusd": {"bbg": "EUR Curncy", "type": "price", "asset_class": "fx"},
    "gbpusd": {"bbg": "GBP Curncy", "type": "price", "asset_class": "fx"},
    "usdjpy": {"bbg": "JPY Curncy", "type": "price", "asset_class": "fx"},
    "audusd": {"bbg": "AUD Curncy", "type": "price", "asset_class": "fx"},
    # commodities
    "oil": {"bbg": "CL1 Comdty", "type": "price", "asset_class": "commodity"},
    "gold": {"bbg": "XAU Curncy", "type": "price", "asset_class": "commodity"},
    # defensive anchor
    "ust_intermediate": {
        "bbg": "IEF US Equity",
        "type": "price",
        "asset_class": "bond_etf",
    },
}


# ---- DERIVED (never edit below) ----

# Bloomberg tickers to request
BBG_TICKERS = [spec["bbg"] for spec in UNIVERSE.values()]

# Mapping for loader rename: bbg_ticker -> friendly_name
TICKER_NAMES = {spec["bbg"]: name for name, spec in UNIVERSE.items()}

# Friendly-name groupings
PRICE_ASSETS = [name for name, spec in UNIVERSE.items() if spec["type"] == "price"]
YIELD_ASSETS = [name for name, spec in UNIVERSE.items() if spec["type"] == "yield"]
FILTER_ASSETS = [name for name, spec in UNIVERSE.items() if spec["type"] == "filter"]

ALL_ASSETS = list(UNIVERSE.keys())
