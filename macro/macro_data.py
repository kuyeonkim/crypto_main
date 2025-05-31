"""Utility functions for fetching macroeconomic indicators.

This module provides helper functions to download common macroeconomic
series from public data providers.  The focus is on using :mod:`pandasdmx`
where possible with :mod:`pandas_datareader` as a fallback for the World
Bank and IMF series.  Each function returns a :class:`pandas.DataFrame`
indexed by year.

The functions are intentionally light wrappers so that callers can easily
assemble custom pipelines.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional
import os

import pandas as pd

# Directory to write downloaded CSV files. Update this path as needed.
SAVE_DIR = "/path/to/save"

# Common country codes used across providers. ISO Alpha-3 codes work for
# World Bank, IMF and ILOSTAT.
MAJOR_COUNTRIES: list[str] = [
    "USA", "CAN", "MEX", "BRA", "ARG",
    "GBR", "FRA", "DEU", "ITA", "ESP",
    "CHN", "JPN", "KOR", "IND", "IDN",
    "AUS", "NZL", "RUS", "TUR", "SAU",
    "ZAF", "NGA", "EGY", "IRN", "PAK",
    "THA", "MYS", "PHL", "SGP", "VNM",
]


def _save_dataframe(df: pd.DataFrame, dataset: str, country: str, save_dir: str) -> None:
    """Save *df* to ``save_dir/dataset/country.csv``.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame to save.
    dataset : str
        Name of the dataset subfolder.
    country : str
        ISO country code.
    save_dir : str
        Root directory where files are written.
    """
    path = os.path.join(save_dir, dataset)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{country}.csv")
    df.to_csv(file_path)


try:
    import pandasdmx
except ImportError:  # pragma: no cover - pandasdmx may not be installed
    pandasdmx = None  # type: ignore

try:
    from pandas_datareader import wb, data as pdr_data
except ImportError:  # pragma: no cover - pandas-datareader may not be installed
    wb = None  # type: ignore
    pdr_data = None  # type: ignore


# ---------------------------------------------------------------------------
# World Bank helpers
# ---------------------------------------------------------------------------

def _wb_series(indicator: str, country: str, start: int, end: Optional[int]) -> pd.DataFrame:
    """Download a World Bank indicator via pandas-datareader.

    Parameters
    ----------
    indicator : str
        Indicator series code.
    country : str
        ISO country code or ``'all'`` for all countries.
    start : int
        First year to request.
    end : int, optional
        Last year to request.  Defaults to the current year.
    """
    if wb is None:
        raise RuntimeError("pandas-datareader is required for World Bank data")
    if end is None:
        end = datetime.now().year
    df = wb.download(indicator=indicator, country=country, start=start, end=end)
    df = df.reset_index().rename(columns={"country": "Country", "year": "Year", indicator: "Value"})
    return df.set_index("Year")


# ---------------------------------------------------------------------------
# ILO helpers using pandasdmx
# ---------------------------------------------------------------------------

def _ilo_series(flow: str, key: str, start: int, end: Optional[int]) -> pd.DataFrame:
    """Download a series from ILOSTAT using pandasdmx.

    Parameters
    ----------
    flow : str
        Dataflow identifier.
    key : str
        SDMX key describing the desired series.
    start : int
        First year of data.
    end : int, optional
        Last year of data.  Defaults to the current year.
    """
    if pandasdmx is None:
        raise RuntimeError("pandasdmx is required for ILO data")
    if end is None:
        end = datetime.now().year
    req = pandasdmx.Request("ILO")
    params = {"startPeriod": start, "endPeriod": end}
    resp = req.data(resource_id=flow, key=key, params=params)
    data = resp.to_pandas()
    data.index.name = "Year"
    return data


# ---------------------------------------------------------------------------
# Public functions for each macro variable
# ---------------------------------------------------------------------------

def real_gdp_per_capita_ppp(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Real GDP per capita (PPP, constant dollars) from World Bank."""
    df = _wb_series("NY.GDP.PCAP.PP.KD", country, start, end)
    if save_dir:
        _save_dataframe(df, "realgdp_capita", country, save_dir)
    return df


def gdp_growth(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Annual real GDP growth rate from World Bank."""
    df = _wb_series("NY.GDP.MKTP.KD.ZG", country, start, end)
    if save_dir:
        _save_dataframe(df, "gdp_growth", country, save_dir)
    return df


def gni_per_capita(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    ppp: bool = True, save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """GNI per capita from World Bank.

    Parameters
    ----------
    ppp : bool
        If ``True``, use the PPP series; otherwise use Atlas method.
    """
    code = "NY.GNP.PCAP.PP.CD" if ppp else "NY.GNP.PCAP.CD"
    df = _wb_series(code, country, start, end)
    if save_dir:
        _save_dataframe(df, "gni_per_capita", country, save_dir)
    return df


def labor_share_gdp(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Labor share of GDP (%) using ILOSTAT SDG 10.4.1 series."""
    key = f"A.{country}.EMP_DWAP_SEX_NOCU_N_OCU_PCAP10.1041.A"  # Example key; adjust as needed
    df = _ilo_series("SDG_1041_NOCU", key, start, end)
    if save_dir:
        _save_dataframe(df, "labor_share_gdp", country, save_dir)
    return df


def unemployment_rate(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Unemployment rate (%) using ILOSTAT modelled estimates."""
    key = f"A.{country}.UNE_TUNE_SEX_AGE_RT.A"  # Example key; adjust as needed
    df = _ilo_series("UNE_TUNE_SEX_AGE_RT", key, start, end)
    if save_dir:
        _save_dataframe(df, "unemployment_rate", country, save_dir)
    return df


def inflation_cpi(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Inflation, CPI (annual %) from IMF IFS."""
    if pdr_data is None:
        raise RuntimeError("pandas-datareader is required for IMF data")
    if end is None:
        end = datetime.now()
    series = f"PCPI_\"{country}\""  # IFS CPI series; may need adjustment
    df = pdr_data.DataReader(series, "imf", start=str(start), end=str(end.year))
    df.index.name = "Year"
    if save_dir:
        _save_dataframe(df, "inflation", country, save_dir)
    return df


def gross_fixed_capital_formation(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Gross fixed capital formation (% of GDP) from World Bank."""
    df = _wb_series("NE.GDI.FTOT.ZS", country, start, end)
    if save_dir:
        _save_dataframe(df, "gross_fixed_capital_formation", country, save_dir)
    return df


def gross_domestic_savings(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Gross domestic savings (% of GDP) from World Bank."""
    df = _wb_series("NY.GDS.TOTL.ZS", country, start, end)
    if save_dir:
        _save_dataframe(df, "gross_domestic_savings", country, save_dir)
    return df


def gini_index(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Income inequality measured by the Gini index from World Bank PIP."""
    df = _wb_series("SI.POV.GINI", country, start, end)
    if save_dir:
        _save_dataframe(df, "gini_index", country, save_dir)
    return df


def labor_productivity(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> pd.DataFrame:
    """Labor productivity (GDP per worker, PPP) from World Bank."""
    df = _wb_series("SL.GDP.PCAP.EM.KD", country, start, end)
    if save_dir:
        _save_dataframe(df, "labor_productivity", country, save_dir)
    return df



def gather_all(
    country: str = "USA", start: int = 2015, end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> dict[str, pd.DataFrame]:
    """Collect all macro series for ``country`` and optionally save them."""
    data = {
        "real_gdp_per_capita_ppp": real_gdp_per_capita_ppp(country, start, end, save_dir),
        "gdp_growth": gdp_growth(country, start, end, save_dir),
        "gni_per_capita": gni_per_capita(country, start, end, save_dir),
        "labor_share_gdp": labor_share_gdp(country, start, end, save_dir),
        "unemployment_rate": unemployment_rate(country, start, end, save_dir),
        "inflation_cpi": inflation_cpi(country, start, end, save_dir),
        "gross_fixed_capital_formation": gross_fixed_capital_formation(country, start, end, save_dir),
        "gross_domestic_savings": gross_domestic_savings(country, start, end, save_dir),
        "gini_index": gini_index(country, start, end, save_dir),
        "labor_productivity": labor_productivity(country, start, end, save_dir),
    }
    return data


def gather_all_for_countries(
    countries: Iterable[str] = MAJOR_COUNTRIES,
    start: int = 2015,
    end: Optional[int] = None,
    save_dir: Optional[str] = SAVE_DIR,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Collect all macro series for multiple ``countries``.

    Returns a nested dictionary ``{country: {dataset: DataFrame}}``.
    """
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for c in countries:
        result[c] = gather_all(c, start, end, save_dir)
    return result
