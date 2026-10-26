"""Download the Open Power System Data (OPSD) hourly file and extract electricity load per country.

The source file (124 MB) is downloaded once into `data/`, which git ignores: the data
comes from ENTSO-E via OPSD and is not redistributed in this repo.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import holidays
import pandas as pd

URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
RAW = Path("data/time_series_60min_singleindex.csv")

# Country code -> (column prefix in the OPSD file, time zone used for holidays, name)
COUNTRIES = {
    "DE": ("DE", "Europe/Berlin", "Germany"),
    "FR": ("FR", "Europe/Paris", "France"),
    "IT": ("IT", "Europe/Rome", "Italy"),
    "ES": ("ES", "Europe/Madrid", "Spain"),
    "GB": ("GB_GBN", "Europe/London", "Great Britain"),
}
DEFAULT_COUNTRIES = list(COUNTRIES)

# Start of the first national COVID-19 lockdown in spring 2020 (stay-at-home orders; in
# Germany the nationwide contact ban). Dates are when the measures were announced or took
# effect, give or take a day. Only used to annotate charts; the model never sees them.
LOCKDOWNS = {"IT": "2020-03-10", "ES": "2020-03-15", "FR": "2020-03-17", "DE": "2020-03-22", "GB": "2020-03-23"}


@dataclass
class PowerData:
    """Hourly frames indexed by UTC time, one column per country, in megawatts (MW).

    `load` is the actual electricity demand. `tso_forecast` is the day-ahead forecast
    the grid operators (TSOs) published, kept as a professional baseline to compare
    against.
    """

    load: pd.DataFrame
    tso_forecast: pd.DataFrame


def download(path: Path = RAW) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} (124 MB, once) ...")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.rename(path)
    return path


def load_power_data(countries: list[str] = DEFAULT_COUNTRIES) -> PowerData:
    columns = {}
    for code in countries:
        prefix = COUNTRIES[code][0]
        columns[f"{prefix}_load_actual_entsoe_transparency"] = ("load", code)
        columns[f"{prefix}_load_forecast_entsoe_transparency"] = ("tso_forecast", code)

    raw = pd.read_csv(download(), usecols=["utc_timestamp", *columns], index_col="utc_timestamp",
                      parse_dates=["utc_timestamp"])
    raw.index.name = "time"
    raw = raw.loc["2015-01-01":]  # the first row is the last hour of 2014, with almost no data
    # The file is already one row per hour; a few hours are missing (NaN), which Timestral handles.
    frames = raw.rename(columns=columns)
    frames.columns = pd.MultiIndex.from_tuples(frames.columns)
    return PowerData(frames["load"][countries], frames["tso_forecast"][countries])


def holiday_indicator(index: pd.DatetimeIndex, countries: list[str]) -> pd.DataFrame:
    """1 for hours that fall on a national public holiday in that country (local time), else 0.

    Holidays are known years ahead, so this is a known-future covariate: it can be
    computed for the forecast horizon too.
    """
    flags = {}
    for code in countries:
        _, tz, _ = COUNTRIES[code]
        local_days = index.tz_convert(tz).date
        calendar = holidays.country_holidays(code, years=range(index.min().year, index.max().year + 1))
        flags[code] = [day in calendar for day in local_days]
    return pd.DataFrame(flags, index=index).astype(float)


def country_name(code: str) -> str:
    return COUNTRIES[code][2]
