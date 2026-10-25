"""Turn the raw `day,path,visitors` export into a complete daily time series per page."""

from pathlib import Path

import pandas as pd

ALL_PAGES = "(all pages)"


def load_daily_visits(csv_path: str | Path, top_pages: int = 4) -> pd.DataFrame:
    """Return a frame indexed by every calendar day, one column per series.

    The export only has rows for days on which a page was visited, so days
    without a row are filled with 0. Days before a page's first visit are left
    as NaN instead: the page probably did not exist yet, and Timestral treats NaN
    as missing rather than as "no visits". Besides the site-wide total, the
    `top_pages` most visited paths each get their own column.
    """
    raw = pd.read_csv(csv_path, parse_dates=["day"])
    wide = raw.pivot_table(index="day", columns="path", values="visitors", aggfunc="sum")

    days = pd.date_range(wide.index.min(), wide.index.max(), freq="D", name="day")
    wide = wide.reindex(days).fillna(0)

    top = wide.sum().sort_values(ascending=False).index[:top_pages]
    series = wide[top].copy()
    series.insert(0, ALL_PAGES, wide.sum(axis=1))
    return series.where(series.cumsum() > 0)
