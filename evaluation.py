"""Scores for the forecasts, and a test of whether the difference between two models is real.

- `score_forecasts`: MAPE, WQL, coverage and width per country, over the usual days.
- `weekly_wql`: the WQL of each week-ahead block, per country.
- `diebold_mariano`: whether the difference in loss between two models over the same weeks
  is larger than chance would give.
- `compare_pairs`: the test and the win counts for every pair of models.

See docs/06-comparing-models.md for what each score means, with examples.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from forecasting import DECILES, QuantileForecast


def mape(predicted: pd.DataFrame, actual: pd.DataFrame) -> pd.Series:
    """Mean absolute percentage error per country: on average, how many % off the forecast was."""
    actual = actual.loc[predicted.index]
    return ((predicted - actual).abs() / actual).mean() * 100


def pinball(actual: pd.DataFrame, quantile: pd.DataFrame, level: float) -> pd.DataFrame:
    """Pinball loss of one quantile for every hour: level × miss above, (1 − level) × miss below."""
    miss = actual - quantile
    return np.maximum(level * miss, (level - 1) * miss)


def score_forecasts(load: pd.DataFrame, fc: QuantileForecast, until: str) -> pd.DataFrame:
    """Forecast quality per country, on the hours up to `until` (the usual days).

    - MAPE: how far the median is from the actual demand, in %. Lower is better.
    - WQL: weighted quantile loss over the 9 deciles, in %. It scores the whole forecast
      distribution, not only the median. Lower is better.
    - in 80%: % of hours between the 10% and the 90% quantile. Best near 80.
    - 80% width: mean width of that range, as % of the actual demand. Narrower is sharper.
    """
    hours = fc.median.loc[:until].index
    actual = load.loc[hours]
    q = {level: frame.loc[hours] for level, frame in fc.levels.items()}
    # Pinball loss of each decile, summed over the deciles and the hours, divided by the total demand.
    loss = sum(pinball(actual, q[level], level) for level in DECILES)
    return pd.DataFrame({
        "MAPE %": mape(q[0.5], load),
        "WQL %": 2 * loss.sum() / (len(DECILES) * actual.sum()) * 100,
        "in 80% range %": ((actual >= q[0.1]) & (actual <= q[0.9])).mean() * 100,
        "80% width %": ((q[0.9] - q[0.1]) / actual).mean() * 100,
    })


def weekly_wql(load: pd.DataFrame, fc: QuantileForecast, until: str, horizon: int) -> pd.DataFrame:
    """WQL (%) of each forecast block, one row per block and one column per country.

    Each block is one week-ahead forecast. These are the paired observations of the test:
    every model forecast exactly the same blocks.
    """
    hours = fc.median.loc[:until].index
    actual = load.loc[hours]
    loss = sum(pinball(actual, fc.levels[level].loc[hours], level) for level in DECILES)
    block = np.arange(len(hours)) // horizon
    starts = hours[::horizon]
    weekly_loss = loss.groupby(block).sum()
    weekly_actual = actual.groupby(block).sum()
    wql = 2 * weekly_loss / (len(DECILES) * weekly_actual) * 100
    wql.index = starts[:len(wql)]
    return wql.rename_axis("week")


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> tuple[float, float]:
    """Diebold–Mariano test that two forecasts have the same expected loss.

    `loss_a` and `loss_b` are the losses of the two models on the same periods, in the same
    order. The test looks at the differences d = loss_a − loss_b. If the models are equally
    good, the mean of d is zero. The statistic is the mean of d divided by its standard
    error. `horizon` is the forecast horizon in periods: here each period is one block that
    is forecast one block ahead, so it is 1, and the differences of consecutive periods
    are taken as uncorrelated. With the small-sample correction of Harvey, Leybourne and
    Newbold (1997), the statistic follows a t distribution with n − 1 degrees of freedom.

    Returns the statistic (negative: model A has the lower loss) and the two-sided p-value.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    mean = d.mean()
    # Long-run variance of the mean: autocovariances up to lag horizon − 1 (none for horizon 1).
    centred = d - mean
    variance = centred @ centred / n
    for lag in range(1, horizon):
        variance += 2 * (centred[lag:] @ centred[:-lag]) / n
    if variance <= 0:
        return 0.0, 1.0
    dm = mean / np.sqrt(variance / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = dm * correction
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p_value)


def compare_pairs(weekly: dict[str, pd.DataFrame], alpha: float = 0.05) -> pd.DataFrame:
    """Every pair of models: mean WQL, win counts and the Diebold–Mariano test.

    The test uses the weekly WQL averaged over the countries (one value per week), so the
    weeks are the independent observations. The win counts also show the single
    (country, week) pairs. The p-values are also adjusted for the number of pairs with the
    Holm method, because testing many pairs makes a small p-value more likely by chance.
    """
    labels = list(weekly)
    pooled = {label: frame.mean(axis=1) for label, frame in weekly.items()}
    rows = []
    for a, b in combinations(labels, 2):
        statistic, p_value = diebold_mariano(pooled[a].to_numpy(), pooled[b].to_numpy())
        cells_a, cells_b = weekly[a].to_numpy().ravel(), weekly[b].to_numpy().ravel()
        rows.append({
            "model A": a, "model B": b,
            "WQL A %": pooled[a].mean(), "WQL B %": pooled[b].mean(),
            "difference A − B": pooled[a].mean() - pooled[b].mean(),
            "weeks A better": f"{int((pooled[a] < pooled[b]).sum())} of {len(pooled[a])}",
            "country-weeks A better": f"{int((cells_a < cells_b).sum())} of {len(cells_a)}",
            "DM statistic": statistic, "p-value": p_value,
        })
    table = pd.DataFrame(rows)
    table["p-value (Holm)"] = _holm(table["p-value"].to_numpy())

    def verdict(row) -> str:
        if row["p-value (Holm)"] >= alpha:
            return "no clear difference"
        return f"{row['model A'] if row['difference A − B'] < 0 else row['model B']} is better"

    table["verdict (5%, Holm)"] = table.apply(verdict, axis=1)
    return table


def _holm(p_values: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni adjustment: the smallest p-value × m, the next × (m − 1), and so on."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted
