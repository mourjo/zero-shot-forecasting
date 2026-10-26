"""Thin wrapper around Timestral that speaks pandas.

Timestral works on arrays shaped `(series, time)` and returns 21 quantiles per
forecast step. Here every column of an hourly frame is one series (one country), and
results come back as frames indexed by time so the rest of the demo never touches raw
arrays.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from timestral import TimestralForecaster

DEFAULT_MODEL = "mistralai/Timestral-209M"
STEP = pd.Timedelta(hours=1)
# Keep every forecast within one "typical swing" of the range the history has seen. The
# library default (10,000) is so loose that, on rare days such as Christmas with the holiday
# covariate, the 99% quantile reached millions of MW. Medians are unaffected.
QUANTILE_CAP = 1.0


@dataclass
class QuantileForecast:
    """Forecast for every series, one frame each: a lower/upper quantile, the median and the mean.

    The median is the best single guess for one hour. To add hours up into energy
    over a period, use the mean.
    """

    low: pd.DataFrame
    median: pd.DataFrame
    high: pd.DataFrame
    mean: pd.DataFrame
    low_level: float
    high_level: float


def load_forecaster(model_id: str = DEFAULT_MODEL) -> TimestralForecaster:
    # Downloads config.json + model.safetensors from Hugging Face once, then reads the local cache.
    return TimestralForecaster.from_pretrained(model_id, quantile_real_cap_k=QUANTILE_CAP)


def future_index(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(history.index[-1] + STEP, periods=horizon, freq=STEP, name=history.index.name)


def forecast(
    forecaster: TimestralForecaster,
    history: pd.DataFrame,
    horizon: int,
    known: pd.DataFrame | None = None,
    low: float = 0.1,
    high: float = 0.9,
) -> QuantileForecast:
    """Forecast `horizon` hours past the end of `history`, for every column at once.

    `known` is an optional known-future covariate, one column per series, covering the
    history and the horizon. Timestral reads at most the last 32,768 hours of history.
    """
    future = future_index(history, horizon)
    covariate = None
    if known is not None:
        covariate = known.loc[history.index[0]:future[-1], history.columns].to_numpy(copy=True).T
    # copy=True: pandas returns read-only arrays, which PyTorch warns about.
    result = _forecast(forecaster, history.to_numpy(copy=True).T, horizon, covariate)

    def frame(values: np.ndarray) -> pd.DataFrame:
        # (series, horizon) -> (horizon, series)
        return pd.DataFrame(values.T, index=future, columns=history.columns)

    quantiles = [result.quantile(q)[:, 0].numpy() for q in (low, 0.5, high)]
    return QuantileForecast(*map(frame, quantiles), frame(_mean(result)[:, 0]), low, high)


def rolling(
    forecaster: TimestralForecaster,
    history: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    horizon: int = 24,
    context: int = 2048,
    gap: int = 0,
    known: pd.DataFrame | None = None,
    low: float = 0.01,
    high: float = 0.99,
    batch_size: int = 64,
) -> QuantileForecast:
    """Forecast `start`..`end` in blocks of `horizon` hours, each block only from earlier hours.

    A last block that would run past `end` is left out.

    Each block is forecast from the `context` hours that end `gap` hours before the
    block starts. With the defaults this is what a forecaster running every midnight
    would have predicted for the next 24 hours; with `gap=16` the forecast is made at
    08:00 the day before, as early as the grid operators publish theirs. Either way the
    model never sees the hours it is judging. Each (block, series) pair becomes one row
    of the batch.
    """
    # Only whole blocks that end by `end`.
    origins = pd.date_range(_in_tz(start, history), _in_tz(end, history) - horizon * STEP, freq=horizon * STEP)
    positions = history.index.get_indexer(origins)
    if (positions < context + gap).any() or positions[-1] + horizon > len(history):
        raise ValueError(f"history must cover {context} hours before {origins[0]} and up to {origins[-1]}")

    values = history.to_numpy().T  # (series, time)
    covariate = None if known is None else known.loc[history.index, history.columns].to_numpy().T
    made = positions - gap  # where each forecast is made: its context ends here
    contexts = np.stack([values[:, m - context:m] for m in made])  # (blocks, series, context)
    contexts = contexts.reshape(-1, context)
    if covariate is not None:
        covariate = np.stack([covariate[:, m - context:p + horizon] for m, p in zip(made, positions)])
        covariate = covariate.reshape(-1, context + gap + horizon)

    quantiles = []
    for first in range(0, len(contexts), batch_size):
        batch = slice(first, first + batch_size)
        result = _forecast(forecaster, contexts[batch], gap + horizon, None if covariate is None else covariate[batch])
        # Keep the block itself, not the `gap` hours between the forecast and the block.
        picked = [result.quantile(q)[:, 0, gap:].numpy() for q in (low, 0.5, high)] + [_mean(result)[:, 0, gap:]]
        quantiles.append(np.stack(picked, axis=-1))  # (rows, horizon, 4)

    n_series = len(history.columns)
    # (blocks * series, horizon, 4) -> (blocks * horizon, series, 4): one row per forecast hour.
    stacked = np.concatenate(quantiles).reshape(len(origins), n_series, horizon, 4)
    stacked = stacked.transpose(0, 2, 1, 3).reshape(len(origins) * horizon, n_series, 4)
    hours = history.index[positions[0]:positions[0] + len(origins) * horizon]

    def frame(i: int) -> pd.DataFrame:
        return pd.DataFrame(stacked[:, :, i], index=hours, columns=history.columns)

    return QuantileForecast(frame(0), frame(1), frame(2), frame(3), low, high)


def _in_tz(when: str | pd.Timestamp, history: pd.DataFrame) -> pd.Timestamp:
    """A plain date like "2020-03-01" means that time in the history's time zone (UTC here)."""
    when = pd.Timestamp(when)
    return when.tz_localize(history.index.tz) if when.tz is None else when


def _mean(result) -> np.ndarray:
    """Approximate the mean of each forecast step from its quantiles.

    The mean is the area under the quantile function. Between two known levels it is
    linear, and in the tails below 1% and above 99% it is flat.
    """
    levels = np.array([0.0, *result.quantile_levels, 1.0])
    q = result.quantiles.numpy()
    q = np.concatenate([q[..., :1], q, q[..., -1:]], axis=-1)
    return (np.diff(levels) * (q[..., 1:] + q[..., :-1]) / 2).sum(axis=-1)


def _forecast(forecaster: TimestralForecaster, context: np.ndarray, horizon: int, known: np.ndarray | None):
    """Run Timestral, optionally with one known-future covariate per series.

    `known` is `(series, time + horizon)`. In this project it is the public-holiday
    indicator: 0/1 rows, which Timestral recognises as an indicator by itself.
    """
    return forecaster.forecast(
        context,
        prediction_length=horizon,
        known_covariates=None if known is None else known[:, None, :],  # (series, 1 covariate, time + horizon)
    )
