"""Thin wrapper around Timestral that speaks pandas.

Timestral works on arrays shaped `(series, time)` and returns 21 quantiles per
forecast step. Here every column of a daily frame is one series, and results come
back as frames indexed by date so the rest of the demo never touches raw arrays.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from timestral import TimestralForecaster

DEFAULT_MODEL = "mistralai/Timestral-209M"


@dataclass
class QuantileForecast:
    """Forecast for every series, one frame each: a lower/upper quantile, the median and the mean.

    The median is the best single guess for one day, but on a sparse page most days
    have 0 visits, so its median is 0 too. To answer "how many visits in total",
    add up the means instead.
    """

    low: pd.DataFrame
    median: pd.DataFrame
    high: pd.DataFrame
    mean: pd.DataFrame
    low_level: float
    high_level: float


def load_forecaster(model_id: str = DEFAULT_MODEL) -> TimestralForecaster:
    # Downloads config.json + model.safetensors from Hugging Face once, then reads the local cache.
    return TimestralForecaster.from_pretrained(model_id)


def forecast(
    forecaster: TimestralForecaster,
    history: pd.DataFrame,
    horizon: int,
    low: float = 0.1,
    high: float = 0.9,
) -> QuantileForecast:
    """Forecast `horizon` days past the end of `history`, for every column at once."""
    context = history.to_numpy().T  # (series, time)
    first_weekday = np.full(len(history.columns), history.index[0].dayofweek)
    result = _forecast(forecaster, context, first_weekday, horizon)

    future = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D", name="day")

    def frame(level: float) -> pd.DataFrame:
        # (series, 1 variate, horizon) -> (horizon, series)
        return pd.DataFrame(result.quantile(level)[:, 0].T.numpy(), index=future, columns=history.columns)

    mean = pd.DataFrame(_mean(result)[:, 0].T, index=future, columns=history.columns)
    return QuantileForecast(frame(low), frame(0.5), frame(high), mean, low, high)


def one_step_ahead(
    forecaster: TimestralForecaster,
    history: pd.DataFrame,
    min_context: int,
    low: float = 0.01,
    high: float = 0.99,
    batch_size: int = 256,
) -> QuantileForecast:
    """For every day after the first `min_context` days, forecast that day from the days before it.

    This is what a forecaster running every night would have predicted for the
    next day, so it can be compared with what actually happened without the model
    ever seeing the day it is judging. Each (day, series) pair becomes one row of
    the batch: the context is everything before that day, left-padded with NaN
    (which Timestral treats as missing) so all rows have the same length.
    """
    values = history.to_numpy().T  # (series, time)
    n_series, n_days = values.shape
    length = n_days - 1  # the longest context, used to judge the last day
    targets = range(min_context, n_days)

    contexts = np.full((len(targets), n_series, length), np.nan)
    first_weekday = np.empty((len(targets), n_series), dtype=int)
    for row, day in enumerate(targets):
        contexts[row, :, length - day:] = values[:, :day]
        # Weekday of the first (padded) position, so the calendar covariate lines up with the real dates.
        first_weekday[row] = (history.index[0].dayofweek + day - length) % 7
    contexts = contexts.reshape(-1, length)
    first_weekday = first_weekday.reshape(-1)

    quantiles = []
    for start in range(0, len(contexts), batch_size):
        batch = slice(start, start + batch_size)
        result = _forecast(forecaster, contexts[batch], first_weekday[batch], horizon=1)
        picked = [result.quantile(q)[:, 0, 0].numpy() for q in (low, 0.5, high)] + [_mean(result)[:, 0, 0]]
        quantiles.append(np.stack(picked, axis=-1))
    quantiles = np.concatenate(quantiles).reshape(len(targets), n_series, 4)

    days = history.index[min_context:]

    def frame(i: int) -> pd.DataFrame:
        return pd.DataFrame(quantiles[:, :, i], index=days, columns=history.columns)

    return QuantileForecast(frame(0), frame(1), frame(2), frame(3), low, high)


def _mean(result) -> np.ndarray:
    """Approximate the mean of each forecast step from its quantiles.

    The mean is the area under the quantile function. Between two known levels it is
    linear, and in the tails below 1% and above 99% it is flat.
    """
    levels = np.array([0.0, *result.quantile_levels, 1.0])
    q = result.quantiles.numpy()
    q = np.concatenate([q[..., :1], q, q[..., -1:]], axis=-1)
    return (np.diff(levels) * (q[..., 1:] + q[..., :-1]) / 2).sum(axis=-1)


def _forecast(forecaster: TimestralForecaster, context: np.ndarray, first_weekday: np.ndarray, horizon: int):
    """Run Timestral with the day of week as a known-future categorical covariate.

    The day of week is known for the forecast horizon too, which lets the model
    pick up weekday/weekend patterns. Codes are 0 (Monday) to 6 (Sunday).
    """
    steps = np.arange(context.shape[-1] + horizon)
    weekday = (first_weekday[:, None] + steps[None, :]) % 7
    return forecaster.forecast(
        context,
        prediction_length=horizon,
        categorical_known_covariates=weekday[:, None, :].astype(float),  # (series, 1 covariate, time + horizon)
    )
