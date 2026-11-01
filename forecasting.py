"""Thin wrapper around two time series foundation models that speaks pandas.

Both models work here in the same way. Each one gets a batch of history rows, shaped
`(rows, time)`, and an optional known-future covariate for each row (the public
holidays). Each one gives back the same 21 quantile levels for each forecast step. Here
every column of an hourly frame is one series (one country), and results come back as
frames indexed by time so the rest of the demo never touches raw arrays.

- Timestral (Mistral AI): `mistralai/Timestral-209M` or `mistralai/Timestral-28M`.
- Chronos-2 (Amazon): `amazon/chronos-2`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Short names for the command line. Any other Hugging Face id also works: an id that
# contains "chronos" loads Chronos-2, all others load Timestral.
MODELS = {
    "timestral": "mistralai/Timestral-209M",
    "timestral-28m": "mistralai/Timestral-28M",
    "chronos-2": "amazon/chronos-2",
}
DEFAULT_MODEL = "timestral"
STEP = pd.Timedelta(hours=1)
# Keep every Timestral forecast within one "typical swing" of the range the history has
# seen. The library default (10,000) is so loose that, on rare days such as Christmas with
# the holiday covariate, the 99% quantile reached millions of MW. Medians are unaffected.
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


@dataclass
class RawForecast:
    """What every model gives back: `quantiles` (rows, horizon, levels) and `mean` (rows, horizon)."""

    levels: np.ndarray
    quantiles: np.ndarray
    mean: np.ndarray

    def quantile(self, level: float) -> np.ndarray:
        match = np.flatnonzero(np.isclose(self.levels, level))
        if not len(match):
            raise ValueError(f"quantile {level} is not one of the model's levels {self.levels.tolist()}")
        return self.quantiles[..., match[0]]


class Model:
    """One interface for both models. `label` is the name in tables, charts and file names."""

    label: str
    model_id: str

    def predict(self, context: np.ndarray, horizon: int, known: np.ndarray | None) -> RawForecast:
        """`context` is (rows, time). `known`, if given, is (rows, time + horizon)."""
        raise NotImplementedError


class TimestralModel(Model):
    def __init__(self, model_id: str, device: str):
        from timestral import TimestralForecaster

        self.label, self.model_id = "Timestral", model_id
        # Downloads config.json + model.safetensors from Hugging Face once, then reads the local cache.
        self.forecaster = TimestralForecaster.from_pretrained(model_id, device=device,
                                                              quantile_real_cap_k=QUANTILE_CAP)

    def predict(self, context, horizon, known):
        # Timestral wants (rows, variates, time); the 0/1 holiday rows are recognised as an indicator.
        result = self.forecaster.forecast(
            context,
            prediction_length=horizon,
            known_covariates=None if known is None else known[:, None, :],
        )
        levels = np.asarray(result.quantile_levels, dtype=float)
        quantiles = result.quantiles[:, 0].cpu().numpy()  # (rows, horizon, 21)
        return RawForecast(levels, quantiles, _mean(levels, quantiles))


class Chronos2Model(Model):
    def __init__(self, model_id: str, device: str):
        from chronos import Chronos2Pipeline

        self.label, self.model_id = "Chronos-2", model_id
        self.pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
        self.levels = np.asarray(self.pipeline.quantiles, dtype=float)

    def predict(self, context, horizon, known):
        # Chronos-2 wants one dict for each row. A known-future covariate is given twice: its
        # past values (aligned with the history) and its future values (for the horizon).
        width = context.shape[1]
        inputs = []
        for i, row in enumerate(context):
            item = {"target": row}
            if known is not None:
                item["past_covariates"] = {"holiday": known[i, :width]}
                item["future_covariates"] = {"holiday": known[i, width:width + horizon]}
            inputs.append(item)
        quantiles, mean = self.pipeline.predict_quantiles(inputs, prediction_length=horizon,
                                                          quantile_levels=self.levels.tolist())
        # Each item is (variates, horizon, levels) and (variates, horizon); we have one variate.
        quantiles = np.stack([q[0].float().cpu().numpy() for q in quantiles])
        mean = np.stack([m[0].float().cpu().numpy() for m in mean])
        return RawForecast(self.levels, quantiles, mean)


def load_forecaster(name: str = DEFAULT_MODEL, device: str = "cpu") -> Model:
    model_id = MODELS.get(name, name)
    return (Chronos2Model if "chronos" in model_id.lower() else TimestralModel)(model_id, device)


def future_index(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(history.index[-1] + STEP, periods=horizon, freq=STEP, name=history.index.name)


def forecast(
    model: Model,
    history: pd.DataFrame,
    horizon: int,
    known: pd.DataFrame | None = None,
    low: float = 0.1,
    high: float = 0.9,
) -> QuantileForecast:
    """Forecast `horizon` hours past the end of `history`, for every column at once.

    `known` is an optional known-future covariate, one column per series, covering the
    history and the horizon. Timestral reads at most the last 32,768 hours of history,
    Chronos-2 at most the last 8,192.
    """
    future = future_index(history, horizon)
    covariate = None
    if known is not None:
        covariate = known.loc[history.index[0]:future[-1], history.columns].to_numpy(copy=True).T
    # copy=True: pandas returns read-only arrays, which PyTorch warns about.
    result = model.predict(history.to_numpy(copy=True).T, horizon, covariate)

    def frame(values: np.ndarray) -> pd.DataFrame:
        # (series, horizon) -> (horizon, series)
        return pd.DataFrame(values.T, index=future, columns=history.columns)

    quantiles = [result.quantile(q) for q in (low, 0.5, high)]
    return QuantileForecast(*map(frame, quantiles), frame(result.mean), low, high)


def rolling(
    model: Model,
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
        result = model.predict(contexts[batch], gap + horizon, None if covariate is None else covariate[batch])
        # Keep the block itself, not the `gap` hours between the forecast and the block.
        picked = [result.quantile(q)[:, gap:] for q in (low, 0.5, high)] + [result.mean[:, gap:]]
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


def _mean(levels: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Approximate the mean of each forecast step from its quantiles (Timestral gives no mean).

    The mean is the area under the quantile function. Between two known levels it is
    linear, and in the tails below 1% and above 99% it is flat.
    """
    levels = np.array([0.0, *levels, 1.0])
    q = np.concatenate([quantiles[..., :1], quantiles, quantiles[..., -1:]], axis=-1)
    return (np.diff(levels) * (q[..., 1:] + q[..., :-1]) / 2).sum(axis=-1)
