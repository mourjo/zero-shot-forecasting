# The `timestral` Python library

A reference for the `timestral` package: how to install it, every argument you can
change, and what each one does. It is based on version **1.0.0** (from
[GitHub](https://github.com/mistralai/timestral)), read from the installed source code.
The timings and effects below were measured on this project's data, on a laptop CPU.

For what the model can do in general, see [02-capabilities.md](02-capabilities.md). For the
statistical terms, see [01-concepts.md](01-concepts.md).

- [Install](#install)
- [The whole API in 10 lines](#the-whole-api-in-10-lines)
- [Loading a model: `from_pretrained`](#loading-a-model-from_pretrained)
- [Forecasting: `forecast`](#forecasting-forecast)
- [Input shapes](#input-shapes)
- [Covariates in practice](#covariates-in-practice)
- [The result: `Forecast`](#the-result-forecast)
- [Behaviour switches: `symmetric_averaging` and `clip_nonnegative`](#behaviour-switches-symmetric_averaging-and-clip_nonnegative)
- [Advanced constructor options](#advanced-constructor-options)
- [The checkpoint config: `TimestralConfig`](#the-checkpoint-config-timestralconfig)
- [Performance tips](#performance-tips)
- [Errors you may hit](#errors-you-may-hit)
- [How this project uses it](#how-this-project-uses-it)

## Install

There is no PyPI package; install from GitHub (Python 3.10+, PyTorch 2.4+):

```sh
pip install git+https://github.com/mistralai/timestral.git
# or, as in this project:
uv add "timestral @ git+https://github.com/mistralai/timestral.git"
```

Dependencies: `torch`, `numpy`, `safetensors`, `huggingface_hub`. The package is
inference only; there is no training code.

## The whole API in 10 lines

```python
import numpy as np
from timestral import TimestralForecaster

forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")

history = np.array([3, 5, 2, 0, 4, 7, 1, 3, 6, 2], dtype=float)  # one series, NaN = missing
result = forecaster.forecast(history, prediction_length=7)

result.median            # (1, 1, 7)      best guess per day
result.quantile(0.9)     # (1, 1, 7)      90% quantile per day
result.quantiles         # (1, 1, 7, 21)  all 21 quantiles
```

The package exports five names: `TimestralForecaster` (what you use), `Forecast` (what
it returns), and three lower-level ones you rarely need: `TimestralConfig`,
`TimestralModel` and `load_model`.


```mermaid
flowchart LR
    A["TimestralForecaster.from_pretrained(id)"] --> F["forecaster"]
    C["context (B, V, T)<br/>+ optional covariates"] --> FC["forecaster.forecast(...)"]
    F --> FC
    FC --> R["Forecast"]
    R --> Q["quantiles (B, V, H, 21)"]
    R --> M["median (B, V, H)"]
    R --> QL["quantile(level) (B, V, H)"]
```

## Loading a model: `from_pretrained`

```python
TimestralForecaster.from_pretrained(
    path_or_repo_id,                 # Hugging Face id or local directory
    device="cpu",
    max_context=None,
    per_shot_horizon_patches=32,
    quantile_real_cap_k=1e4,
    revision=None,
    cache_dir=None,
    token=None,
    local_files_only=False,
)
```

| Argument | Default | What it does | When to change it |
|---|---|---|---|
| `path_or_repo_id` | (required) | A Hugging Face repo id such as `"mistralai/Timestral-28M"`, or a local directory containing `config.json` and `model.safetensors` | Pick the size; use a directory for offline or air-gapped setups |
| `device` | `"cpu"` | Where the model runs: `"cpu"`, `"cuda"`, `"cuda:1"`, or any other PyTorch device | You have a GPU and large batches |
| `max_context` | `None` (= 32,768) | The most past steps the model reads; older steps are dropped | To cap the cost of very long histories, or to test whether old data helps |
| `per_shot_horizon_patches` | `32` | Patches (of 32 steps) predicted per pass, capped at 32. Longer horizons take several passes | Rarely. Lower values mean more passes, each feeding its median back |
| `quantile_real_cap_k` | `1e4` | Clamps forecasts to `[min - k*s, max + k*s]` of the history, where `s` is the scale of the latest values. `0` turns it off | When extreme quantiles blow up. This project uses `1`; see [below](#capping-extreme-quantiles-quantile_real_cap_k) |
| `revision` | `None` | A Hub branch, tag or commit to pin | Pin a version for reproducibility |
| `cache_dir` | `None` | Where downloads are stored (default: the Hugging Face cache, `~/.cache/huggingface/hub`) | Shared or read-only machines |
| `token` | `None` | Hugging Face token | Only for gated or private repos |
| `local_files_only` | `False` | Never use the network; load from the cache only | Offline runs. The env var `HF_HUB_OFFLINE=1` does the same |

The first load downloads two files once: `config.json` and `model.safetensors`.


```mermaid
flowchart TD
    A["from_pretrained(path_or_repo_id)"] --> B{"Is it a local directory?"}
    B -- "yes" --> L["Load config.json and<br/>model.safetensors from it"]
    B -- "no" --> D{"local_files_only=True or<br/>HF_HUB_OFFLINE=1?"}
    D -- "yes" --> CO["Load from the Hugging Face cache only<br/>error if it was never downloaded"]
    D -- "no" --> G["Fetch from the Hub<br/>files already cached are reused"]
    CO --> LM["Build the model on device, in float32"]
    G --> LM
    L --> LM
```

| Model | Download | 150 day-ahead forecasts, with holidays | without holidays |
|---|---|---|---|
| `mistralai/Timestral-5M` | 21 MB | not measured | not measured |
| `mistralai/Timestral-28M` | 111 MB | 8.3 s | 2.2 s |
| `mistralai/Timestral-123M` | 490 MB | not measured | not measured |
| `mistralai/Timestral-209M` | 837 MB | 49.0 s | 12.0 s |

Measured on a laptop CPU: 5 countries x 30 days of September 2020, each forecast from
2,048 hours of history, 24-hour horizon, best of two runs. The average error was
1.86% with 28M and 1.81% with 209M.

**Precision.** Models load in float32, the precision Mistral validated. The lower-level
`timestral.load_model(path, device, dtype)` accepts another `dtype`, but anything other
than float32 is untested territory.

## Forecasting: `forecast`

```python
forecaster.forecast(
    context,                              # (B, V, T), (B, T) or (T,)
    prediction_length,                    # horizon, in steps
    observation_mask=None,
    past_covariates=None,                 # (B, C, T)
    known_covariates=None,                # (B, C, T + prediction_length)
    categorical_past_covariates=None,     # (B, C, T)
    categorical_known_covariates=None,    # (B, C, T + prediction_length)
    symmetric_averaging=True,
    clip_nonnegative=True,
) -> Forecast
```

`B` = number of series, `V` = variates per series, `T` = context length, `C` = number
of covariates.

| Argument | What it is |
|---|---|
| `context` | The history. NumPy array or torch tensor. `NaN` marks a missing value |
| `prediction_length` | How many steps to forecast. Any positive integer; above 1,024 the model works block by block |
| `observation_mask` | Optional boolean array shaped like `context`. `False` marks extra values to treat as missing, without editing the data |
| `past_covariates` | Numeric covariates known only over the history |
| `known_covariates` | Numeric covariates known over the history **and** the horizon. A row of only 0s and 1s is treated as an indicator |
| `categorical_past_covariates` | Category codes known only over the history |
| `categorical_known_covariates` | Category codes known over the history and the horizon |
| `symmetric_averaging` | See [behaviour switches](#behaviour-switches-symmetric_averaging-and-clip_nonnegative) |
| `clip_nonnegative` | See [behaviour switches](#behaviour-switches-symmetric_averaging-and-clip_nonnegative) |

## Input shapes

`context` accepts three shapes:

| Shape | Meaning | Example |
|---|---|---|
| `(T,)` | one series | Germany's hourly load |
| `(B, T)` | `B` independent series | 5 countries, one row each, as in this project |
| `(B, V, T)` | `B` series, each with `V` variates forecast jointly | 10 pumps x (vibration, temperature, pressure) |


```mermaid
flowchart LR
    A["(T,)<br/>one series"] -- "read as (1, 1, T)" --> N["(B, V, T)"]
    B["(B, T)<br/>B series"] -- "read as (B, 1, T)" --> N
    C["(B, V, T)"] --> N
    N --> O["quantiles<br/>(B, V, H, 21)"]
```

Rules worth knowing:

- **All rows share one length `T`.** For histories of different lengths, pad the start
  with `NaN`. (`rolling()` in `forecasting.py` avoids the issue by giving every row the
  same 2,048-hour window.)
- **The time steps must be regular**: one per day, per hour, etc. Timestral has no
  notion of dates; fill missing slots with `NaN` (unknown) or 0 (truly zero),
  whichever is true.
- **Missing values**: `NaN` anywhere in the history is fine. But a series with *no*
  observed value at all does **not** raise an error: it silently comes back as a
  forecast of 0. Filter out empty series yourself.
- **Covariates may drop the `C` axis**: a `(B, L)` covariate is read as a single
  covariate, `(B, 1, L)`.

## Covariates in practice

```python
# Public holidays, known for the future too, as a 0/1 indicator (what this project does;
# see holiday_indicator() in data.py). One row per country, covering history + horizon.
holiday = holiday_flags[None, None, :]                      # (1, 1, T + H), values 0 or 1
result = forecaster.forecast(context, H, known_covariates=holiday)

# Day of week as a categorical covariate.
weekday = (first_weekday + np.arange(T + H)) % 7            # codes 0..6
result = forecaster.forecast(
    context, H,
    categorical_known_covariates=weekday[None, None, :].astype(float),  # (1, 1, T + H)
)

# Several covariates: stack them along the C axis.
known = np.concatenate([holiday, temperature_forecast], axis=1)     # (B, 2, T + H)
```

What to know:

- **Length must match exactly**: past covariates cover `T` steps, known ones
  `T + prediction_length`. A mismatch raises an `AssertionError` stating the expected
  shape.
- **Covariates are never forecast.** They only inform the target forecast;
  `result.quantiles` contains targets only.
- **Categorical codes can be any numbers.** Use `pandas.factorize` or a fixed mapping.
  `NaN` means a missing category. For univariate targets, the library *target-encodes*
  the codes: each category becomes the average target value seen with it in that
  series' history (helper: `timestral.covariate_encoding.target_encode`). A category
  that never appears in the history gets the series average. For multivariate targets
  there is no single target to average, so the raw codes are passed through.
- **Check support**: `forecaster.supports_covariates` and
  `forecaster.supports_known_covariates` say whether a checkpoint accepts covariates.
  Both are `True` for the 28M and 209M checkpoints.
- **Cost**: any covariate makes a forecast about **4x slower** (12.0 s → 49.0 s for the
  209M batch above; numeric and categorical alike), because the model then runs its
  full target-plus-covariate attention. Only add covariates that help; backtest with
  and without. The holiday covariate here is worth it: it makes no difference in
  September, but cuts the day-ahead error over Christmas from 4.25% to 3.30%.

## The result: `Forecast`

`forecast()` returns a `Forecast`:

| Member | Shape | What it is |
|---|---|---|
| `quantiles` | `(B, V, H, 21)` | All quantiles, sorted along the last axis |
| `quantile_levels` | tuple of 21 | `(0.01, 0.05, 0.1, 0.15, ..., 0.9, 0.95, 0.99)` |
| `median` | `(B, V, H)` | The 50% quantile |
| `quantile(level)` | `(B, V, H)` | One quantile; `level` must be one of `quantile_levels` exactly, or it raises `ValueError` |

The values are **torch tensors**, even if you passed NumPy, and live on the model's
`device`. Convert with `.numpy()` on CPU, or `.cpu().numpy()` on a GPU. `V` is always
present: a `(B, T)` input comes back as `(B, 1, H)`, hence the `[:, 0]` in
`forecasting.py`.

There is no `mean` member. If you need totals over a horizon, estimate the mean from
the quantiles; `_mean()` in `forecasting.py` shows one way.

## Behaviour switches: `symmetric_averaging` and `clip_nonnegative`

Two options on `forecast()`, both on by default. Mistral calls the pair *clampsym*.


```mermaid
flowchart TD
    A["context + covariates"] --> E["Target-encode categorical covariates"]
    E --> P1["Model pass on the series"]
    E -- "symmetric_averaging=True" --> P2["Model pass on the negated series<br/>covariates negated too"]
    P2 --> FL["Negate back and<br/>mirror the quantiles"]
    P1 --> AV["Average the two passes"]
    FL --> AV
    AV --> G{"clip_nonnegative=True and<br/>history all >= 0?"}
    G -- "yes" --> CL["Clamp the forecast at 0"]
    G -- "no" --> R["Forecast"]
    CL --> R
```

### `symmetric_averaging=True`

The model forecasts the series **and its upside-down copy** (every value negated, and
every covariate too), flips the second forecast back, and averages the two. This
cancels any tendency of the model to lean upwards or downwards, and usually makes
forecasts a little more stable.

- **Cost**: two model passes instead of one, so turning it off **halves the time**.
- **Effect here**: negligible. On the 150 day-ahead forecasts above (209M, with
  holidays), the average error went from 1.81% to 1.79% with it *off*, and 80%
  coverage from 86% to 84%. Time went from 49.0 s to 26.4 s.

Turn it off when speed matters more than the last bit of accuracy, such as large
batches or the anomaly replay. Backtest first.

### `clip_nonnegative=True`

If every observed value in a series' history is 0 or more, its forecast is clamped
at 0. Series with any negative history value are left alone.

- **Effect here**: none. Electricity load never gets anywhere near 0, so there is
  nothing to clip. It matters for counts close to 0: an earlier version of this project
  forecast daily website visits (0-5 a day), and turning it off let the 1% quantile
  drop to -0.25 visits, which is impossible.
- Turn it off only for data that *can* go negative but happened not to in the
  history: temperatures in summer, profit and loss, deviations from a baseline.

### Capping extreme quantiles: `quantile_real_cap_k`

Not a `forecast()` switch but a `from_pretrained()` one, and the one setting this
project changes. Forecasts are clamped to `[min - k*s, max + k*s]` of the history,
where `s` is the scale of the latest values. With the default `k = 10,000` the clamp
almost never applies.

Why it matters here: with the holiday covariate, the week-ahead forecast for
Christmas Day in Great Britain had a **99% quantile of 37.8 million MW**, for a country
that peaks around 45,000 MW. The median stayed sensible (47,030 MW), but the mean,
which is computed from all quantiles, jumped to 1.2 million MW, and the anomaly band
became useless for that week. The same week without the covariate looked normal.
Rare indicator values seem to make the outer quantiles unstable.

| `quantile_real_cap_k` | Great Britain's highest 99% quantile, Christmas week | Medians |
|---|---|---|
| `1e4` (default) | 37,784,240 MW | unchanged |
| `10` | 125,787 MW | unchanged |
| `3` | 73,023 MW | unchanged |
| `1` (this project) | 57,948 MW | unchanged |

Spain and Italy were unaffected at every value. A tight cap is safe when the series
stays within its historical range, as electricity demand does. Be careful with
strongly trending series, where future values legitimately leave the historical range.

## Advanced constructor options

`TimestralForecaster(model, ...)` takes the same `max_context`,
`per_shot_horizon_patches` and `quantile_real_cap_k`, plus one more:

| Argument | Default | What it does |
|---|---|---|
| `extreme_value_threshold` | `1e30` | For horizons over 1,024 steps, a fed-back median at or above this size is replaced by 0 |

It is only reachable by building the forecaster yourself:

```python
from timestral import TimestralForecaster, load_model

model = load_model("mistralai/Timestral-209M", device="cpu")
forecaster = TimestralForecaster(model, max_context=4096, extreme_value_threshold=1e12)
```

## The checkpoint config: `TimestralConfig`

Each checkpoint ships a `config.json` describing its architecture. You can read it as
`forecaster.config`. These values are fixed by the trained weights, and **changing
them breaks loading** (weights load with `strict=True`). They are useful to read,
not to tweak.

| Field | 28M | 209M | Meaning |
|---|---|---|---|
| `patch_size` | 32 | 32 | Time steps per patch; histories are left-padded to a multiple of it |
| `block_size` | 32 | 32 | Max patches predicted per pass, so 32 x 32 = 1,024 steps |
| `context_length` | 32,768 | 32,768 | Trained context; the default for `max_context` |
| `quantile_knots` | 21 levels | 21 levels | The quantile levels returned |
| `d_model` | 384 | 768 | Width of the network |
| `num_heads` | 6 | 12 | Attention heads per layer |
| `mlp_hidden_dim` | 1,024 | 2,048 | Width of each feed-forward layer |
| `block_schedule` | 12 layers | 24 layers | Layer types: `time` (attends across time) and `time_variate` (across time, then across variates/covariates), with every 4th layer a `time_variate` |
| `covariate_enabled` | true | true | Whether the checkpoint accepts covariates |

Useful derived properties: `forecaster.patch_size`, `forecaster.quantile_levels`,
`forecaster.config.num_quantiles`, `forecaster.supports_known_covariates`.

Fine-tuning, or a different architecture, is not possible with this package; Mistral
offers that through Mistral Forge.

## Performance tips

From the most to the least impact, measured on the 150 day-ahead forecasts above:

| Change | Speed-up | Accuracy cost here |
|---|---|---|
| Use 28M instead of 209M | ~6x (49.0 s → 8.3 s) | 1.81% → 1.86% |
| Skip covariates that don't help | ~4x (49.0 s → 12.0 s) | none in September; holidays matter around holidays |
| Turn off `symmetric_averaging` | ~2x (49.0 s → 26.4 s) | none measurable |
| All three | ~49x (49.0 s → 1.0 s) | 1.81% → 1.83% in September |

Also:

- **Batch.** One call with many rows beats many calls with one row. Watch memory;
  chunks of a few dozen to a few hundred rows work well on a laptop (`rolling()` uses
  64 rows of 2,048 hours).
- **Cap the context** with `max_context`, or pass less history: time grows with
  history length. With 8,192 hours instead of 2,048, day-ahead forecasts were ~5x
  slower and only slightly more accurate (1.71% vs 1.77% in September 2019).
- **Use a GPU** (`device="cuda"`) for large volumes.

Backtest after each change to check that accuracy holds where it matters to you.

## Errors you may hit

| Message (abridged) | Cause | Fix |
|---|---|---|
| `context must be (T,), (B, T) or (B, V, T)` | Wrong number of dimensions | Reshape the input |
| `known_covariates must be (B=..., C, L)` | Covariate length or batch doesn't match | Known: `T + prediction_length`; past: `T` |
| `level 0.97 is not one of (...)` | Asked for an untrained quantile | Use one of the 21 levels |
| `this checkpoint cannot condition on known-future covariates` | The checkpoint doesn't support it | Check `supports_known_covariates`; use a past covariate or another checkpoint |
| `forecast produced N NaN values; the input is degenerate` | Numerically broken input | Check for infinities or absurd magnitudes |
| `... is neither a local directory nor a Hugging Face Hub repo you can access` | A typo in the id, or no network with nothing cached | Check the id; download once while online |

## How this project uses it

| Where | What it calls |
|---|---|
| `forecasting.load_forecaster` | `from_pretrained(model_id, quantile_real_cap_k=1.0)`; `--model` picks the size |
| `forecasting._forecast` | `forecast(context, prediction_length, known_covariates=holidays)`; symmetric averaging and clipping left on |
| `forecasting.forecast` | A `(5, T)` batch, one row per country, with up to 32,768 hours of history and a 720-hour horizon; keeps the 10%, 50% and 90% quantiles, plus the mean from all 21 |
| `forecasting.rolling` | One row per (block, country), each with the same 2,048-hour window, in chunks of 64. Day-ahead: 24-hour blocks, made 16 hours early in the backtest (`gap=16`); week-ahead: 168-hour blocks. Keeps the 1%, 50% and 99% quantiles (10% / 90% in the backtest) |
