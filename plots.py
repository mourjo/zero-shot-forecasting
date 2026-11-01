"""Matplotlib charts: one small panel per country, sharing the time axis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import country_name
from forecasting import QuantileForecast

ACTUAL = "#2a78d6"
FORECAST = "#eb6834"
# One fixed colour per model, so that a model has the same colour in every chart.
MODEL_COLORS = {"Timestral": "#eb6834", "Chronos-2": "#6250d6", "TimesFM 3": "#1baf7a"}
SPARE_COLORS = ["#eda100", "#e87ba4"]


def model_color(label: str) -> str:
    if label not in MODEL_COLORS:
        MODEL_COLORS[label] = SPARE_COLORS[len(MODEL_COLORS) - 3]
    return MODEL_COLORS[label]
ANOMALY = "#e34948"
INK = "#52514e"
GRID = "#e4e3df"
USUAL_SHADE = "#efeee9"


def _panels(series: list[str], title: str, unit: str):
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.3 * len(series)), sharex=True, squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color="#0b0b0b")
    for ax, code in zip(axes[:, 0], series):
        ax.set_title(f"{country_name(code)} ({unit})", loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=8, length=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    return fig, axes[:, 0]


def _band(ax, fc: QuantileForecast, code: str, label: str, color: str = FORECAST):
    pct = round((fc.high_level - fc.low_level) * 100)
    ax.fill_between(fc.median.index, fc.low[code], fc.high[code], color=color, alpha=0.18, linewidth=0,
                    label=f"{label}: {pct}% interval")
    ax.plot(fc.median.index, fc.median[code], color=color, linewidth=1.2, label=f"{label}: median")


def _save(fig, axes, path: Path):
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, ncols=4, bbox_to_anchor=(0, 1.35))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_signal(load: pd.Series, threshold: float, average: pd.Series, spread: pd.Series, lockdown: str | None,
                path: Path):
    """Two panels of the same hourly load: against a fixed threshold, and against a moving-average band."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    fig.suptitle(f"{country_name(load.name)}: hourly load, and two simple ways to flag anomalies", x=0.01, ha="left",
                 fontsize=13, color="#0b0b0b")
    titles = [f"Fixed threshold: an anomaly is any hour below {threshold / 1000:,.0f}k MW",
              "Moving average: an anomaly is any hour outside the past week's average \u00b1 2 standard deviations"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=8, length=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        ax.plot(load.index, load, color=ACTUAL, linewidth=0.8, label="actual")
        if lockdown:
            ax.axvline(pd.Timestamp(lockdown, tz="UTC"), color=INK, linewidth=0.8, linestyle=(0, (4, 3)),
                       label="lockdown starts")

    below = load[load < threshold]
    axes[0].axhline(threshold, color=FORECAST, linewidth=1.2, label="threshold")
    axes[0].scatter(below.index, below, s=4, color=ANOMALY, zorder=3, label="flagged hour")
    axes[1].fill_between(average.index, average - 2 * spread, average + 2 * spread, color=FORECAST, alpha=0.18,
                         linewidth=0, label="average \u00b1 2 std")
    axes[1].plot(average.index, average, color=FORECAST, linewidth=1.2, label="7-day moving average")
    for ax in axes:
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncols=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_forecast(history: pd.DataFrame, fc: QuantileForecast, path: Path, label: str = "Timestral",
                  show_days: int = 28):
    recent = history.iloc[-show_days * 24:]
    fig, axes = _panels(list(history.columns),
                        f"Hourly load: last {show_days} days and the next {len(fc.median) // 24} days by {label}", "MW")
    for ax, code in zip(axes, history.columns):
        ax.plot(recent.index, recent[code], color=ACTUAL, linewidth=0.8, label="actual")
        _band(ax, fc, code, label, model_color(label))
    _save(fig, axes, path)


def plot_anomalies(actual: pd.DataFrame, expected: pd.DataFrame, days: pd.DataFrame, lockdowns: dict[str, str],
                   horizon: int, path: Path, label: str = "Timestral"):
    ahead = {24: "day-ahead", 168: "week-ahead"}.get(horizon, f"{horizon}-hours-ahead")
    fig, axes = _panels(list(actual.columns), f"Daily energy vs the {ahead} forecast by {label}; red = anomalous day",
                        "MWh")
    for ax, code in zip(axes, actual.columns):
        ax.plot(expected.index, expected[code], color=model_color(label), linewidth=1.2, label="expected (forecast)")
        ax.plot(actual.index, actual[code], color=ACTUAL, linewidth=1, label="actual")
        hits = days[days["country"] == code]
        ax.scatter(hits["day"], hits["actual MWh"], s=24, color=ANOMALY, edgecolor="white", linewidth=1, zorder=3,
                   label="anomalous day")
        if code in lockdowns:
            ax.axvline(pd.Timestamp(lockdowns[code], tz="UTC"), color=INK, linewidth=0.8, linestyle=(0, (4, 3)),
                       label="lockdown starts")
    _save(fig, axes, path)


def plot_compare_anomalies(flags: pd.DataFrame, lockdowns: dict[str, str], path: Path, normal_until: str,
                           window_days: int):
    """One panel per country: a row of marks for each model, one mark for each anomalous day.

    Two shaded periods tell the reader what a mark means. Before `normal_until` the days
    are usual, so a mark there is a holiday or a false alert: fewer is better. In the first
    `window_days` of a lockdown the days are known anomalies, so a mark there is a hit.
    """
    codes = list(flags["country"].unique())
    models = [c for c in flags.columns if c not in ("day", "country")]
    first, last = flags["day"].min(), flags["day"].max()
    normal_end = pd.Timestamp(normal_until, tz="UTC") + pd.Timedelta(days=1)
    fig, axes = plt.subplots(len(codes), 1, figsize=(11, (0.55 + 0.4 * len(models)) * len(codes) + 1.6), sharex=True,
                             squeeze=False)
    fig.text(0.01, 0.99, "Days that each model flagged as anomalous (one mark = one day)", ha="left", va="top",
             fontsize=13, color="#0b0b0b")
    fig.text(0.01, 0.955, "Grey area: usual days, so marks are holidays or false alerts (fewer is better).   "
             "Red area: the first two weeks of a lockdown, so marks are hits (more is better).",
             ha="left", va="top", fontsize=9, color=INK)
    for ax, code in zip(axes[:, 0], codes):
        part = flags[flags["country"] == code]
        ax.axvspan(first, normal_end, color=USUAL_SHADE, linewidth=0, zorder=0)
        start = None
        if code in lockdowns:
            start = pd.Timestamp(lockdowns[code], tz="UTC")
            ax.axvspan(start, start + pd.Timedelta(days=window_days), color=ANOMALY, alpha=0.16, linewidth=0, zorder=0)
        labels = []
        for row, model in enumerate(models):
            hits = part.loc[part[model], "day"]
            ax.scatter(hits, [row] * len(hits), marker="|", s=110, linewidths=2, color=model_color(model), zorder=2)
            usual = int((hits < normal_end).sum())
            caught = 0 if start is None else int(((hits >= start) & (hits < start + pd.Timedelta(days=window_days))).sum())
            labels.append(f"{model}\n{usual} usual · {caught} lockdown")
        ax.set_title(country_name(code), loc="left", fontsize=10, color=INK)
        ax.set_yticks(range(len(models)), labels)
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.set_xlim(first, last + pd.Timedelta(days=1))
        ax.invert_yaxis()
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=8, length=0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_compare_forecast(history: pd.DataFrame, forecasts: dict[str, QuantileForecast], path: Path,
                          show_days: int = 14):
    """The last `show_days` of actual load, then each model's median forecast for the days after the data."""
    recent = history.iloc[-show_days * 24:]
    days = len(next(iter(forecasts.values())).median) // 24
    fig, axes = _panels(list(history.columns), f"The next {days} days: the median forecast of each model", "MW")
    for ax, code in zip(axes, history.columns):
        ax.plot(recent.index, recent[code], color=ACTUAL, linewidth=0.8, label="actual")
        for label, fc in forecasts.items():
            ax.plot(fc.median.index, fc.median[code], color=model_color(label), linewidth=1, label=label)
    _save(fig, axes, path)


def plot_compare_week(load: pd.DataFrame, forecasts: dict[str, QuantileForecast], start: str, path: Path):
    """One usual week: the actual load and each model's week-ahead median, made before the week started."""
    first = pd.Timestamp(start, tz="UTC")
    hours = pd.date_range(first, periods=7 * 24, freq="h")
    fig, axes = _panels(list(load.columns),
                        f"Week-ahead forecasts for the week from {first:%Y-%m-%d}, made before it started", "MW")
    for ax, code in zip(axes, load.columns):
        ax.plot(hours, load.loc[hours, code], color=ACTUAL, linewidth=1.6, label="actual")
        for label, fc in forecasts.items():
            ax.plot(hours, fc.median.loc[hours, code], color=model_color(label), linewidth=1, label=f"{label} median")
    _save(fig, axes, path)
