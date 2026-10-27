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
OPERATOR = "#1baf7a"
ANOMALY = "#e34948"
INK = "#52514e"
GRID = "#e4e3df"


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


def _band(ax, fc: QuantileForecast, code: str, label: str):
    pct = round((fc.high_level - fc.low_level) * 100)
    ax.fill_between(fc.median.index, fc.low[code], fc.high[code], color=FORECAST, alpha=0.18, linewidth=0,
                    label=f"{pct}% interval")
    ax.plot(fc.median.index, fc.median[code], color=FORECAST, linewidth=1.2, label=label)


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


def plot_forecast(history: pd.DataFrame, fc: QuantileForecast, path: Path, show_days: int = 28):
    recent = history.iloc[-show_days * 24:]
    fig, axes = _panels(list(history.columns),
                        f"Hourly load: last {show_days} days and next {len(fc.median) // 24} days", "MW")
    for ax, code in zip(axes, history.columns):
        ax.plot(recent.index, recent[code], color=ACTUAL, linewidth=0.8, label="actual")
        _band(ax, fc, code, "forecast (median)")
    _save(fig, axes, path)


def plot_backtest(seen: pd.DataFrame, hidden: pd.DataFrame, day: QuantileForecast, operator: pd.DataFrame,
                  path: Path, show_days: int = 7):
    context = seen.iloc[-show_days * 24:]
    fig, axes = _panels(list(seen.columns),
                        f"Backtest: day-ahead forecasts from {hidden.index[0]:%Y-%m-%d} vs what happened", "MW")
    for ax, code in zip(axes, seen.columns):
        ax.plot(context.index, context[code], color=ACTUAL, linewidth=0.8, label="before the backtest")
        ax.plot(hidden.index, hidden[code], color=ACTUAL, linewidth=0.8, linestyle=(0, (2, 2)), label="actual")
        ax.axvline(hidden.index[0], color=INK, linewidth=0.8)
        _band(ax, day, code, "Timestral day-ahead")
        ax.plot(operator.index, operator[code], color=OPERATOR, linewidth=0.8, label="grid operator day-ahead")
    _save(fig, axes, path)


def plot_anomalies(actual: pd.DataFrame, expected: pd.DataFrame, days: pd.DataFrame, lockdowns: dict[str, str],
                   horizon: int, path: Path):
    ahead = {24: "day-ahead", 168: "week-ahead"}.get(horizon, f"{horizon}-hours-ahead")
    fig, axes = _panels(list(actual.columns), f"Daily energy vs the {ahead} forecast; red = anomalous day", "MWh")
    for ax, code in zip(axes, actual.columns):
        ax.plot(expected.index, expected[code], color=FORECAST, linewidth=1.2, label="expected (forecast)")
        ax.plot(actual.index, actual[code], color=ACTUAL, linewidth=1, label="actual")
        hits = days[days["country"] == code]
        ax.scatter(hits["day"], hits["actual MWh"], s=24, color=ANOMALY, edgecolor="white", linewidth=1, zorder=3,
                   label="anomalous day")
        if code in lockdowns:
            ax.axvline(pd.Timestamp(lockdowns[code], tz="UTC"), color=INK, linewidth=0.8, linestyle=(0, (4, 3)),
                       label="lockdown starts")
    _save(fig, axes, path)
