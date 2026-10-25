"""Matplotlib charts: one small panel per series, sharing the date axis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from forecasting import QuantileForecast

ACTUAL = "#2a78d6"
FORECAST = "#eb6834"
ANOMALY = "#e34948"
INK = "#52514e"
GRID = "#e4e3df"


def _panels(series: list[str], title: str):
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.3 * len(series)), sharex=True, squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color="#0b0b0b")
    for ax, name in zip(axes[:, 0], series):
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=8, length=0)
    return fig, axes[:, 0]


def _band(ax, fc: QuantileForecast, name: str, label: str):
    pct = round((fc.high_level - fc.low_level) * 100)
    ax.fill_between(fc.median.index, fc.low[name], fc.high[name], color=FORECAST, alpha=0.18, linewidth=0,
                    label=f"{pct}% interval")
    ax.plot(fc.median.index, fc.median[name], color=FORECAST, linewidth=2, label=label)


def _save(fig, axes, path: Path):
    for ax in axes:
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, ncols=4, bbox_to_anchor=(0, 1.35))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_forecast(history: pd.DataFrame, fc: QuantileForecast, path: Path, show_days: int = 120):
    recent = history.iloc[-show_days:]
    fig, axes = _panels(list(history.columns), f"Daily visits: last {show_days} days and next {len(fc.median)} days")
    for ax, name in zip(axes, history.columns):
        ax.plot(recent.index, recent[name], color=ACTUAL, linewidth=1.5, label="actual")
        _band(ax, fc, name, "forecast (median)")
    _save(fig, axes, path)


def plot_backtest(history: pd.DataFrame, actual: pd.DataFrame, fc: QuantileForecast, path: Path, show_days: int = 90):
    context = history.iloc[-show_days:]
    fig, axes = _panels(list(history.columns), f"Backtest: forecast from {history.index[-1]:%Y-%m-%d} vs what happened")
    for ax, name in zip(axes, history.columns):
        ax.plot(context.index, context[name], color=ACTUAL, linewidth=1.5, label="seen by the model")
        ax.plot(actual.index, actual[name], color=ACTUAL, linewidth=1.5, linestyle=(0, (2, 2)),
                label="hidden from the model")
        ax.axvline(history.index[-1], color=INK, linewidth=0.8)
        _band(ax, fc, name, "forecast (median)")
    _save(fig, axes, path)


def plot_anomalies(history: pd.DataFrame, fc: QuantileForecast, anomalies: pd.DataFrame, path: Path):
    fig, axes = _panels(list(history.columns), "Anomalies: days outside the one-day-ahead forecast interval")
    for ax, name in zip(axes, history.columns):
        ax.fill_between(fc.median.index, fc.low[name], fc.high[name], color=FORECAST, alpha=0.18, linewidth=0,
                        label=f"{round((fc.high_level - fc.low_level) * 100)}% interval")
        ax.plot(history.index, history[name], color=ACTUAL, linewidth=1, label="actual")
        hits = anomalies[anomalies["series"] == name]
        ax.scatter(hits["day"], hits["visitors"], s=36, color=ANOMALY, edgecolor="white", linewidth=1.5, zorder=3,
                   label="anomaly")
    _save(fig, axes, path)
