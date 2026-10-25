"""Forecast, backtest and find anomalies in daily website visits with Timestral.

    uv run main.py forecast   # next 30 days
    uv run main.py backtest   # hide the last 30 days, forecast them, compare
    uv run main.py anomalies  # days that fell outside the one-day-ahead forecast
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_daily_visits
from forecasting import DEFAULT_MODEL, forecast, load_forecaster, one_step_ahead
from plots import plot_anomalies, plot_backtest, plot_forecast

OUTPUT = Path("output")


def run_forecast(history: pd.DataFrame, args: argparse.Namespace):
    fc = forecast(load_forecaster(args.model), history, args.horizon)

    table = pd.concat({"low": fc.low, "median": fc.median, "high": fc.high, "mean": fc.mean}, axis=1).round(2)
    table.to_csv(OUTPUT / "forecast.csv")
    plot_forecast(history, fc, OUTPUT / "forecast.png")

    print(f"Forecast for {fc.median.index[0]:%Y-%m-%d} .. {fc.median.index[-1]:%Y-%m-%d}\n")
    summary = pd.DataFrame({
        f"last {args.horizon} days (actual)": history.iloc[-args.horizon:].sum(),
        f"next {args.horizon} days (forecast)": fc.mean.sum(),
        f"{fc.low_level:.0%} quantile": fc.low.sum(),
        f"{fc.high_level:.0%} quantile": fc.high.sum(),
    })
    print(summary.round(0).astype(int).to_string())
    print("\nThe forecast total adds up the expected (mean) visits per day. The quantile columns add up the")
    print("daily quantiles, so they give a rough range for the total, not an exact one.")
    print(f"Wrote {OUTPUT / 'forecast.csv'} and {OUTPUT / 'forecast.png'}")


def run_backtest(history: pd.DataFrame, args: argparse.Namespace):
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else history.index[-1] - pd.Timedelta(days=args.horizon)
    seen = history.loc[:cutoff]
    hidden = history.loc[cutoff + pd.Timedelta(days=1):].iloc[:args.horizon]
    if len(hidden) < args.horizon:
        raise SystemExit(f"Only {len(hidden)} days after {cutoff:%Y-%m-%d}; pick an earlier --cutoff")

    fc = forecast(load_forecaster(args.model), seen, args.horizon)
    # Baseline to beat: repeat the last week the model saw ("same weekday last week").
    naive = pd.DataFrame(np.resize(seen.iloc[-7:].to_numpy(), hidden.shape), index=hidden.index, columns=hidden.columns)

    inside = (hidden >= fc.low) & (hidden <= fc.high)
    report = pd.DataFrame({
        "actual total": hidden.sum(),
        "forecast total": fc.mean.sum(),
        "MAE/day timestral": (hidden - fc.median).abs().mean(),
        "MAE/day last-week": (hidden - naive).abs().mean(),
        f"in {fc.high_level - fc.low_level:.0%} interval": inside.mean().map("{:.0%}".format),
    })

    days = pd.concat({"actual": hidden, "median": fc.median, "low": fc.low, "high": fc.high, "mean": fc.mean},
                     axis=1).round(2)
    days.to_csv(OUTPUT / "backtest.csv")
    plot_backtest(seen, hidden, fc, OUTPUT / "backtest.png")

    print(f"Model saw data up to {cutoff:%Y-%m-%d}; forecast {hidden.index[0]:%Y-%m-%d} .. {hidden.index[-1]:%Y-%m-%d}\n")
    print(report.round(2).to_string())
    print("\nforecast total = sum of expected (mean) visits per day.")
    print("MAE = mean absolute error of the daily median forecast in visits per day, lower is better.")
    print("last-week = baseline that repeats the last 7 days the model saw.")
    print(f"Wrote {OUTPUT / 'backtest.csv'} and {OUTPUT / 'backtest.png'}")


def run_anomalies(history: pd.DataFrame, args: argparse.Namespace):
    fc = one_step_ahead(load_forecaster(args.model), history, args.min_context)
    actual = history.loc[fc.median.index]

    rows = []
    for name in history.columns:
        expected = pd.DataFrame({"visitors": actual[name], "low": fc.low[name], "median": fc.median[name],
                                 "high": fc.high[name]})
        # A page added later (like /talks) needs its own `min_context` days of history before it is judged.
        days_seen = history[name].notna().cumsum().shift(fill_value=0)
        expected = expected[days_seen.loc[expected.index] >= args.min_context].round(2)
        # Outside the 1%..99% interval, and by at least `min_excess` visits so that 1 visit
        # on a page that usually gets 0 does not count as an anomaly.
        spike = expected["visitors"] - expected["high"]
        drop = expected["low"] - expected["visitors"]
        expected["kind"] = np.select([spike >= args.min_excess, drop >= args.min_excess], ["spike", "drop"], "")
        expected["excess"] = np.maximum(spike, drop)
        hits = expected[expected["kind"] != ""].reset_index()
        hits.insert(1, "series", name)
        rows.append(hits)
    anomalies = pd.concat(rows).sort_values("excess", ascending=False)

    anomalies.to_csv(OUTPUT / "anomalies.csv", index=False)
    plot_anomalies(history, fc, anomalies, OUTPUT / "anomalies.png")

    print(f"Checked {fc.median.index[0]:%Y-%m-%d} .. {fc.median.index[-1]:%Y-%m-%d}, "
          f"each day forecast only from the days before it.\n")
    print(f"{len(anomalies)} anomalies, largest first:\n")
    print(anomalies.head(args.show).to_string(index=False))
    print(f"\nWrote {OUTPUT / 'anomalies.csv'} and {OUTPUT / 'anomalies.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="daily_pages.csv", help="export with day,path,visitors columns")
    parser.add_argument("--pages", type=int, default=4, help="forecast the N most visited pages besides the total")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face id, e.g. mistralai/Timestral-28M")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("forecast", help="forecast the days after the last day in the data")
    p.add_argument("--horizon", type=int, default=30, help="days to forecast")
    p.set_defaults(run=run_forecast)

    p = commands.add_parser("backtest", help="forecast a past month from the data before it and score it")
    p.add_argument("--horizon", type=int, default=30, help="days to hide and forecast")
    p.add_argument("--cutoff", help="last day the model may see (default: horizon days before the end)")
    p.set_defaults(run=run_backtest)

    p = commands.add_parser("anomalies", help="flag days that fell outside the one-day-ahead forecast")
    p.add_argument("--min-context", type=int, default=56, help="days of history before the first checked day")
    p.add_argument("--min-excess", type=float, default=2, help="visits beyond the interval needed to flag a day")
    p.add_argument("--show", type=int, default=20, help="rows to print")
    p.set_defaults(run=run_anomalies)

    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    history = load_daily_visits(args.csv, args.pages)
    args.run(history, args)


if __name__ == "__main__":
    main()
