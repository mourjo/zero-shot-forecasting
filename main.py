"""Forecast, backtest and find anomalies in European electricity demand with Timestral.

    uv run main.py forecast   # the next 30 days, hour by hour
    uv run main.py backtest   # hide a past month, forecast it, compare with the grid operators
    uv run main.py anomalies  # days that fell outside the week-ahead forecast, e.g. COVID lockdowns
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import DEFAULT_COUNTRIES, LOCKDOWNS, holiday_indicator, load_power_data
from forecasting import DEFAULT_MODEL, forecast, future_index, load_forecaster, rolling
from plots import plot_anomalies, plot_backtest, plot_forecast

OUTPUT = Path("output")
DAY = pd.Timedelta(days=1)
CONTEXT = 2048  # hours of history for each rolling (day- or week-ahead) forecast, ~12 weeks
# Grid operators publish day-ahead forecasts by 10:00 CET the day before (2 hours before the
# 12:00 day-ahead market closes). Timestral forecasts from 08:00 UTC the day before, so in the
# backtest it never knows more than they did.
OPERATOR_GAP = 16


def holidays_for(index: pd.DatetimeIndex, args: argparse.Namespace) -> pd.DataFrame | None:
    return None if args.no_holidays else holiday_indicator(index, args.countries)


def mape(predicted: pd.DataFrame, actual: pd.DataFrame) -> pd.Series:
    """Mean absolute percentage error per country: on average, how many % off the forecast was."""
    actual = actual.loc[predicted.index]
    return ((predicted - actual).abs() / actual).mean() * 100


def run_forecast(power, args: argparse.Namespace):
    history = power.load
    horizon = args.days * 24
    known = holidays_for(history.index.append(future_index(history, horizon)), args)
    fc = forecast(load_forecaster(args.model), history, horizon, known)

    table = pd.concat({"low": fc.low, "median": fc.median, "high": fc.high, "mean": fc.mean}, axis=1).round(0)
    table.to_csv(OUTPUT / "forecast.csv")
    plot_forecast(history, fc, OUTPUT / "forecast.png")

    print(f"Forecast for {fc.median.index[0]:%Y-%m-%d %H:%M} .. {fc.median.index[-1]:%Y-%m-%d %H:%M} UTC\n")
    summary = pd.DataFrame({
        f"last {args.days} days GWh (actual)": history.iloc[-horizon:].sum() / 1000,
        f"next {args.days} days GWh (forecast)": fc.mean.sum() / 1000,
        "peak MW (actual)": history.iloc[-horizon:].max(),
        "peak MW (forecast)": fc.median.max(),
    })
    print(summary.round(0).astype(int).to_string())
    print("\nGWh = energy over the period (sum of the hourly mean forecasts); MW = power in one hour.")
    print(f"Wrote {OUTPUT / 'forecast.csv'} and {OUTPUT / 'forecast.png'}")


def run_backtest(power, args: argparse.Namespace):
    load = power.load
    last = load.index[-1].normalize()
    cutoff = pd.Timestamp(args.cutoff, tz="UTC") if args.cutoff else last - args.days * DAY
    start, end = cutoff + DAY, cutoff + (args.days + 1) * DAY  # hidden: the days after the cutoff
    if end > load.index[-1] + pd.Timedelta(hours=1):
        raise SystemExit(f"Not enough data after {cutoff:%Y-%m-%d}; pick an earlier --cutoff")
    seen = load.loc[:start - pd.Timedelta(hours=1)]
    hidden = load.loc[start:end - pd.Timedelta(hours=1)]
    horizon = len(hidden)
    known = holidays_for(load.index, args)
    forecaster = load_forecaster(args.model)

    # 1. Month ahead: one forecast for the whole hidden period, made at the cutoff.
    month = forecast(forecaster, seen, horizon, known)
    # 2. Day ahead: at 08:00 UTC every day, forecast the next day's 24 hours, like the grid operators do.
    day = rolling(forecaster, load, start, end, horizon=24, context=args.context, gap=OPERATOR_GAP, known=known,
                  low=0.1, high=0.9)
    # Baseline to beat: "same hour one week earlier".
    last_week = load.shift(7 * 24).loc[hidden.index]

    inside = (hidden >= month.low) & (hidden <= month.high)
    report = pd.DataFrame({
        "month-ahead timestral": mape(month.median, load),
        "month-ahead last-week": mape(pd.DataFrame(np.resize(seen.iloc[-7 * 24:].to_numpy(), hidden.shape),
                                                   index=hidden.index, columns=hidden.columns), load),
        "day-ahead timestral": mape(day.median, load),
        "day-ahead grid operator": mape(power.tso_forecast.loc[hidden.index], load),
        "day-ahead last-week": mape(last_week, load),
        f"month-ahead in {month.high_level - month.low_level:.0%} interval": inside.mean().map("{:.0%}".format),
    })

    days = pd.concat({"actual": hidden, "month-ahead": month.median, "day-ahead": day.median,
                      "grid operator": power.tso_forecast.loc[hidden.index]}, axis=1).round(0)
    name = f"backtest-{cutoff:%Y-%m-%d}"
    days.to_csv(OUTPUT / f"{name}.csv")
    plot_backtest(seen, hidden, month, day, OUTPUT / f"{name}.png")

    print(f"Model saw data up to {cutoff:%Y-%m-%d}; forecast {start:%Y-%m-%d} .. {end - DAY:%Y-%m-%d} (UTC)\n")
    print("Mean absolute percentage error (MAPE), lower is better:\n")
    print(report.round(2).to_string())
    print("\nmonth-ahead = one forecast for the whole period, made at the cutoff.")
    print("day-ahead   = every day at 08:00 UTC, a forecast for the next day from the history before it.")
    print("grid operator = the day-ahead forecast the national grid operator published (from the data).")
    print("last-week   = baseline that repeats the same hour one week earlier.")
    print(f"Wrote {OUTPUT / name}.csv and {OUTPUT / name}.png")


def run_anomalies(power, args: argparse.Namespace):
    load = power.load
    known = holidays_for(load.index, args)
    end = pd.Timestamp(args.end, tz="UTC") if args.end else load.index[-1].normalize() + DAY
    fc = rolling(load_forecaster(args.model), load, args.start, end, horizon=args.horizon, context=args.context,
                 known=known)
    actual = load.loc[fc.median.index]

    # Per hour: is it outside the 1%..99% interval? Per day: how many such hours, and how
    # far is the day's energy from the forecast?
    below = (actual < fc.low).resample("D").sum()
    above = (actual > fc.high).resample("D").sum()
    actual_day = actual.resample("D").sum()
    expected_day = fc.mean.resample("D").sum()
    deviation = (actual_day - expected_day) / expected_day * 100

    rows = []
    for code in load.columns:
        daily = pd.DataFrame({"country": code, "hours below": below[code], "hours above": above[code],
                              "actual MWh": actual_day[code], "expected MWh": expected_day[code],
                              "deviation %": deviation[code]})
        # A day is anomalous when at least `min_hours` of its 24 hours fell outside the interval.
        flagged = daily[(daily["hours below"] + daily["hours above"]) >= args.min_hours].copy()
        flagged["kind"] = np.where(flagged["hours below"] >= flagged["hours above"], "drop", "spike")
        rows.append(flagged)
    days = pd.concat(rows).round(1).rename_axis("day").reset_index()
    name = f"anomalies-{args.horizon}h"
    days.to_csv(OUTPUT / f"{name}.csv", index=False)
    plot_anomalies(actual_day, expected_day, days, LOCKDOWNS, args.horizon, OUTPUT / f"{name}.png")

    print(f"Checked {fc.median.index[0]:%Y-%m-%d} .. {fc.median.index[-1]:%Y-%m-%d} (UTC), {args.horizon} hours at a "
          f"time: each block forecast at its start, from the {args.context} hours before it.\n")
    # Calibration: before the pandemic, ~1% of hours should be above the 99% line and ~1% below the 1% line.
    normal = slice(None, "2020-02-29")
    above_share = (actual.loc[normal] > fc.high.loc[normal]).to_numpy().mean()
    below_share = (actual.loc[normal] < fc.low.loc[normal]).to_numpy().mean()
    if len(actual.loc[normal]):
        print(f"Calibration up to 2020-02-29: {above_share:.1%} of hours above the {fc.high_level:.0%} line, "
              f"{below_share:.1%} below the {fc.low_level:.0%} line (expected: about 1% each).\n")
    print(_episodes(days).to_string(index=False))
    print(f"\nA day is anomalous when at least {args.min_hours} of its hours fell outside the "
          f"{fc.low_level:.0%}..{fc.high_level:.0%} interval. Consecutive anomalous days form one episode.")
    print(f"Wrote {OUTPUT / name}.csv ({len(days)} days) and {OUTPUT / name}.png")


def _episodes(days: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive anomalous days of the same country and kind into episodes."""
    days = days.sort_values(["country", "day"])
    new = (days["day"].diff() != DAY) | (days["country"] != days["country"].shift()) | (days["kind"] != days["kind"].shift())
    days = days.assign(episode=new.cumsum())
    episodes = days.groupby("episode").agg(country=("country", "first"), kind=("kind", "first"),
                                           first=("day", "first"), last=("day", "last"), days=("day", "size"),
                                           deviation=("deviation %", "mean"))
    episodes["first"] = episodes["first"].dt.strftime("%Y-%m-%d")
    episodes["last"] = episodes["last"].dt.strftime("%Y-%m-%d")
    episodes = episodes.rename(columns={"deviation": "avg deviation %"}).round(1)
    return episodes.sort_values(["days", "country"], ascending=[False, True])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES),
                        help="comma-separated, from DE,FR,IT,ES,GB (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face id, e.g. mistralai/Timestral-28M")
    parser.add_argument("--no-holidays", action="store_true", help="don't pass public holidays as a covariate")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("forecast", help="forecast the days after the last hour in the data")
    p.add_argument("--days", type=int, default=30, help="days to forecast")
    p.set_defaults(run=run_forecast)

    p = commands.add_parser("backtest", help="forecast a past period and score it against the grid operators")
    p.add_argument("--days", type=int, default=30, help="days to hide and forecast")
    p.add_argument("--cutoff", help="last day the model may see, YYYY-MM-DD (default: days before the end)")
    p.add_argument("--context", type=int, default=CONTEXT, help="hours of history for each day-ahead forecast")
    p.set_defaults(run=run_backtest)

    p = commands.add_parser("anomalies", help="flag days that fell outside the week-ahead forecast")
    p.add_argument("--start", default="2019-09-30", help="first day to check, YYYY-MM-DD (a Monday)")
    p.add_argument("--end", help="day after the last day to check (default: end of the data)")
    p.add_argument("--context", type=int, default=CONTEXT, help="hours of history for each rolling forecast")
    p.add_argument("--horizon", type=int, default=168,
                   help="hours forecast at a time: 168 = week-ahead (sees shifts that build up over days), "
                        "24 = day-ahead (sharp one-day events)")
    p.add_argument("--min-hours", type=int, default=6, help="hours outside the interval that make a day anomalous")
    p.set_defaults(run=run_anomalies)

    args = parser.parse_args()
    args.countries = args.countries.split(",")
    OUTPUT.mkdir(exist_ok=True)
    args.run(load_power_data(args.countries), args)


if __name__ == "__main__":
    main()
