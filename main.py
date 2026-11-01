"""Forecast, backtest and find anomalies in European electricity demand with Timestral or Chronos-2.

    uv run main.py signal     # the raw signal, with a fixed threshold and a moving average (no model)
    uv run main.py forecast   # the next 30 days, hour by hour
    uv run main.py backtest   # replay a past month day by day, compare with the grid operators
    uv run main.py anomalies  # days that fell outside the week-ahead forecast, e.g. COVID lockdowns
    uv run main.py compare    # the backtest and the anomaly check with Timestral and Chronos-2, side by side

Add --model chronos-2 to any command to use Chronos-2 instead of Timestral.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import DEFAULT_COUNTRIES, LOCKDOWNS, country_name, holiday_indicator, load_power_data
from forecasting import (DEFAULT_MODEL, MODELS, Model, QuantileForecast, TimestralModel, forecast, future_index,
                         load_forecaster, rolling)
from plots import (plot_anomalies, plot_backtest, plot_compare_anomalies, plot_compare_backtest, plot_forecast,
                   plot_signal)

OUTPUT = Path("output")
DAY = pd.Timedelta(days=1)
CONTEXT = 2048  # hours of history for each rolling (day- or week-ahead) forecast, ~12 weeks
# Grid operators publish day-ahead forecasts by 10:00 CET the day before (2 hours before the
# 12:00 day-ahead market closes). The model forecasts from 08:00 UTC the day before, so in the
# backtest it never knows more than they did.
OPERATOR_GAP = 16
NORMAL_UNTIL = "2020-02-29"  # last day before the pandemic: used to check calibration
LOCKDOWN_WINDOW = 14  # days from the start of each lockdown that `compare` checks


def holidays_for(index: pd.DatetimeIndex, args: argparse.Namespace) -> pd.DataFrame | None:
    return None if args.no_holidays else holiday_indicator(index, args.countries)


def mape(predicted: pd.DataFrame, actual: pd.DataFrame) -> pd.Series:
    """Mean absolute percentage error per country: on average, how many % off the forecast was."""
    actual = actual.loc[predicted.index]
    return ((predicted - actual).abs() / actual).mean() * 100


def run_signal(power, args: argparse.Namespace):
    """Why simple rules struggle: a fixed threshold and a moving-average band, without any model."""
    code = args.country
    load = power.load[code].loc[args.start:args.end]
    week = 7 * 24
    # Only the past week, so that the band at each hour uses no future data.
    full = power.load[code]
    average = full.rolling(week).mean().loc[load.index]
    spread = full.rolling(week).std().loc[load.index]
    path = OUTPUT / f"signal-{code}.png"
    plot_signal(load, args.threshold, average, spread, LOCKDOWNS.get(code), path)

    lockdown = pd.Timestamp(LOCKDOWNS[code], tz="UTC") if code in LOCKDOWNS else load.index[-1]
    periods = {"before the lockdown": load.loc[:lockdown - pd.Timedelta(hours=1)],
               "from the lockdown on": load.loc[lockdown:]}
    outside = (load - average).abs() > 2 * spread
    print(f"{country_name(code)}, {load.index[0]:%Y-%m-%d} .. {load.index[-1]:%Y-%m-%d} (UTC)\n")
    for label, part in periods.items():
        print(f"{label}: {(part < args.threshold).mean():.0%} of hours below {args.threshold:,.0f} MW, "
              f"{outside.loc[part.index].mean():.0%} outside the moving-average band")
    print(f"Wrote {path}")


def run_forecast(power, args: argparse.Namespace):
    history = power.load
    horizon = args.days * 24
    known = holidays_for(history.index.append(future_index(history, horizon)), args)
    model = load_forecaster(args.model, args.device)
    fc = forecast(model, history, horizon, known)

    table = pd.concat({"low": fc.low, "median": fc.median, "high": fc.high, "mean": fc.mean}, axis=1).round(0)
    name = f"forecast{_suffix(model)}"
    table.to_csv(OUTPUT / f"{name}.csv")
    plot_forecast(history, fc, OUTPUT / f"{name}.png")

    print(f"Forecast by {model.label} for {fc.median.index[0]:%Y-%m-%d %H:%M} .. {fc.median.index[-1]:%Y-%m-%d %H:%M} UTC\n")
    summary = pd.DataFrame({
        f"last {args.days} days GWh (actual)": history.iloc[-horizon:].sum() / 1000,
        f"next {args.days} days GWh (forecast)": fc.mean.sum() / 1000,
        "peak MW (actual)": history.iloc[-horizon:].max(),
        "peak MW (forecast)": fc.median.max(),
    })
    print(summary.round(0).astype(int).to_string())
    print("\nGWh = energy over the period (sum of the hourly mean forecasts); MW = power in one hour.")
    print(f"Wrote {OUTPUT / name}.csv and {OUTPUT / name}.png")


def backtest(model: Model, power, args: argparse.Namespace) -> dict:
    """Day-ahead forecasts for the `args.days` days after the cutoff, each made at 08:00 UTC the day before."""
    load = power.load
    last = load.index[-1].normalize()
    cutoff = pd.Timestamp(args.cutoff, tz="UTC") if args.cutoff else last - args.days * DAY
    start, end = cutoff + DAY, cutoff + (args.days + 1) * DAY  # hidden: the days after the cutoff
    if end > load.index[-1] + pd.Timedelta(hours=1):
        raise SystemExit(f"Not enough data after {cutoff:%Y-%m-%d}; pick an earlier --cutoff")
    hidden = load.loc[start:end - pd.Timedelta(hours=1)]
    clock = time.perf_counter()
    # Every day at 08:00 UTC, forecast the next day's 24 hours, like the grid operators do.
    day = rolling(model, load, start, end, horizon=24, context=args.context, gap=OPERATOR_GAP,
                  known=holidays_for(load.index, args), low=0.1, high=0.9)
    return {"cutoff": cutoff, "start": start, "end": end, "seen": load.loc[:start - pd.Timedelta(hours=1)],
            "hidden": hidden, "day": day, "seconds": time.perf_counter() - clock,
            "operator": power.tso_forecast.loc[hidden.index],
            # Baseline to beat: "same hour one week earlier".
            "last week": load.shift(7 * 24).loc[hidden.index]}


def coverage(actual: pd.DataFrame, fc: QuantileForecast) -> pd.Series:
    """Share of hours inside the forecast interval, per country."""
    return ((actual >= fc.low) & (actual <= fc.high)).mean()


def run_backtest(power, args: argparse.Namespace):
    model = load_forecaster(args.model, args.device)
    bt = backtest(model, power, args)
    load, hidden, day = power.load, bt["hidden"], bt["day"]
    report = pd.DataFrame({
        model.label: mape(day.median, load),
        "grid operator": mape(bt["operator"], load),
        "last week": mape(bt["last week"], load),
        f"{model.label} in {day.high_level - day.low_level:.0%} interval": coverage(hidden, day).map("{:.0%}".format),
    })

    days = pd.concat({"actual": hidden, model.label.lower(): day.median, "grid operator": bt["operator"]},
                     axis=1).round(0)
    name = f"backtest-{bt['cutoff']:%Y-%m-%d}{_suffix(model)}"
    days.to_csv(OUTPUT / f"{name}.csv")
    plot_backtest(bt["seen"], hidden, day, bt["operator"], OUTPUT / f"{name}.png", label=model.label)

    print(f"Day-ahead forecasts by {model.label} for {bt['start']:%Y-%m-%d} .. {bt['end'] - DAY:%Y-%m-%d} (UTC), "
          f"each made at 08:00 UTC the day before from the {args.context} hours up to then\n")
    print("Mean absolute percentage error (MAPE), lower is better:\n")
    print(report.round(2).to_string())
    print("\ngrid operator = the day-ahead forecast the national grid operator published (from the data).")
    print("last week     = baseline that repeats the same hour one week earlier.")
    print(f"Wrote {OUTPUT / name}.csv and {OUTPUT / name}.png")


def anomalies(model: Model, power, args: argparse.Namespace) -> dict:
    """Forecast every block of `args.horizon` hours from the hours before it; flag the days outside 1%..99%."""
    load = power.load
    end = pd.Timestamp(args.end, tz="UTC") if args.end else load.index[-1].normalize() + DAY
    clock = time.perf_counter()
    fc = rolling(model, load, args.start, end, horizon=args.horizon, context=args.context,
                 known=holidays_for(load.index, args))
    seconds = time.perf_counter() - clock
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
    # Calibration: before the pandemic, ~1% of hours should be above the 99% line and ~1% below the 1% line.
    normal = slice(None, NORMAL_UNTIL)
    calibration = pd.DataFrame({"below": (actual.loc[normal] < fc.low.loc[normal]).mean(),
                                "above": (actual.loc[normal] > fc.high.loc[normal]).mean()})
    return {"fc": fc, "actual": actual, "actual day": actual_day, "expected day": expected_day, "days": days,
            "calibration": calibration, "seconds": seconds}


def run_anomalies(power, args: argparse.Namespace):
    model = load_forecaster(args.model, args.device)
    an = anomalies(model, power, args)
    fc, days = an["fc"], an["days"]
    name = f"anomalies-{args.horizon}h{_suffix(model)}"
    days.to_csv(OUTPUT / f"{name}.csv", index=False)
    plot_anomalies(an["actual day"], an["expected day"], days, LOCKDOWNS, args.horizon, OUTPUT / f"{name}.png")

    print(f"Checked {fc.median.index[0]:%Y-%m-%d} .. {fc.median.index[-1]:%Y-%m-%d} (UTC) with {model.label}, "
          f"{args.horizon} hours at a time: each block forecast at its start, from the {args.context} hours before it.\n")
    if len(an["actual"].loc[:NORMAL_UNTIL]):
        overall = an["calibration"].mean()
        print(f"Calibration up to {NORMAL_UNTIL}: {overall['above']:.1%} of hours above the {fc.high_level:.0%} line, "
              f"{overall['below']:.1%} below the {fc.low_level:.0%} line (expected: about 1% each).\n")
    print(_episodes(days).to_string(index=False))
    print(f"\nA day is anomalous when at least {args.min_hours} of its hours fell outside the "
          f"{fc.low_level:.0%}..{fc.high_level:.0%} interval. Consecutive anomalous days form one episode.")
    print(f"Wrote {OUTPUT / name}.csv ({len(days)} days) and {OUTPUT / name}.png")


def run_compare(power, args: argparse.Namespace):
    """Run the same backtest and the same anomaly check with both models, and compare them."""
    load = power.load
    models = [load_forecaster(name, args.device) for name in (args.model, args.against)]
    labels = [m.label for m in models]
    if labels[0] == labels[1]:
        labels = [args.model, args.against]

    # 1. Forecasting: the day-ahead backtest.
    bts = {label: backtest(model, power, args) for label, model in zip(labels, models)}
    first = bts[labels[0]]
    hidden = first["hidden"]
    accuracy = pd.DataFrame({**{f"{label} MAPE %": mape(bt["day"].median, load) for label, bt in bts.items()},
                             "grid operator MAPE %": mape(first["operator"], load),
                             "last week MAPE %": mape(first["last week"], load)})
    for label, bt in bts.items():
        accuracy[f"{label} in 80% range"] = coverage(hidden, bt["day"]) * 100
    for label, bt in bts.items():
        # Sharpness: how wide the 80% range is, as % of the actual load. Narrower is better, if coverage holds.
        accuracy[f"{label} 80% width %"] = ((bt["day"].high - bt["day"].low) / hidden).mean() * 100
    accuracy.loc["mean"] = accuracy.mean()
    accuracy.round(2).rename_axis("country").to_csv(OUTPUT / "compare-backtest.csv")
    plot_compare_backtest(hidden, {label: bt["day"] for label, bt in bts.items()}, OUTPUT / "compare-backtest.png")

    print(f"1. FORECASTING: day-ahead forecasts for {first['start']:%Y-%m-%d} .. {first['end'] - DAY:%Y-%m-%d} (UTC), "
          f"each made at 08:00 UTC the day before from the {args.context} hours up to then\n")
    print(accuracy.round(2).to_string())
    print("\nMAPE = mean absolute percentage error, lower is better. '80% range' should contain about 80% of hours.")
    print("80% width = average width of the 10%-90% range, as % of the actual load: narrower is sharper.")
    print("Time: " + ", ".join(f"{label} {bt['seconds']:.1f} s" for label, bt in bts.items()) + "\n")

    # 2. Anomaly detection: the week-ahead check.
    ans = {label: anomalies(model, power, args) for label, model in zip(labels, models)}
    calibration = pd.concat({label: an["calibration"] * 100 for label, an in ans.items()}, axis=1)
    calibration.loc["mean"] = calibration.mean()

    days = ans[labels[0]]["actual day"].index
    flags = []
    for code in load.columns:
        frame = pd.DataFrame({"day": days, "country": code})
        for label, an in ans.items():
            hit = an["days"].loc[an["days"]["country"] == code, "day"]
            frame[label] = frame["day"].isin(hit)
        flags.append(frame)
    flags = pd.concat(flags, ignore_index=True)
    flagged = flags[flags[labels].any(axis=1)]
    flagged.assign(day=flagged["day"].dt.strftime("%Y-%m-%d")).to_csv(OUTPUT / "compare-anomalies.csv", index=False)
    plot_compare_anomalies(flags, LOCKDOWNS, OUTPUT / "compare-anomalies.png")

    a, b = labels
    agreement = flags.groupby("country", sort=False).apply(lambda f: pd.Series({
        f"{a} days": f[a].sum(), f"{b} days": f[b].sum(), "both": (f[a] & f[b]).sum(),
        f"only {a}": (f[a] & ~f[b]).sum(), f"only {b}": (f[b] & ~f[a]).sum()}), include_groups=False)
    agreement.loc["total"] = agreement.sum()

    lockdown = {}
    for code, when in LOCKDOWNS.items():
        if code not in load.columns:
            continue
        first_day = pd.Timestamp(when, tz="UTC")
        window = flags[(flags["country"] == code) & (flags["day"] >= first_day)
                       & (flags["day"] < first_day + LOCKDOWN_WINDOW * DAY)]
        lockdown[code] = {f"{label} days": window[label].sum() for label in labels}
        for label in labels:
            hit = window.loc[window[label], "day"]
            lockdown[code][f"{label} first"] = f"{hit.iloc[0]:%m-%d}" if len(hit) else "-"
    lockdown = pd.DataFrame(lockdown).T

    print(f"2. ANOMALY DETECTION: {args.start} .. {days[-1]:%Y-%m-%d}, {args.horizon} hours at a time; a day is "
          f"anomalous when {args.min_hours}+ hours fall outside the 1%-99% range\n")
    print(f"Calibration up to {NORMAL_UNTIL}: % of hours below the 1% line and above the 99% line "
          "(expected: about 1% each)\n")
    print(calibration.round(1).to_string())
    print("\nAnomalous days, and how far the two models agree:\n")
    print(agreement.astype(int).to_string())
    print(f"\nLockdowns: anomalous days in the first {LOCKDOWN_WINDOW} days from the start, and the first one\n")
    print(lockdown.to_string())
    print("\nTime: " + ", ".join(f"{label} {an['seconds']:.1f} s" for label, an in ans.items()))
    print("\nWrote " + ", ".join(str(OUTPUT / f) for f in ("compare-backtest.csv", "compare-backtest.png",
                                                          "compare-anomalies.csv", "compare-anomalies.png")))


def _suffix(model: Model) -> str:
    """Timestral keeps the plain file names; another model adds its name, so it never overwrites them."""
    return "" if isinstance(model, TimestralModel) else f"-{model.label.lower()}"


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
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"one of {', '.join(MODELS)}, or a Hugging Face id (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, mps or cuda (default: cpu)")
    parser.add_argument("--no-holidays", action="store_true", help="don't pass public holidays as a covariate")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("signal", help="plot the raw signal against a fixed threshold and a moving average")
    p.add_argument("--country", default="IT", help="one of DE,FR,IT,ES,GB")
    p.add_argument("--start", default="2020-02-17", help="first day to plot, YYYY-MM-DD")
    p.add_argument("--end", default="2020-03-29", help="last day to plot, YYYY-MM-DD")
    p.add_argument("--threshold", type=float, default=25000, help="MW: hours below this are flagged")
    p.set_defaults(run=run_signal)

    p = commands.add_parser("forecast", help="forecast the days after the last hour in the data")
    p.add_argument("--days", type=int, default=30, help="days to forecast")
    p.set_defaults(run=run_forecast)

    p = commands.add_parser("backtest", help="replay day-ahead forecasts for a past period and score them against the grid operators")
    p.add_argument("--days", type=int, default=30, help="days to hide and forecast")
    p.add_argument("--cutoff", help="day before the first forecast day, YYYY-MM-DD (default: days before the end)")
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

    p = commands.add_parser("compare", help="run the backtest and the anomaly check with two models and compare them")
    p.add_argument("--against", default="chronos-2", help="the second model (the first is --model)")
    p.add_argument("--days", type=int, default=30, help="backtest: days to hide and forecast")
    p.add_argument("--cutoff", help="backtest: day before the first forecast day (default: days before the end)")
    p.add_argument("--start", default="2019-09-30", help="anomalies: first day to check, YYYY-MM-DD (a Monday)")
    p.add_argument("--end", help="anomalies: day after the last day to check (default: end of the data)")
    p.add_argument("--context", type=int, default=CONTEXT, help="hours of history for each rolling forecast")
    p.add_argument("--horizon", type=int, default=168, help="anomalies: hours forecast at a time")
    p.add_argument("--min-hours", type=int, default=6, help="anomalies: hours outside the interval that make a day anomalous")
    p.set_defaults(run=run_compare)

    args = parser.parse_args()
    args.countries = args.countries.split(",")
    OUTPUT.mkdir(exist_ok=True)
    args.run(load_power_data(args.countries), args)


if __name__ == "__main__":
    main()
