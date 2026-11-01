"""Forecast and find anomalies in European electricity demand with time series foundation models.

    uv run main.py signal     # the raw signal, with a fixed threshold and a moving average (no model)
    uv run main.py forecast   # the next 30 days, hour by hour
    uv run main.py anomalies  # days that fell outside the week-ahead forecast, e.g. COVID lockdowns
    uv run main.py compare    # forecasts and anomaly checks with every model, scored side by side

The models are Timestral (default), Chronos-2 and TimesFM 3. Add --model chronos-2 or
--model timesfm-3 to a command to use another model.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import DEFAULT_COUNTRIES, LOCKDOWNS, country_name, holiday_indicator, load_power_data
from forecasting import DECILES, DEFAULT_MODEL, MODELS, Model, QuantileForecast, forecast, future_index, \
    load_forecaster, rolling
from plots import (plot_anomalies, plot_compare_anomalies, plot_compare_forecast, plot_compare_week, plot_forecast,
                   plot_signal)

OUTPUT = Path("output")
DAY = pd.Timedelta(days=1)
CONTEXT = 2048  # hours of history for each rolling (week-ahead) forecast, ~12 weeks
NORMAL_UNTIL = "2020-02-29"  # last day before the pandemic: the forecasts are scored and calibrated up to here
LOCKDOWN_WINDOW = 14  # days from the start of each lockdown that `compare` checks
SAMPLE_WEEK = "2019-11-18"  # a usual week, for the chart of the week-ahead forecasts
COMPARE_MODELS = "timestral,chronos-2,timesfm-3"


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


def make_forecast(model: Model, power, args: argparse.Namespace) -> QuantileForecast:
    """Forecast `args.days` days after the last hour in the data, and write the CSV file and the chart."""
    history = power.load
    horizon = args.days * 24
    known = holidays_for(history.index.append(future_index(history, horizon)), args)
    fc = forecast(model, history, horizon, known)
    table = pd.concat({"low": fc.low, "median": fc.median, "high": fc.high, "mean": fc.mean}, axis=1).round(0)
    name = f"forecast{_suffix(model)}"
    table.to_csv(OUTPUT / f"{name}.csv")
    plot_forecast(history, fc, OUTPUT / f"{name}.png", label=model.label)
    return fc


def run_forecast(power, args: argparse.Namespace):
    history = power.load
    horizon = args.days * 24
    model = load_forecaster(args.model, args.device)
    fc = make_forecast(model, power, args)
    name = f"forecast{_suffix(model)}"

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


def week_ahead(model: Model, power, args: argparse.Namespace) -> tuple[QuantileForecast, float]:
    """Forecast every block of `args.horizon` hours from the `args.context` hours before it, and time it."""
    load = power.load
    end = pd.Timestamp(args.end, tz="UTC") if args.end else load.index[-1].normalize() + DAY
    clock = time.perf_counter()
    fc = rolling(model, load, args.start, end, horizon=args.horizon, context=args.context,
                 known=holidays_for(load.index, args))
    return fc, time.perf_counter() - clock


def score_forecasts(load: pd.DataFrame, fc: QuantileForecast) -> pd.DataFrame:
    """Forecast quality per country, on the usual days before the pandemic.

    - MAPE: how far the median is from the actual demand, in %. Lower is better.
    - WQL: weighted quantile loss over the 9 deciles, in %. It scores the whole forecast
      distribution, not only the median. Lower is better.
    - in 80%: % of hours between the 10% and the 90% quantile. Best near 80.
    - 80% width: mean width of that range, as % of the actual demand. Narrower is sharper.
    """
    hours = fc.median.loc[:NORMAL_UNTIL].index
    actual = load.loc[hours]
    q = {level: frame.loc[hours] for level, frame in fc.levels.items()}
    # Pinball loss of each decile, summed over the deciles and the hours, divided by the total demand.
    loss = sum(np.maximum(level * (actual - q[level]), (level - 1) * (actual - q[level])) for level in DECILES)
    return pd.DataFrame({
        "MAPE %": mape(q[0.5], load),
        "WQL %": 2 * loss.sum() / (len(DECILES) * actual.sum()) * 100,
        "in 80% range %": ((actual >= q[0.1]) & (actual <= q[0.9])).mean() * 100,
        "80% width %": ((q[0.9] - q[0.1]) / actual).mean() * 100,
    })


def find_anomalies(load: pd.DataFrame, fc: QuantileForecast, min_hours: int) -> tuple[pd.DataFrame, dict]:
    """Days with `min_hours` or more hours outside the low..high interval, and the daily totals."""
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
        flagged = daily[(daily["hours below"] + daily["hours above"]) >= min_hours].copy()
        flagged["kind"] = np.where(flagged["hours below"] >= flagged["hours above"], "drop", "spike")
        rows.append(flagged)
    days = pd.concat(rows).round(1).rename_axis("day").reset_index()
    # Calibration: before the pandemic, ~1% of hours should be above the 99% line and ~1% below the 1% line.
    normal = slice(None, NORMAL_UNTIL)
    calibration = pd.DataFrame({"below": (actual.loc[normal] < fc.low.loc[normal]).mean() * 100,
                                "above": (actual.loc[normal] > fc.high.loc[normal]).mean() * 100})
    return days, {"actual day": actual_day, "expected day": expected_day, "calibration": calibration}


def write_anomalies(model: Model, load: pd.DataFrame, fc: QuantileForecast, args: argparse.Namespace):
    days, daily = find_anomalies(load, fc, args.min_hours)
    name = f"anomalies-{args.horizon}h{_suffix(model)}"
    days.to_csv(OUTPUT / f"{name}.csv", index=False)
    plot_anomalies(daily["actual day"], daily["expected day"], days, LOCKDOWNS, args.horizon, OUTPUT / f"{name}.png",
                   label=model.label)
    return days, daily, name


def can_find_anomalies(model: Model) -> bool:
    return model.has_levels(0.01, 0.99)


def run_anomalies(power, args: argparse.Namespace):
    model = load_forecaster(args.model, args.device)
    if not can_find_anomalies(model):
        raise SystemExit(f"{model.label} gives only the quantiles {model.levels.tolist()}: it has no 1% and 99% "
                         "quantile, so it cannot do this anomaly check. Use --model timestral or chronos-2.")
    fc, _ = week_ahead(model, power, args)
    days, daily, name = write_anomalies(model, power.load, fc, args)

    print(f"Checked {fc.median.index[0]:%Y-%m-%d} .. {fc.median.index[-1]:%Y-%m-%d} (UTC) with {model.label}, "
          f"{args.horizon} hours at a time: each block forecast at its start, from the {args.context} hours before it.\n")
    if len(daily["calibration"]):
        overall = daily["calibration"].mean()
        print(f"Calibration up to {NORMAL_UNTIL}: {overall['above']:.1f}% of hours above the {fc.high_level:.0%} line, "
              f"{overall['below']:.1f}% below the {fc.low_level:.0%} line (expected: about 1% each).\n")
    print(_episodes(days).to_string(index=False))
    print(f"\nA day is anomalous when at least {args.min_hours} of its hours fell outside the "
          f"{fc.low_level:.0%}..{fc.high_level:.0%} interval. Consecutive anomalous days form one episode.")
    print(f"Wrote {OUTPUT / name}.csv ({len(days)} days) and {OUTPUT / name}.png")


def run_compare(power, args: argparse.Namespace):
    """Run the forecast and the week-ahead check with each model, score them, and say which is better."""
    load = power.load
    results = {}
    for name in args.models.split(","):
        model = load_forecaster(name, args.device)
        print(f"{model.label}: 30-day forecast, then week-ahead forecasts from {args.start} ...", flush=True)
        result = {"forecast": make_forecast(model, power, args)}
        result["week"], result["seconds"] = week_ahead(model, power, args)
        result["score"] = score_forecasts(load, result["week"])
        if can_find_anomalies(model):
            result["days"], result["daily"], _ = write_anomalies(model, load, result["week"], args)
        results[model.label] = result
        del model  # free the memory before the next model loads
    labels = list(results)
    finders = [label for label in labels if "days" in results[label]]

    # 1. Forecasting: every model, scored on the same hours with the same deciles.
    forecasting = pd.concat({label: r["score"] for label, r in results.items()}, axis=1)
    forecasting.loc["mean"] = forecasting.mean()
    forecasting.round(2).rename_axis("country").to_csv(OUTPUT / "compare-forecasting.csv")
    plot_compare_forecast(load, {label: r["forecast"] for label, r in results.items()},
                          OUTPUT / "compare-forecast.png")
    plot_compare_week(load, {label: r["week"] for label, r in results.items()}, SAMPLE_WEEK,
                      OUTPUT / "compare-week.png")

    first = results[labels[0]]["week"].median.index
    print(f"\n1. FORECASTING: week-ahead forecasts from {args.context} hours of history, scored on the usual days "
          f"{first[0]:%Y-%m-%d} .. {NORMAL_UNTIL}\n")
    # Baseline that any model must beat: "each hour is the same as one week earlier".
    hours = first[first <= pd.Timestamp(NORMAL_UNTIL, tz="UTC") + DAY - pd.Timedelta(hours=1)]
    baseline = mape(load.shift(7 * 24).loc[hours], load)
    for metric in forecasting.columns.levels[1]:
        table = forecasting.xs(metric, axis=1, level=1)
        if metric == "MAPE %":
            table = table.assign(**{"last week (baseline)": [*baseline, baseline.mean()]})
        print(f"{metric}:")
        print(table.round(2).to_string(), "\n")

    # 2. Anomaly detection: only the models that give the 1% and 99% quantiles.
    calibration, flags, lockdown = None, None, None
    if finders:
        calibration = pd.concat({label: results[label]["daily"]["calibration"] for label in finders}, axis=1)
        calibration.loc["mean"] = calibration.mean()
        days = results[finders[0]]["daily"]["actual day"].index
        flags = pd.concat([pd.DataFrame({"day": days, "country": code, **{
            label: days.isin(results[label]["days"].loc[results[label]["days"]["country"] == code, "day"])
            for label in finders}}) for code in load.columns], ignore_index=True)
        flagged = flags[flags[finders].any(axis=1)]
        flagged.assign(day=flagged["day"].dt.strftime("%Y-%m-%d")).to_csv(OUTPUT / "compare-anomalies.csv",
                                                                          index=False)
        plot_compare_anomalies(flags, LOCKDOWNS, OUTPUT / "compare-anomalies.png")
        lockdown = _lockdown_hits(flags, finders, load.columns)

        print(f"2. ANOMALY DETECTION: {args.start} .. {days[-1]:%Y-%m-%d}, {args.horizon} hours at a time; a day is "
              f"anomalous when {args.min_hours}+ hours fall outside the 1%-99% range\n")
        print(f"Calibration up to {NORMAL_UNTIL}: % of hours below the 1% line and above the 99% line "
              "(expected: about 1% each)\n")
        print(calibration.round(1).to_string())
        normal = flags["day"] <= pd.Timestamp(NORMAL_UNTIL, tz="UTC")
        counts = flags.groupby("country", sort=False)[finders].sum()
        counts.loc["total"] = counts.sum()
        counts["all models"] = flags[flags[finders].all(axis=1)].groupby("country").size().reindex(counts.index)
        counts.loc["total", "all models"] = counts["all models"].iloc[:-1].sum()
        print("\nAnomalous days, and the days that all of these models found:\n")
        print(counts.fillna(0).astype(int).to_string())
        print(f"\nLockdowns: anomalous days in the first {LOCKDOWN_WINDOW} days from the start, and the first one\n")
        print(lockdown.to_string())
    skipped = [label for label in labels if label not in finders]
    if skipped:
        print(f"\nNo anomaly check for {', '.join(skipped)}: no 1% and 99% quantile.")

    # 3. Which model is better.
    summary = _summary(results, labels, finders, flags, normal if finders else None, lockdown)
    summary.insert(len(labels), "last week (baseline)", "")
    summary.loc[("forecasting", "MAPE % of the median (lower is better)"), "last week (baseline)"] = \
        f"{baseline.mean():.2f}"
    summary.to_csv(OUTPUT / "compare-summary.csv")
    print("\n3. SUMMARY: the best value in each row is the winner of that row\n")
    print(summary.to_string())
    print("\nWrote " + ", ".join(str(OUTPUT / f) for f in (
        "compare-summary.csv", "compare-forecasting.csv", "compare-forecast.png", "compare-week.png",
        *(("compare-anomalies.csv", "compare-anomalies.png") if finders else ()))))


def _lockdown_hits(flags: pd.DataFrame, labels: list[str], countries) -> pd.DataFrame:
    """For each lockdown: the anomalous days in its first LOCKDOWN_WINDOW days, and the first one, per model."""
    rows = {}
    for code, when in LOCKDOWNS.items():
        if code not in countries:
            continue
        start = pd.Timestamp(when, tz="UTC")
        window = flags[(flags["country"] == code) & (flags["day"] >= start) & (flags["day"] < start + LOCKDOWN_WINDOW * DAY)]
        rows[code] = {}
        for label in labels:
            hit = window.loc[window[label], "day"]
            rows[code][f"{label} days"] = int(hit.size)
            rows[code][f"{label} first"] = f"{hit.iloc[0]:%m-%d}" if len(hit) else "-"
    return pd.DataFrame(rows).T


def _summary(results: dict, labels: list[str], finders: list[str], flags, normal, lockdown) -> pd.DataFrame:
    """One row per criterion, one column per model, and the winner of each row.

    For each row the rule is: lowest, highest, or nearest to a target value.
    """
    rows = []

    def row(section, criterion, values: dict, rule, target=None, fmt="{:.2f}"):
        known = {label: v for label, v in values.items() if v is not None and not pd.isna(v)}
        if rule == "low":
            best = min(known, key=known.get)
        elif rule == "high":
            best = max(known, key=known.get)
        else:
            best = min(known, key=lambda label: abs(known[label] - target))
        ties = [label for label in known if np.isclose(known[label], known[best])]
        cells = {label: fmt.format(values[label]) if label in known else "not possible" for label in labels}
        rows.append({"section": section, "criterion": criterion, **cells,
                     "better": "tie" if len(ties) > 1 else best})

    mean = {label: r["score"].mean() for label, r in results.items()}
    row("forecasting", "WQL % (main score, lower is better)", {l: mean[l]["WQL %"] for l in labels}, "low")
    row("forecasting", "MAPE % of the median (lower is better)", {l: mean[l]["MAPE %"] for l in labels}, "low")
    row("forecasting", "hours in the 80% range (best: 80)", {l: mean[l]["in 80% range %"] for l in labels},
        "near", 80, "{:.1f}")
    row("forecasting", "width of the 80% range % (lower is better)", {l: mean[l]["80% width %"] for l in labels},
        "low")
    row("cost", "seconds for all week-ahead forecasts (CPU)", {l: results[l]["seconds"] for l in labels}, "low",
        fmt="{:.0f}")
    if finders:
        outside = {l: results[l]["daily"]["calibration"].sum(axis=1).mean() if l in finders else None for l in labels}
        row("anomalies", "hours outside 1%-99% before the pandemic % (main score, best: 2)", outside, "near", 2.0,
            "{:.1f}")
        before = {l: int(flags.loc[normal, l].sum()) if l in finders else None for l in labels}
        row("anomalies", "anomalous days before the pandemic (fewer = fewer false alerts)", before, "low", fmt="{:d}")
        found = {l: int(lockdown[f"{l} days"].sum()) if l in finders else None for l in labels}
        row("anomalies", f"lockdown days found, first {LOCKDOWN_WINDOW} days (more is better)", found, "high",
            fmt="{:d}")
        countries = {l: int((lockdown[f"{l} days"] > 0).sum()) if l in finders else None for l in labels}
        row("anomalies", "lockdowns found, of 5 countries (more is better)", countries, "high", fmt="{:d}")
    return pd.DataFrame(rows).set_index(["section", "criterion"])


def _suffix(model: Model) -> str:
    """Timestral keeps the plain file names; another model adds its name, so it never overwrites them."""
    return "" if model.label == "Timestral" else "-" + model.label.lower().replace(" ", "-")


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

    def week_ahead_options(p):
        p.add_argument("--start", default="2019-09-30", help="first day to check, YYYY-MM-DD (a Monday)")
        p.add_argument("--end", help="day after the last day to check (default: end of the data)")
        p.add_argument("--context", type=int, default=CONTEXT, help="hours of history for each rolling forecast")
        p.add_argument("--horizon", type=int, default=168,
                       help="hours forecast at a time: 168 = week-ahead (sees shifts that build up over days), "
                            "24 = day-ahead (sharp one-day events)")
        p.add_argument("--min-hours", type=int, default=6, help="hours outside the interval that make a day anomalous")

    p = commands.add_parser("anomalies", help="flag days that fell outside the week-ahead forecast")
    week_ahead_options(p)
    p.set_defaults(run=run_anomalies)

    p = commands.add_parser("compare", help="forecast and check for anomalies with several models, and score them")
    p.add_argument("--models", default=COMPARE_MODELS, help=f"comma-separated (default: {COMPARE_MODELS})")
    p.add_argument("--days", type=int, default=30, help="days to forecast after the end of the data")
    week_ahead_options(p)
    p.set_defaults(run=run_compare)

    args = parser.parse_args()
    args.countries = args.countries.split(",")
    OUTPUT.mkdir(exist_ok=True)
    args.run(load_power_data(args.countries), args)


if __name__ == "__main__":
    main()
