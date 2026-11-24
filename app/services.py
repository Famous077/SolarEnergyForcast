"""Everything the pages need, computed once per (site, replay hour) and cached.

Pages never touch the models directly. They call get_context() and read from the dict.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import LOW_OUTPUT_RATIO, SITES  # noqa: E402
from features import build_features  # noqa: E402
from io_utils import ROOT, load_processed  # noqa: E402
from models import predict_quantiles  # noqa: E402
from pipeline import ForecastPipeline  # noqa: E402

SITE_NAMES = {"primary": "Primary plant (Alice Springs)", "site_a": "Site A (Black Sea coast)"}
NEAR = [1, 2, 3, 4, 5, 6]

# How each model input reads to an operator. Anything not listed counts as sun position / time.
def feature_group(col: str) -> str:
    if col.startswith(("cf_", "kt_at_origin")):
        return "recent output"
    if col.startswith("w_"):
        return "observed weather"
    return "sun position and time of day"


# ------------------------------------------------------------------ cached loaders
@st.cache_resource
def pipeline() -> ForecastPipeline:
    return ForecastPipeline.load()


@st.cache_data
def data(site: str) -> pd.DataFrame:
    return load_processed(site)


@st.cache_data
def test_scores():
    res = pd.read_csv(ROOT / "reports" / "lgbm_results.csv")
    q = pd.read_csv(ROOT / "reports" / "lgbm_quantile_results.csv")
    al = pd.read_csv(ROOT / "reports" / "alert_results.csv")
    return res, q, al


@st.cache_resource
def explainer(site: str, h: int):
    return shap.TreeExplainer(pipeline().sites[site]["models"][h]["p50"])


# ------------------------------------------------------------------ main context
@st.cache_data(show_spinner="Running forecast...")
def get_context(site: str, origin: pd.Timestamp) -> dict:
    pipe, d = pipeline(), data(site)
    info = pipe.sites[site]
    cap = info["capacity_kw"]
    history = d.loc[:origin]
    fc = pipe.forecast(site, history)
    prev = pipe.forecast(site, history.iloc[:-1]).iloc[0]      # what we expected for "now", an hour ago
    now = history.iloc[-1]
    ext = pipe._extend(site, history)
    clear_sky_kw = (ext["ghi_clearsky"] * info["cs_scale"] * cap).clip(upper=cap).iloc[-(48 + 48):]
    return {
        "site": site, "tz": SITES[site]["tz"], "origin": origin, "cap": cap, "info": info,
        "history": history, "fc": fc, "now": now, "prev_fc": prev,
        "actual_after": d["power_kw"].loc[origin:fc.index[-1]].iloc[1:],
        "recent_rmse": pipe.recent_rmse(site, history),
        "drivers": drivers(site, history, fc),
        "health": health(site, history, info),
        "clear_sky_kw": clear_sky_kw,
    }


# ------------------------------------------------------------------ explanations
def drivers(site, history, fc) -> pd.DataFrame:
    """SHAP contributions (kW) of each input group for the 1-6 h forecasts.
    'base' is the model's average daytime forecast; base + groups = P50."""
    pipe = pipeline()
    cap = pipe.sites[site]["capacity_kw"]
    df = pipe._extend(site, history)
    rows = []
    for h in NEAR:
        t = history.index[-1] + pd.Timedelta(hours=h)
        X = build_features(df.loc[:t].tail(400), h, weather_mode="lagged").loc[[t]].drop(columns="y")
        if X["is_day"].iloc[0] == 0:
            continue
        ex = explainer(site, h)
        sv = pd.Series(ex.shap_values(X)[0], index=X.columns)
        g = sv.groupby(feature_group).sum() * cap
        rows.append({"target": t, "horizon_h": h, "base": float(np.ravel(ex.expected_value)[0]) * cap, **g})
    return pd.DataFrame(rows).set_index("target") if rows else pd.DataFrame()


def driver_sentence(row: pd.Series, p50_kw: float) -> str:
    groups = row.drop(["horizon_h", "base"]).sort_values(key=abs, ascending=False)
    top, val = groups.index[0], groups.iloc[0]
    diff = p50_kw - row["base"]
    direction = "above" if diff > 0 else "below"
    s = (f"{p50_kw:.1f} kW is {abs(diff):.1f} kW {direction} what the model expects on an average daytime hour. "
         f"The biggest reason is **{top}** ({val:+.1f} kW)")
    if len(groups) > 1 and abs(groups.iloc[1]) > 0.5:
        s += f", then {groups.index[1]} ({groups.iloc[1]:+.1f} kW)"
    return s + "."


# ------------------------------------------------------------------ data and model health
def health(site, history, info) -> dict:
    last24 = history.tail(24)
    day = last24["is_day"] == 1
    missing = int(last24.loc[day, "capacity_factor"].isna().sum())
    flat = int(last24.loc[day, "flag_flatline"].sum())
    last_good = history["capacity_factor"].last_valid_index()
    stale_h = (history.index[-1] - last_good) / pd.Timedelta(hours=1) if last_good is not None else np.inf
    return {"missing_day_hours_24h": missing, "flatline_hours_24h": flat, "hours_since_data": stale_h,
            "last_good": last_good}


@st.cache_data(show_spinner="Checking recent accuracy...")
def error_trend(site: str, origin: pd.Timestamp, days: int = 30) -> pd.DataFrame:
    """Daily daytime RMSE (% of capacity) of the 1 h model and of plain persistence, last `days` days."""
    pipe, d = pipeline(), data(site)
    h = d.loc[:origin].tail(days * 24 + 400).copy()
    h.attrs["tz"] = SITES[site]["tz"]
    X = build_features(h, 1, weather_mode="lagged").tail(days * 24)
    p50 = predict_quantiles(pipe.sites[site]["models"][1], X)["p50"]
    pers = h["capacity_factor"].shift(1).reindex(X.index)
    ok = (X["is_day"] == 1) & X["y"].notna() & pers.notna()
    day = X.index[ok].tz_convert(SITES[site]["tz"]).date
    err = pd.DataFrame({"model": (p50 - X["y"])[ok] ** 2, "persistence": (pers - X["y"])[ok] ** 2})
    return np.sqrt(err.groupby(day).mean()) * 100


# ------------------------------------------------------------------ alerts as operator cards
SEVERITY = {"ramp": "High", "low_output": "Medium", "low_confidence": "Caution", "drift": "Caution",
            "data": "Caution"}
SEV_ORDER = {"High": 0, "Medium": 1, "Caution": 2}


def _track_record(site, alert, h):
    _, _, al = test_scores()
    r = al[(al["site"] == site) & (al["alert"] == alert) & (al["horizon_h"] == (1 if h <= 6 else 24))]
    if r.empty or pd.isna(r["precision"].iloc[0]):
        return "No test record for this alert at this horizon."
    r = r.iloc[0]
    return (f"In testing, {r['precision']:.0%} of these alerts were correct and they caught "
            f"{r['recall']:.0%} of real events ({int(r['events'])} events).")


def alert_cards(ctx) -> list[dict]:
    fc, tz, cap, site = ctx["fc"], ctx["tz"], ctx["cap"], ctx["site"]
    info = ctx["info"]
    hhmm = lambda t: t.tz_convert(tz).strftime("%a %H:%M")
    cards = []

    hit = fc[fc["ramp"]]
    if len(hit):
        t = hit.index[0]
        before = fc["p50_kw"].shift(1).get(t, ctx["now"]["power_kw"])
        if pd.isna(before):
            before = ctx["now"]["power_kw"]
        cards.append({
            "kind": "ramp", "title": "Sudden drop in output", "when": hhmm(t),
            "impact": f"From about {before:.1f} kW to {hit['p50_kw'].iloc[0]:.1f} kW within one hour.",
            "why": "The forecast falls by more than 20 % of capacity in an hour, and the sun's path "
                   "only explains a small part of it.",
            "record": _track_record(site, "ramp", int(hit["horizon_h"].iloc[0])) +
                      " Treat this alert as experimental.",
            "check": "Have reserve ready for that hour. If you have sky-camera or satellite images, "
                     "check for incoming cloud."})

    hit = fc[fc["low_output"]]
    if len(hit):
        t = hit.index[0]
        cs_kw = float(ctx["clear_sky_kw"].get(t, np.nan))
        cards.append({
            "kind": "low_output", "title": "Low generation expected",
            "when": ", ".join(hhmm(x) for x in hit.index[:4]) + (" ..." if len(hit) > 4 else ""),
            "impact": f"{hit['p50_kw'].iloc[0]:.1f} kW expected at {hhmm(t)}, against about "
                      f"{cs_kw:.1f} kW on a clear day.",
            "why": f"Forecast is under {LOW_OUTPUT_RATIO:.0%} of clear-sky output.",
            "record": _track_record(site, "low_output", int(hit["horizon_h"].iloc[0])),
            "check": "Plan for the shortfall. If the sky is actually clear, look for an inverter "
                     "fault or curtailment instead."})

    hit = fc[fc["low_confidence"]]
    if len(hit):
        r = hit.iloc[0]
        cards.append({
            "kind": "low_confidence", "title": "Forecast less certain than usual",
            "when": ", ".join(hhmm(x) for x in hit.index[:4]) + (" ..." if len(hit) > 4 else ""),
            "impact": f"Likely range at {hhmm(hit.index[0])} is {r['p10_kw']:.1f} to {r['p90_kw']:.1f} kW.",
            "why": "The range is wider than 90 % of forecasts the model made on training data.",
            "record": _track_record(site, "low_confidence", int(r["horizon_h"])),
            "check": "Don't plan on the middle value alone for these hours. Keep some margin."})

    if fc["drift"].any():
        cards.append({
            "kind": "drift", "title": "Forecast error higher than normal",
            "when": "past 7 days",
            "impact": f"1-hour forecast error is {ctx['recent_rmse'] * 100:.1f} % of capacity, against "
                      f"{info['cv_rmse'][1] * 100:.1f} % in validation.",
            "why": "Recent error is more than 1.5 times the validation error.",
            "record": "No labelled test data for this alert.",
            "check": "Check the plant and sensor data for faults. If it lasts, the model needs retraining."})

    hl = ctx["health"]
    if hl["hours_since_data"] > 1 or hl["missing_day_hours_24h"] or hl["flatline_hours_24h"]:
        bits = []
        if hl["hours_since_data"] > 1:
            bits.append(f"last valid reading {hl['hours_since_data']:.0f} h ago")
        if hl["missing_day_hours_24h"]:
            bits.append(f"{hl['missing_day_hours_24h']} daytime hours missing in the last 24 h")
        if hl["flatline_hours_24h"]:
            bits.append(f"{hl['flatline_hours_24h']} hours stuck at one value")
        cards.append({
            "kind": "data", "title": "Input data problem", "when": "now",
            "impact": "; ".join(bits).capitalize() + ".",
            "why": "The model uses the last few hours of output, so gaps make the next hours less reliable.",
            "record": "", "check": "Check the plant's data link and meter."})

    for c in cards:
        c["severity"] = SEVERITY[c["kind"]]
    return sorted(cards, key=lambda c: SEV_ORDER[c["severity"]])


def interval_hit_rate(site: str) -> float:
    """Share of test hours where the actual fell inside P10-P90 (1-6 h, realistic model)."""
    _, q, _ = test_scores()
    r = q[(q["site"] == site) & (q["mode"] == "lagged") & (q["horizon_h"] <= 6)]
    return float(r["coverage"].mean())


# ------------------------------------------------------------------ one-line situation summary
def situation(ctx, hours: int = 3) -> str:
    """What an operator would say out loud: where output is heading over the next few hours and why."""
    fc, tz, now_kw = ctx["fc"], ctx["tz"], ctx["now"]["power_kw"]
    near = fc[fc["horizon_h"] <= hours]
    hm = lambda t: t.tz_convert(tz).strftime("%H:%M")
    if pd.isna(now_kw):
        now_kw = 0.0

    if near["p50_kw"].max() < 0.5 and now_kw < 0.5:
        day = fc[fc["p50_kw"] > 0.5]
        nxt = f" The day-ahead forecast for {day.index[0].tz_convert(tz):%a %H:%M} is {day['p50_kw'].iloc[0]:.1f} kW." \
            if len(day) else ""
        return f"The plant is dark and stays dark for the next {hours} hours.{nxt}"

    end = near["p50_kw"].iloc[-1]
    change = end - now_kw
    cs = ctx["clear_sky_kw"]
    cs_change = cs.get(near.index[-1], np.nan) - cs.get(ctx["origin"], np.nan)
    if abs(change) < 0.05 * ctx["cap"]:
        s = f"Output should stay close to {now_kw:.1f} kW until {hm(near.index[-1])}."
    else:
        verb = "rise" if change > 0 else "fall"
        s = f"Output is expected to {verb} from {now_kw:.1f} kW now to about {end:.1f} kW by {hm(near.index[-1])}."
        if np.sign(cs_change) == np.sign(change) and abs(cs_change) >= 0.6 * abs(change):
            s += " Most of this is the sun's normal daily path."
        elif change < 0:
            s += " That is more than the sun's path explains, so cloud or a plant issue is likely."
    steps = pd.concat([pd.Series([now_kw], index=[ctx["origin"]]), near["p50_kw"]]).diff().iloc[1:]
    if steps.min() < -0.1 * ctx["cap"]:
        t = steps.idxmin()
        s += f" The steepest drop is expected in the hour to {hm(t)} ({steps.min():.1f} kW)."
    return s
