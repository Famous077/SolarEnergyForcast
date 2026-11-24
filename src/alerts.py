"""Weeks 9-10: alert engine for grid operators.

Four alerts, each raised from a forecast P = DataFrame[p10, p50, p90] indexed by target time:
  ramp           forecast drop > RAMP_THRESHOLD in one hour that clear-sky output does not explain
  low_output     forecast below LOW_OUTPUT_RATIO x clear-sky output
  low_confidence P10-P90 interval wider than usual
  drift          rolling forecast error well above the error seen in cross-validation
The first three have a matching "actual event", so they are scored with precision/recall/F1.
"""
import numpy as np
import pandas as pd

from config import (DRIFT_FACTOR, DRIFT_WINDOW_D, LARGE_ERROR_FACTOR, LOW_OUTPUT_RATIO,
                    MIN_CLEARSKY_CF, RAMP_THRESHOLD, WIDE_INTERVAL_Q)


def _day_pair(df):
    """Hours where both this hour and the previous one are daytime (a 1-hour change is defined)."""
    return (df["is_day"] == 1) & (df["is_day"].shift(1) == 1)


def weather_ramp(cf: pd.Series, df: pd.DataFrame, cs_max: float | None = None) -> pd.Series:
    """A drop > RAMP_THRESHOLD in one hour, where the clear-sky curve explains less than half of it
    (same definition as the EDA, notebook 03 section 8). Pass cs_max when df is a short window."""
    drop = cf.diff() < -RAMP_THRESHOLD
    clear_sky_drop = df["ghi_clearsky"].diff() / (cs_max or df["ghi_clearsky"].max())
    return drop & (clear_sky_drop > -RAMP_THRESHOLD / 2) & _day_pair(df)


def low_output(cf: pd.Series, df: pd.DataFrame, cs_scale: float) -> pd.Series:
    cf_cs = df["ghi_clearsky"] * cs_scale
    return (cf < LOW_OUTPUT_RATIO * cf_cs) & (cf_cs > MIN_CLEARSKY_CF) & (df["is_day"] == 1)


def raise_alerts(P: pd.DataFrame, df: pd.DataFrame, cs_scale: float, width_limit: float,
                 cv_rmse: float, cs_max: float | None = None) -> pd.DataFrame:
    """Alerts from one horizon's forecasts. width_limit comes from training data (see width_threshold)."""
    day = df["is_day"] == 1
    err = (P["p50"] - df["capacity_factor"]).where(day)
    rolling_rmse = np.sqrt((err ** 2).rolling(f"{DRIFT_WINDOW_D}D", min_periods=24).mean())
    return pd.DataFrame({
        "ramp": weather_ramp(P["p50"], df, cs_max),
        "low_output": low_output(P["p50"], df, cs_scale),
        "low_confidence": ((P["p90"] - P["p10"]) > width_limit) & day,
        "drift": (rolling_rmse > DRIFT_FACTOR * cv_rmse) & day,
    }, index=P.index)


def actual_events(P: pd.DataFrame, df: pd.DataFrame, cs_scale: float, cv_rmse: float) -> pd.DataFrame:
    """Ground truth for the alerts that have one. A 'low confidence' alert is correct when the
    P50 error turns out large (> LARGE_ERROR_FACTOR x CV RMSE)."""
    cf = df["capacity_factor"]
    return pd.DataFrame({
        "ramp": weather_ramp(cf, df),
        "low_output": low_output(cf, df, cs_scale),
        "low_confidence": ((P["p50"] - cf).abs() > LARGE_ERROR_FACTOR * cv_rmse) & (df["is_day"] == 1),
    }, index=P.index)


def width_threshold(P_train: pd.DataFrame, is_day: pd.Series) -> float:
    w = (P_train["p90"] - P_train["p10"])[is_day.reindex(P_train.index) == 1]
    return float(w.quantile(WIDE_INTERVAL_Q))


def score(alerts: pd.Series, events: pd.Series, valid: pd.Series) -> dict:
    """Precision / recall / F1 of one alert type over the rows in `valid`."""
    a, e = alerts[valid].astype(bool), events[valid].astype(bool)
    tp, fp, fn = int((a & e).sum()), int((a & ~e).sum()), int((~a & e).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else np.nan
    return {"events": int(e.sum()), "alerts": int(a.sum()), "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "F1": f1}
