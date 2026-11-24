"""Step 5: causal, horizon-aware feature engineering + time-series splits.

Direct multi-horizon setup: for forecast origin t and horizon h, the target is
capacity_factor(t+h). Lag/rolling features use only data <= t. Calendar/solar-geometry
features at t+h are deterministic (known in advance). Weather at t+h is an optional
"perfect forecast" proxy (upper bound); set weather_mode="lagged" for a realistic variant.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import TEST_FRAC

LAGS = [1, 2, 3, 24, 168]          # relative to origin t (lag 1 == value at t-1... see below)
WINDOWS = {"3h": 3, "24h": 24, "7d": 168}
DETERMINISTIC = ["solar_zenith", "solar_elevation", "solar_azimuth", "ghi_clearsky", "is_day"]


def weather_cols(df):
    cand = ["ghi", "dhi", "dni", "gti", "dti", "temp_c", "humidity_pct", "cloud_cover_pct",
            "wind_speed_kmh", "wind_dir", "rain_mm", "clearsky_index"]
    return [c for c in cand if c in df.columns]


def build_features(df: pd.DataFrame, horizon: int, weather_mode: str = "forecast") -> pd.DataFrame:
    """Rows are indexed by TARGET time (t+h). Returns features + 'y'."""
    h = horizon
    y = df["capacity_factor"]
    X = pd.DataFrame(index=df.index)
    # --- past generation, as seen from origin t = target - h
    for lag in LAGS:
        X[f"cf_lag_{lag}"] = y.shift(h + lag - 1)           # lag 1 -> value at origin t
    base = y.shift(h)                                        # value at origin
    for name, w in WINDOWS.items():
        r = base.rolling(w, min_periods=max(1, w // 2))
        X[f"cf_roll_mean_{name}"] = r.mean(); X[f"cf_roll_std_{name}"] = r.std()
        X[f"cf_roll_min_{name}"] = r.min(); X[f"cf_roll_max_{name}"] = r.max()
    X["kt_at_origin"] = df["clearsky_index"].shift(h)
    # same hour yesterday relative to target (only valid if h <= 24, else shift further)
    X["cf_same_hour_prev_day"] = y.shift(24 * int(np.ceil(h / 24)))
    # --- deterministic features at target time
    for c in DETERMINISTIC:
        X[c] = df[c]
    local = df.index.tz_convert(df.attrs.get("tz", "UTC")) if df.attrs.get("tz") else df.index
    X["hour_sin"] = np.sin(2 * np.pi * local.hour / 24); X["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25); X["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    X["month_sin"] = np.sin(2 * np.pi * local.month / 12); X["month_cos"] = np.cos(2 * np.pi * local.month / 12)
    # --- weather
    for c in weather_cols(df):
        X[f"w_{c}"] = df[c] if weather_mode == "forecast" else df[c].shift(h)
    if "temp_c" in df and "ghi" in df:
        # panel-temperature efficiency proxy (NOCT model, -0.4%/K above 25C)
        t_cell = df["temp_c"] + df["ghi"] / 800 * 20
        X["w_eff_proxy"] = 1 - 0.004 * (t_cell - 25)
        if weather_mode != "forecast":
            X["w_eff_proxy"] = X["w_eff_proxy"].shift(h)
    if "rain_mm" in df:
        X["w_rain_flag"] = (X["w_rain_mm"] > 0).astype(int)
    X["y"] = y
    return X


def chronological_split(X: pd.DataFrame, test_frac=TEST_FRAC):
    cut = X.index[int(len(X) * (1 - test_frac))]
    return X[X.index < cut], X[X.index >= cut]


def rolling_folds(train: pd.DataFrame, n_splits=5, gap=48):
    """Rolling-origin folds inside the training set, with a gap to avoid lag overlap."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap).split(train)


def daytime_trainable(X: pd.DataFrame) -> pd.DataFrame:
    """Night hours are masked from training AND metrics; target must be present."""
    return X[(X["is_day"] == 1) & X["y"].notna()]


def leakage_check(df: pd.DataFrame, horizon: int) -> None:
    """Perturb the future (>= origin+1) and assert no past-based feature changes."""
    X1 = build_features(df, horizon, weather_mode="lagged")
    k = len(df) // 2
    df2 = df.copy()
    df2.iloc[k:, df2.columns.get_loc("capacity_factor")] += 0.5
    df2.iloc[k:, df2.columns.get_loc("clearsky_index")] += 0.5
    X2 = build_features(df2, horizon, weather_mode="lagged")
    # any row whose target time is <= k-1+h only uses origin <= k-1 -> must be identical
    rows = X1.index[: k - 1 + horizon + 1]
    past_cols = [c for c in X1.columns if c not in ("y",) + tuple(DETERMINISTIC)]
    a, b = X1.loc[rows, past_cols], X2.loc[rows, past_cols]
    assert a.fillna(-999).equals(b.fillna(-999)), f"LEAKAGE at horizon {horizon}"
    corr = X1.drop(columns="y").corrwith(X1["y"]).abs().dropna()
    assert (corr < 0.999).all(), f"Suspicious perfect correlation: {corr[corr >= 0.999].index.tolist()}"
