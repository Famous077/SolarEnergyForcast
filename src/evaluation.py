"""Baselines and metrics (reused when the ML models are compared in later stages).

All forecasts are for horizon h: issued at origin t, predicting capacity factor at t+h.
Series are indexed by TARGET time, so a value at index T was forecast at T-h.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- metrics
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_res, ss_tot = (err ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return {"RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(np.abs(err).mean()),
            "MBE": float(err.mean()), "R2": float(1 - ss_res / ss_tot), "n": int(len(y))}


def pinball(y, q, alpha) -> float:
    """Quantile (pinball) loss; lower is better."""
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def interval_metrics(y, lo, hi, nominal=0.8) -> dict:
    """Coverage and mean width of a central prediction interval (P10-P90 -> nominal 80 %)."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "nominal": nominal,
            "width": float(np.mean(hi - lo))}


def skill(rmse_model, rmse_ref) -> float:
    """Skill score = 1 - RMSE_model / RMSE_reference (>0 means better than the reference)."""
    return 1 - rmse_model / rmse_ref


# ---------------------------------------------------------------- clear-sky CF
def fit_clearsky_scale(train: pd.DataFrame) -> float:
    """CF per W/m2 of clear-sky GHI, fitted on clear daytime training hours (kt > 0.9)."""
    m = (train["is_day"] == 1) & (train["clearsky_index"] > 0.9) & (train["ghi_clearsky"] > 200)
    return float((train.loc[m, "capacity_factor"] / train.loc[m, "ghi_clearsky"]).median())


# ---------------------------------------------------------------- baselines
def persistence(df, h):
    """CF(t+h) = CF(t)."""
    return df["capacity_factor"].shift(h)


def smart_persistence(df, h, scale):
    """Hold the clear-sky index of the last daylight hour and rescale by clear-sky output at t+h."""
    cf_cs = df["ghi_clearsky"] * scale
    kc = (df["capacity_factor"] / cf_cs).where(cf_cs > 0.05).clip(0, 1.2)
    kc_origin = kc.ffill(limit=18).shift(h)      # carry the last daylight value over the night
    return (kc_origin * cf_cs).clip(0, 1.05)


def seasonal_naive(df, h):
    """Same hour on the most recent day that is known at the origin."""
    return df["capacity_factor"].shift(24 * int(np.ceil(h / 24)))


def climatology(df, train, tz):
    """Mean CF per (month, local hour), from training data only."""
    loc_tr = train.index.tz_convert(tz)
    table = train["capacity_factor"].groupby([loc_tr.month, loc_tr.hour]).mean()
    loc = df.index.tz_convert(tz)
    keys = pd.MultiIndex.from_arrays([loc.month, loc.hour])
    return pd.Series(table.reindex(keys).to_numpy(), index=df.index)


def all_baselines(df, train, h, tz, scale) -> pd.DataFrame:
    return pd.DataFrame({
        "persistence": persistence(df, h),
        "smart_persistence": smart_persistence(df, h, scale),
        "seasonal_naive": seasonal_naive(df, h),
        "climatology": climatology(df, train, tz),
    })


def evaluate(df, preds: pd.DataFrame, mask) -> pd.DataFrame:
    """Score every column of preds on the SAME rows (mask & all present): daytime, test set."""
    rows = mask & df["capacity_factor"].notna() & preds.notna().all(axis=1)
    y = df.loc[rows, "capacity_factor"]
    return pd.DataFrame({c: metrics(y, preds.loc[rows, c]) for c in preds}).T
