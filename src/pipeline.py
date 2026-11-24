"""Weeks 11-12: the deployable forecaster.

ForecastPipeline bundles, per site, the realistic (`lagged`) LightGBM quantile models for every
horizon plus the thresholds the alert engine needs. forecast() only receives history up to the
origin, so nothing after the origin can leak in, whether the data is live or a replay.

Build once (about 3 minutes):  python src/pipeline.py   ->  models/forecaster.pkl
"""
import json

import joblib
import numpy as np
import pandas as pd

from alerts import raise_alerts, width_threshold
from config import SITES, TEST_FRAC
from evaluation import fit_clearsky_scale
from features import build_features
from io_utils import ROOT, load_processed
from models import predict_quantiles, train_from_params
from solar_geometry import add_solar_features

HORIZONS = [1, 2, 3, 4, 5, 6, 24, 36, 48]
PKL = ROOT / "models" / "forecaster.pkl"
DRIFT_LOOKBACK_H = 7 * 24


class ForecastPipeline:
    def __init__(self):
        self.sites = {}

    def fit_site(self, site: str, df: pd.DataFrame, params: dict):
        cut = df.index[int(len(df) * (1 - TEST_FRAC))]
        models, limits, cv = {}, {}, {}
        for h in HORIZONS:
            p = params[f"{site}|lagged|{h}"]
            models[h], X = train_from_params(df, h, "lagged", p)
            P = predict_quantiles(models[h], X)
            limits[h] = width_threshold(P[P.index < cut], df["is_day"])
            cv[h] = p["cv_rmse"]
        cap = (df["power_kw"] / df["capacity_factor"]).replace([np.inf, -np.inf], np.nan)
        self.sites[site] = {
            "models": models, "width_limit": limits, "cv_rmse": cv,
            "cs_scale": fit_clearsky_scale(df[df.index < cut]),
            "cs_max": float(df.loc[df.index < cut, "ghi_clearsky"].max()),
            "capacity_kw": float(cap[df["capacity_factor"] > 0.2].median()),
            "test_start": cut, "columns": list(df.columns),
        }

    def _extend(self, site: str, history: pd.DataFrame) -> pd.DataFrame:
        """Append future hourly rows with only the deterministic columns (sun position, clear sky)."""
        meta = SITES[site]
        last = history.index[-1]
        future = pd.DataFrame(index=pd.date_range(last + pd.Timedelta(hours=1), periods=max(HORIZONS),
                                                  freq="h", tz="UTC"))
        future["ghi"] = np.nan
        future = add_solar_features(future, meta["lat"], meta["lon"], meta["altitude"], ghi_col="ghi",
                                    offset_min=meta["solar_offset_min"])
        out = pd.concat([history, future.reindex(columns=history.columns)])
        out.attrs["tz"] = meta["tz"]
        return out

    def forecast(self, site: str, history: pd.DataFrame) -> pd.DataFrame:
        """Forecast every horizon from the last timestamp in `history`.
        Returns one row per horizon: target time, P10/P50/P90 (CF and kW) and alert flags."""
        s = self.sites[site]
        df = self._extend(site, history)
        origin = history.index[-1]
        rows = []
        for h in HORIZONS:
            target = origin + pd.Timedelta(hours=h)
            X = build_features(df.loc[:target].tail(400), h, weather_mode="lagged").loc[[target]]
            P = predict_quantiles(s["models"][h], X).iloc[0]
            rows.append({"horizon_h": h, "target": target, **P.to_dict()})
        out = pd.DataFrame(rows).set_index("target")
        for q in ("p10", "p50", "p90"):
            out[f"{q}_kw"] = out[q] * s["capacity_kw"]
        out = out.join(self._alerts(site, df, out, history))
        return out

    def _alerts(self, site, df, out, history) -> pd.DataFrame:
        """Alerts per horizon. Ramp only makes sense between consecutive hours, so it is computed on
        an hourly grid (1-6 h) and is NaN-free only there. Drift comes from the last 7 days of h=1."""
        s = self.sites[site]
        grid = out[["p10", "p50", "p90"]].reindex(pd.date_range(history.index[-1], out.index[-1],
                                                                  freq="h", tz="UTC"))
        grid.iloc[0] = history["capacity_factor"].iloc[-1]          # value at origin starts the ramp
        flags = []
        for h, t in zip(out["horizon_h"], out.index):
            a = raise_alerts(grid, df.reindex(grid.index), s["cs_scale"], s["width_limit"][h],
                             s["cv_rmse"][h], cs_max=s["cs_max"]).loc[t]
            flags.append(a.drop("drift"))
        A = pd.DataFrame(flags, index=out.index).astype(bool)
        A["drift"] = self.recent_rmse(site, history) > 1.5 * s["cv_rmse"][1]
        return A

    def recent_rmse(self, site: str, history: pd.DataFrame) -> float:
        """Daytime RMSE of the 1-hour model over the last 7 days of history."""
        s = self.sites[site]
        h = history.tail(DRIFT_LOOKBACK_H + 400).copy()
        h.attrs["tz"] = SITES[site]["tz"]
        X = build_features(h, 1, weather_mode="lagged").tail(DRIFT_LOOKBACK_H)
        P = predict_quantiles(s["models"][1], X)
        day = (X["is_day"] == 1) & X["y"].notna()
        return float(np.sqrt(((P["p50"] - X["y"])[day] ** 2).mean())) if day.any() else np.nan

    def save(self, path=PKL):
        path.parent.mkdir(exist_ok=True)
        joblib.dump(self, path, compress=3)     # 71 MB -> 30 MB, under GitHub's 100 MB limit

    @staticmethod
    def load(path=PKL) -> "ForecastPipeline":
        return joblib.load(path)

    def model_card(self) -> str:
        """Readable description of what is inside the pkl, written next to it as MODEL_CARD.md."""
        any_site = next(iter(self.sites.values()))
        n_feat = len(any_site["models"][1]["p50"].feature_name_)
        lines = [
            "# What is inside `forecaster.pkl`",
            "",
            "This file is generated by `python src/pipeline.py`. Don't edit it, rebuild it. "
            "It is a compressed Python pickle, so it can't be read in a text editor. This card lists "
            "what it holds. The numbers below are read from the pkl itself.",
            "",
            f"It holds {len(self.sites)} plants x {len(HORIZONS)} horizons x 3 quantiles (P10, P50, P90) = "
            f"{len(self.sites) * len(HORIZONS) * 3} LightGBM models, each using {n_feat} input features, "
            "plus the thresholds the alert engine needs.",
            "",
            "## How to check it",
            "",
            "1. Rebuild it from the code and data in this repo: `python src/pipeline.py` (about a minute). "
            "Training uses the tuned settings in `reports/lgbm_params.json`, which notebook 05 produced.",
            "2. The test-set scores in `reports/lgbm_results.csv` and notebooks 05 and 08 come from these same "
            "settings, so a rebuilt model reproduces them.",
            "3. Only load a pkl you built yourself or got from this repo. Loading a pickle runs code.",
            "",
        ]
        for site, s in self.sites.items():
            lines += [
                f"## {site}",
                "",
                f"- capacity used for kW figures: {s['capacity_kw']:.1f} kW",
                f"- training data ends / test period starts: {s['test_start']:%Y-%m-%d %H:%M} UTC",
                f"- clear-sky scale (CF per W/m2 of clear-sky GHI): {s['cs_scale']:.6f}",
                "",
                "| horizon | trees per model | validation RMSE (% of capacity) | low-confidence width limit (% of capacity) |",
                "|---|---|---|---|",
            ]
            for h in HORIZONS:
                trees = s["models"][h]["p50"].n_estimators
                lines.append(f"| +{h} h | {trees} | {s['cv_rmse'][h] * 100:.2f} | {s['width_limit'][h] * 100:.1f} |")
            lines.append("")
        lines += ["## Input features", "", ", ".join(f"`{f}`" for f in any_site["models"][1]["p50"].feature_name_), ""]
        return "\n".join(lines)


if __name__ == "__main__":
    # import the class from the module, not __main__, so the pickle loads from anywhere
    from pipeline import ForecastPipeline as Pipeline
    params = json.load(open(ROOT / "reports" / "lgbm_params.json"))
    pipe = Pipeline()
    for site in ("primary", "site_a"):
        d = load_processed(site)
        d.attrs["tz"] = SITES[site]["tz"]
        pipe.fit_site(site, d, params)
        print(f"{site}: fitted {len(HORIZONS)} horizons")
    pipe.save()
    (PKL.parent / "MODEL_CARD.md").write_text(pipe.model_card(), encoding="utf-8")
    print(f"saved {PKL} and MODEL_CARD.md")
