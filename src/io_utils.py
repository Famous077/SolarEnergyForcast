"""Paths and data loaders shared by all notebooks."""
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"


def load_primary_raw(path=RAW / "solar_dataset.csv") -> pd.DataFrame:
    """5-min site data. Every line is wrapped in double quotes -> strip them first."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip().strip('"') for ln in text.splitlines() if ln.strip()]
    df = pd.read_csv(io.StringIO("\n".join(lines)), parse_dates=["timestamp"])
    return df.set_index("timestamp").sort_index()


def load_site_a_raw(path=RAW / "site_a_hourly_weather.csv") -> pd.DataFrame:
    """Hourly Site A data, indexed on UTC."""
    df = pd.read_csv(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.drop(columns=["datetime_local"]).set_index("datetime_utc").sort_index()


def load_processed(site: str) -> pd.DataFrame:
    """Cleaned hourly data written by notebook 02 (CSV, UTC index)."""
    df = pd.read_csv(PROCESSED / f"{site}_hourly_clean.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime_utc"
    return df
