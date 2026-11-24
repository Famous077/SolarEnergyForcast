"""Solar position + clear-sky irradiance via pvlib."""
import pandas as pd
import pvlib


def add_solar_features(df: pd.DataFrame, lat, lon, altitude, ghi_col, offset_min=30) -> pd.DataFrame:
    """df must have a tz-aware UTC index. Hourly-mean data is labelled at hour start,
    so geometry is evaluated at the interval midpoint (offset_min=30). Use 0 for instantaneous stamps."""
    loc = pvlib.location.Location(lat, lon, altitude=altitude)
    mid = df.index + pd.Timedelta(minutes=offset_min)
    pos = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen")
    out = df.copy()
    out["solar_zenith"] = pos["apparent_zenith"].to_numpy()
    out["solar_elevation"] = pos["apparent_elevation"].to_numpy()
    out["solar_azimuth"] = pos["azimuth"].to_numpy()
    out["ghi_clearsky"] = cs["ghi"].to_numpy()
    out["is_day"] = (out["solar_elevation"] > 0).astype(int)
    kt = out[ghi_col] / out["ghi_clearsky"].where(out["ghi_clearsky"] > 20)
    out["clearsky_index"] = kt.clip(0, 1.5)
    return out
