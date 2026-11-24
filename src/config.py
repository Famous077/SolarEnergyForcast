"""Site metadata and thresholds. Locations were INFERRED from the data (see reports/data_quality.md)."""

SITES = {
    "primary": {
        # DKASC-style columns; southern hemisphere, local clock with no DST, peak GHI in Jan.
        # Assumed Alice Springs, NT (confirm with mentor).
        "lat": -23.76, "lon": 133.87, "tz": "Australia/Darwin", "altitude": 546,
        "power_col": "Active_Power", "ghi_col": "Global_Horizontal_Radiation",
        "capacity_kw": None,  # estimated from data (p99.9) at cleaning time
        "solar_offset_min": 30,  # hourly means labelled at hour start -> geometry at :30
    },
    "site_a": {
        # Peak solar elevation 68.4 deg at the June solstice -> lat ~45N; solar noon ~10:00 UTC -> lon ~30E.
        "lat": 45.0, "lon": 30.0, "tz": "Europe/Bucharest", "altitude": 50,
        "power_col": "active_power_kw", "ghi_col": "ghi_wm2",
        "capacity_kw": 55.0,  # nameplate AC limit (clipping at 55 kW)
        "solar_offset_min": 0,   # instantaneous hourly stamps
    },
}

SHORT_GAP_H = 3            # interpolate gaps shorter than this
MIN_COVERAGE = 0.8         # fraction of 5-min samples needed for a valid hour
FLATLINE_H = 3             # identical daytime values for >= this many hours -> flag
RAMP_THRESHOLD = 0.20      # drop > 20% of capacity within 1 h = ramp event
TEST_FRAC = 0.20

# Alert engine (weeks 9-10)
LOW_OUTPUT_RATIO = 0.30    # output below 30% of clear-sky output = low-output event
MIN_CLEARSKY_CF = 0.10     # ignore low-sun hours where clear-sky output itself is tiny
WIDE_INTERVAL_Q = 0.90     # P10-P90 width above this training quantile = low confidence
LARGE_ERROR_FACTOR = 2.0   # |error| > 2 x CV RMSE = a forecast that deserved low confidence
DRIFT_WINDOW_D = 7         # rolling window for drift monitoring (days)
DRIFT_FACTOR = 1.5         # rolling RMSE > 1.5 x CV RMSE = drift
