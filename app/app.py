"""Solar forecast dashboard for grid operators.

    streamlit run app/app.py        (from the OJT_Project folder)

Needs models/forecaster.pkl (python src/pipeline.py). The data ends in 2026, so the app runs on a
replay clock: the operator picks "now" and the model only sees data up to that hour.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from components.ui import load_css  # noqa: E402
from services import SITE_NAMES, SITES, data, get_context, pipeline  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
st.set_page_config(page_title="SolarCast", page_icon=str(STATIC / "logo-mark.png"), layout="wide")
st.logo(str(STATIC / "logo-mark.png"), size="large")
load_css()

pipe = pipeline()

with st.sidebar:
    st.markdown("### Solar forecast desk")
    # A link like ?site=site_a&t=2026-03-22T09:00Z opens the app at that plant and hour.
    qp = st.query_params
    if "site" not in st.session_state and qp.get("site") in pipe.sites:
        st.session_state.plant = qp["site"]
    site = st.selectbox("Plant", list(pipe.sites), format_func=SITE_NAMES.get, key="plant")
    d, info = data(site), pipe.sites[site]
    lo, hi = d.index[200], d.index[-49]

    tz = SITES[site]["tz"]
    if st.session_state.get("site") != site:            # new plant: start at 10:00 local, 20 days into the test period
        st.session_state.site = site
        first = (info["test_start"] + pd.Timedelta(days=20)).tz_convert(tz).normalize() + pd.Timedelta(hours=10)
        st.session_state.origin = first.tz_convert("UTC")
        if "t" in qp and st.session_state.get("t_used") is None:
            st.session_state.t_used = True
            try:
                st.session_state.origin = pd.Timestamp(qp["t"]).tz_convert("UTC")
            except (ValueError, TypeError):
                pass

    # The clock lives in st.session_state.origin. Buttons shift it; the date/hour widgets mirror it
    # and write back through a callback, so the two never fight.
    def from_widgets():
        st.session_state.origin = pd.Timestamp(st.session_state.w_day).replace(
            hour=st.session_state.w_hour).tz_localize(tz, nonexistent="shift_forward", ambiguous=False).tz_convert("UTC")

    st.markdown("**Replay clock**")
    c = st.columns(4)
    for col, (lab, hrs) in zip(c, [("−1d", -24), ("−1h", -1), ("+1h", 1), ("+1d", 24)]):
        if col.button(lab, use_container_width=True):
            st.session_state.origin += pd.Timedelta(hours=hrs)
    # snap onto the data's own hourly grid (the primary plant's hours fall on :30 local time)
    o = min(max(st.session_state.origin, lo), hi)
    st.session_state.origin = d.index[d.index.searchsorted(o)]
    cur = st.session_state.origin.tz_convert(tz)
    st.session_state.w_day, st.session_state.w_hour = cur.date(), cur.hour
    st.date_input("Jump to date", key="w_day", on_change=from_widgets,
                  min_value=lo.tz_convert(tz).date(), max_value=hi.tz_convert(tz).date())
    st.slider("Hour (plant local time)", 0, 23, key="w_hour", on_change=from_widgets)

    st.caption(f"Now = **{cur:%a %d %b %Y, %H:%M}** plant time. "
               f"Test period from {info['test_start'].tz_convert(tz):%d %b %Y}; earlier dates were used for training.")

st.session_state.ctx = get_context(site, st.session_state.origin)

pages = {
    "Operator": [
        st.Page("views/overview.py", title="Overview", default=True),
        st.Page("views/forecast.py", title="Forecast"),
        st.Page("views/alerts.py", title="Alerts"),
        st.Page("views/weather.py", title="Weather and drivers"),
        st.Page("views/health.py", title="System health"),
    ],
    "Analyst": [st.Page("views/analyst.py", title="Model and reports")],
}
st.navigation(pages).run()
