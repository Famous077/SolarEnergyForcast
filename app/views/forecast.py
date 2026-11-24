import pandas as pd
import streamlit as st

from components.charts import forecast_chart
from components.ui import note, tile
from services import data, interval_hit_rate, situation

ctx = st.session_state.ctx
fc, tz, cap = ctx["fc"], ctx["tz"], ctx["cap"]

st.title("Forecast")
view = st.radio("Look ahead", ["Next 6 hours", "Next 2 days"], horizontal=True, label_visibility="collapsed")
view = "6h" if view == "Next 6 hours" else "48h"

st.markdown(f"<div class='summary'>{situation(ctx, 6 if view == '6h' else 3)}</div>", unsafe_allow_html=True)
st.plotly_chart(forecast_chart(ctx, view), use_container_width=True, config={"displayModeBar": False})

near = fc[fc["horizon_h"] <= 6]
c = st.columns(3)
day = near[near["p50_kw"] > 0.5]
with c[0]:
    tile("Highest in next 6 h", f"{near['p50_kw'].max():.1f} kW", f"at {near['p50_kw'].idxmax().tz_convert(tz):%H:%M}", "info")
with c[1]:
    tile("Lowest daylight hour", f"{day['p50_kw'].min():.1f} kW" if len(day) else "dark",
         f"at {day['p50_kw'].idxmin().tz_convert(tz):%H:%M}" if len(day) else "no daylight in the next 6 h", "info")
with c[2]:
    tile("Energy, next 6 h", f"{near['p50_kw'].sum():.0f} kWh",
         f"likely {near['p10_kw'].sum():.0f} to {near['p90_kw'].sum():.0f} kWh", "info")

if view == "48h":
    note("Beyond 6 hours we only forecast three single hours (+24, +36 and +48 h). "
         "The model has no weather forecast, so day-ahead values lean on the time of year and recent days. "
         "In testing they were only slightly better than a seasonal average.")

st.markdown("### Hour by hour")
d = data(ctx["site"])
table = pd.DataFrame({
    "Time": fc.index.tz_convert(tz).strftime("%a %H:%M"),
    "Ahead": [f"+{h} h" for h in fc["horizon_h"]],
    "Forecast (kW)": fc["p50_kw"].round(1),
    "Likely low": fc["p10_kw"].round(1),
    "Likely high": fc["p90_kw"].round(1),
    "Clear-sky max": ctx["clear_sky_kw"].reindex(fc.index).round(1).to_numpy(),
    "What happened (replay)": d["power_kw"].reindex(fc.index).round(1).to_numpy(),
})
st.dataframe(table, hide_index=True, use_container_width=True)
st.caption(f"'Likely' range = P10 to P90. Real output fell inside it {interval_hit_rate(ctx['site']):.0%} of "
           "the time in testing (the target is 80 %). The last column is only available because this is a replay.")
