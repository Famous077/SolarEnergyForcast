import streamlit as st

from components.charts import driver_chart, weather_chart
from components.ui import note, tile
from services import driver_sentence

ctx = st.session_state.ctx
hist, tz, fc = ctx["history"], ctx["tz"], ctx["fc"]
now = hist.iloc[-1]

st.title("Weather and what drives the forecast")
note("All weather here is <b>measured</b>, up to now. The model is not fed a weather forecast, "
     "so it can't see a cloud front before it reaches the plant.")

kt = now["clearsky_index"]
if now["is_day"] == 0 or kt != kt:
    sky, tone = ("night" if now["is_day"] == 0 else "unknown"), "info"
elif kt > 0.8:
    sky, tone = "clear", "ok"
elif kt > 0.4:
    sky, tone = "partly cloudy", "warn"
else:
    sky, tone = "overcast", "bad"

c = st.columns(4)
with c[0]:
    tile("Sky right now", sky, f"sunlight at {kt:.0%} of clear sky" if kt == kt and now["is_day"] else "", tone)
with c[1]:
    tile("Sunlight (GHI)", f"{now['ghi']:.0f} W/m²" if now["ghi"] == now["ghi"] else "n/a",
         f"clear sky would be {now['ghi_clearsky']:.0f}", "info")
with c[2]:
    tile("Air temperature", f"{now['temp_c']:.0f} °C" if now["temp_c"] == now["temp_c"] else "n/a",
         "panels lose about 0.4 % per °C above 25 °C cell temperature", "info")
with c[3]:
    if "cloud_cover_pct" in hist:
        tile("Cloud cover", f"{now['cloud_cover_pct']:.0f} %", "", "info")
    else:
        tile("Rain, last 24 h", f"{hist['rain_mm'].tail(24).sum():.1f} mm", "no cloud sensor at this plant", "info")

st.markdown("### Last 48 hours")
st.plotly_chart(weather_chart(hist, tz, {
    "ghi": ("Sunlight", "W/m²"), "cloud_cover_pct": ("Cloud cover", "%"),
    "temp_c": ("Temperature", "°C"), "wind_speed_kmh": ("Wind", "km/h")}),
    use_container_width=True, config={"displayModeBar": False})

st.markdown("### Why the forecast says what it says")
drv = ctx["drivers"]
if drv.empty:
    st.info("No daylight hours in the next 6 hours, so there is nothing to explain.")
else:
    st.plotly_chart(driver_chart(drv, tz), use_container_width=True, config={"displayModeBar": False})
    st.caption("Each bar shows how far an input group pushes the forecast above or below an average daytime hour.")
    for t, row in drv.iterrows():
        st.markdown(f"**{t.tz_convert(tz):%H:%M}** (+{int(row['horizon_h'])} h): "
                    + driver_sentence(row, fc.loc[t, "p50_kw"]))
