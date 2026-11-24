import pandas as pd
import streamlit as st

from components.charts import error_chart
from components.ui import tile
from io_utils import ROOT
from services import error_trend, interval_hit_rate

ctx = st.session_state.ctx
info, tz, h = ctx["info"], ctx["tz"], ctx["health"]

st.title("System health")

c = st.columns(4)
with c[0]:
    ok = h["hours_since_data"] <= 1
    tile("Latest reading", f"{h['last_good'].tz_convert(tz):%H:%M}" if h["last_good"] is not None else "none",
         "current" if ok else f"{h['hours_since_data']:.0f} h old", "ok" if ok else "bad")
with c[1]:
    m = h["missing_day_hours_24h"] + h["flatline_hours_24h"]
    tile("Bad daytime hours, 24 h", str(m),
         f"{h['missing_day_hours_24h']} missing · {h['flatline_hours_24h']} stuck", "ok" if m == 0 else "warn")
with c[2]:
    r = ctx["recent_rmse"]
    v = info["cv_rmse"][1]
    tone = "ok" if r == r and r <= 1.5 * v else "warn"
    tile("1 h error, last 7 days", f"{r * 100:.1f} %" if r == r else "n/a",
         f"validation level {v * 100:.1f} % of capacity", tone)
with c[3]:
    built = pd.Timestamp((ROOT / "models" / "forecaster.pkl").stat().st_mtime, unit="s")
    tile("Model", "LightGBM", f"9 horizons · built {built:%d %b %Y}", "info")

st.markdown("### Daily forecast error, last 30 days")
trend = error_trend(ctx["site"], ctx["origin"])
st.plotly_chart(error_chart(trend, info["cv_rmse"][1], info["test_start"].tz_convert(tz).date()), use_container_width=True, config={"displayModeBar": False})
trend = trend[trend.index >= info["test_start"].tz_convert(tz).date()]    # score unseen days only
if len(trend):
    better = (trend["model"] < trend["persistence"]).mean()
    skill = 1 - (trend["model"] ** 2).mean() ** .5 / (trend["persistence"] ** 2).mean() ** .5
    st.markdown(f"On the {len(trend)} unseen days in this window the 1-hour forecast beat a plain *same as last hour* guess on "
                f"**{better:.0%} of days**, and its overall error was **{skill:.0%} lower**.")

st.markdown("### How far to trust the likely range")
st.markdown(f"On the test set, real output fell inside the P10–P90 range **{interval_hit_rate(ctx['site']):.0%}** "
            "of the time for the next 6 hours. A calibrated range would hit 80 %. Ours is too narrow, "
            "so allow some extra margin when planning on the low end.")
st.caption(f"Training data ends {info['test_start'].tz_convert(tz):%d %b %Y}. Any replay hour before that date "
           "was seen by the model during training, so its accuracy there will look better than it really is.")
