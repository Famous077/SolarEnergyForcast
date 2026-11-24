import streamlit as st

from components.charts import forecast_chart
from components.ui import alert_card, pills, tile
from services import SITE_NAMES, alert_cards, interval_hit_rate, situation

ctx = st.session_state.ctx
fc, now, prev, cap, tz = ctx["fc"], ctx["now"], ctx["prev_fc"], ctx["cap"], ctx["tz"]
cards = alert_cards(ctx)

head, status = st.columns([3, 2])
head.title(SITE_NAMES[ctx["site"]])
with status:
    h = ctx["health"]
    data_ok = h["hours_since_data"] <= 1 and not h["missing_day_hours_24h"]
    drift = fc["drift"].any()
    pills([("REPLAY", "info"), ("DATA OK" if data_ok else "DATA GAP", "ok" if data_ok else "warn"),
           ("MODEL DRIFT" if drift else "MODEL OK", "warn" if drift else "ok")])
    st.caption(f"Data up to {ctx['origin'].tz_convert(tz):%d %b %H:%M} plant time · capacity {cap:.0f} kW")

# --- four numbers an operator checks first
now_kw = now["power_kw"]
c = st.columns(4)
with c[0]:
    if now_kw != now_kw:            # NaN
        tile("Generating now", "no data", "last reading missing", "warn")
    elif prev["p10_kw"] <= now_kw <= prev["p90_kw"] or max(prev["p90_kw"], now_kw) < 0.5:
        tile("Generating now", f"{now_kw:.1f} kW", "as expected an hour ago", "ok")
    else:
        word = "above" if now_kw > prev["p90_kw"] else "below"
        tile("Generating now", f"{now_kw:.1f} kW",
             f"{word} the {prev['p10_kw']:.1f}–{prev['p90_kw']:.1f} kW we expected", "warn")
with c[1]:
    n1 = fc.iloc[0]
    tile(f"Next hour ({fc.index[0].tz_convert(tz):%H:%M})", f"{n1['p50_kw']:.1f} kW",
         f"likely {n1['p10_kw']:.1f} to {n1['p90_kw']:.1f} kW", "info")
with c[2]:
    d3 = fc.iloc[2]["p50_kw"] - (now_kw if now_kw == now_kw else 0)
    tone = "bad" if d3 < -0.2 * cap else "warn" if d3 < -0.1 * cap else "ok"
    tile("Change over 3 hours", f"{d3:+.1f} kW", f"{d3 / cap:+.0%} of capacity", tone)
with c[3]:
    n_high = sum(x["severity"] == "High" for x in cards)
    n_med = sum(x["severity"] == "Medium" for x in cards)
    tone = "bad" if n_high else "warn" if cards else "ok"
    tile("Active alerts", str(len(cards)),
         f"{n_high} high · {n_med} medium · {len(cards) - n_high - n_med} caution" if cards else "nothing to act on",
         tone)

st.markdown(f"<div class='summary'>{situation(ctx)}</div>", unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    st.markdown("### Next 6 hours")
    st.plotly_chart(forecast_chart(ctx, "6h"), use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Shaded area is the likely range (P10–P90). In testing the real output landed inside it "
               f"{interval_hit_rate(ctx['site']):.0%} of the time, so it is a bit too narrow.")
with right:
    st.markdown("### Alerts")
    if not cards:
        st.success("No alerts for the next 48 hours.")
    for card in cards[:3]:
        alert_card(card, compact=True)
    if len(cards) > 3:
        st.caption(f"{len(cards) - 3} more on the Alerts page.")
    if cards:
        st.page_link("views/alerts.py", label="Open alert details →")
