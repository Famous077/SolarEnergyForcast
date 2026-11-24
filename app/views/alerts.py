import streamlit as st

from components.ui import alert_card
from config import DRIFT_FACTOR, LOW_OUTPUT_RATIO, RAMP_THRESHOLD
from services import alert_cards

ctx = st.session_state.ctx
cards = alert_cards(ctx)

st.title("Alerts")
if not cards:
    st.success("Nothing to act on for the next 48 hours.")
for c in cards:
    alert_card(c)

st.caption("These are advice only. The system never sends control commands to the grid.")

with st.expander("What each alert means and how it is triggered"):
    st.markdown(f"""
| Alert | Level | Triggered when |
|---|---|---|
| Sudden drop | High | forecast falls by more than {RAMP_THRESHOLD:.0%} of capacity in one hour and the sun's path explains less than half of that |
| Low generation | Medium | forecast is under {LOW_OUTPUT_RATIO:.0%} of what a clear sky would give |
| Less certain than usual | Caution | likely range is wider than 90 % of ranges seen on training data |
| Error higher than normal | Caution | 1-hour forecast error over the last 7 days is {DRIFT_FACTOR}x the validation error |
| Input data problem | Caution | the last reading is old, daytime hours are missing, or the meter is stuck on one value |

The sudden-drop alert only uses past output, so it can't see clouds coming. In testing it caught
18 % of weather-driven drops at Site A and none at the primary plant. Read it as a hint, not a warning you can rely on.
""")
