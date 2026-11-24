import plotly.graph_objects as go
import streamlit as st

from components.charts import _layout
from io_utils import FIG
from report import forecast_figure, pdf_report
from services import data, test_scores

ctx = st.session_state.ctx
site, tz = ctx["site"], ctx["tz"]
res, q, al = test_scores()

st.title("Model and reports")
st.caption("For analysts. Numbers come from the daytime test set in notebooks 04, 05 and 08.")

st.markdown("### Error by method and horizon (% of capacity)")
r = res[(res["site"] == site) & (res["model"] != "lgbm_forecast")]
names = {"lgbm_lagged": "LightGBM", "smart_persistence": "Smart persistence", "persistence": "Persistence",
         "seasonal_naive": "Seasonal naive", "climatology": "Climatology"}
table = r.pivot_table(index="model", columns="horizon_h", values="RMSE_%cap").round(1).rename(index=names)
table.columns = [f"+{c} h" for c in table.columns]
st.dataframe(table, use_container_width=True)

sk = r[r["model"] == "lgbm_lagged"]
fig = go.Figure(go.Bar(x=[f"+{h} h" for h in sk["horizon_h"]], y=sk["skill_vs_best_baseline"] * 100,
                       marker_color="#f0a830", hovertemplate="%{y:.1f} %"))
st.markdown("### LightGBM improvement over the best simple method")
st.plotly_chart(_layout(fig, height=260, yaxis="% lower error"), use_container_width=True,
                config={"displayModeBar": False})
up = res[(res["site"] == site) & (res["model"] == "lgbm_forecast")]
st.caption(f"With the real weather at the target hour fed in (an upper bound, not deployable) the error drops to "
           f"{up['RMSE_%cap'].min():.1f}–{up['RMSE_%cap'].max():.1f} %. That gap is what a good weather forecast would be worth.")

c = st.columns(2)
with c[0]:
    st.markdown("### Likely-range hit rate")
    cov = q[(q["site"] == site) & (q["mode"] == "lagged")].set_index("horizon_h")[["coverage", "width"]]
    cov.index = [f"+{h} h" for h in cov.index]
    st.dataframe((cov * 100).round(0).rename(columns={"coverage": "hit rate %", "width": "width, % cap"}),
                 use_container_width=True)
with c[1]:
    st.markdown("### Alert scores")
    a = al[al["site"] == site][["horizon_h", "alert", "events", "alerts", "precision", "recall", "F1"]]
    st.dataframe(a.round(2), hide_index=True, use_container_width=True)

st.markdown("### Which inputs matter (SHAP)")
st.image(str(FIG / "15_shap_importance.png"), use_container_width=True)

st.markdown("### Export")
fig_pdf = forecast_figure(site, ctx["history"], ctx["fc"], data(site)["power_kw"].loc[ctx["origin"]:ctx["fc"].index[-1]], tz)
st.download_button("Download PDF of the current forecast", pdf_report(site, fig_pdf, ctx["fc"], ctx["recent_rmse"], tz),
                   file_name=f"forecast_{site}_{ctx['origin'].tz_convert(tz):%Y%m%d_%H%M}.pdf", mime="application/pdf")
