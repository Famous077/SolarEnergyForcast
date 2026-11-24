"""Forecast chart and one-page PDF report, shared by the dashboard."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ALERTS = ["ramp", "low_output", "low_confidence", "drift"]


def forecast_figure(site, history, fc, actual=None, tz="UTC"):
    """Last 48 h of history, the forecast band at each horizon, and the actuals when replaying."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    hist = history["power_kw"].tail(48)
    ax.plot(hist.index.tz_convert(tz), hist, color="k", lw=1.5, label="measured")
    # 1-6 h form an hourly run and are drawn as a band; 24/36/48 h are single hours, so they get
    # error bars instead of a line that would invent values for the hours in between.
    run, far = fc[fc["horizon_h"] <= 6], fc[fc["horizon_h"] > 6]
    t = run.index.tz_convert(tz)
    ax.fill_between(t, run["p10_kw"], run["p90_kw"], color="C3", alpha=.2, label="P10–P90")
    ax.plot(t, run["p50_kw"], "o-", color="C3", ms=4, label="P50 forecast")
    ax.errorbar(far.index.tz_convert(tz), far["p50_kw"],
                yerr=[far["p50_kw"] - far["p10_kw"], far["p90_kw"] - far["p50_kw"]],
                fmt="o", color="C3", ms=4, capsize=4)
    if actual is not None and len(actual):
        ax.plot(actual.index.tz_convert(tz), actual, color="k", lw=1, ls=":", label="actual (replay)")
    ax.axvline(history.index[-1].tz_convert(tz), color="grey", ls="--", lw=1)
    ax.set_ylabel("power (kW)")
    ax.set_title(f"{site}: forecast from {history.index[-1].tz_convert(tz):%Y-%m-%d %H:%M} ({tz})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def pdf_report(site, fig, fc, recent_rmse, tz="UTC") -> bytes:
    """Page 1 is the chart, page 2 the forecast table with alert flags."""
    table = fc.copy()
    table.index = table.index.tz_convert(tz).strftime("%Y-%m-%d %H:%M")
    table = table[["horizon_h", "p10_kw", "p50_kw", "p90_kw"] + ALERTS]
    table[["p10_kw", "p50_kw", "p90_kw"]] = table[["p10_kw", "p50_kw", "p90_kw"]].round(1)
    table[ALERTS] = table[ALERTS].replace({True: "YES", False: ""})

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig)
        page, ax = plt.subplots(figsize=(11, 4.2))
        ax.axis("off")
        ax.table(cellText=table.reset_index().values, colLabels=["target"] + list(table.columns),
                 loc="center", cellLoc="center").scale(1, 1.3)
        rmse_txt = f"{recent_rmse * 100:.1f} % of capacity" if pd.notna(recent_rmse) else "n/a"
        ax.set_title(f"{site}: forecast table. 1 h model RMSE over the last 7 days: {rmse_txt}", fontsize=10)
        pdf.savefig(page)
        plt.close(page)
    return buf.getvalue()
