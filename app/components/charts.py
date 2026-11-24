"""Plotly charts. One palette for the whole app, defined here."""
import pandas as pd
import plotly.graph_objects as go

C = {"actual": "#e6edf3", "forecast": "#f0a830", "band": "rgba(240,168,48,0.18)",
     "clear": "#5b7a99", "grid": "#26344a", "now": "#8b9bb0", "replay": "#8b9bb0"}


def _layout(fig, height=380, yaxis="kW"):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color="#c9d4e0"),
        hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
        yaxis=dict(title=yaxis, gridcolor=C["grid"], zeroline=False, rangemode="tozero"),
        xaxis=dict(gridcolor=C["grid"]),
    )
    return fig


def forecast_chart(ctx, view: str = "6h") -> go.Figure:
    """History, the 1-6 h band, the day-ahead points, clear-sky ceiling and (replay) what really happened."""
    tz, fc, hist = ctx["tz"], ctx["fc"], ctx["history"]
    back = 12 if view == "6h" else 24
    end = fc.index[5] if view == "6h" else fc.index[-1]
    loc = lambda idx: idx.tz_convert(tz)

    fig = go.Figure()
    cs = ctx["clear_sky_kw"].loc[hist.index[-back]:end]
    fig.add_trace(go.Scatter(x=loc(cs.index), y=cs, name="Clear-sky ceiling", mode="lines", line=dict(color=C["clear"], dash="dot", width=1),
                             hovertemplate="%{y:.1f} kW"))
    h = hist["power_kw"].iloc[-back:]
    fig.add_trace(go.Scatter(x=loc(h.index), y=h, name="Measured", mode="lines", line=dict(color=C["actual"], width=2.5),
                             hovertemplate="%{y:.1f} kW"))

    near = fc[fc["horizon_h"] <= 6]
    x = loc(pd.DatetimeIndex([hist.index[-1]]).append(near.index))
    start = hist["power_kw"].iloc[-1]
    fig.add_trace(go.Scatter(x=x, y=[start, *near["p90_kw"]], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[start, *near["p10_kw"]], fill="tonexty", fillcolor=C["band"], mode="lines", line=dict(width=0),
                             name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[start, *near["p50_kw"]], name="Forecast", mode="lines+markers",
                             line=dict(color=C["forecast"], width=2.5), marker=dict(size=6),
                             customdata=[[start, start], *near[["p10_kw", "p90_kw"]].to_numpy()],
                             hovertemplate="%{y:.1f} kW (range %{customdata[0]:.1f} to %{customdata[1]:.1f})"))
    if view == "48h":
        far = fc[fc["horizon_h"] > 6]
        fig.add_trace(go.Scatter(
            x=loc(far.index), y=far["p50_kw"], mode="markers", name="Day-ahead forecast",
            marker=dict(color=C["forecast"], size=9, symbol="diamond"),
            error_y=dict(type="data", symmetric=False, array=far["p90_kw"] - far["p50_kw"],
                         arrayminus=far["p50_kw"] - far["p10_kw"], color=C["forecast"], thickness=1.5, width=6),
            hovertemplate="%{y:.1f} kW"))

    act = ctx["actual_after"].loc[:end]
    if len(act):
        fig.add_trace(go.Scatter(x=loc(act.index), y=act, name="What happened (replay)", mode="lines",
                                 line=dict(color=C["replay"], dash="dash", width=1.5), hovertemplate="%{y:.1f} kW"))

    now = loc(pd.DatetimeIndex([hist.index[-1]]))[0]
    fig.add_vline(x=now, line=dict(color=C["now"], width=1, dash="dot"))
    fig.add_annotation(x=now, y=1, yref="paper", text="now", showarrow=False, yshift=8, font=dict(color=C["now"]))
    fig.update_yaxes(range=[0, ctx["cap"] * 1.08])
    return _layout(fig)


def weather_chart(hist: pd.DataFrame, tz: str, cols: dict, hours: int = 48) -> go.Figure:
    """Small multiples of observed weather. cols maps column -> (label, unit)."""
    from plotly.subplots import make_subplots
    cols = {k: v for k, v in cols.items() if k in hist}
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=[f"{lab} ({unit})" for lab, unit in cols.values()])
    h = hist.iloc[-hours:]
    x = h.index.tz_convert(tz)
    for i, (c, (lab, unit)) in enumerate(cols.items(), start=1):
        fig.add_trace(go.Scatter(x=x, y=h[c], name=lab, mode="lines", line=dict(color=C["actual"], width=1.8),
                                 hovertemplate=f"%{{y:.0f}} {unit}"), row=i, col=1)
        if c == "ghi":
            fig.add_trace(go.Scatter(x=x, y=h["ghi_clearsky"], name="clear sky", mode="lines", line=dict(color=C["clear"], dash="dot", width=1),
                                     hovertemplate="%{y:.0f} W/m²"), row=i, col=1)
    _layout(fig, height=170 * len(cols), yaxis="")
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=28, b=10))
    fig.update_yaxes(gridcolor=C["grid"], rangemode="tozero")
    fig.update_annotations(font=dict(size=12, color="#8b9bb0"), x=0, xanchor="left")
    return fig


def driver_chart(drv: pd.DataFrame, tz: str) -> go.Figure:
    """Stacked contribution (kW) of each input group to the 1-6 h forecasts, relative to an average hour."""
    groups = [c for c in drv.columns if c not in ("horizon_h", "base")]
    colors = {"recent output": "#f0a830", "sun position and time of day": "#5b7a99", "observed weather": "#4cb782"}
    x = [f"+{h} h" for h in drv["horizon_h"]]
    fig = go.Figure([go.Bar(x=x, y=drv[g], name=g, marker_color=colors.get(g, "#888"),
                            hovertemplate="%{y:+.1f} kW") for g in groups])
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, line=dict(color="#8b9bb0", width=1))
    return _layout(fig, height=300, yaxis="kW vs average hour")


def error_chart(trend: pd.DataFrame, validation_rmse: float, test_start=None) -> go.Figure:
    fig = go.Figure()
    if test_start is not None and len(trend) and trend.index[0] < test_start:
        fig.add_vrect(x0=trend.index[0], x1=min(test_start, trend.index[-1]), fillcolor="#8b9bb0", opacity=0.08,
                      line_width=0, annotation_text="training data", annotation_position="top left",
                      annotation_font_color="#8b9bb0")
    fig.add_trace(go.Scatter(x=trend.index, y=trend["model"], name="Our 1 h forecast", mode="lines", line=dict(color=C["forecast"], width=2),
                             hovertemplate="%{y:.1f} %"))
    fig.add_trace(go.Scatter(x=trend.index, y=trend["persistence"], name="'Same as last hour'", mode="lines", line=dict(color=C["clear"], width=1.5, dash="dot"),
                             hovertemplate="%{y:.1f} %"))
    fig.add_hline(y=validation_rmse * 100, line=dict(color=C["now"], dash="dash", width=1),
                  annotation_text="validation level", annotation_font_color=C["now"])
    return _layout(fig, height=300, yaxis="daily error, % of capacity")
