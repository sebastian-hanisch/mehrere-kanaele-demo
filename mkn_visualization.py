"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""


import plotly.graph_objects as go

import mkn_constants as C
import mkn_corr as R


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_trajectory(traj, stat_name: str):
    """Statistik der Karte über die Vektornummer t mit Schwelle, Änderungszeitpunkt (nach Vektor ν) und erstem Alarm."""
    n = len(traj.stat)
    ts = list(range(1, n + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=list(traj.stat), mode="lines", line=dict(color=C.COLOR_LINE, width=1.6), name=stat_name, hovertemplate="Vektor %{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1, n], y=[traj.threshold, traj.threshold], mode="lines", line=dict(color=C.COLOR_THRESH, dash="dash", width=1.5), name="Schwelle (ARL₀ 370,4)", hoverinfo="skip"))
    top = max(float(max(traj.stat)), traj.threshold) * 1.08
    if 0 < traj.nu < n:
        fig.add_trace(go.Scatter(x=[traj.nu + 0.5, traj.nu + 0.5], y=[0, top], mode="lines", line=dict(color=C.COLOR_CHANGE, dash="dot", width=1.5), name="Änderung", hoverinfo="skip"))
    if traj.alarm_t is not None:
        fig.add_trace(go.Scatter(x=[traj.alarm_t], y=[traj.stat[traj.alarm_t - 1]], mode="markers", marker=dict(color="#c0392b", size=11, symbol="x"), name="erster Alarm",
                                 hovertemplate="Alarm bei Vektor %{x}<extra></extra>"))
    fig.update_xaxes(title_text="Vektornummer t", range=[0, n + 1])
    fig.update_yaxes(title_text=stat_name, range=[0, top])
    return _base(fig)


def build_penalty(ps, series: dict, d: float):
    """ARL₁ gegen die Kanalzahl p (log-Achse). `series`: Name -> (Farbe, [(p, ARL₁)]); Karten mit Gedächtnis haben nur die gerechneten p."""
    fig = go.Figure()
    for name, (color, pts) in series.items():
        fig.add_trace(go.Scatter(x=[a for a, _ in pts], y=[b for _, b in pts], mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=7), name=name,
                                 hovertemplate=f"{name}, p = %{{x}}: %{{y:.1f}}<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der Kanäle p", tickmode="array", tickvals=[1, 2, 5, 10, 20])
    fig.update_yaxes(title_text=f"ARL₁ bei d = {d:.1f}".replace(".", ",") + " (Vektoren)", type="log")
    return _base(fig)


def build_plane(rho: float, ucl: float, z: float, z_new: float, mu):
    """Ebene (x₁, x₂) für p = 2: T²-Ellipse, Rechteck der getrennten Karten (neu geeicht und Šidák), Sprungvektor; Achsen fest bei ±6."""
    ex, ey = R.ellipse_points(rho, ucl)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ex), y=list(ey), mode="lines", line=dict(color=C.COLOR_T2, width=2.5), name="T²-Grenze", hoverinfo="skip"))
    for zz, nm, dash, col in ((z_new, "getrennt, neu geeicht", "solid", C.COLOR_SHEW), (z, "getrennt, Šidák", "dot", "#a07010")):
        fig.add_trace(go.Scatter(x=[-zz, zz, zz, -zz, -zz], y=[-zz, -zz, zz, zz, -zz], mode="lines", line=dict(color=col, width=2, dash=dash), name=nm, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, float(mu[0])], y=[0, float(mu[1])], mode="lines+markers", line=dict(color=C.COLOR_CUSUM, width=3), marker=dict(size=[4, 10], symbol=["circle", "diamond"]),
                             name="Sprung μ (Kanäle 1 und 2)", hovertemplate="μ = (%{x:.2f}, %{y:.2f})<extra></extra>"))
    fig.update_xaxes(title_text="Kanal 1", range=[-6, 6], zeroline=True)
    fig.update_yaxes(title_text="Kanal 2", range=[-6, 6], zeroline=True)
    return _base(fig, height=420)


def build_bars(items, title: str):
    """Waagerechte Balken (lineare Achse, Zahl am Balken): `items` = [(Name, Wert, Farbe)]."""
    fig = go.Figure(go.Bar(x=[v for _, v, _ in items], y=[n for n, _, _ in items], orientation="h", marker_color=[c for _, _, c in items],
                           text=[f"{v:.2f}".replace(".", ",") for _, v, _ in items], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    top = max(v for _, v, _ in items)
    fig.update_xaxes(title_text=title, range=[0, top * 1.18])
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=40 * len(items) + 110)


def build_dilution(ms, series: dict, title: str):
    """ARL₁ gegen die Zahl springender Kanäle m (log-Achse); `series`: Name -> (Farbe, [(m, ARL₁)])."""
    fig = go.Figure()
    for name, (color, pts) in series.items():
        fig.add_trace(go.Scatter(x=[a for a, _ in pts], y=[b for _, b in pts], mode="lines+markers", line=dict(color=color, width=2.5), marker=dict(size=7), name=name,
                                 hovertemplate=f"{name}, m = %{{x}}: %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der springenden Kanäle m", tickmode="array", tickvals=list(ms))
    fig.update_yaxes(title_text=title, type="log")
    return _base(fig)
