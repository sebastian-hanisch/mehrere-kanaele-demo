"""Viele Kanäle: was eine gemeinsame Alarmregel kostet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der Reihe „Änderungserkennung“: p Messreihen („Kanäle“) laufen gleichzeitig, es gibt einen gemeinsamen Alarm, und die Fehlalarmrate (ARL₀ 370,4) soll für den
gemeinsamen Alarm gelten. Gezählt werden Beobachtungsvektoren bis zum ersten Alarm; Mittel und Kovarianz sind bekannt, die Kanäle sind ausgedacht.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import mkn_cards as K
import mkn_constants as C
import mkn_corr as R
import mkn_evaluation as EV
import mkn_reference as REF
import mkn_scenario as SC
import mkn_texts as TX
import mkn_visualization as V
from mkn_layout import dunkles_design, kopf_css
from mkn_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Viele Kanäle – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _trajectory(p, m, d, nu, seed, karte):
    t = SC.run_trajectory(p, m, d, nu, seed, karte)
    t.x = None
    return t


@st.cache_data(show_spinner=False)
def _mewma_live(p, d, R_, seed):
    return SC.mewma_live(p, d, R_, seed)


@st.cache_data(show_spinner=False)
def _exp3(model, rho, richtung, sprung, gleich):
    return EV.exp3_row(EV.Settings(korr_modell=model, rho=rho, richtung=richtung, sprung=sprung, gleich=gleich))


de, de_s, pct = EV.de, EV.de_s, EV.pct

# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mkn_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📡 Viele Kanäle: was eine gemeinsame Alarmregel kostet")
st.markdown(TX.INTRO)

with st.expander("So funktioniert die gemeinsame Alarmregel", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Alle Experimente")
    p = st.select_slider("Zahl der Kanäle p", options=C.P_OPTIONS, key="p_select",
                         help="Unabhängige standardnormale Kanäle. Gilt für Experiment 1, 2 (Kurven laufen über alle p), 5. Die Karten mit Gedächtnis sind für p = 1, 2, 5, 10, 20 geeicht.")
    d = st.select_slider("Sprung d je springendem Kanal (σ)", options=C.D_OPTIONS, key="d_select", format_func=lambda v: f"{v:.1f}".replace(".", ","),
                         help="Höhe des Mittelwertsprungs in Standardabweichungen. Die Karten mit Gedächtnis sind für d = 0,5 / 1 / 2 / 3 gefroren.")
    st.header("⚙️ Experiment 1: ein Verlauf")
    m = st.slider("Springende Kanäle m", *bounds("m_slider"), step=1, key="m_slider", help="Wie viele der p Kanäle springen (die ersten m). Ist m größer als p, springen alle p.")
    karte = st.radio("Karte", C.KARTEN, key="karte_radio", help="Welche Karte den Verlauf überwacht; alle Schwellen sind auf die gemeinsame ARL₀ 370,4 geeicht.")
    nu = st.slider("Vektoren vor der Änderung ν", *bounds("nu_slider"), step=C.NU_STEP, key="nu_slider", help="Die Änderung beginnt mit Vektor ν + 1. Bei ν = 0 gilt das Maß der exakten ARL₁.")
    st.header("⚙️ Experiment 3: Korrelation und Richtung")
    korr_modell = st.radio("Modell", C.KORR_MODELLE, key="korr_modell_radio", help="Zwei Kanäle mit jeder Korrelation bis ±0,95, oder fünf gleichkorrelierte Kanäle (ρ ≥ 0).")
    rho = st.slider("Korrelation ρ", *bounds("rho_slider"), step=C.RHO_STEP, key="rho_slider", format="%.2f",
                    help="Korrelation zwischen den Kanälen (Einheitsvarianzen). Für p = 5 gilt nur ρ ≥ 0.")
    richtung = st.radio("Richtung des Sprungs", C.RICHTUNGEN, key="richtung_radio",
                        help="entlang: alle Kanäle gleichsinnig (1, 1, …); gegen: Kanal 1 und 2 gegeneinander (1, −1); ein Kanal: nur Kanal 1; schief: (1, −0,3).")
    gleich = st.checkbox("Gleiche Nichtzentralität λ = 4", key="gleich_check", help="Der Sprung wird so skaliert, dass μ'Σ⁻¹μ = 4: T² hat dann in jeder Richtung dieselbe ARL₁, getrennte Karten nicht.")
    sprung = st.slider("Skala des Sprungs d (σ)", *bounds("sprung_slider"), step=C.SPRUNG_STEP, key="sprung_slider", format="%.2f", disabled=gleich,
                       help="Der Sprungvektor ist d mal die Richtung. Gesperrt, solange „Gleiche Nichtzentralität“ gewählt ist (dann bestimmt λ = 4 die Skala).")
    st.header("⚙️ Experiment 4: Verdünnung")
    vd_p = st.select_slider("Zahl der Kanäle p (Verdünnung)", options=C.VD_P_OPTIONS, key="vd_p_select", help="Fünf oder zehn unabhängige Kanäle.")
    vd_m = st.slider("Springende Kanäle m (Verdünnung)", *bounds("vd_m_slider"), step=1, key="vd_m_slider",
                     help="Wie viele Kanäle springen. Die CUSUM ist nur für m = 1 bis 5 (p = 5) bzw. 1, 2, 4, 5, 10 (p = 10) gefroren; ist m größer als p, springen alle.")
    vd_vergleich = st.radio("Vergleich", C.VD_VERGLEICHE, key="vd_vergleich_radio", help="Gleiche Höhe: jeder springende Kanal um 1 σ. Gleiche Energie: λ = m·d² = 4.")
    vd_summe = st.checkbox("Summenkarte zeigen (Richtung bekannt)", key="vd_summe_check", help="Die Summenkarte setzt voraus, dass alle springenden Kanäle gleichsinnig springen.")
    st.header("⚙️ Experiment 5: Live-Simulation")
    r_live = st.slider("Läufe R der MEWMA-Simulation", *bounds("r_slider"), step=C.R_STEP, key="r_slider", help="Wie viele Läufe der MEWMA live simuliert werden (ARL₁, ein Kanal springt um d).")
    st.header("⚙️ Zufall")
    seed = st.number_input("Zufalls-Seed (Verlauf und MEWMA-Lauf)", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

sync_query_params({"p_select": int(p), "d_select": float(d), "m_slider": int(m), "karte_radio": karte, "nu_slider": int(nu), "seed_input": int(seed),
                   "korr_modell_radio": korr_modell, "rho_slider": float(rho), "richtung_radio": richtung, "sprung_slider": float(sprung), "gleich_check": bool(gleich),
                   "vd_p_select": int(vd_p), "vd_m_slider": int(vd_m), "vd_vergleich_radio": vd_vergleich, "vd_summe_check": bool(vd_summe), "r_slider": int(r_live)})
S = EV.Settings(int(p), float(d), int(m), karte, int(nu), int(seed), korr_modell, round(float(rho), 4), richtung, float(sprung), bool(gleich),
                int(vd_p), int(vd_m), vd_vergleich, bool(vd_summe), int(r_live))
STAT_NAME = {C.KARTE_T2: "T² (x'x)", C.KARTE_SHEW: "größter |x_i|", C.KARTE_CUSUM: "größter CUSUM-Wert", C.KARTE_EWMA: "größter |Z_i|"}

# =====================================================================================================================================================
# Experiment 1
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 1: Ein Verlauf, eine Alarmzeit")
st.markdown(TX.EXP1_FRAGE)
m1 = S.m_eff
if S.m > S.p:
    st.caption(f"m = {S.m} ist größer als p = {S.p}: es springen alle {S.p} Kanäle.")
traj = _trajectory(S.p, m1, S.d, S.nu, S.seed, S.karte)
arl1 = EV.arl1_of_card(S.karte, S.p, m1, S.d)
rl = EV.run_length_summary(S.karte, S.p, m1, S.d)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Erster Alarm bei Vektor", "kein Alarm" if traj.alarm_t is None else str(traj.alarm_t), delta="Fehlalarm vor der Änderung" if traj.false_alarm else None, delta_color="off")
c2.metric("Vektoren nach dem Beginn der Änderung", "–" if traj.delay is None else str(traj.delay), delta=f"Änderung ab Vektor {S.nu + 1}", delta_color="off")
c3.metric("Alarmierender Kanal", "–" if traj.alarm_channel is None else str(traj.alarm_channel), delta="T² nennt keinen Kanal" if S.karte == C.KARTE_T2 else None, delta_color="off")
c4.metric("Exakte ARL₁ dieser Karte", "nicht gefroren" if arl1 is None else de(arl1, 1), delta="ν = 0, zero-state", delta_color="off")
st.plotly_chart(V.build_trajectory(traj, STAT_NAME[S.karte]), width="stretch", key="traj_chart")
st.caption(f"Zufalls-Seed {S.seed}: gleicher Zufalls-Seed, gleiche Vektoren (SplitMix64). Gestrichelt: Schwelle; gepunktet: Beginn der Änderung; ✕: erster Alarm.")
kind, text = EV.verdict_exp1(S.karte, S.p, m1, S.d, S.nu, traj.alarm_t, traj.alarm_channel, traj.delay, arl1, rl["median"] if rl else None)
getattr(st, kind)(text)
if S.nu > 0 and S.karte in (C.KARTE_CUSUM, C.KARTE_EWMA):
    st.caption(TX.EXP1_HINWEIS)

st.markdown("**Verteilung der Lauflänge bis zum Alarm** (Vektoren; Shewhart-Typ exakt geometrisch, CUSUM ohne Änderung aus dem Gitter m = 48 ohne Extrapolation, gefroren, das Mittel liegt bei p = 20 um 1,2 % unter 370,4):")
rows = []
for nm, summ in (("ohne Änderung (ARL₀ 370,4)", EV.null_summary(S.karte, S.p)), ("nach dem Sprung (ν = 0)", rl)):
    if summ is None:
        rows.append({"Lauflänge": nm, "Mittel": "nicht gemessen", "Median": "–", "5 %-Quantil": "–", "95 %-Quantil": "–", "Alarm bis Vektor 100": "–"})
    else:
        rows.append({"Lauflänge": nm, "Mittel": de(summ["mean"], 1), "Median": str(summ["median"]), "5 %-Quantil": str(summ["q05"]), "95 %-Quantil": str(summ["q95"]),
                     "Alarm bis Vektor 100": pct(summ["p100"], 2 if nm.startswith("ohne") else 1)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Ohne Änderung ist die Verteilung bei Shewhart-Typ für jedes p dieselbe (nur die Alarmwahrscheinlichkeit je Vektor zählt); bei CUSUM ähnlich, nicht geometrisch. "
           "Nach dem Sprung ist die Verteilung der Karten mit Gedächtnis nicht gerechnet.")

# =====================================================================================================================================================
# Experiment 2
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 2: Die Strafe der Kanalzahl")
st.markdown(TX.EXP2_FRAGE)
all_p = tuple(range(1, 21))
series = {"T²": (C.COLOR_T2, [(q, K.arl_t2(q, S.d * S.d)) for q in all_p]),
          "Getrennte Shewhart (Šidák)": (C.COLOR_SHEW, [(q, K.arl_sep(q, 1, S.d)) for q in all_p]),
          "Getrennte CUSUM": (C.COLOR_CUSUM, [(q, REF.cusum_arl1(q, 1, S.d)) for q in C.P_OPTIONS]),
          "Getrennte EWMA": (C.COLOR_EWMA, [(q, REF.ewma_arl1(q, S.d)) for q in C.P_OPTIONS])}
st.plotly_chart(V.build_penalty(all_p, series, S.d), width="stretch", key="penalty_chart")
rows = []
for q in C.P_OPTIONS:
    pr = K.penalty_row(q, S.d)
    rows.append({"p": q, "Schranke T²": de(pr["ucl"], 2), "z je Kanal (Šidák)": de(pr["z"], 4), "ARL₀ je Kanal": de(pr["arl0_kanal"], 1), "h* CUSUM": de(REF.cusum_h(q), 4), "H* EWMA": de(REF.ewma_H(q), 4),
                 "ARL₁ T²": de(pr["t2"], 2), "ARL₁ Shewhart": de(pr["sep"], 2), "ARL₁ CUSUM": de(REF.cusum_arl1(q, 1, S.d), 2), "ARL₁ EWMA": de(REF.ewma_arl1(q, S.d), 2)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"Ein Kanal springt um {de(S.d, 1)} σ; gemeinsame ARL₀ 370,4 (Vektoren). T² und Shewhart exakt; CUSUM und EWMA Gitter mit Richardson, auf etwa 0,4 % genau (gefroren, Stand {REF.meta()['stand']}); "
           "zero-state. Die Karten mit Gedächtnis sind nur für p = 1, 2, 5, 10, 20 gerechnet.")
faster = K.t2_faster_cells()
kind, text = EV.verdict_exp2(S.d, 1, 20, K.arl_t2(1, S.d * S.d), K.arl_t2(20, S.d * S.d), K.arl_sep(20, 1, S.d), REF.cusum_arl1(1, 1, S.d), REF.cusum_arl1(20, 1, S.d), len(faster), 16)
getattr(st, kind)(text)
st.markdown(TX.EXP2_NEGATIV)

# =====================================================================================================================================================
# Experiment 3
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 3: Korrelation und Richtung")
st.markdown(TX.EXP3_FRAGE)
res3 = _exp3(S.korr_modell, S.rho, S.richtung, S.sprung, S.gleich)
if res3 is None:
    st.warning("Für p = 5 sind nur Korrelationen ρ ≥ 0 gerechnet (negative Gleichkorrelation bei p ≥ 3 ist nicht gemessen); die Seitenleiste verlangt hier ρ ≥ 0.")
else:
    mu, row = res3
    p3 = R.dim(S.korr_modell)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nichtzentralität λ = μ'Σ⁻¹μ", de(row["lam"], 2), delta=f"Sprung μ = ({', '.join(de(float(v), 2) for v in mu)})", delta_color="off")
    c2.metric("ARL₁ T²", de(row["t2"], 2), delta="hängt nur an λ", delta_color="off")
    c3.metric("ARL₁ getrennt (neu geeicht)", de(row["sep_new"], 2), delta=f"{de_s(row['sep_new'] - row['t2'], 2)} gegen T²", delta_color="off")
    c4.metric("ARL₀ der Šidák-Schranke", de(row["arl0_sidak"], 1), delta=f"{de_s(row['arl0_sidak'] - C.GAMMA0, 1)} gegen 370,4", delta_color="off")
    left, right = st.columns(2)
    with left:
        if S.korr_modell == C.KORR_P2:
            st.plotly_chart(V.build_plane(S.rho, K.ucl_t2(2), row["z_sidak"], row["z_new"], mu), width="stretch", key="plane_chart")
            st.caption("Ebene der zwei Kanäle: ein Vektor außerhalb der roten Ellipse alarmiert T², einer außerhalb des Rechtecks die getrennten Karten. Der Sprung μ ist der grüne Pfeil.")
        else:
            st.info("Für p = 5 gibt es keine Ebene zu zeichnen; die Balken rechts zeigen die ARL₁ der drei Karten.")
    with right:
        st.plotly_chart(V.build_bars([("T²", row["t2"], C.COLOR_T2), ("getrennt, neu geeicht", row["sep_new"], C.COLOR_SHEW), ("getrennt, Šidák-Schranke", row["sep_sidak"], "#a07010")],
                                     "ARL₁ (Vektoren bis zum Alarm)"), width="stretch", key="corr_bars")
    kind, text = EV.verdict_exp3(S.korr_modell, S.rho, S.richtung, row["lam"], row["t2"], row["sep_new"], row["sep_sidak"], row["arl0_sidak"])
    getattr(st, kind)(text)
    st.markdown("**Gleiche Nichtzentralität (λ = 4) in jeder Richtung** (bei dieser Korrelation; Sprung so skaliert, dass μ'Σ⁻¹μ = 4):")
    rows = []
    for nm in C.RICHTUNGEN:
        mu_g, r_g = _exp3(S.korr_modell, S.rho, nm, 1.0, True)
        rows.append({"Richtung": nm, "Sprung μ": "(" + ", ".join(de(float(v), 2) for v in mu_g) + ")", "λ": de(r_g["lam"], 2), "ARL₁ T²": de(r_g["t2"], 2), "ARL₁ getrennt (neu geeicht)": de(r_g["sep_new"], 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Schranken bei ρ = {de(S.rho, 2)}: z neu geeicht {de(row['z_new'], 4)}, z nach Šidák {de(row['z_sidak'], 4)}. Zähleinheit Vektoren bis zum Alarm; Σ bekannt; gleiche Varianz, Mittel 0.")
st.warning(TX.EXP3_FALLE)

# =====================================================================================================================================================
# Experiment 4
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 4: Verdünnung, wie viele Kanäle springen?")
st.markdown(TX.EXP4_FRAGE)
mv = S.vd_m_eff
if S.vd_m > S.vd_p:
    st.caption(f"m = {S.vd_m} ist größer als p = {S.vd_p}: es springen alle {S.vd_p} Kanäle.")
dv = K.dilution_d(mv, S.vd_vergleich)
rowv = K.dilution_row(S.vd_p, mv, dv)
cu_v = REF.cusum_arl1(S.vd_p, mv, dv)
ms = C.VD_M_OPTIONS[S.vd_p]
ser = {"T²": (C.COLOR_T2, [(k, K.dilution_row(S.vd_p, k, K.dilution_d(k, S.vd_vergleich))["t2"]) for k in ms]),
       "Getrennte Shewhart": (C.COLOR_SHEW, [(k, K.dilution_row(S.vd_p, k, K.dilution_d(k, S.vd_vergleich))["sep"]) for k in ms])}
if S.vd_summe:
    ser["Summenkarte (Richtung bekannt)"] = (C.COLOR_SUM, [(k, K.dilution_row(S.vd_p, k, K.dilution_d(k, S.vd_vergleich))["sum"]) for k in ms])
ser["Getrennte CUSUM"] = (C.COLOR_CUSUM, [(k, REF.cusum_arl1(S.vd_p, k, K.dilution_d(k, S.vd_vergleich))) for k in ms])
st.plotly_chart(V.build_dilution(ms, ser, "ARL₁ (Vektoren)"), width="stretch", key="dilution_chart")
rows = []
for k in ms:
    dk = K.dilution_d(k, S.vd_vergleich)
    rk = K.dilution_row(S.vd_p, k, dk)
    rows.append({"m": k, "d": de(dk, 3), "λ = m·d²": de(rk["lam"], 2), "ARL₁ T²": de(rk["t2"], 2), "ARL₁ getrennt Shewhart": de(rk["sep"], 2),
                 "ARL₁ Summenkarte": de(rk["sum"], 2) if S.vd_summe else "ausgeblendet", "ARL₁ getrennte CUSUM": de(REF.cusum_arl1(S.vd_p, k, dk), 2)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"p = {S.vd_p} unabhängige Kanäle, gemeinsame ARL₀ 370,4, Vergleich: {S.vd_vergleich}. T², Shewhart und Summenkarte exakt; CUSUM Gitter, auf etwa 0,4 % genau, zero-state.")
kind, text = EV.verdict_exp4(S.vd_p, mv, S.vd_vergleich, dv, rowv["lam"], rowv["t2"], rowv["sep"], rowv["sum"] if S.vd_summe else None, cu_v)
getattr(st, kind)(text)
st.markdown(TX.EXP4_NEGATIV)

# =====================================================================================================================================================
# Experiment 5
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 5: Was bringt das Gedächtnis?")
st.markdown(TX.EXP5_FRAGE)
mem = EV.memory_rows(S.p, S.d)
tab = [{"Karte": r_["karte"], "ARL₁ zero-state": de(r_["arl1"], 2), "ARL₁ steady-state": ("nicht gemessen" if r_["steady"] is None else de(r_["steady"], 2)), "Herkunft": r_["art"]} for r_ in mem]
st.plotly_chart(V.build_bars([(r_["karte"], r_["arl1"], c_) for r_, c_ in zip(mem, (C.COLOR_T2, C.COLOR_SHEW, C.COLOR_CUSUM, C.COLOR_EWMA, C.COLOR_MEWMA))], "ARL₁ (Vektoren bis zum Alarm)"),
                width="stretch", key="memory_bars")
st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
sim = REF.mewma_arl1_sim(S.p, S.d)
steady = REF.cusum_steady(S.p, S.d)
kind, text = EV.verdict_exp5(S.p, S.d, mem[0]["arl1"], mem[1]["arl1"], mem[2]["arl1"], mem[3]["arl1"], sim[0], sim[1], None if steady is None else steady[1])
getattr(st, kind)(text)
with st.spinner(f"Simuliere {S.r} Läufe der MEWMA (p = {S.p}, d = {de(S.d, 1)}) …"):
    live = _mewma_live(S.p, S.d, S.r, S.seed + 1)
l1, l2, l3 = st.columns(3)
l1.metric("MEWMA ARL₁, Live-Lauf", f"{de(live.mean, 2)} ± {de(live.se, 2)}", delta=f"{live.R} Läufe, Zufalls-Seed {S.seed + 1}", delta_color="off")
l2.metric("MEWMA ARL₁, gefroren", f"{de(sim[0], 2)} ± {de(sim[1], 2)}", delta=f"{sim[2]} Läufe", delta_color="off")
l3.metric("Rechenzeit des Live-Laufs", f"{de(live.seconds, 2)} s", delta="Rechenzeit dieser Maschine, kein Befund", delta_color="off")
st.caption("Standardfehler der Simulation; der Live-Lauf und die gefrorene Zahl sollen im Rahmen ihrer Standardfehler übereinstimmen. Die MEWMA-Schwelle ist aus der radialen Markov-Kette geeicht (ARL₀ aus der Kette mit n = 400 Ringen: bei p = 1 0,13 % Abweichung gegen n = 800, sonst unter 0,02 %).")
st.markdown(TX.EXP5_NEGATIV)

# =====================================================================================================================================================
# Grenzen und Formeln
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATH_INTRO)
    st.markdown(TX.CONVENTIONS_MARKDOWN)
    st.markdown(TX.MATH_MODEL)
    st.markdown(TX.MATH_TARGET)
    st.markdown(TX.MATH_T2_1)
    st.markdown(TX.MATH_T2_2)
    st.markdown(TX.MATH_SEP_1)
    st.markdown(TX.MATH_SEP_2)
    st.markdown(TX.MATH_SUM)
    st.markdown(TX.MATH_GEO_1)
    st.markdown(TX.MATH_GEO_2)
    st.markdown(TX.MATH_MEM_1)
    st.markdown(TX.MATH_CUSUM)
    st.markdown(TX.MATH_EWMA)
    st.markdown(TX.MATH_MEWMA)
    st.markdown(TX.MATH_MIN_1)
    st.markdown(TX.MATH_MIN_2)
    st.markdown(TX.MATH_RICH)
    st.markdown(TX.MATH_CORR_1)
    st.markdown(TX.MATH_CORR_2)
    st.markdown(TX.MATH_CORR_3)
    st.markdown(TX.MATH_LIT)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html)."
)
