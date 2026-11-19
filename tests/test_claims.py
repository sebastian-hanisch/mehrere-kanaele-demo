"""Claims: jede Zahl der README wird hier aus der Rechnung bzw. der gefrorenen Tabelle neu erzeugt und muss wörtlich in der README stehen.

Die Tabelle „Zahl → Test“ der README nennt je Zeile die Funktion. Die Zahlentabellen der README stammen aus `tools/mkn_readme_tables.py` (dieselben Funktionen wie die App);
exakte Zahlen (Karten ohne Gedächtnis, korrelierte Karten) werden bei jedem Lauf gerechnet, Gitterzahlen (CUSUM, EWMA) und Simulationen (MEWMA-ARL₁) stehen in
`mkn_cusum_frozen.json` und werden in tests/test_tables.py und tests/test_oracle_memory.py gegen Neurechnung und Simulation geprüft."""

import importlib.util
import math
import re
import time
from pathlib import Path

import mkn_cards as K
import mkn_constants as C
import mkn_corr as R
import mkn_evaluation as EV
import mkn_memory as M
import mkn_normal
import mkn_reference as REF
import mkn_texts as TX

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
de = EV.de


def _im_readme(text):
    assert text in README, f"nicht in der README: {text!r}"


def _tool():
    spec = importlib.util.spec_from_file_location("mkn_readme_tables", ROOT / "tools" / "mkn_readme_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_tabellen_stehen_woertlich_in_der_readme():
    T = _tool()
    for name, f in T.ALL.items():
        _im_readme(f())
    for mode in C.VD_VERGLEICHE:
        _im_readme(T.tab_verduennung(5, mode))
    assert len(T.ALL) == 12


def test_kernbotschaft_ohne_gedaechtnis():
    t2, sep, eins = K.arl_t2(5, 1.0), K.arl_sep(5, 1, 1.0), K.arl_t2(1, 1.0)
    _im_readme(f"{de(t2, 1)} Vektoren (T²) bzw. {de(sep, 1)} (fünf Šidák-Shewhart-Karten) gegen {de(eins, 1)} bei einer Karte für den bekannten Kanal")
    _im_readme(f"(T²: {de(K.arl_t2(20, 1.0), 1)} bei p = 20)")


def test_t2_schneller_in_4_von_16_zellen():
    cells = K.t2_faster_cells()
    assert len(cells) == 4 and all(d == 0.5 for _, d in cells) and [p for p, _ in cells] == [2, 5, 10, 20]
    total = 4 * len(C.D_OPTIONS)          # p = 2, 5, 10, 20 mal d = 0,5 / 1 / 2 / 3
    assert total == 16
    _im_readme("T² ist dabei in 4 von 16 Zellen")
    _im_readme("**4 von 16 Zellen**")
    _im_readme("in 12 von 16 Zellen")                      # 16 - 4: in diesen ist T² nicht schneller
    assert total - len(cells) == 12
    assert K.arl_t2(10, 4.0) == K.arl_t2(10, 2.0 ** 2) and f"{de(K.arl_t2(10, 4.0), 2)} gegen {de(K.arl_sep(10, 1, 2.0), 2)}" == "31,26 gegen 19,02"
    _im_readme("31,26 gegen 19,02")


def test_gedaechtnis_zahlen():
    cu, ew = REF.cusum_arl1(5, 1, 1.0), REF.ewma_arl1(5, 1.0)
    mew, se, R_ = REF.mewma_arl1_sim(5, 1.0)
    assert R_ == 20000
    _im_readme(f"getrennte zweiseitige CUSUM brauchen {de(cu, 1)} Vektoren")
    _im_readme(f"CUSUM {de(cu, 2)}, EWMA {de(ew, 2)}, MEWMA {de(mew, 2)} ± {de(se, 2)} gegen {de(K.arl_t2(5, 1.0), 2)} und {de(K.arl_sep(5, 1, 1.0), 2)}")
    z, s = REF.cusum_steady(5, 1.0)
    assert de(s, 2) == "12,11"
    ratios = [REF.cusum_steady(p, d)[1] / REF.cusum_steady(p, d)[0] for p in (1, 5) for d in (0.5, 1.0, 2.0)]
    assert 0.04 <= 1 - max(ratios) and 1 - min(ratios) <= 0.083          # „4 bis 8 %“: 4,3 bis 8,2 %
    _im_readme("liegt 4 bis 8 % unter dem zero-state-Maß")
    # p = 1, d = 3: Shewhart 2,00 schneller als CUSUM 2,49 und EWMA 2,76
    assert (de(K.arl_t2(1, 9.0), 2), de(REF.cusum_arl1(1, 1, 3.0), 2), de(REF.ewma_arl1(1, 3.0), 2)) == ("2,00", "2,49", "2,76")
    _im_readme("bei p = 1 und d = 3 sind T² und Shewhart-Karte mit 2,00 schneller als CUSUM (2,49) und EWMA (2,76)")
    # Dimensionsstrafe
    _im_readme(f"(9,93 bei p = 1, 15,71 bei p = 20), die der Karten ohne Gedächtnis groß ({de(K.arl_t2(1, 1.0), 2)} auf {de(K.arl_t2(20, 1.0), 2)})")
    assert (de(REF.cusum_arl1(1, 1, 1.0), 2), de(REF.cusum_arl1(20, 1, 1.0), 2)) == ("9,93", "15,71")


def test_korrelation_zahlen():
    mu = {n: R.jump_vector(C.KORR_P2, 0.8, n, 1.0, False) for n in (C.RICHTUNG_ENTLANG, C.RICHTUNG_GEGEN, C.RICHTUNG_EIN)}
    r = {n: R.arl_row(C.KORR_P2, 0.8, m) for n, m in mu.items()}
    g, e, ein = r[C.RICHTUNG_GEGEN], r[C.RICHTUNG_ENTLANG], r[C.RICHTUNG_EIN]
    _im_readme(f"braucht T² gegen die Korrelationsstruktur {de(g['t2'], 2)} Vektoren gegen {de(g['sep_new'], 2)} bei getrennten Karten (etwa 15-mal schneller)")
    assert round(g["sep_new"] / g["t2"]) == 15
    _im_readme(f"bei einem Sprung in einem Kanal {de(ein['t2'], 2)} gegen {de(ein['sep_new'], 2)} ({de(ein['sep_new'] / ein['t2'], 1)}-mal schneller)")
    _im_readme(f"entlang der Korrelation aber {de(e['t2'], 2)} gegen {de(e['sep_new'], 2)} (langsamer)")
    assert e["t2"] > e["sep_new"] and g["t2"] < g["sep_new"] and ein["t2"] < ein["sep_new"]
    _im_readme(f"({de(g['arl0_sidak'], 1)} bei ρ = 0,8, p = 2)")
    _im_readme(f"(p = 5, ρ = 0,6, ein Kanal: {de(R.arl_row(C.KORR_P5, 0.6, R.jump_vector(C.KORR_P5, 0.6, C.RICHTUNG_EIN, 1.0, False))['t2'], 2)} gegen "
               f"{de(R.arl_row(C.KORR_P5, 0.6, R.jump_vector(C.KORR_P5, 0.6, C.RICHTUNG_EIN, 1.0, False))['sep_new'], 2)})")
    _im_readme("(ρ = 0,8: 59,71 gegen 41,27)")
    # Kipppunkt entlang (p = 2, d = 1): T² gewinnt bei ρ = 0,2, verliert bei ρ = 0,3; die Nullstelle der Differenz liegt bei 0,24
    def diff(rho):
        row = R.arl_row(C.KORR_P2, rho, R.jump_vector(C.KORR_P2, rho, C.RICHTUNG_ENTLANG, 1.0, False))
        return row["t2"] - row["sep_new"]
    lo, hi = 0.2, 0.3
    assert diff(lo) < 0 < diff(hi)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if diff(mid) < 0 else (lo, mid)
    assert round(lo, 2) == 0.24
    _im_readme("erst ab ρ ≈ 0,24 (p = 2, d = 1; bei ρ = 0 gewinnt T² mit 27,73 gegen 36,64)")
    _im_readme("für ρ → ±1 melden beide Kanäle dasselbe Ereignis, ARL₀ → 1/q = " + de(K.arl0_per_channel(2), 1))
    _im_readme("T² gibt in jeder Richtung 9,41")


def test_sidak_zaehlung():
    zellen = [(C.KORR_P2, r) for r in (-0.9, -0.5, 0.0, 0.5, 0.8, 0.95)] + [(C.KORR_P5, r) for r in (0.0, 0.3, 0.6, 0.9)]
    assert len(zellen) == 10 and all(R.arl0_sidak(m, r) >= C.GAMMA0 - 1e-9 for m, r in zellen)
    _im_readme("in den 10 von 10 Zellen der Tabelle und in 59 von 59 Zellen des dichten Rasters")
    import numpy as np
    n = 0
    for rho in np.round(np.arange(-0.95, 0.9501, 0.05), 2):
        n += R.arl0_sidak(C.KORR_P2, float(rho)) >= C.GAMMA0 - 1e-6
    for rho in np.round(np.arange(0.0, 0.9501, 0.05), 2):
        n += R.arl0_sidak(C.KORR_P5, float(rho)) >= C.GAMMA0 - 1e-6
    assert n == 59


def test_verduennung_zahlen():
    _im_readme("T² hängt nur an λ (17,93 für alle Zeilen der Energie-Tabelle)")
    assert all(de(K.dilution_row(5, m, K.dilution_d(m, C.VD_GLEICH_ENERGIE))["t2"], 2) == "17,93" for m in C.VD_M_OPTIONS[5])
    _im_readme("177,73 gegen 114,38 bei T²)")
    assert (de(K.arl_sum(5, 1, 1.0), 2), de(K.arl_t2(5, 1.0), 2)) == ("177,73", "114,38")
    # p = 10, λ = 4
    soll = [de(K.dilution_row(10, m, K.dilution_d(m, C.VD_GLEICH_ENERGIE))["sep"], 2) for m in (1, 2, 4, 5, 10)]
    _im_readme("T² bei λ = 4: " + de(K.dilution_row(10, 1, 2.0)["t2"], 2) + "; getrennt " + " / ".join(soll) + " bei m = 1 / 2 / 4 / 5 / 10 mit d = 2/√m")
    # Summenkarte ist bei m = 1 die langsamste der vier, wenn ein Kanal springt (gleiche Höhe)
    row = K.dilution_row(5, 1, 1.0)
    assert row["sum"] > max(row["t2"], row["sep"], REF.cusum_arl1(5, 1, 1.0))
    # je mehr Kanäle den Sprung teilen (gleiche Energie), desto besser die Summenkarte und desto schlechter getrennte Shewhart-Karten
    for p_, ms in ((5, (1, 2, 3, 4, 5)), (10, (1, 2, 4, 5, 10))):
        s = [K.dilution_row(p_, m, K.dilution_d(m, C.VD_GLEICH_ENERGIE)) for m in ms]
        assert [x["sum"] for x in s] == sorted([x["sum"] for x in s], reverse=True) and [x["sep"] for x in s] == sorted(x["sep"] for x in s), p_
        for mode in C.VD_VERGLEICHE:       # ein springender Kanal: die Summenkarte ist die langsamste der vier
            r1 = K.dilution_row(p_, 1, K.dilution_d(1, mode))
            assert r1["sum"] > max(r1["t2"], r1["sep"], REF.cusum_arl1(p_, 1, K.dilution_d(1, mode))), (p_, mode)
    _im_readme("Bei **gleicher Energie** gilt: je mehr Kanäle den Sprung teilen")


def test_kleinstmodell_zahlen_der_readme():
    """Die Brüche und Prozente stehen in tests/test_handbeispiele.py (exakt); hier nur, dass die README dieselben Zahlen nennt."""
    for s in ("ARL₀ 12, ARL₁ 9/2", "Median der Lauflänge ohne Änderung 9", "ARL₀ = 3152/451 = 6,9889 (um 41,8 % kleiner als 12)", "14308/3629 = 3,9427 (um 12,4 % kleiner als 4,5)"):
        _im_readme(s)
    assert (round(3152 / 451, 4), round(14308 / 3629, 4), round(100 * (1 - 3152 / 451 / 12), 1), round(100 * (1 - 14308 / 3629 / 4.5), 1)) == (6.9889, 3.9427, 41.8, 12.4)


def test_befunde_und_korrekturen_zahlen():
    mew, se, _ = REF.mewma_arl1_sim(5, 1.0)
    _im_readme(f"p = 5, ein Kanal um 1 σ: {de(mew, 2)} ± {de(se, 2)} (Vormessung 14,68 ± 0,05")
    assert abs(mew - 14.68) < 2 * math.hypot(se, 0.05)                      # „innerhalb von zwei kombinierten Standardfehlern“
    paar = REF.mewma_lambda_paar()
    a, b = paar[REF.case_key(1, 2.0)], paar[REF.case_key(4, 1.0)]
    _im_readme(f"{de(a[0], 2)} ± {de(a[1], 2)} (m = 1, d = 2) und {de(b[0], 2)} ± {de(b[1], 2)} (m = 4, d = 1); die Vormessung nennt 5,92 und 5,90")
    _im_readme(f"p = 2 / 5 gibt {de(REF.mewma_H(2), 4)} / {de(REF.mewma_H(5), 4)} statt 10,0764 / 16,2902")
    assert abs(REF.mewma_H(2) / 10.0764 - 1) < 2e-4 and abs(REF.mewma_H(5) / 16.2902 - 1) < 2e-4          # „unter 0,02 %“
    # p = 1: n = 150 lag um 0,7 % daneben
    assert abs(M.chain_arl(M.mewma_chain(1, 0.1, 7.3162, 400)) / C.GAMMA0 - 1.0) > 0.006
    _im_readme(f"mit n = 400 ist H* = {de(REF.mewma_H(1), 4)} (ARL₀ bei n = 800: {de(REF.tables()['mewma']['1']['arl0']['800'], 1)})")
    _im_readme("(ARL₀ 372,9 bei n = 400;")
    assert de(M.chain_arl(M.mewma_chain(1, 0.1, 7.3162, 400)), 1) == "372,9"
    _im_readme(f"CUSUM bei p = 10, d = 3 gibt {de(REF.cusum_arl1(10, 1, 3.0), 3)} (die Vormessung druckt 3,40, diese README 3,39)")
    assert de(REF.cusum_arl1(10, 1, 3.0), 2) == "3,39"
    m3 = {mode: K.dilution_row(5, 3, K.dilution_d(3, mode)) for mode in C.VD_VERGLEICHE}
    h, e = m3[C.VD_GLEICH_HOEHE], m3[C.VD_GLEICH_ENERGIE]
    _im_readme(f"gleiche Höhe T² {de(h['t2'], 2)} / getrennt {de(h['sep'], 2)} / Summe {de(h['sum'], 2)} / CUSUM {de(REF.cusum_arl1(5, 3, 1.0), 2)}; "
               f"gleiche Energie {de(e['t2'], 2)} / {de(e['sep'], 2)} / {de(e['sum'], 2)} / {de(REF.cusum_arl1(5, 3, 2 / math.sqrt(3)), 2)}")
    # der EWMA bei p = 1 gleich der MEWMA-Grenze
    assert abs(math.sqrt(REF.mewma_H(1) * 0.1 / 1.9) / REF.ewma_H(1) - 1) < 3e-3
    # Zufalls-Seed 15: bei p = 5, ν = 50, ein Kanal um 1 σ alarmiert keine der vier Karten vor der Änderung
    import mkn_scenario as SC
    for k in C.KARTEN:
        t = SC.run_trajectory(5, 1, 1.0, 50, 15, k)
        assert t.alarm_t is not None and t.alarm_t > 50 and not t.false_alarm, k
    assert C.DEFAULT_SEED == 15 and C.DEFAULT_NU == 50 and C.DEFAULT_P == 5
    _im_readme("Der voreingestellte Zufalls-Seed 15")


def test_zufallsquelle_ist_schnell():
    t = time.perf_counter()
    mkn_normal.Stream(1).normal(1_000_000)
    assert time.perf_counter() - t < 1.0          # die README nennt rund 0,04 s (Messung dieser Maschine); hier nur die Größenordnung
    _im_readme("10⁶ Normalzahlen in rund 0,04 s")
    _im_readme("rund 82 µs je Zahl")


def test_zahlen_in_texten_und_presets():
    """Zahlen, die im Text der App und in den Hilfetexten der Presets stehen, stammen aus der Rechnung."""
    assert de(K.arl0_per_channel(20), 0).replace(".", "") in "7398" and "7 398" in TX.EXP2_FRAGE
    assert "von 370,4 bei p = 1 auf 7 398 bei p = 20" in TX.EXP2_FRAGE and de(K.arl0_per_channel(1), 1) == "370,4"
    assert "12 von 16" in TX.EXP2_NEGATIV and 16 - len(K.t2_faster_cells()) == 12
    h = C.PRESET_HELP
    assert f"{de(K.arl_t2(20, 1.0), 1)} statt {de(K.arl_t2(5, 1.0), 1)}" in h["p = 20 Kanäle"]
    g = R.arl_row(C.KORR_P2, 0.8, R.jump_vector(C.KORR_P2, 0.8, C.RICHTUNG_GEGEN, 1.0, False))
    e = R.arl_row(C.KORR_P2, 0.8, R.jump_vector(C.KORR_P2, 0.8, C.RICHTUNG_ENTLANG, 1.0, False))
    assert f"T² braucht {de(g['t2'], 2)} Vektoren, getrennte Karten {de(g['sep_new'], 2)}" in h["Korreliert, Sprung gegen die Korrelation (ρ = 0,8)"]
    assert f"T² braucht {de(e['t2'], 2)} Vektoren, getrennte Karten {de(e['sep_new'], 2)}" in h["Korreliert, Sprung entlang (ρ = 0,8)"]
    assert f"T² braucht dann {de(K.arl_t2(5, 5.0), 1)} Vektoren" in h["Alle fünf springen"]
    assert f"braucht sie {de(REF.cusum_arl1(5, 1, 1.0), 1)} Vektoren statt {de(K.arl_t2(5, 1.0), 1)} bei T²" in h["Mit Gedächtnis: getrennte CUSUM"]
    # Energie-Preset: Shewhart langsam, Summenkarte und CUSUM schnell (m = 4, p = 5)
    r4 = K.dilution_row(5, 4, 1.0)
    assert r4["sep"] > r4["t2"] > r4["sum"] and r4["sep"] > REF.cusum_arl1(5, 4, 1.0) and REF.cusum_arl1(5, 4, 1.0) < r4["t2"]
    # Verlaufstext: 400 Vektoren
    assert C.N_VERLAUF == 400 and "400 Vektoren" in TX.EXP1_FRAGE


def test_app_texte_stehen_in_der_readme():
    """Zwei Fassungen desselben Textes (App und README) werden verglichen, nicht von Hand synchron gehalten."""
    _im_readme(TX.CONVENTIONS_MARKDOWN.strip())
    # die Quellen-Aussagen der README und des Formelblocks: dieselben Stufen
    for s in ("Xie und Siegmund", "NIST pmc543", "Vanhatalo und Kulahci", "Lowry u. a. 1992", "**S**"):
        assert s in README and s.replace("**", "") in TX.MATH_LIT.replace("**", "")


def test_sprachregeln_in_readme_und_app_texten():
    alle = README + " ".join(v for k, v in vars(TX).items() if isinstance(v, str)) + (ROOT / "app.py").read_text(encoding="utf-8")
    for verboten in ("Saatwert", "Schwanz", "trägt", "tragen", "geplant", "TODO"):
        stellen = [m.start() for m in re.finditer(verboten, alle)]
        # das TODO im Platzhalter des Live-Links (HTML-Kommentar) ist beabsichtigt
        if verboten == "TODO":
            stellen = [s for s in stellen if "TODO nach dem Deploy" not in alle[s: s + 25]]
        assert not stellen, (verboten, [alle[s - 30: s + 30] for s in stellen])
    # „Hotelling“ nur als Hotelling-T² (und als Jahreszahl-Quelle „Hotelling 1947“)
    ohne = [m.start() for m in re.finditer(r"Hotelling(?!-T²|“)", alle) if not alle[m.start(): m.start() + 14].startswith("Hotelling 1947")]
    assert not ohne, [alle[s - 20: s + 30] for s in ohne]
    # echte Umlaute: keine ae/oe/ue-Schreibung typischer Wörter
    for ascii_wort in ("Aenderung", "Kanaele", "Verzoegerung", "Laeufe"):
        assert ascii_wort not in README
    assert "Zufalls-Seed" in README
