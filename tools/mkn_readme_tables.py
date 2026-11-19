"""Erzeugt die Zahlentabellen der README aus den Rechnungen der Demo (exakte Funktionen und gefrorene Tabellen).

    python tools/mkn_readme_tables.py        # druckt alle Tabellen als Markdown

tests/test_claims.py prüft, dass jede dieser Tabellen wörtlich in der README steht: eine Zahl in der README ist damit immer eine Zahl der Rechnung. Format: Dezimalkomma,
Zähleinheit Beobachtungsvektoren bis zum ersten Alarm, gemeinsame ARL₀ 370,398."""

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mkn_cards as K  # noqa: E402
import mkn_constants as C  # noqa: E402
import mkn_corr as R  # noqa: E402
import mkn_evaluation as EV  # noqa: E402
import mkn_reference as REF  # noqa: E402

de = EV.de


def _tab(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(r) + " |" for r in rows])


def tab_eichung() -> str:
    rows = []
    for p in C.P_OPTIONS:
        z, q = K.z_sidak(p)
        rows.append([str(p), de(K.ucl_t2(p), 3), de(z, 4), de(1 / q, 1), de(REF.cusum_h(p), 4), de(REF.ewma_H(p), 4), de(REF.mewma_H(p), 4)])
    return _tab(["p", "Schranke T² (χ²)", "z je Kanal (Šidák)", "ARL₀ je Kanal", "h* CUSUM", "H* EWMA", "H* MEWMA"], rows)


def tab_ohne_gedaechtnis() -> str:
    rows = []
    for p in C.P_OPTIONS:
        rows.append([str(p)] + [f"{de(K.arl_t2(p, d * d), 2)} / {de(K.arl_sep(p, 1, d), 2)}" for d in C.D_OPTIONS])
    return _tab(["p"] + [f"d = {de(d, 1)}: T² / getrennt" for d in C.D_OPTIONS], rows)


def tab_mit_gedaechtnis() -> str:
    rows = []
    for p in C.P_OPTIONS:
        rows.append([str(p), " / ".join(de(REF.cusum_arl1(p, 1, d), 2) for d in C.D_OPTIONS), " / ".join(de(REF.ewma_arl1(p, d), 2) for d in C.D_OPTIONS)])
    return _tab(["p", "CUSUM je Kanal: d = 0,5 / 1 / 2 / 3", "EWMA je Kanal (λ = 0,1): d = 0,5 / 1 / 2 / 3"], rows)


def tab_vergleich_d1() -> str:
    rows = []
    for p in C.P_OPTIONS:
        t2, sep, cu = K.arl_t2(p, 1.0), K.arl_sep(p, 1, 1.0), REF.cusum_arl1(p, 1, 1.0)
        rows.append([str(p), de(t2, 2), de(sep, 2), de(cu, 2), de(cu / t2, 3)])
    return _tab(["p", "T²", "getrennt (Šidák)", "getrennte CUSUM", "Verhältnis CUSUM / T²"], rows)


def tab_verteilung_ohne_aenderung() -> str:
    s = K.geo_summary(C.P_ALARM0)
    rows = [["Shewhart-Typ (T², getrennt, Summe), jedes p", de(s["mean"], 1), str(s["median"]), str(s["q05"]), str(s["q95"]), de(100 * s["p100"], 2) + " %"]]
    for p in C.P_OPTIONS:
        q = REF.cusum_quantiles0(p)
        rows.append([f"getrennte CUSUM, p = {p}", de(q["mean"], 1), str(q["median"]), str(q["q05"]), str(q["q95"]), de(100 * q["p100"], 1) + " %"])
    return _tab(["Karte", "Mittel", "Median", "5 %-Quantil", "95 %-Quantil", "Alarm bis Vektor 100"], rows)


def tab_korrelation_arl0() -> str:
    rows = []
    for rho in (-0.9, -0.5, 0.0, 0.5, 0.8, 0.95):
        rows.append(["p = 2", de(rho, 2), de(R.arl0_sidak(C.KORR_P2, rho), 1)])
    for rho in (0.0, 0.3, 0.6, 0.9):
        rows.append(["p = 5 gleichkorreliert", de(rho, 2), de(R.arl0_sidak(C.KORR_P5, rho), 1)])
    return _tab(["Kanäle", "ρ", "ARL₀ der getrennten Karten mit Šidák-Schranke"], rows)


def tab_korrelation_richtung() -> str:
    rows = []
    for rho in (0.0, 0.5, 0.8, 0.95):
        for name in (C.RICHTUNG_ENTLANG, C.RICHTUNG_GEGEN, C.RICHTUNG_EIN):
            row = R.arl_row(C.KORR_P2, rho, R.jump_vector(C.KORR_P2, rho, name, 1.0, False))
            rows.append([de(rho, 2), name, de(row["lam"], 2), de(row["t2"], 2), de(row["sep_new"], 2), de(row["sep_sidak"], 2)])
    return _tab(["ρ", "Richtung (d = 1)", "λ", "T²", "getrennt (neu geeicht)", "getrennt (Šidák-Schranke)"], rows)


def tab_gleiche_lambda() -> str:
    rows = []
    for name in (C.RICHTUNG_GEGEN, C.RICHTUNG_SCHIEF, C.RICHTUNG_EIN, C.RICHTUNG_ENTLANG):
        mu = R.jump_vector(C.KORR_P2, 0.8, name, 1.0, True)
        row = R.arl_row(C.KORR_P2, 0.8, mu)
        rows.append([name, de(row["lam"], 2), de(row["t2"], 2), de(row["sep_new"], 2)])
    return _tab(["Richtung (ρ = 0,8)", "λ", "T²", "getrennt (neu geeicht)"], rows)


def tab_p5_korreliert() -> str:
    rows = []
    for rho in (0.0, 0.6, 0.9):
        out = []
        for name in (C.RICHTUNG_ENTLANG, C.RICHTUNG_EIN):
            row = R.arl_row(C.KORR_P5, rho, R.jump_vector(C.KORR_P5, rho, name, 1.0, False))
            out += [de(row["t2"], 2), de(row["sep_new"], 2)]
        rows.append([de(rho, 1)] + out)
    return _tab(["ρ", "alle fünf: T²", "alle fünf: getrennt (neu geeicht)", "ein Kanal: T²", "ein Kanal: getrennt (neu geeicht)"], rows)


def tab_verduennung(p: int, mode: str) -> str:
    rows = []
    for m in C.VD_M_OPTIONS[p]:
        d = K.dilution_d(m, mode)
        r = K.dilution_row(p, m, d)
        cu = REF.cusum_arl1(p, m, d)
        rows.append([str(m), de(d, 3), de(r["lam"], 2), de(r["t2"], 2), de(r["sep"], 2), de(r["sum"], 2), de(cu, 2)])
    return _tab(["m", "d", "λ = m·d²", "T²", "getrennt (Šidák)", "Summenkarte", "getrennte CUSUM"], rows)


def tab_steady() -> str:
    rows = []
    for p in (1, 5):
        cells = []
        for d in (0.5, 1.0, 2.0):
            z, s = REF.cusum_steady(p, d)
            cells.append(f"{de(z, 2)} / {de(s, 2)} ({de(s / z, 3)})")
        rows.append([str(p)] + cells)
    return _tab(["p", "d = 0,5: zero / steady (Verhältnis)", "d = 1: zero / steady (Verhältnis)", "d = 2: zero / steady (Verhältnis)"], rows)


def tab_mewma() -> str:
    rows = []
    for p in C.P_OPTIONS:
        rows.append([str(p)] + [f"{de(REF.mewma_arl1_sim(p, d)[0], 2)} ± {de(REF.mewma_arl1_sim(p, d)[1], 2)}" for d in C.D_OPTIONS])
    return _tab(["p", "d = 0,5", "d = 1", "d = 2", "d = 3"], rows)


def tab_lambda_paar() -> str:
    paar = REF.mewma_lambda_paar()
    rows = []
    for (m, d), nm in (((1, 2.0), "m = 1, d = 2 (λ = 4)"), ((4, 1.0), "m = 4, d = 1 (λ = 4)"), ((1, 1.0), "m = 1, d = 1 (λ = 1)"), ((5, 1.0), "m = 5, d = 1 (λ = 5)")):
        v = paar[REF.case_key(m, d)]
        rows.append([nm, f"{de(v[0], 2)} ± {de(v[1], 2)}"])
    return _tab(["Fall (p = 5, MEWMA)", "ARL₁ (Simulation, 20 000 Läufe)"], rows)


ALL = {
    "eichung": tab_eichung, "ohne_gedaechtnis": tab_ohne_gedaechtnis, "mit_gedaechtnis": tab_mit_gedaechtnis, "vergleich_d1": tab_vergleich_d1,
    "verteilung_ohne_aenderung": tab_verteilung_ohne_aenderung, "korrelation_arl0": tab_korrelation_arl0, "korrelation_richtung": tab_korrelation_richtung,
    "gleiche_lambda": tab_gleiche_lambda, "p5_korreliert": tab_p5_korreliert, "steady": tab_steady, "mewma": tab_mewma, "lambda_paar": tab_lambda_paar,
}

if __name__ == "__main__":
    for name, f in ALL.items():
        print(f"\n### {name}\n{f()}")
    for p in (5, 10):
        for mode in C.VD_VERGLEICHE:
            print(f"\n### verduennung p = {p}, {mode}\n{tab_verduennung(p, mode)}")
