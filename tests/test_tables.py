"""Gefrorene Tabellen (mkn_cusum_frozen.json): Aufbau, Anschluss an die Vormessung (Plan 06 / V15), Neurechnung mit Toleranz, Konsistenz.

Die Zahlen der Vormessung stehen hier als Erwartung mit ihrer Toleranz (Rundung auf die gedruckten Stellen); wo der Bau davon abweicht, steht es in der README unter „Befunde und
Korrekturen“. Die Gitterzahlen sind auf etwa 0,4 % genau; die Neurechnung vergleicht mit relativer Toleranz, nicht bitgenau (BLAS und libm weichen plattformabhängig in der
letzten Stelle ab)."""

import importlib.util
import math
from pathlib import Path

import pytest

import mkn_constants as C
import mkn_reference as REF

ROOT = Path(__file__).resolve().parent.parent

V15_H = {1: 4.7749, 2: 5.4498, 5: 6.3507, 10: 7.0360, 20: 7.7230}
V15_CUSUM = {1: (35.27, 9.93, 3.86, 2.49), 2: (42.03, 11.24, 4.31, 2.75), 5: (52.07, 13.00, 4.91, 3.12), 10: (60.35, 14.35, 5.37, 3.40), 20: (69.13, 15.71, 5.82, 3.67)}
V15_EWMA_H = {1: 0.6198, 2: 0.6771, 5: 0.7462, 10: 0.7943, 20: 0.8394}
V15_EWMA = {1: (28.23, 9.74, 4.18, 2.76), 2: (34.72, 11.07, 4.59, 3.00), 5: (45.64, 12.93, 5.10, 3.29), 10: (56.18, 14.41, 5.48, 3.50), 20: (69.17, 15.98, 5.84, 3.70)}


def test_aufbau_und_meta():
    T = REF.tables()
    assert set(T["cusum"]) == set(T["ewma"]) == set(T["mewma"]) == {str(p) for p in C.P_OPTIONS}
    assert set(T["steady"]) == {"1", "5"}
    for k in ("quelle", "stand", "python", "numpy", "scipy", "massstab", "cusum", "ewma", "mewma", "steady", "laufzeit_sekunden"):
        assert T["meta"][k] not in (None, "")
    assert T["meta"]["stand"] == C.FROZEN_STAND and "370,398" in T["meta"]["massstab"]


@pytest.mark.parametrize("p", C.P_OPTIONS)
def test_cusum_schwelle_und_arl1_gegen_vormessung(p):
    assert abs(REF.cusum_h(p) - V15_H[p]) < 5e-4
    for d, soll in zip(C.D_OPTIONS, V15_CUSUM[p]):
        assert abs(REF.cusum_arl1(p, 1, d) - soll) < 0.006, (p, d)
    assert abs(REF.cusum_arl0(p) / C.GAMMA0 - 1) < 1e-4


@pytest.mark.parametrize("p", C.P_OPTIONS)
def test_ewma_grenze_und_arl1_gegen_vormessung(p):
    assert abs(REF.ewma_H(p) - V15_EWMA_H[p]) < 5e-4
    for d, soll in zip(C.D_OPTIONS, V15_EWMA[p]):
        assert abs(REF.ewma_arl1(p, d) - soll) < 0.006, (p, d)
    assert abs(REF.ewma_arl0(p) / C.GAMMA0 - 1) < 1e-4
    assert abs(REF.tables()["ewma"][str(p)]["arl1_kontrolle_d1"] / REF.ewma_arl1(p, 1.0) - 1) < 1e-5      # Kontrollgitter 160/320 gleich


def test_verduennung_cusum_p5_gegen_vormessung():
    soll = {(1, 2.0): 4.909, (2, math.sqrt(2)): 6.159, (4, 1.0): 7.690, (5, math.sqrt(0.8)): 8.238, (1, 1.0): 13.002, (2, 1.0): 9.705, (5, 1.0): 7.207}
    for (m, d), s in soll.items():
        assert abs(REF.cusum_arl1(5, m, d) - s) < 0.002, (m, d)
    assert REF.cusum_arl1(5, 3, 1.0) is not None and REF.cusum_arl1(5, 3, 2 / math.sqrt(3)) is not None
    assert REF.cusum_arl1(5, 1, 0.123) is None          # nicht gefrorene Fälle melden None, nie eine erfundene Zahl


def test_mewma_schwelle_arl0_und_arl1_gegen_vormessung():
    """H* mit n = 400 Ringen: 10,0750 / 16,2895 (p = 2 / 5), die Vormessung (n = 150) nennt 10,0764 / 16,2902: Abweichung unter 0,02 %. ARL₁ Simulation, 20 000 Läufe."""
    assert abs(REF.mewma_H(2) - 10.0764) < 2e-3 and abs(REF.mewma_H(5) - 16.2902) < 2e-3
    for p in C.P_OPTIONS:
        a = REF.tables()["mewma"][str(p)]["arl0"]
        assert abs(a["400"] / C.GAMMA0 - 1) < 1e-4 and abs(a["800"] / C.GAMMA0 - 1) < 2e-3
    mean, se, R_ = REF.mewma_arl1_sim(5, 1.0)
    assert R_ == 20000 and abs(mean - 14.68) < 4 * math.hypot(se, 0.05)            # Vormessung: 14,68 ± 0,05


def test_mewma_lambda_paar_haengt_an_lambda():
    paar = REF.mewma_lambda_paar()
    a, b = paar[REF.case_key(1, 2.0)], paar[REF.case_key(4, 1.0)]
    assert abs(a[0] - b[0]) < 4 * math.hypot(a[1], b[1])          # (m = 1, d = 2) und (m = 4, d = 1): beide λ = 4
    assert paar[REF.case_key(5, 1.0)][0] < b[0] < paar[REF.case_key(1, 1.0)][0]


def test_steady_state_liegt_4_bis_8_prozent_unter_zero_state():
    ratios = []
    for p in (1, 5):
        for d in (0.5, 1.0, 2.0):
            z, s = REF.cusum_steady(p, d)
            assert abs(z / REF.cusum_arl1(p, 1, d) - 1) < 2e-3          # zero-state aus Abschnitt „steady“ (Gitter 24/32) gleich dem der Haupttabelle (32/48)
            ratios.append(s / z)
    assert len(ratios) == 6 and 0.917 < min(ratios) and max(ratios) < 0.958
    assert REF.cusum_steady(2, 1.0) is None and REF.cusum_steady(5, 3.0) is None


def test_lauflaenge_ohne_aenderung_der_cusum_gegen_vormessung():
    soll = {1: (258, 25, 22.9), 2: (258, 26, 22.7), 5: (258, 27, 22.5), 10: (258, 29, 22.3), 20: (257, 30, 22.2)}
    for p, (med, q05, p100) in soll.items():
        q = REF.cusum_quantiles0(p)
        assert q["median"] == med and q["q05"] == q05 and abs(100 * q["p100"] - p100) < 0.05, p
        assert abs(q["mean"] - 370) < 5          # Gitter m = 48 ohne Extrapolation, Mittel liegt 1 % unter 370,4


def _tool():
    spec = importlib.util.spec_from_file_location("mkn_make_cusum", ROOT / "tools" / "mkn_make_cusum.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_neu_rechnen_an_den_gespeicherten_schwellen_stimmt_mit_tolerance():
    """`tools/mkn_make_cusum.py --check`: ARL₀ aller Karten an den gespeicherten Schwellen (CUSUM, EWMA, MEWMA) und je ein ARL₁, relative Toleranz 0,1 bis 0,2 %."""
    assert _tool().check() == 0

