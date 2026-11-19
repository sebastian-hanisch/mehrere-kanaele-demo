"""Handbeispiele (Plan 06, `handbeispiele_06.py`) als exakte Tests, ohne den Code der Demo zu benutzen, wo die Handrechnung steht.

Teil 1 bis 3: ein Kleinstmodell, das man von Hand rechnet: jeder Kanal liefert je Beobachtung eine ganze Zahl ξ ∈ {-1, 0, +1}, ohne Änderung mit den
Wahrscheinlichkeiten (1/4, 1/2, 1/4), nach der Änderung mit (1/8, 3/8, 1/2); je Kanal ein einseitiger CUSUM S' = max(0, S + ξ), Alarm bei S' ≥ h; gemeinsame Karte:
Alarm, sobald IRGENDEIN Kanal alarmiert. Wege: lineares Gleichungssystem in Brüchen (`fractions`), vollständige Aufzählung aller Pfade (exakte Überlebenswahrscheinlichkeit),
Minimum unabhängiger Lauflängen (die Produktfunktionen der Demo `mkn_memory.surv_iter` und `prod_arl` laufen hier auf der diskreten Kette: sie sind nicht an die Normalverteilung gebunden).
Teil 4 bis 7: Gaußmodell in geschlossener Form (T² bei p = 2, Summenkarte, Grenzfall der Korrelation, geometrische Lauflänge)."""

import math
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from scipy.stats import norm

import mkn_cards as K
import mkn_constants as C
import mkn_corr as R
import mkn_memory as M

VALS = (-1, 0, 1)
P0 = {-1: F(1, 4), 0: F(1, 2), 1: F(1, 4)}
P1 = {-1: F(1, 8), 0: F(3, 8), 1: F(1, 2)}


def solve_frac(A, b):
    """Gauß-Elimination in Fraction (A quadratisch)."""
    n = len(A)
    Mx = [list(map(F, A[i])) + [F(b[i])] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if Mx[r][c] != 0)
        Mx[c], Mx[piv] = Mx[piv], Mx[c]
        pv = Mx[c][c]
        Mx[c] = [x / pv for x in Mx[c]]
        for r in range(n):
            if r != c and Mx[r][c] != 0:
                f = Mx[r][c]
                Mx[r] = [x - f * y for x, y in zip(Mx[r], Mx[c])]
    return [Mx[i][n] for i in range(n)]


def chan_step(s, x, h):
    t = max(0, s + x)
    return None if t >= h else t


def chan_Q(P, h):
    Q = [[F(0)] * h for _ in range(h)]
    for s in range(h):
        for x in VALS:
            t = chan_step(s, x, h)
            if t is not None:
                Q[s][t] += P[x]
    return Q


def arl_single(P, h):
    Q = chan_Q(P, h)
    A = [[(1 if i == j else 0) - Q[i][j] for j in range(h)] for i in range(h)]
    return solve_frac(A, [1] * h)[0]


def joint_arl(Ps, h):
    """Maximum-Regel über len(Ps) Kanäle mit je eigener Verteilung: Zustand = Tupel der Kanalzustände; ARL vom Nullzustand."""
    p = len(Ps)
    states = list(product(range(h), repeat=p))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    A = [[F(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for s in states:
        for xs in product(VALS, repeat=p):
            pr, nxt, alarm = F(1), [], False
            for k in range(p):
                pr *= Ps[k][xs[k]]
                t = chan_step(s[k], xs[k], h)
                alarm = alarm or t is None
                nxt.append(t)
            if not alarm:
                A[idx[s]][idx[tuple(nxt)]] -= pr
    return solve_frac(A, [1] * n)[idx[tuple([0] * p)]]


def surv_enum(Ps, h, nmax):
    """Aufzählung ALLER Pfade: exakte P(RL > n), n = 0..nmax, Maximum-Regel."""
    p = len(Ps)
    cnt = [F(0)] * (nmax + 1)
    cnt[0] = F(1)

    def rec(state, pr, n):
        if n == nmax:
            return
        for xs in product(VALS, repeat=p):
            q, nxt, alarm = pr, [], False
            for k in range(p):
                q *= Ps[k][xs[k]]
                t = chan_step(state[k], xs[k], h)
                alarm = alarm or t is None
                nxt.append(t)
            if alarm:
                continue
            cnt[n + 1] += q
            rec(tuple(nxt), q, n + 1)
    rec(tuple([0] * p), F(1), 0)
    return cnt


def surv_matrix(P, h, nmax):
    Q = chan_Q(P, h)
    v = [F(1)] * h
    out = [v[0]]
    for _ in range(nmax):
        v = [sum(Q[i][j] * v[j] for j in range(h)) for i in range(h)]
        out.append(v[0])
    return out


def _float_Q(P, h):
    return np.array([[float(x) for x in row] for row in chan_Q(P, h)])


# ------------------------------------------------------------------------------------------------------------------ Teil 1: ein Kanal
def test_h1_ein_kanal_arl0_12_und_arl1_9_halbe():
    assert arl_single(P0, 2) == 12 and arl_single(P1, 2) == F(9, 2)
    # Handrechnung: L0 = 1 + 3/4 L0 + 1/4 L1; L1 = 1 + 1/4 L0 + 1/2 L1  =>  L1 = 2 + L0/2  =>  L0 = 12
    assert chan_Q(P0, 2) == [[F(3, 4), F(1, 4)], [F(1, 4), F(1, 2)]]
    assert chan_Q(P1, 2) == [[F(1, 2), F(1, 2)], [F(1, 8), F(3, 8)]]


def test_h1_aufzaehlung_aller_pfade_gleich_matrixpotenz():
    for P in (P0, P1):
        assert surv_enum([P], 2, 8) == surv_matrix(P, 2, 8)
    e0 = surv_enum([P0], 2, 3)
    assert e0[0] == 1 and e0[1] == 1 and e0[2] == 1 - F(1, 16)      # ein Alarm ist erst bei Beobachtung 2 möglich (+1, +1)


def test_h1_produktfunktion_der_demo_gibt_12_und_9_halbe_auf_der_diskreten_kette():
    """`mkn_memory.surv_iter` und `prod_arl` (die Wege der Demo für die Normalkanäle) auf der diskreten Kette: ARL = Σ_n P(RL > n)."""
    for P, soll in ((P0, 12.0), (P1, 4.5)):
        S = M.surv_iter(_float_Q(P, 2), 0, 2000)
        assert abs(M.prod_arl(S, S, 1, 1) - soll) < 1e-9


def test_h1_median_9_gegen_mittel_12():
    m = surv_matrix(P0, 2, 60)
    med = next(n for n in range(61) if 1 - m[n] >= F(1, 2))
    assert med == 9 and 1 - m[8] < F(1, 2) <= 1 - m[9]
    q = M.joint_quantiles(M.surv_iter(_float_Q(P0, 2), 0, 2000), 1)
    assert q["median"] == 9 and abs(q["mean"] - 12) < 1e-9


# ------------------------------------------------------------------------------------------------------------------ Teil 2: zwei Kanäle
def test_h2_zwei_kanaele_arl0_3152_durch_451_und_arl1_14308_durch_3629():
    a0 = joint_arl([P0, P0], 2)
    assert a0 == F(3152, 451)
    LA, LB, LC = F(3152, 451), F(2208, 451), F(1600, 451)
    assert 7 * LA - 6 * LB - LC == 16 and -3 * LA + 9 * LB - 2 * LC == 16 and -LA - 4 * LB + 12 * LC == 16       # die Handgleichungen
    assert joint_arl([P1, P0], 2) == F(14308, 3629)
    assert float(a0) < 7.0 and float(a0) > 6.98


def test_h2_zwei_kanaele_aufzaehlung_gleich_produktweg_als_brueche_und_floats():
    s0, s1 = surv_matrix(P0, 2, 5), surv_matrix(P1, 2, 5)
    assert surv_enum([P0, P0], 2, 5) == [x * x for x in s0]
    assert surv_enum([P1, P0], 2, 5) == [x * y for x, y in zip(s1, s0)]
    S0 = M.surv_iter(_float_Q(P0, 2), 0, 800)
    S1 = M.surv_iter(_float_Q(P1, 2), 0, len(S0) - 1)
    assert abs(M.prod_arl(S0, S1, 2, 0) - 3152 / 451) < 1e-9          # m = 0: kein Kanal geändert (ARL₀)
    assert abs(M.prod_arl(S0, S1, 2, 1) - 14308 / 3629) < 1e-9        # m = 1: ein Kanal geändert (ARL₁)


def test_h2_ein_kanal_mehr_bei_gleicher_schwelle_senkt_arl0_staerker_als_arl1():
    a0 = joint_arl([P0, P0], 2)
    a1 = joint_arl([P1, P0], 2)
    assert a1 <= arl_single(P1, 2) and float(a0) / 12 < float(a1) / float(arl_single(P1, 2))
    assert round(100 * (1 - float(a0) / 12), 1) == 41.8 and round(100 * (1 - float(a1) / 4.5), 1) == 12.4


# ------------------------------------------------------------------------------------------------------------------ Teil 3: Schwelle anheben
def test_h3_tabelle_schwellen_2_3_4_fuer_p_1_bis_3():
    rows = {}
    for p in (1, 2, 3):
        for h in (2, 3, 4):
            rows[(p, h)] = (joint_arl([P0] * p, h), joint_arl([P1] + [P0] * (p - 1), h))
    soll = {(1, 2): (12, 4.500), (1, 3): (24, 7.125), (1, 4): (40, 9.781), (2, 2): (6.989, 3.943), (2, 3): (14.059, 6.578), (2, 4): (23.487, 9.311),
            (3, 2): (5.304, 3.579), (3, 3): (10.694, 6.172), (3, 4): (17.888, 8.931)}
    for k, (a, b) in soll.items():
        assert abs(float(rows[k][0]) - a) < 6e-4 and abs(float(rows[k][1]) - b) < 6e-4, k
    for p in (1, 2, 3):
        assert rows[(p, 2)][0] < rows[(p, 3)][0] < rows[(p, 4)][0]
    for h in (2, 3, 4):
        assert rows[(1, h)][0] > rows[(2, h)][0] > rows[(3, h)][0]
    # (p = 3, h = 3) gegen (p = 1, h = 2): kleinere ARL₀ und zugleich größere ARL₁, also auf beiden Maßen schlechter
    assert rows[(3, 3)][0] < rows[(1, 2)][0] and rows[(3, 3)][1] > rows[(1, 2)][1]


# ------------------------------------------------------------------------------------------------------------------ Teil 4 bis 7: Gaußmodell
def test_h4_t2_bei_p_2_ucl_ist_2_ln_gamma_und_arl1_67_32():
    assert abs(K.ucl_t2(2) - 2 * math.log(C.GAMMA0)) < 1e-9          # χ² mit zwei Freiheitsgraden ist exponentiell: P(T² > c) = exp(-c/2)
    assert abs(math.exp(K.ucl_t2(2) / 2) - C.GAMMA0) < 1e-7 and abs(C.GAMMA0 - 370.3983) < 1e-3
    assert abs(K.arl_t2(2, 1.0) - 67.32) < 0.006
    z, q = K.z_sidak(2)
    assert abs(q - (1 - math.sqrt(1 - 1 / C.GAMMA0))) < 1e-15 and abs(z - 3.2050) < 5e-4 and abs(1 / q - 740.30) < 0.01
    assert abs(K.arl_sep(2, 1, 1.0) - 66.34) < 0.006 and abs(K.arl_sep(5, 1, 1.0) - 109.89) < 0.006 and abs(K.arl_t2(5, 1.0) - 114.38) < 0.006


def test_h5_grenzfall_der_korrelation_und_nichtzentralitaeten_bei_rho_08():
    z, q = K.z_sidak(2)
    assert abs(1 / (1 - (1 - q) ** 2) - C.GAMMA0) < 1e-9             # ρ = 0: Definition der Šidák-Eichung
    assert C.GAMMA0 < 1 / q < 2 * C.GAMMA0                           # ρ = ±1: beide Kanäle melden dasselbe Ereignis, ARL₀ = 1/q
    for mu, lam in (((1, 1), 2 / 1.8), ((1, -1), 2 / 0.2), ((1, 0), 1 / 0.36)):
        assert abs(R.noncentrality(C.KORR_P2, 0.8, mu) - lam) < 1e-12
    assert abs(K.arl_t2(2, 10.0) - 2.22) < 0.006 and abs(K.arl_t2(2, 2 / 1.8) - 59.71) < 0.006
    ev = np.linalg.eigvalsh(R.cov_p2(0.8))
    assert abs(ev[0] - 0.2) < 1e-12 and abs(ev[1] - 1.8) < 1e-12


def test_h6_summenkarte_4_50_177_73_und_8_86():
    assert abs(K.arl_sum(5, 5, 1.0) - 4.50) < 0.006 and abs(K.arl_sum(5, 1, 1.0) - 177.73) < 0.006 and abs(K.arl_sum(5, 4, 1.0) - 8.86) < 0.006
    assert K.arl_sum(5, 1, 1.0) > K.arl_sum(5, 4, 1.0) > K.arl_sum(5, 5, 1.0)
    a = math.sqrt(5)                                                  # von Hand: s ~ N(√5 d, 1)
    assert abs(1 / (norm.sf(3 - a) + norm.cdf(-3 - a)) - K.arl_sum(5, 5, 1.0)) < 1e-9
    assert abs(1 / (norm.sf(3) + norm.cdf(-3)) - C.GAMMA0) < 1e-9     # ohne Änderung ARL₀ = 370,398 für jedes p


def test_h7_geometrische_lauflaenge_median_257_q05_19_q95_1109_und_23_69_prozent():
    s = K.geo_summary(C.P_ALARM0)
    assert (s["median"], s["q05"], s["q95"]) == (257, 19, 1109) and abs(s["p100"] - 0.2369) < 5e-5
    for p in (1, 2, 5, 10, 20):      # gleiche Verteilung für T² und getrennte Karten bei jedem p: die Alarmwahrscheinlichkeit je Vektor ist 1/370,398
        for pa in (K.pa_t2(p, 0.0), K.pa_sep(p, 0, 0.0)):
            g = K.geo_summary(pa)
            assert (g["median"], g["q05"], g["q95"]) == (257, 19, 1109) and abs(g["mean"] / s["mean"] - 1) < 1e-10 and abs(g["p100"] - s["p100"]) < 1e-12


@pytest.mark.parametrize("p", [1, 2, 5, 10, 20])
def test_h7_alarmwahrscheinlichkeit_ohne_aenderung_ist_fuer_jedes_p_gleich(p):
    assert abs(K.pa_t2(p, 0.0) - C.P_ALARM0) < 1e-14
    assert abs(K.pa_sep(p, 1, 0.0) - C.P_ALARM0) < 1e-14
    assert abs(K.pa_sum(p, p, 0.0) - C.P_ALARM0) < 1e-14
