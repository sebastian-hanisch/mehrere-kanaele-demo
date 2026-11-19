"""Orakel für Karten mit Gedächtnis je Kanal (CUSUM, EWMA, MEWMA): andere Rechenwege als Markov-Gitter und Produktweg der Demo (`mkn_memory`).

Orakel: (a) Simulation der Lauflängen mit dem numpy-Generator (Gegenprobe; Band z <= 4 Standardfehler gegen die Gitterzahl, die Gitterzahl ist auf etwa 0,4 % genau, also
kleiner als die Bänder); (b) der EWMA eines Kanals aus der Integralgleichung (Nyström mit Gauß-Legendre), ein anderer Weg als das Zellengitter; (c) die MEWMA bei p = 1 ist ein
EWMA: ihre Schwelle folgt aus der EWMA-Grenze; (d) Konvergenz der Gitter (m = 24 / 32 / 48, Richardson) und der radialen Kette (n = 100 bis 800); (e) die diskrete Kette
des Handbeispiels (tests/test_handbeispiele.py) prüft die Produktfunktionen exakt in Brüchen."""

import math

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

import mkn_constants as C
import mkn_memory as M
import mkn_reference as REF


def _sim_cusum(p, h, shift, R, rng, k=C.CUSUM_K):
    cp, cn = np.zeros((R, p)), np.zeros((R, p))
    rl = np.zeros(R)
    idx = np.arange(R)
    t = 0
    while idx.size:
        t += 1
        z = rng.standard_normal((idx.size, p)) + shift
        cp, cn = np.maximum(0.0, cp + z - k), np.maximum(0.0, cn - z - k)
        hit = ((cp > h) | (cn > h)).any(axis=1)
        rl[idx[hit]] = t
        idx, cp, cn = idx[~hit], cp[~hit], cn[~hit]
    return rl


def _sim_ewma(p, H, shift, R, rng, lam=C.EWMA_LAMBDA):
    Z = np.zeros((R, p))
    rl = np.zeros(R)
    idx = np.arange(R)
    t = 0
    while idx.size:
        t += 1
        Z = lam * (rng.standard_normal((idx.size, p)) + shift) + (1 - lam) * Z
        hit = (np.abs(Z) > H).any(axis=1)
        rl[idx[hit]] = t
        idx, Z = idx[~hit], Z[~hit]
    return rl


def _im_band(rl, soll, z=4.0):
    se = rl.std(ddof=1) / math.sqrt(rl.size)
    assert abs(rl.mean() - soll) < z * se, (rl.mean(), soll, se)


def _shift(p, m, d):
    s = np.zeros(p)
    s[:m] = d
    return s


@pytest.mark.parametrize("p,m,d,R", [(1, 1, 1.0, 40000), (5, 1, 1.0, 40000), (5, 1, 0.5, 20000), (5, 2, 1.0, 40000), (5, 5, 1.0, 40000), (10, 4, 1.0, 20000)])
def test_cusum_arl1_gitter_gegen_simulation(p, m, d, R):
    rng = np.random.default_rng(16000 + 10 * p + m)
    rl = _sim_cusum(p, REF.cusum_h(p), _shift(p, m, d), R, rng)
    _im_band(rl, REF.cusum_arl1(p, m, d))


@pytest.mark.parametrize("p,R", [(1, 20000), (5, 20000)])
def test_cusum_arl0_gitter_gegen_simulation(p, R):
    rl = _sim_cusum(p, REF.cusum_h(p), np.zeros(p), R, np.random.default_rng(16100 + p))
    _im_band(rl, C.GAMMA0)


@pytest.mark.parametrize("p,m,d,R", [(1, 1, 1.0, 40000), (5, 1, 1.0, 40000), (5, 1, 2.0, 20000), (20, 1, 1.0, 20000)])
def test_ewma_arl1_gitter_gegen_simulation(p, m, d, R):
    rl = _sim_ewma(p, REF.ewma_H(p), _shift(p, m, d), R, np.random.default_rng(16200 + p))
    _im_band(rl, REF.ewma_arl1(p, d))


@pytest.mark.parametrize("p", [2, 5])
def test_ewma_arl0_gitter_gegen_simulation(p):
    rl = _sim_ewma(p, REF.ewma_H(p), np.zeros(p), 20000, np.random.default_rng(16300 + p))
    _im_band(rl, C.GAMMA0)


def _ewma_integral_arl(lam, H, delta, N=300, u0=0.0):
    """Integralgleichung L(u) = 1 + ∫_{-H}^{H} φ((y - (1-λ)u)/λ - δ)/λ · L(y) dy, Nyström mit Gauß-Legendre: ein anderer Weg als das Zellengitter."""
    x, w = leggauss(N)
    y, w = H * x, H * w
    A = norm.pdf((y[None, :] - (1 - lam) * y[:, None]) / lam - delta) / lam * w[None, :]
    L = np.linalg.solve(np.eye(N) - A, np.ones(N))
    return float(1 + (norm.pdf((y - (1 - lam) * u0) / lam - delta) / lam * w * L).sum())


def test_ewma_ein_kanal_gitter_gleich_integralgleichung():
    for d in (0.0, 0.5, 1.0, 3.0):
        a = _ewma_integral_arl(C.EWMA_LAMBDA, REF.ewma_H(1), d)
        b = REF.ewma_arl0(1) if d == 0.0 else REF.ewma_arl1(1, d)
        assert abs(a / b - 1) < 2e-4, (d, a, b)
    assert abs(_ewma_integral_arl(C.EWMA_LAMBDA, REF.ewma_H(1), 0.0) / C.GAMMA0 - 1) < 2e-4


def test_ewma_produktweg_bei_p_kanaelen_gleich_minimum_aus_integralgleichung_ueberlebensfunktion():
    """Die Überlebensfunktion des Kanals aus dem Nyström-Operator (n Schritte), dann das Produkt für p = 5: ein anderer Weg als Gitter plus Produktfunktionen der Demo."""
    lam, H, N = C.EWMA_LAMBDA, REF.ewma_H(5), 200
    x, w = leggauss(N)
    y, w = H * x, H * w
    out = {}
    for d in (0.0, 1.0):
        A = norm.pdf((y[None, :] - (1 - lam) * y[:, None]) / lam - d) / lam * w[None, :]
        r0 = norm.pdf(y / lam - d) / lam * w          # Start in Z = 0: Dichte von Z_1 = λ x_1
        v = np.ones(N)
        S = [1.0]
        for _ in range(40000):
            S.append(float(r0 @ v))
            v = A @ v
            if d == 0.0 and S[-1] ** 5 < 1e-14:       # die gemeinsame Überlebensfunktion (fünf Kanäle) ist dann null
                break
            if d != 0.0 and len(S) >= len(out[0.0]):  # der geänderte Kanal läuft so lange wie der ungeänderte
                break
        out[d] = np.array(S)
    n = min(len(out[0.0]), len(out[1.0]))
    arl0 = float((out[0.0][:n] ** 5).sum())
    arl1 = float((out[0.0][:n] ** 4 * out[1.0][:n]).sum())
    assert abs(arl0 / C.GAMMA0 - 1) < 2e-3 and abs(arl1 / REF.ewma_arl1(5, 1.0) - 1) < 2e-3, (arl0, arl1)


def test_gitterkonvergenz_cusum_richardson_liegt_hinter_den_gittern():
    """ARL₁ (p = 1, d = 1) auf Gittern m = 24 / 32 / 48: der Unterschied schrumpft wie 1/m², die Richardson-Zahl steht in der gefrorenen Tabelle."""
    h = REF.cusum_h(1)
    v = {}
    for g in (24, 32, 48):
        S = M.surv_iter(M.cusum_two_R(h, 1.0, g), 0, 20000)
        v[g] = M.prod_arl(S, S, 1, 1)
    assert abs(v[24] - v[48]) > abs(v[32] - v[48]) > 0
    r = M.richardson(v[32], 32, v[48], 48)
    assert abs(r - REF.cusum_arl1(1, 1, 1.0)) < 1e-9 and abs(v[48] - r) < 0.01


def test_mewma_radiale_kette_konvergiert_und_p1_ist_ein_ewma():
    for p in C.P_OPTIONS:
        a = [M.chain_arl(M.mewma_chain(p, C.EWMA_LAMBDA, REF.mewma_H(p), n)) for n in (100, 200, 400)]
        assert abs(a[2] / C.GAMMA0 - 1) < 1e-4 and abs(a[1] - C.GAMMA0) < abs(a[0] - C.GAMMA0) + 1e-9, (p, a)
    # p = 1: |Z|² · (2-λ)/λ > H  <=>  |Z| > sqrt(H λ/(2-λ)) = Grenze des EWMA
    h1 = math.sqrt(REF.mewma_H(1) * C.EWMA_LAMBDA / (2 - C.EWMA_LAMBDA))
    assert abs(h1 / REF.ewma_H(1) - 1) < 3e-3


def test_mewma_arl1_simulation_gefroren_gegen_neue_simulation_mit_numpy():
    """Unabhängiger Generator und eigene Schleife (p = 5, ein Kanal um 1 σ, 20 000 Läufe): die gefrorene SplitMix64-Zahl liegt im Band."""
    p, R = 5, 20000
    rng = np.random.default_rng(16400)
    Z = np.zeros((R, p))
    rl = np.zeros(R)
    idx = np.arange(R)
    t = 0
    lam, Hc = C.EWMA_LAMBDA, REF.mewma_H(p)
    while idx.size:
        t += 1
        Z = lam * (rng.standard_normal((idx.size, p)) + _shift(p, 1, 1.0)) + (1 - lam) * Z
        hit = (Z * Z).sum(axis=1) * (2 - lam) / lam > Hc
        rl[idx[hit]] = t
        idx, Z = idx[~hit], Z[~hit]
    mean, se, _ = REF.mewma_arl1_sim(p, 1.0)
    assert abs(rl.mean() - mean) < 4 * math.hypot(rl.std(ddof=1) / math.sqrt(R), se)


def test_cusum_arl0_steigt_mit_h_und_arl1_faellt_mit_d():
    h = REF.cusum_h(1)
    a = [M.cusum_arl0(hh, 1, mg=(24, 32)) for hh in (h - 0.5, h, h + 0.5)]
    assert a == sorted(a) and a[0] < C.GAMMA0 < a[2]
    for p in C.P_OPTIONS:
        v = [REF.cusum_arl1(p, 1, d) for d in C.D_OPTIONS]
        assert v == sorted(v, reverse=True)
    for d in C.D_OPTIONS:
        v = [REF.cusum_arl1(p, 1, d) for p in C.P_OPTIONS]
        assert v == sorted(v)          # mehr Kanäle: größere Schwelle je Kanal, größere Verzögerung
