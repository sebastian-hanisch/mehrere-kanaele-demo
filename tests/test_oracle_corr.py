"""Orakel für korrelierte Kanäle (bekanntes Σ): andere Rechenwege als die Integrale der Demo (`mkn_corr`).

Weg der Demo: p = 2 mit adaptiver Quadratur über x₁, p = 5 gleichkorreliert mit Ein-Faktor-Darstellung und Gauß-Hermite. Orakel: (a) `scipy.stats.multivariate_normal.cdf`
(Genz-Verfahren) mit Einschluss-Ausschluss bzw. `lower_limit`, Toleranz 1e-4 (das Verfahren ist eine Zufallsquadratur, kein exaktes Integral); (b) Simulation mit dem
numpy-Generator (Gegenprobe) für Alarmanteile je Vektor; (c) Handwerte (Grenzfälle, Nichtzentralitäten); (d) Eigenschaften: Šidák-ARL₀ nie unter 370,398 in den gerechneten Zellen,
T²-ARL₀ für jedes Σ, T²-ARL₁ nur an λ, Umnummerieren der Kanäle ändert getrennte Karten nicht."""

import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import mkn_cards as K
import mkn_constants as C
import mkn_corr as R


def _mvn_noalarm(z, mus, cov):
    p = len(mus)
    rv = multivariate_normal(mean=np.asarray(mus, float), cov=cov, abseps=1e-12, releps=1e-12, maxpts=4_000_000)
    return float(rv.cdf(np.full(p, z), lower_limit=np.full(p, -z)))


@pytest.mark.parametrize("rho", [-0.9, -0.5, 0.0, 0.5, 0.8, 0.95])
def test_p2_quadratur_gleich_mvn_ohne_und_mit_sprung(rho):
    z = K.z_sidak(2)[0]
    for mu in ((0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (1.0, 1.0)):
        a = R.noalarm_p2(z, mu[0], mu[1], rho)
        b = _mvn_noalarm(z, mu, R.cov_p2(rho))
        assert abs(a - b) < 1e-4, (rho, mu, a, b)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_p5_faktor_integral_gleich_mvn(rho):
    z = K.z_sidak(5)[0]
    for mu in (np.zeros(5), np.array([1.0, 0, 0, 0, 0]), np.ones(5)):
        a = R.noalarm_equi(z, mu, rho)
        b = _mvn_noalarm(z, mu, R.cov_equi(5, rho))
        assert abs(a - b) < 1e-4 * max(1.0, abs(b)) and abs(a - b) < 5e-4, (rho, mu, a, b)


def test_arl0_der_sidak_karte_bei_korrelation_p2_und_p5_gegen_mvn():
    """Zellen: p = 2 mit ρ = -0,9 / -0,5 / 0 / 0,5 / 0,8 / 0,95 und p = 5 gleichkorreliert mit ρ = 0 / 0,3 / 0,6 / 0,9: Rechenwege stimmen, ARL₀ >= 370,398 in 10 von 10 Zellen."""
    soll = {(C.KORR_P2, -0.9): 470.4, (C.KORR_P2, -0.5): 379.0, (C.KORR_P2, 0.0): 370.4, (C.KORR_P2, 0.5): 379.0, (C.KORR_P2, 0.8): 422.7, (C.KORR_P2, 0.95): 520.7,
            (C.KORR_P5, 0.0): 370.4, (C.KORR_P5, 0.3): 375.2, (C.KORR_P5, 0.6): 414.1, (C.KORR_P5, 0.9): 669.8}
    ge = 0
    for (model, rho), s in soll.items():
        a = R.arl0_sidak(model, rho)
        p = R.dim(model)
        b = 1.0 / (1.0 - _mvn_noalarm(K.z_sidak(p)[0], np.zeros(p), R.sigma(model, rho)))
        assert abs(a - s) < 0.06 and abs(a / b - 1) < 5e-4, (model, rho, a, b)
        ge += a >= C.GAMMA0 - 1e-9
    assert ge == len(soll) == 10


def test_sidak_ungleichung_auf_einem_dichten_raster_p2_und_p5():
    """Zählung (die Ungleichung selbst ist nicht aus der Quelle gelesen): ARL₀ der Šidák-Karte >= 370,398 für p = 2 auf 39 Werten von ρ und p = 5 auf 20 Werten."""
    n = 0
    for rho in np.round(np.arange(-0.95, 0.9501, 0.05), 2):
        assert R.arl0_sidak(C.KORR_P2, float(rho)) >= C.GAMMA0 - 1e-6, rho
        n += 1
    for rho in np.round(np.arange(0.0, 0.9501, 0.05), 2):
        assert R.arl0_sidak(C.KORR_P5, float(rho)) >= C.GAMMA0 - 1e-6, rho
        n += 1
    assert n == 39 + 20


def test_randfall_rho_gegen_eins_arl0_gegen_ein_durch_q():
    """Von Hand: ρ → ±1, beide Kanäle melden dasselbe Ereignis, ARL₀ → 1/q = 740,3 (p = 2). Numerisch bei 1 - ρ = 1e-3 und 1e-5 darunter und monoton steigend."""
    z, q = K.z_sidak(2)
    a3 = 1 / (1 - R.noalarm_p2(z, 0, 0, 1 - 1e-3))
    a5 = 1 / (1 - R.noalarm_p2(z, 0, 0, 1 - 1e-5))
    assert a3 < a5 < 1 / q and (1 / q - a5) / (1 / q) < 0.01
    assert abs(1 / q - 740.30) < 0.01


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.95])
def test_neu_geeichte_schranke_gibt_arl0_370_398_und_ist_kleiner_als_sidak(rho):
    z_new = R.z_calibrated(C.KORR_P2, rho)
    assert abs(1 / (1 - _mvn_noalarm(z_new, [0, 0], R.cov_p2(rho))) / C.GAMMA0 - 1) < 5e-4
    assert z_new < K.z_sidak(2)[0]
    assert abs(R.z_calibrated(C.KORR_P2, 0.0) - K.z_sidak(2)[0]) < 1e-9
    assert abs(R.z_calibrated(C.KORR_P5, 0.0) - K.z_sidak(5)[0]) < 1e-9


def test_tabelle_rho_08_p2_gegen_plan_und_mvn():
    """ARL₁ je Richtung bei ρ = 0,8 (Plan 06, Tabelle C): T² und getrennt neu geeicht; der Weg `multivariate_normal` bestätigt die getrennte Zeile `gegen`."""
    soll = {C.RICHTUNG_ENTLANG: (1.11, 59.71, 41.27, 45.39), C.RICHTUNG_GEGEN: (10.0, 2.22, 32.99, 36.42), C.RICHTUNG_EIN: (2.78, 16.92, 62.22, 68.82)}
    for name, (lam, t2, sep_new, sep_sidak) in soll.items():
        mu = R.jump_vector(C.KORR_P2, 0.8, name, 1.0, False)
        row = R.arl_row(C.KORR_P2, 0.8, mu)
        assert abs(row["lam"] - lam) < 0.006 and abs(row["t2"] - t2) < 0.006 and abs(row["sep_new"] - sep_new) < 0.006 and abs(row["sep_sidak"] - sep_sidak) < 0.006, name
    mu = R.jump_vector(C.KORR_P2, 0.8, C.RICHTUNG_GEGEN, 1.0, False)
    z = R.z_calibrated(C.KORR_P2, 0.8)
    assert abs(1 / (1 - _mvn_noalarm(z, mu, R.cov_p2(0.8))) - 32.985) < 0.01


def test_simulation_der_alarmanteile_bei_rho_08_im_band():
    """2 Mio. Vektoren mit Korrelation 0,8 (numpy, Cholesky): Alarmanteil je Vektor von T² (mit Σ⁻¹) und der neu geeichten getrennten Karten gegen die exakte Zahl, z <= 4."""
    rng = np.random.default_rng(15003)
    N = 2_000_000
    S = R.cov_p2(0.8)
    L, Si = np.linalg.cholesky(S), np.linalg.inv(S)
    E = rng.standard_normal((N, 2)) @ L.T
    z = R.z_calibrated(C.KORR_P2, 0.8)
    for name in (C.RICHTUNG_ENTLANG, C.RICHTUNG_GEGEN, C.RICHTUNG_EIN):
        mu = R.jump_vector(C.KORR_P2, 0.8, name, 1.0, False)
        X = E + mu
        row = R.arl_row(C.KORR_P2, 0.8, mu)
        for k, pa in ((int((np.einsum("ij,jk,ik->i", X, Si, X) > K.ucl_t2(2)).sum()), 1 / row["t2"]), (int((np.abs(X) > z).any(1).sum()), 1 / row["sep_new"])):
            assert abs(k / N - pa) < 4 * math.sqrt(pa * (1 - pa) / N), (name, k / N, pa)
    X0 = E
    k0 = int((np.einsum("ij,jk,ik->i", X0, Si, X0) > K.ucl_t2(2)).sum())
    assert abs(k0 / N - C.P_ALARM0) < 4 * math.sqrt(C.P_ALARM0 / N)          # ohne Sprung: T² hat bei jedem Σ die ARL₀ 370,398


def test_t2_arl0_fuer_jedes_sigma_und_arl1_nur_an_lambda():
    """Zufällige positiv definite Σ (p = 2 bis 5, fester Zufalls-Seed): Whitening zeigt x'Σ⁻¹x = ε'ε; ARL₁ ändert sich nicht, wenn der Sprung rotiert oder die Kanäle umnummeriert werden."""
    rng = np.random.default_rng(15004)
    for _ in range(30):
        p = int(rng.integers(2, 6))
        A = rng.standard_normal((p, p))
        S = A @ A.T + 0.3 * np.eye(p)
        L = np.linalg.cholesky(S)
        e = rng.standard_normal(p)
        x = L @ e
        assert abs(x @ np.linalg.solve(S, x) - e @ e) < 1e-9 * max(1.0, e @ e)
        mu = rng.standard_normal(p)
        lam = float(mu @ np.linalg.solve(S, mu))
        perm = rng.permutation(p)
        lam2 = float(mu[perm] @ np.linalg.solve(S[np.ix_(perm, perm)], mu[perm]))
        assert abs(lam - lam2) < 1e-9 * max(1.0, lam) and abs(K.arl_t2(p, lam) / K.arl_t2(p, lam2) - 1) < 1e-8


def test_gleiche_nichtzentralitaet_gibt_gleiche_t2_arl1_aber_nicht_fuer_getrennte_karten():
    """ρ = 0,8, λ = 4 in vier Richtungen: T² 9,41 in allen, getrennte Karten 88,4 (gegen) / 63,9 (schief) / 39,2 (ein Kanal) / 6,81 (entlang)."""
    soll = {C.RICHTUNG_GEGEN: 88.39, C.RICHTUNG_SCHIEF: 63.92, C.RICHTUNG_EIN: 39.20, C.RICHTUNG_ENTLANG: 6.81}
    for name, s in soll.items():
        mu = R.jump_vector(C.KORR_P2, 0.8, name, 1.0, True)
        row = R.arl_row(C.KORR_P2, 0.8, mu)
        assert abs(row["lam"] - 4.0) < 1e-9 and abs(row["t2"] - 9.4071) < 5e-4 and abs(row["sep_new"] - s) < 0.01, name


def test_umnummerieren_der_kanaele_aendert_getrennte_karten_nicht():
    z = R.z_calibrated(C.KORR_P5, 0.6)
    mu = np.array([1.0, 0.5, 0, 0, 0.2])
    a = R.noalarm_equi(z, mu, 0.6)
    assert abs(a - R.noalarm_equi(z, mu[::-1], 0.6)) < 1e-13 and abs(a - R.noalarm_equi(z, mu[[2, 0, 4, 1, 3]], 0.6)) < 1e-13


def test_p5_gleichkorreliert_tabelle_aus_der_vormessung():
    """T² / getrennt (neu geeicht), d = 1: alle fünf springen: ρ = 0: 12,11 / 29,16; ρ = 0,6: 76,70 / 35,07; ρ = 0,9: 105,70 / 41,30; ein Kanal: 114,38 / 109,89, 50,25 / 106,51, 5,09 / 81,19."""
    soll = {(0.0, C.RICHTUNG_ENTLANG): (12.11, 29.16), (0.6, C.RICHTUNG_ENTLANG): (76.70, 35.07), (0.9, C.RICHTUNG_ENTLANG): (105.70, 41.30),
            (0.0, C.RICHTUNG_EIN): (114.38, 109.89), (0.6, C.RICHTUNG_EIN): (50.25, 106.51), (0.9, C.RICHTUNG_EIN): (5.09, 81.19)}
    for (rho, name), (t2, sep) in soll.items():
        row = R.arl_row(C.KORR_P5, rho, R.jump_vector(C.KORR_P5, rho, name, 1.0, False))
        assert abs(row["t2"] - t2) < 0.006 and abs(row["sep_new"] - sep) < 0.006, (rho, name)
