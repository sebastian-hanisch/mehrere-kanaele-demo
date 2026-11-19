"""Orakel für die Karten ohne Gedächtnis (T², getrennte Šidák-Karten, Summenkarte): andere Rechenwege als der der Demo.

Weg der Demo: `scipy.stats.ncx2.sf` und geschlossene Formeln (`mkn_cards`). Orakel: (a) Poisson-Mischung zentraler χ²-Verteilungen (nichtzentrales χ² = Poisson(λ/2)-Mischung
von χ²(p + 2j)); (b) für p = 2 ein Doppelintegral der Normaldichte über die Kreisscheibe; (c) getrennte Karten als Produkt eindimensionaler Integrale der Normaldichte
(`scipy.integrate.quad`, nicht die Formel mit `norm.sf`); (d) eine Simulation der Alarmwahrscheinlichkeit je Vektor mit dem numpy-Generator (Gegenprobe, nicht Ergebnis der App;
der SplitMix64-Weg der App wird in test_scenario.py geprüft); (e) die Verteilung der Lauflänge (geometrisch) gegen eine Simulation der Lauflängen."""

import math

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.stats import chi2, norm, poisson

import mkn_cards as K
import mkn_constants as C


def _pa_poisson(p, lam):
    j = np.arange(0, 600)
    return float((poisson.pmf(j, lam / 2.0) * chi2.sf(K.ucl_t2(p), p + 2 * j)).sum())


@pytest.mark.parametrize("p", [1, 2, 5, 10, 20])
@pytest.mark.parametrize("d", [0.5, 1.0, 2.0, 3.0])
def test_t2_ncx2_gleich_poisson_mischung(p, d):
    assert abs(K.pa_t2(p, d * d) / _pa_poisson(p, d * d) - 1) < 1e-9


@pytest.mark.parametrize("lam", [0.25, 1.0, 4.0, 9.0])
def test_t2_p2_gleich_doppelintegral_ueber_die_kreisscheibe(lam):
    ucl, mu = K.ucl_t2(2), math.sqrt(lam)
    def f(th, r):
        return r * math.exp(-0.5 * ((r * math.cos(th) - mu) ** 2 + (r * math.sin(th)) ** 2)) / (2 * math.pi)

    inner, _ = dblquad(f, 0.0, math.sqrt(ucl), 0.0, 2 * math.pi, epsabs=1e-13, epsrel=1e-13)
    assert abs((1 - inner) / K.pa_t2(2, lam) - 1) < 1e-8


@pytest.mark.parametrize("p", [1, 2, 5, 10, 20])
def test_t2_alarmwahrscheinlichkeit_ohne_aenderung_gleich_chi2_integral(p):
    """Zentrales χ²: der Anteil jenseits der Schranke als Integral der χ²-Dichte; unabhängig von `chi2.sf`."""
    ucl = K.ucl_t2(p)
    val = quad(lambda x: chi2.pdf(x, p), ucl, np.inf, epsabs=1e-15, epsrel=1e-12)[0]
    assert abs(val / C.P_ALARM0 - 1) < 1e-8


@pytest.mark.parametrize("p,m,d", [(1, 1, 1.0), (2, 1, 0.5), (5, 1, 1.0), (5, 2, 1.414), (10, 5, 1.0), (20, 1, 3.0), (5, 5, 0.0)])
def test_getrennte_karten_gleich_produkt_eindimensionaler_integrale(p, m, d):
    """P(kein Alarm) = Produkt über die Kanäle von ∫_{-z}^{z} φ(x - μ_i) dx, mit quad statt Verteilungsfunktion."""
    z = K.z_sidak(p)[0]
    def inner(mu):
        return quad(lambda x: norm.pdf(x - mu), -z, z, epsabs=1e-15, epsrel=1e-13)[0]

    kein_alarm = inner(d) ** m * inner(0.0) ** (p - m)
    assert abs((1 - kein_alarm) / K.pa_sep(p, m, d) - 1) < 1e-9


@pytest.mark.parametrize("p", [1, 2, 5, 10, 20])
def test_sidak_schranke_gibt_gemeinsame_arl0_370_398(p):
    """Gegenprobe der Eichung: P(irgendein Kanal alarmiert) mit der Schranke z aus quad statt aus der Formel."""
    z = K.z_sidak(p)[0]
    inner = quad(lambda x: norm.pdf(x), -z, z, epsabs=1e-16, epsrel=1e-13)[0]
    assert abs(1.0 / (1.0 - inner ** p) / C.GAMMA0 - 1) < 1e-8


def test_simulation_der_alarmwahrscheinlichkeit_je_vektor_im_band():
    """3 Mio. Vektoren je Fall mit dem numpy-Generator (Gegenprobe); Band z <= 4 Standardfehler gegen die exakte Zahl."""
    rng = np.random.default_rng(15001)
    N = 3_000_000
    for p, m, d in ((5, 1, 1.0), (5, 2, math.sqrt(2.0)), (2, 1, 1.0)):
        X = rng.standard_normal((N, p))
        X[:, :m] += d
        for name, k, pa in (("T²", int(((X * X).sum(1) > K.ucl_t2(p)).sum()), K.pa_t2(p, m * d * d)),
                            ("getrennt", int((np.abs(X) > K.z_sidak(p)[0]).any(1).sum()), K.pa_sep(p, m, d)),
                            ("Summe", int((np.abs(X.sum(1) / math.sqrt(p)) > 3).sum()), K.pa_sum(p, m, d))):
            se = math.sqrt(pa * (1 - pa) / N)
            assert abs(k / N - pa) < 4 * se, (name, p, m, d, k / N, pa)


def test_simulation_ohne_aenderung_alarmanteil_fuer_jedes_p_gleich_der_zielzahl():
    rng = np.random.default_rng(15002)
    N = 2_000_000
    for p in (2, 5, 20):
        X = rng.standard_normal((N, p))
        for name, k in (("T²", int(((X * X).sum(1) > K.ucl_t2(p)).sum())), ("getrennt", int((np.abs(X) > K.z_sidak(p)[0]).any(1).sum())),
                        ("Summe", int((np.abs(X.sum(1) / math.sqrt(p)) > 3).sum()))):
            se = math.sqrt(C.P_ALARM0 * (1 - C.P_ALARM0) / N)
            assert abs(k / N - C.P_ALARM0) < 4 * se, (name, p)


def test_lauflaenge_verteilung_median_aus_simulierten_laeufen():
    """Median der Lauflänge ohne Änderung (T², p = 2) aus 20 000 simulierten Läufen nahe 257; Mittel im Band um 370,398."""
    rng = np.random.default_rng(15799)
    R = 20000
    T = np.zeros(R)
    alive = np.arange(R)
    t = 0
    while alive.size:
        t += 1
        al = (rng.standard_normal((alive.size, 2)) ** 2).sum(axis=1) > K.ucl_t2(2)
        T[alive[al]] = t
        alive = alive[~al]
    assert abs(np.median(T) - 257) <= 8 and abs(T.mean() - C.GAMMA0) < 4 * T.std(ddof=1) / math.sqrt(R)


def test_t2_ist_affin_invariant_und_haengt_nur_an_lambda():
    """m = 1, d = 2 und m = 4, d = 1 haben dieselbe Nichtzentralität (λ = 4): dieselbe ARL₁ von T² (exakt), getrennte Karten nicht."""
    assert abs(K.arl_t2(5, 1 * 2.0 ** 2) - K.arl_t2(5, 4 * 1.0 ** 2)) < 1e-12
    assert abs(K.arl_sep(5, 1, 2.0) - K.arl_sep(5, 4, 1.0)) > 10
    assert K.arl_t2(5, 4.0) == pytest.approx(17.934, abs=5e-4)


def test_monotonie_arl1_faellt_in_d_und_steigt_in_p():
    for p in (1, 2, 5, 10, 20):
        v = [K.arl_t2(p, d * d) for d in (0.25, 0.5, 1.0, 2.0, 3.0)]
        w = [K.arl_sep(p, 1, d) for d in (0.25, 0.5, 1.0, 2.0, 3.0)]
        assert v == sorted(v, reverse=True) and w == sorted(w, reverse=True)
    for d in (0.5, 1.0, 2.0, 3.0):
        t2 = [K.arl_t2(p, d * d) for p in range(1, 21)]
        sep = [K.arl_sep(p, 1, d) for p in range(1, 21)]
        assert t2 == sorted(t2) and sep == sorted(sep)
    assert K.arl0_per_channel(2) < K.arl0_per_channel(5) < K.arl0_per_channel(20)


def test_p1_alle_drei_karten_sind_gleich_und_ohne_aenderung_gleich_dem_ziel():
    for d in (0.5, 1.0, 2.0, 3.0):
        assert abs(K.arl_t2(1, d * d) / K.arl_sep(1, 1, d) - 1) < 1e-9 and abs(K.arl_sum(1, 1, d) / K.arl_sep(1, 1, d) - 1) < 5e-3
    assert abs(K.ucl_t2(1) - 9.0) < 1e-3 and abs(K.z_sidak(1)[0] - 3.0) < 1e-9     # χ²₁-Schranke 9 = 3²
