"""Szenario mit dem SplitMix64 der Demo: Daten, Verlauf je Karte, Alarmzeit und Kanal, MEWMA-Live-Lauf. Die Statistik jeder Karte wird mit einer eigenen Rekursion neu gerechnet."""

import math

import numpy as np
import pytest

import mkn_cards as K
import mkn_constants as C
import mkn_reference as REF
import mkn_scenario as SC
from mkn_normal import Stream


def test_daten_sind_reproduzierbar_und_der_sprung_beginnt_nach_nu():
    x = SC.make_data(5, 2, 3.0, 40, 7)
    assert x.shape == (C.N_VERLAUF, 5) and np.array_equal(x, SC.make_data(5, 2, 3.0, 40, 7)) and not np.array_equal(x, SC.make_data(5, 2, 3.0, 40, 8))
    basis = Stream(7).normal((C.N_VERLAUF, 5))
    assert np.array_equal(x[:40], basis[:40])                           # Vektoren 1 bis ν unverändert
    assert np.allclose(x[40:, :2], basis[40:, :2] + 3.0) and np.array_equal(x[40:, 2:], basis[40:, 2:])      # ab Vektor ν + 1: nur die ersten m Kanäle um d


def test_m_ausserhalb_von_1_bis_p_ist_ein_fehler():
    for m in (0, 6):
        with pytest.raises(ValueError):
            SC.make_data(5, m, 1.0, 0, 1)


def test_t2_statistik_ist_die_quadratsumme_und_alarm_ist_der_erste_ueber_der_schranke():
    t = SC.run_trajectory(5, 5, 3.0, 20, 15, C.KARTE_T2)
    x = SC.make_data(5, 5, 3.0, 20, 15)
    assert np.allclose(t.stat, (x * x).sum(axis=1)) and t.threshold == K.ucl_t2(5) and t.alarm_channel is None
    assert t.alarm_t == int(np.flatnonzero(t.stat > t.threshold)[0]) + 1 and t.alarm_t > 20 and t.delay == t.alarm_t - 20 and not t.false_alarm


def test_shewhart_nennt_den_alarmierenden_kanal():
    t = SC.run_trajectory(5, 1, 3.0, 20, 15, C.KARTE_SHEW)
    x = SC.make_data(5, 1, 3.0, 20, 15)
    assert np.allclose(t.stat, np.abs(x).max(axis=1)) and t.threshold == K.z_sidak(5)[0]
    assert t.alarm_t is not None and t.alarm_channel == int(np.argmax(np.abs(x[t.alarm_t - 1]))) + 1
    if t.alarm_t > 20:
        assert t.alarm_channel == 1          # nach dem Sprung von Kanal 1 um 3 σ ist es dieser Kanal


def test_cusum_statistik_gleich_eigener_rekursion():
    p, h = 5, REF.cusum_h(5)
    t = SC.run_trajectory(p, 1, 1.0, 50, 15, C.KARTE_CUSUM)
    x = SC.make_data(p, 1, 1.0, 50, 15)
    sp_, sn, stat = np.zeros(p), np.zeros(p), []
    for row in x:
        for i in range(p):
            sp_[i] = max(0.0, sp_[i] + row[i] - 0.5)
            sn[i] = max(0.0, sn[i] - row[i] - 0.5)
        stat.append(max(sp_.max(), sn.max()))
    assert np.allclose(t.stat, stat) and t.threshold == h
    assert t.alarm_t == next(i + 1 for i, s in enumerate(stat) if s > h)


def test_ewma_statistik_gleich_eigener_rekursion():
    p, H = 5, REF.ewma_H(5)
    t = SC.run_trajectory(p, 1, 1.0, 50, 15, C.KARTE_EWMA)
    x = SC.make_data(p, 1, 1.0, 50, 15)
    z, stat = [0.0] * p, []
    for row in x:
        z = [0.1 * row[i] + 0.9 * z[i] for i in range(p)]
        stat.append(max(abs(v) for v in z))
    assert np.allclose(t.stat, stat) and t.threshold == H
    assert t.alarm_t == next(i + 1 for i, s in enumerate(stat) if s > H)


def test_fehlalarm_und_verzoegerung_sind_getrennte_aussagen():
    """ν = 300 und Zufalls-Seed, bei dem T² früh alarmiert: Alarm vor der Änderung ist ein Fehlalarm und hat keine Verzögerung."""
    for seed in range(1, 200):
        t = SC.run_trajectory(20, 1, 0.5, 300, seed, C.KARTE_SHEW)
        if t.alarm_t is not None and t.alarm_t <= 300:
            assert t.false_alarm and t.delay is None
            return
    pytest.fail("kein Fehlalarm in 199 Verläufen: bei ARL₀ 370,4 und 300 Vektoren zu erwarten wären rund 160")


def test_ohne_alarm_hat_der_verlauf_weder_fehlalarm_noch_verzoegerung():
    t = SC.run_trajectory(5, 1, 0.5, 300, 15, C.KARTE_CUSUM)
    if t.alarm_t is None:
        assert not t.false_alarm and t.delay is None and t.alarm_channel is None


def test_unbekannte_karte_ist_ein_fehler():
    with pytest.raises(ValueError):
        SC.run_trajectory(5, 1, 1.0, 0, 1, "Unsinn")


def test_alarmanteil_je_vektor_mit_dem_splitmix_strom_im_band():
    """Die Zufallsquelle der App (SplitMix64, Inversionsverfahren) trifft die exakten Alarmwahrscheinlichkeiten: 300 000 Vektoren, p = 5, ein Kanal um 1 σ, Band z <= 4."""
    N, p = 300_000, 5
    x = Stream(24680).normal((N, p))
    x[:, 0] += 1.0
    for name, k, pa in (("T²", int(((x * x).sum(1) > K.ucl_t2(p)).sum()), K.pa_t2(p, 1.0)), ("getrennt", int((np.abs(x) > K.z_sidak(p)[0]).any(1).sum()), K.pa_sep(p, 1, 1.0))):
        assert abs(k / N - pa) < 4 * math.sqrt(pa * (1 - pa) / N), name


def test_mewma_live_gegen_gefrorene_zahl_und_gegen_zufalls_seed():
    p, d, R = 5, 1.0, 2000
    live = SC.mewma_live(p, d, R, 777)
    mean, se, _ = REF.mewma_arl1_sim(p, d)
    assert live.R == R and live.censored == 0 and abs(live.mean - mean) < 4 * math.hypot(live.se, se)
    assert live.mean == SC.mewma_live(p, d, R, 777).mean and live.mean != SC.mewma_live(p, d, R, 778).mean
    assert live.seconds < 30          # Rechenzeit dieser Maschine, nur eine grobe Obergrenze (gemessen: unter einer Sekunde)


def test_mewma_live_grenzen():
    for R in (0, C.R_MAX + 1):
        with pytest.raises(ValueError):
            SC.mewma_live(5, 1.0, R, 1)
    # kleinster und größter Fall der Regler laufen: p = 20, d = 0,5, R = 2000 ist der teuerste
    big = SC.mewma_live(20, 0.5, C.R_MAX, 1)
    mean, se, _ = REF.mewma_arl1_sim(20, 0.5)
    assert abs(big.mean - mean) < 4 * math.hypot(big.se, se) and big.censored == 0
