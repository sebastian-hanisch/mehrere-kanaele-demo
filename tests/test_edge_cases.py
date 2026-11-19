"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): p = 1, p = 20, m = p, d = 0, ρ am Rand, Σ nicht positiv definit, ν hinter dem Ende, nicht gefrorene Fälle."""

import math

import numpy as np
import pytest

import mkn_cards as K
import mkn_constants as C
import mkn_corr as R
import mkn_evaluation as ev
import mkn_reference as REF
import mkn_scenario as SC


def test_p1_alle_karten_sind_dieselbe_karte():
    """p = 1: T², getrennte Karte und Summenkarte sind dieselbe Schranke 3 σ; die Šidák-Schranke ist genau 3 und die ARL₀ je Kanal 370,398."""
    assert abs(K.z_sidak(1)[0] - 3.0) < 1e-9 and abs(K.arl0_per_channel(1) - C.GAMMA0) < 1e-9
    for d in C.D_OPTIONS:
        assert abs(K.arl_t2(1, d * d) / K.arl_sep(1, 1, d) - 1) < 1e-9 and abs(K.arl_sum(1, 1, d) / K.arl_sep(1, 1, d) - 1) < 1e-12


def test_p20_arl0_je_kanal_7398():
    assert abs(K.arl0_per_channel(20) - 7398.46) < 0.01 and abs(K.arl0_per_channel(2) - 740.30) < 0.01 and abs(K.arl0_per_channel(5) - 1849.99) < 0.01


def test_d_null_arl1_gleich_arl0_fuer_jede_karte():
    for p in C.P_OPTIONS:
        for a in (K.arl_t2(p, 0.0), K.arl_sep(p, 1, 0.0), K.arl_sep(p, p, 0.0), K.arl_sum(p, p, 0.0)):
            assert abs(a / C.GAMMA0 - 1) < 1e-9


def test_m_gleich_p_und_grosser_sprung_arl1_nahe_eins():
    assert K.arl_t2(20, 20 * 9.0) < 1.0001 and K.arl_sep(20, 20, 3.0) < 1.02 and K.arl_sum(20, 20, 3.0) < 1.0001
    assert K.arl_t2(20, 20 * 9.0) >= 1.0 and K.arl_sep(20, 20, 3.0) >= 1.0


def test_sprung_aller_kanaele_ist_fuer_summenkarte_am_schnellsten_bei_gleicher_hoehe():
    """m = p = 5, d = 1: Summe 4,50 < T² 12,11 < getrennt 29,16 (Aussage „alle springen“, nicht allgemein)."""
    assert K.arl_sum(5, 5, 1.0) < K.arl_t2(5, 5.0) < K.arl_sep(5, 5, 1.0)


def test_rho_am_rand_der_regler_rechnet_und_rho_eins_ist_ausserhalb():
    for rho in (C.RHO_MIN, C.RHO_MAX, 0.0):
        row = R.arl_row(C.KORR_P2, rho, R.jump_vector(C.KORR_P2, rho, C.RICHTUNG_EIN, 1.0, False))
        assert all(math.isfinite(row[k]) and row[k] > 0 for k in ("lam", "t2", "sep_new", "sep_sidak", "arl0_sidak"))
    assert C.RHO_MAX < 1.0 and C.RHO_MIN > -1.0
    with pytest.raises((ZeroDivisionError, np.linalg.LinAlgError, ValueError)):
        R.noalarm_p2(3.0, 0.0, 0.0, 1.0)          # ρ = 1 ist keine Korrelationsmatrix mit Rang 2: die Regler lassen es nicht zu
    with pytest.raises(np.linalg.LinAlgError):
        R.ellipse_points(1.0, 11.8)


def test_p5_negative_gleichkorrelation_wird_nicht_gerechnet():
    assert ev.exp3_row(ev.Settings(korr_modell=C.KORR_P5, rho=-0.1)) is None
    assert ev.exp3_row(ev.Settings(korr_modell=C.KORR_P5, rho=0.0)) is not None and ev.exp3_row(ev.Settings(korr_modell=C.KORR_P2, rho=-0.95)) is not None
    # p = 5, ρ = -0,3 ist nicht positiv definit (kleinster Eigenwert 1 - 4·0,3 + ... < 0 ab ρ < -1/4): auch die Matrix selbst bestätigt die Grenze
    assert np.linalg.eigvalsh(R.cov_equi(5, -0.3)).min() < 0 < np.linalg.eigvalsh(R.cov_equi(5, -0.2)).min()


def test_unbekannte_richtung_ist_ein_fehler():
    with pytest.raises(ValueError):
        R.direction(C.KORR_P2, "Unsinn")


def test_gleiche_nichtzentralitaet_skaliert_auch_bei_rho_extrem():
    for rho in (C.RHO_MIN, 0.0, C.RHO_MAX):
        for name in C.RICHTUNGEN:
            mu = R.jump_vector(C.KORR_P2, rho, name, 1.0, True)
            assert abs(R.noncentrality(C.KORR_P2, rho, mu) - C.LAMBDA_GLEICH) < 1e-9


def test_aenderung_hinter_dem_ende_des_verlaufs_hat_keinen_sprung_und_keine_verzoegerung():
    for nu in (C.N_VERLAUF, C.N_VERLAUF + 50):
        t = SC.run_trajectory(5, 1, 3.0, nu, 15, C.KARTE_SHEW)
        assert t.delay is None
        assert np.allclose(SC.make_data(5, 1, 3.0, nu, 15), SC.make_data(5, 1, 0.0, 0, 15))     # der Sprung kommt nie vor, die Daten sind die ohne Sprung


def test_nicht_gefrorene_faelle_melden_none_und_die_app_zeigt_es():
    assert REF.cusum_arl1(5, 3, 0.5) is None and ev.arl1_of_card(C.KARTE_EWMA, 5, 2, 1.0) is None and ev.arl1_of_card(C.KARTE_CUSUM, 2, 2, 1.0) is None
    assert ev.arl1_of_card(C.KARTE_T2, 20, 7, 0.5) is not None and ev.arl1_of_card(C.KARTE_SHEW, 20, 7, 0.5) is not None        # ohne Gedächtnis jedes m exakt
    with pytest.raises(ValueError):
        ev.arl1_of_card("Unsinn", 5, 1, 1.0)


def test_verteilungszeilen_je_karte():
    assert ev.null_summary(C.KARTE_EWMA, 5) is None and ev.run_length_summary(C.KARTE_CUSUM, 5, 1, 1.0) is None
    assert ev.null_summary(C.KARTE_CUSUM, 5)["median"] == 258 and ev.null_summary(C.KARTE_T2, 20)["median"] == 257


def test_geometrische_quantile_bei_sehr_grosser_und_sehr_kleiner_alarmwahrscheinlichkeit():
    assert K.geo_quantile(0.999999, 0.5) == 1 and K.geo_quantile(1e-6, 0.5) == 693147
    assert K.geo_summary(1.0 - 1e-12)["median"] == 1


def test_memory_rows_alle_karten_und_steady_nur_wo_gemessen():
    rows = ev.memory_rows(5, 1.0)
    assert [r["karte"] for r in rows] == [C.KARTE_T2, C.KARTE_SHEW, C.KARTE_CUSUM, C.KARTE_EWMA, "MEWMA (Simulation)"] and rows[2]["steady"] is not None
    assert all(r["steady"] is None for r in ev.memory_rows(5, 3.0)) and all(r["steady"] is None for r in ev.memory_rows(10, 1.0))


def test_verduennung_energie_und_hoehe():
    assert K.dilution_d(4, C.VD_GLEICH_ENERGIE) == 1.0 and abs(K.dilution_d(1, C.VD_GLEICH_ENERGIE) - 2.0) < 1e-12 and K.dilution_d(5, C.VD_GLEICH_HOEHE) == 1.0
    for m in (1, 2, 4, 5, 10):
        assert abs(K.dilution_row(10, m, K.dilution_d(m, C.VD_GLEICH_ENERGIE))["lam"] - 4.0) < 1e-12
