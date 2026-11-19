"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Zufalls-Seeds. Damit prüft der Test die Grenzen und jede genannte Zahl direkt."""

import mkn_constants as C
import mkn_evaluation as ev


# ------------------------------------------------------------------------------------------------------------------ Experiment 1
def test_exp1_alarm_nach_der_aenderung_ist_success_und_nennt_verzoegerung_kanal_und_arl():
    kind, text = ev.verdict_exp1(C.KARTE_SHEW, 5, 1, 1.0, 50, 88, 1, 38, 109.89, 76)
    assert kind == "success"
    for s in ("Getrennte Shewhart-Karten (Šidák), p = 5, 1 Kanal um 1,0 σ ab Vektor 51", "erster Alarm bei Vektor 88 (Kanal 1)", "38 Vektoren nach dem Beginn der Änderung",
              "ARL₁ dieser Karte 109,9 Vektoren, Median 76", "Ein Verlauf ist eine Stichprobe"):
        assert s in text, s


def test_exp1_ein_vektor_verzoegerung_steht_im_singular_und_t2_nennt_keinen_kanal():
    kind, text = ev.verdict_exp1(C.KARTE_T2, 5, 5, 3.0, 0, 1, None, 1, 3.5, 2)
    assert kind == "success" and "1 Vektor nach dem Beginn" in text and "Kanal" not in text.split("erster Alarm")[1].split("nach dem")[0] and "5 Kanäle um 3,0 σ" in text


def test_exp1_alarm_vor_der_aenderung_ist_ein_fehlalarm_und_warnt():
    kind, text = ev.verdict_exp1(C.KARTE_T2, 5, 1, 1.0, 50, 6, None, None, 114.38, 79)
    assert kind == "warning" and "Fehlalarm bei Vektor 6, noch vor der Änderung" in text and "ν = 0" not in text
    assert ev.verdict_exp1(C.KARTE_T2, 5, 1, 1.0, 50, 50, None, None, 114.38, 79)[0] == "warning"        # Vektor ν ist der letzte vor der Änderung
    assert ev.verdict_exp1(C.KARTE_T2, 5, 1, 1.0, 50, 51, None, 1, 114.38, 79)[0] == "success"          # Vektor ν + 1 ist der erste geänderte


def test_exp1_ohne_alarm_ist_info_und_nennt_die_zahl_der_vektoren():
    kind, text = ev.verdict_exp1(C.KARTE_CUSUM, 5, 1, 0.5, 300, None, None, None, 52.07, None)
    assert kind == "info" and f"in diesen {C.N_VERLAUF} Vektoren kein Alarm" in text and "ARL₁ dieser Karte 52,1 Vektoren (Änderung" in text


def test_exp1_ohne_gefrorene_arl_nennt_es():
    kind, text = ev.verdict_exp1(C.KARTE_EWMA, 5, 2, 1.0, 50, 60, 1, 10, None, None)
    assert "keine ARL₁ gefroren" in text and "2 Kanäle um 1,0 σ" in text


# ------------------------------------------------------------------------------------------------------------------ Experiment 2
def test_exp2_nennt_die_drei_zahlen_und_die_zaehlung():
    kind, text = ev.verdict_exp2(1.0, 1, 20, 43.89, 206.87, 200.63, 9.93, 15.71, 4, 16)
    assert kind == "info"
    for s in ("d = 1,0", "von p = 1 auf p = 20", "bei T² von 43,9 auf 206,9", "getrennten Shewhart-Karten auf 200,6", "getrennten CUSUM nur von 9,9 auf 15,7",
              "in 4 von 16 gemessenen Zellen"):
        assert s in text, s


def test_exp2_ohne_cusum_zahlen_laesst_den_cusum_teil_weg():
    assert "CUSUM" not in ev.verdict_exp2(1.0, 1, 20, 43.89, 206.87, 200.63, None, None, 4, 16)[1]


# ------------------------------------------------------------------------------------------------------------------ Experiment 3
def test_exp3_vergleich_schwelle_10_prozent():
    assert ev.compare_words(100.0, 110.5) == "t2"             # getrennt mehr als 10 % langsamer: T² schneller
    assert ev.compare_words(100.0, 109.5) == "gleich" and ev.compare_words(109.5, 100.0) == "gleich" and ev.compare_words(100.0, 100.0) == "gleich"
    assert ev.compare_words(110.5, 100.0) == "sep"


def test_exp3_t2_schneller_success_mit_faktor():
    kind, text = ev.verdict_exp3(C.KORR_P2, 0.8, "gegen", 10.0, 2.22, 32.99, 36.42, 422.7)
    assert kind == "success" and "p = 2, ρ = 0,80, Sprung „gegen“ (λ = 10,00)" in text
    assert "T² braucht 2,22 Vektoren" in text and "T² ist 14,9-mal schneller" in text and "ARL₀ 422,7 statt 370,4" in text and "ARL₁ dann 36,42" in text


def test_exp3_getrennt_schneller_warning_nennt_t2_langsamer():
    kind, text = ev.verdict_exp3(C.KORR_P2, 0.8, "entlang", 1.11, 59.71, 41.27, 45.39, 422.7)
    assert kind == "warning" and "getrennte Karten sind 1,4-mal schneller" in text and "nur 41,27" in text


def test_exp3_gleich_info_und_arl0_370_4_wie_vorgesehen_bei_rho_0():
    kind, text = ev.verdict_exp3(C.KORR_P2, 0.0, "ein Kanal", 1.0, 67.32, 66.34, 66.34, 370.3983)
    assert kind == "info" and "innerhalb von 10 %" in text and "ARL₀ 370,4, wie vorgesehen" in text and "statt" not in text


def test_exp3_p5_wird_als_gleichkorreliert_benannt():
    assert "p = 5 gleichkorreliert, ρ = 0,60" in ev.verdict_exp3(C.KORR_P5, 0.6, "ein Kanal", 2.06, 50.25, 106.51, 111.0, 414.1)[1]


# ------------------------------------------------------------------------------------------------------------------ Experiment 4
def test_exp4_nennt_alle_vier_karten_und_die_schnellste():
    kind, text = ev.verdict_exp4(5, 4, C.VD_GLEICH_ENERGIE, 1.0, 4.0, 17.93, 35.64, 8.86, 7.69)
    assert kind == "info"
    for s in ("p = 5, 4 Kanäle springen um je 1,00 σ (Vergleich: gleiche Energie (λ = m·d² = 4); λ = 4,00)", "T² 17,93", "getrennte Shewhart-Karten 35,64", "Summenkarte (Richtung bekannt) 8,86", "getrennte CUSUM 7,69",
              "am schnellsten ist getrennte CUSUM", "Die Summe von CUSUMs ist nicht gemessen"):
        assert s in text, s


def test_exp4_ein_kanal_singular_und_summenkarte_ausgeblendet_und_cusum_nicht_gefroren():
    kind, text = ev.verdict_exp4(10, 1, C.VD_GLEICH_HOEHE, 1.0, 1.0, 159.91, 152.90, None, None)
    assert "1 Kanal springt um je 1,00 σ" in text and "Summenkarte (Richtung bekannt)" not in text and "CUSUM nicht gefroren" in text and "am schnellsten ist getrennte Shewhart-Karten" in text


def test_exp4_summenkarte_gewinnt_bei_allen_springenden_kanaelen():
    assert "am schnellsten ist Summenkarte (Richtung bekannt)" in ev.verdict_exp4(5, 5, C.VD_GLEICH_HOEHE, 1.0, 5.0, 12.11, 29.16, 4.50, 7.21)[1]


# ------------------------------------------------------------------------------------------------------------------ Experiment 5
def test_exp5_gedaechtnis_schneller_success_mit_steady_state():
    kind, text = ev.verdict_exp5(5, 1.0, 114.38, 109.89, 13.00, 12.93, 14.61, 0.05, 12.11)
    assert kind == "success"
    for s in ("T² 114,38", "getrennte Shewhart-Karten 109,89", "CUSUM 13,00 (steady-state 12,11)", "EWMA 12,93", "MEWMA 14,61 ± 0,05 (Simulation)", "mehr als die Wahl zwischen T² und getrennten Karten",
              "die schnellste Karte mit Gedächtnis braucht 12,93", "die schnellere der beiden ohne Gedächtnis 109,89", "4 bis 8 %", "Lorden- und Pollak-Maß sind nicht gemessen"):
        assert s in text, s


def test_exp5_ohne_steady_state_kein_klammerzusatz():
    assert "steady-state" not in ev.verdict_exp5(10, 1.0, 159.91, 152.90, 14.35, 14.41, 18.18, 0.06, None)[1].split("MEWMA")[0]


def test_exp5_grosser_sprung_ein_kanal_gedaechtnis_bringt_nichts_warning():
    kind, text = ev.verdict_exp5(1, 3.0, 2.00, 2.00, 2.49, 2.76, 2.77, 0.00, None)
    assert kind == "warning" and "bringt hier nichts" in text and "braucht 2,00" in text and "mit Gedächtnis 2,49" in text


def test_exp5_schwelle_gleichstand_ist_nicht_besser():
    assert ev.verdict_exp5(1, 3.0, 2.0, 2.0, 2.0, 2.5, 2.8, 0.0, None)[0] == "warning"          # genau gleich schnell: das Gedächtnis „bringt“ nichts
    assert ev.verdict_exp5(1, 3.0, 2.0, 2.0, 1.99, 2.5, 2.8, 0.0, None)[0] == "success"


# ------------------------------------------------------------------------------------------------------------------ Formate und Zeilen
def test_zahlenformat_dezimalkomma_und_vorzeichen():
    assert ev.de(1234.5678, 2) == "1234,57" and ev.de_s(0.0, 1) == "+0,0"
    assert ev.de_s(2.5, 1) == "+2,5" and ev.de_s(-2.5, 1) == "-2,5" and ev.pct(0.2369, 2) == "23,69 %"


def test_settings_begrenzt_m_auf_p():
    assert ev.Settings(p=2, m=7).m_eff == 2 and ev.Settings(p=20, m=7).m_eff == 7 and ev.Settings(vd_p=5, vd_m=10).vd_m_eff == 5 and ev.Settings().m_eff == 1
