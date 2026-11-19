"""Je Schnellstart die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a, Punkt 7): ein Preset ohne Effekt meldet keinen Effekt, und die Zahlen im Hilfetext des Presets stehen in der Verdikt-Zeile.

Die Verdikte kommen aus der App (AppTest) mit dem Standard-Zufalls-Seed; erwartet werden Art und Wortlaut des Satzteils, der von den exakten Zahlen abhängt (die Art des Verdikts
von Experiment 1 hängt vom Verlauf ab und wird nicht erwartet)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import mkn_constants as C

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _run(name):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _verdikt(at, anfang):
    treffer = [(type(w).__name__.lower(), w.value) for w in list(at.warning) + list(at.info) + list(at.success) if w.value.startswith(anfang)]
    assert len(treffer) == 1, (anfang, [w.value[:70] for w in list(at.warning) + list(at.info) + list(at.success)])
    return treffer[0]


def test_standard_preset_alle_verdikte():
    at = _run("Ein Kanal springt, p = 5")
    assert "ARL₁ dieser Karte 114,4 Vektoren, Median 79" in _verdikt(at, "T²-Karte, p = 5, 1 Kanal um 1,0 σ ab Vektor 51")[1]
    assert "von p = 1 auf p = 20 bei T² von 43,9 auf 206,9" in _verdikt(at, "Bei gemeinsamer ARL₀ 370,4 steigt ARL₁")[1]
    kind, text = _verdikt(at, "p = 2, ρ = 0,00, Sprung „ein Kanal“")           # kein Effekt: ρ = 0, Šidák ist exakt, beide Karten gleich schnell
    assert kind == "info" and "innerhalb von 10 %" in text and "ARL₀ 370,4, wie vorgesehen" in text
    kind, text = _verdikt(at, "p = 5, 1 Kanal springt um je 2,00 σ")
    assert kind == "info" and "am schnellsten ist getrennte CUSUM" in text
    assert _verdikt(at, "p = 5, ein Kanal springt um 1,0 σ, gemeinsame ARL₀ 370,4")[0] == "success"


def test_p20_nennt_die_strafe_der_kanalzahl():
    text = _verdikt(_run("p = 20 Kanäle"), "T²-Karte, p = 20, 1 Kanal um 1,0 σ")[1]
    assert "ARL₁ dieser Karte 206,9 Vektoren" in text


def test_korreliert_gegen_t2_gewinnt_mit_faktor():
    kind, text = _verdikt(_run("Korreliert, Sprung gegen die Korrelation (ρ = 0,8)"), "p = 2, ρ = 0,80, Sprung „gegen“")
    assert kind == "success" and "T² braucht 2,22 Vektoren" in text and "32,99" in text and "ARL₀ 422,7 statt 370,4" in text


def test_korreliert_entlang_t2_verliert():
    kind, text = _verdikt(_run("Korreliert, Sprung entlang (ρ = 0,8)"), "p = 2, ρ = 0,80, Sprung „entlang“")
    assert kind == "warning" and "T² braucht 59,71 Vektoren" in text and "nur 41,27" in text and "getrennte Karten sind 1,4-mal schneller" in text


def test_alle_fuenf_springen_nennt_5_kanaele_und_aendert_die_anderen_experimente_nicht():
    at = _run("Alle fünf springen")
    assert "5 Kanäle um 1,0 σ" in _verdikt(at, "T²-Karte, p = 5, 5 Kanäle")[1] and "ARL₁ dieser Karte 12,1 Vektoren" in _verdikt(at, "T²-Karte, p = 5, 5 Kanäle")[1]
    assert "innerhalb von 10 %" in _verdikt(at, "p = 2, ρ = 0,00")[1]            # Experiment 3 ist unberührt
    assert "p = 5, 1 Kanal springt um je 2,00 σ" in _verdikt(at, "p = 5, 1 Kanal springt")[1]      # Experiment 4 ist unberührt


def test_gleiche_energie_vier_kanaele():
    kind, text = _verdikt(_run("Gleiche Energie, verteilt auf vier Kanäle"), "p = 5, 4 Kanäle springen um je 1,00 σ")
    assert kind == "info" and "T² 17,93" in text and "getrennte Shewhart-Karten 35,64" in text and "Summenkarte (Richtung bekannt) 8,86" in text and "getrennte CUSUM 7,69" in text


def test_mit_gedaechtnis_cusum_im_verlauf():
    text = _verdikt(_run("Mit Gedächtnis: getrennte CUSUM"), "Getrennte CUSUM, p = 5, 1 Kanal um 1,0 σ")[1]
    assert "ARL₁ dieser Karte 13,0 Vektoren" in text


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_liefert_alle_fuenf_verdikte(name):
    at = _run(name)
    texte = [w.value for w in list(at.warning) + list(at.info) + list(at.success)]
    for stueck in ("Ein Verlauf ist eine Stichprobe", "steigt ARL₁", "Die Šidák-Schranke", "am schnellsten ist", "Das Gedächtnis bringt hier"):
        assert any(stueck in t for t in texte), (name, stueck)
