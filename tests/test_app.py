"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Erweiterungen dieser Demo (der Anfang der Datei bleibt die Vorlage aus tools/templates/test_app_smoke.py) -----------------------------------
import pytest  # noqa: E402

import mkn_constants as C  # noqa: E402
import mkn_presets as P  # noqa: E402


def _texte(at) -> str:
    return " ".join([w.value for w in at.success] + [w.value for w in at.info] + [w.value for w in at.warning])


def test_seite_hat_fuenf_experimente_und_titel():
    at = _fresh_app()
    assert at.title and sum(1 for s in at.subheader if s.value.startswith("🔬")) == 5


def test_standard_zeigt_alle_fuenf_verdikte_und_die_diagramme():
    at = _fresh_app()
    text = _texte(at)
    for stueck in ("Ein Verlauf ist eine Stichprobe", "steigt ARL₁", "Die Šidák-Schranke", "am schnellsten ist", "Das Gedächtnis bringt hier"):
        assert stueck in text, stueck
    assert len(at.get("plotly_chart")) == 6          # Verlauf, Strafe der Kanalzahl, Ebene, Balken (Korrelation), Verdünnung, Balken (Gedächtnis)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_und_setzt_alle_regler(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, state_key in P.PRESET_KEYS.items():
        soll = C.PRESETS[name].get(key, P.SETTING_SPECS[state_key].default)
        ist = at.session_state[state_key]
        assert abs(ist - soll) < 1e-9 if isinstance(soll, float) else ist == soll, (name, state_key)


def test_zufalls_button_aendert_den_seed():
    at = _fresh_app()
    alt = at.session_state["seed_input"]
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["seed_input"] != alt


def _regler(at, key):
    for art in ("select_slider", "radio", "slider", "checkbox"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_every_option_of_every_selection_does_not_raise():
    """Jede Option jeder Auswahl (select_slider, radio) einzeln; die Werte stammen aus SETTING_SPECS (AppTest nennt die Optionen als Text)."""
    gefunden = 0
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.options is None or spec.caster is P.to_bool:
            continue
        for option in spec.options:
            at = _fresh_app()
            _regler(at, state_key).set_value(option).run(timeout=TIMEOUT)
            gefunden += 1
            assert not at.exception, (state_key, option, [str(e) for e in at.exception])
            assert at.session_state[state_key] == option
    # p (5), d (4), Karte (4), Korrelationsmodell (2), Richtung (4), Verdünnung-p (2), Vergleich (2)
    assert gefunden == 5 + 4 + 4 + 2 + 4 + 2 + 2


def test_beide_wahrheitswerte_der_kontrollkaesten_laufen():
    for key in ("gleich_check", "vd_summe_check"):
        for wert in (True, False):
            at = _fresh_app()
            at.sidebar.checkbox(key=key).set_value(wert).run(timeout=TIMEOUT)
            assert not at.exception and at.session_state[key] is wert, (key, wert)


def test_gleiche_nichtzentralitaet_sperrt_den_sprungregler():
    at = _fresh_app()
    assert not at.sidebar.slider(key="sprung_slider").disabled
    at.sidebar.checkbox(key="gleich_check").set_value(True).run(timeout=TIMEOUT)
    assert at.sidebar.slider(key="sprung_slider").disabled and not at.exception


def test_p5_mit_negativer_korrelation_nennt_es_statt_zu_rechnen():
    at = _fresh_app()
    at.sidebar.radio(key="korr_modell_radio").set_value(C.KORR_P5).run(timeout=TIMEOUT)
    at.sidebar.slider(key="rho_slider").set_value(-0.5).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("nur Korrelationen ρ ≥ 0 gerechnet" in w.value for w in at.warning)


def test_m_groesser_p_springen_alle_kanaele():
    at = _fresh_app()
    at.sidebar.select_slider(key="p_select").set_value(2).run(timeout=TIMEOUT)
    at.sidebar.slider(key="m_slider").set_value(7).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("es springen alle 2 Kanäle" in c.value for c in at.caption)


def test_aenderung_spaet_im_verlauf_laeuft_ohne_fehler():
    """ν = 300 und kleiner Sprung: bei 400 Vektoren bleibt wenig Zeit; die App meldet Alarm oder keinen Alarm, aber keinen Fehler."""
    at = _fresh_app()
    at.sidebar.slider(key="nu_slider").set_value(300).run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="d_select").set_value(0.5).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Ein Verlauf ist eine Stichprobe" in _texte(at)
