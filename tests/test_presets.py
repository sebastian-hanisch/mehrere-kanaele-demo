"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset einzigartig; Permalink rastet ein und begrenzt."""

import mkn_constants as C
import mkn_evaluation as ev
import mkn_presets as P
import mkn_reference as REF


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def _wert(preset, key):
    return preset.get(key, P.SETTING_SPECS[P.PRESET_KEYS[key]].default)


def test_preset_values_are_inside_the_regler_bounds_and_options():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec, v = P.SETTING_SPECS[state_key], _wert(preset, key)
            if spec.options is not None:
                assert v in spec.options, (name, key)
            elif spec.lo is not None:
                assert spec.lo <= v <= spec.hi, (name, key)


def test_default_preset_is_the_first_and_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {} and ev.Settings() == ev.Settings(**{k: _wert(C.PRESETS[first], k) for k in P.PRESET_KEYS})


def test_every_preset_differs_from_every_other():
    """Zwei Presets mit denselben Reglerwerten wären nicht unterscheidbar (aktives_preset nähme immer das erste)."""
    stand = [tuple(_wert(p, k) for k in P.PRESET_KEYS) for p in C.PRESETS.values()]
    assert len(set(stand)) == len(stand)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == set(ev.Settings.__dataclass_fields__)      # jeder Regler steht auch in Settings


def test_preset_werte_liegen_auf_den_gefrorenen_gittern():
    """Jedes Preset braucht für Karten mit Gedächtnis gefrorene Zeilen: p und d aus den Optionen, Verdünnungs-p aus den gerechneten."""
    for name, preset in C.PRESETS.items():
        p, d = _wert(preset, "p"), _wert(preset, "d")
        assert REF.cusum_arl1(p, 1, d) is not None and REF.ewma_arl1(p, d) is not None and REF.mewma_arl1_sim(p, d) is not None, name
        assert REF.cusum_arl1(_wert(preset, "vd_p"), 1, 1.0) is not None, name


def test_permalink_rastet_ein_und_begrenzt():
    s = P.SETTING_SPECS
    assert P._snap(0.47, s["rho_slider"]) == 0.45 and P._snap(2.0, s["rho_slider"]) == 0.95 and P._snap(-3.0, s["rho_slider"]) == -0.95
    assert P._snap(7, s["p_select"]) == 5 and P._snap(100, s["p_select"]) == 20
    assert P._snap(1.4, s["d_select"]) == 1.0 and P._snap(9.0, s["d_select"]) == 3.0
    assert P._snap("Unsinn", s["karte_radio"]) is None and P._snap(C.KARTE_CUSUM, s["karte_radio"]) == C.KARTE_CUSUM
    assert P._snap(260, s["r_slider"]) == 200 and P._snap(340, s["r_slider"]) == 400
    assert P._snap(9999, s["r_slider"]) == 2000
    assert P._snap(94, s["nu_slider"]) == 90 and P._snap(0, s["m_slider"]) == 1 and P._snap(99, s["m_slider"]) == 20


def test_wahrheitswerte_im_permalink():
    assert P.to_bool("True") is True and P.to_bool("False") is False and P.to_bool("1") is True and P.to_bool("0") is False and P.to_bool(True) is True
    s = P.SETTING_SPECS["gleich_check"]
    assert P._snap(P.to_bool("True"), s) is True and P._snap(P.to_bool("False"), s) is False
