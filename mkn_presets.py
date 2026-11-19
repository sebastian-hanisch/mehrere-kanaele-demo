"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import mkn_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # Auswahl (select_slider, radio): nur diese Werte sind gültig
    step: float | None = None         # Raster eines Zahlenreglers (Permalink rastet ein)


def to_bool(v) -> bool:
    """Permalink-Wert als Wahrheitswert („True“/„False“, „1“/„0“)."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "ja")
    return bool(v)


SETTING_SPECS = {
    "p_select": SettingSpec("p", int, C.DEFAULT_P, C.P_OPTIONS[0], C.P_OPTIONS[-1], options=C.P_OPTIONS),
    "d_select": SettingSpec("d", float, C.DEFAULT_D, C.D_OPTIONS[0], C.D_OPTIONS[-1], options=C.D_OPTIONS),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX, step=1),
    "karte_radio": SettingSpec("karte", str, C.DEFAULT_KARTE, options=C.KARTEN),
    "nu_slider": SettingSpec("nu", int, C.DEFAULT_NU, C.NU_MIN, C.NU_MAX, step=C.NU_STEP),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "korr_modell_radio": SettingSpec("km", str, C.DEFAULT_KORR_MODELL, options=C.KORR_MODELLE),
    "rho_slider": SettingSpec("rho", float, C.DEFAULT_RHO, C.RHO_MIN, C.RHO_MAX, step=C.RHO_STEP),
    "richtung_radio": SettingSpec("richtung", str, C.DEFAULT_RICHTUNG, options=C.RICHTUNGEN),
    "sprung_slider": SettingSpec("sprung", float, C.DEFAULT_SPRUNG, C.SPRUNG_MIN, C.SPRUNG_MAX, step=C.SPRUNG_STEP),
    "gleich_check": SettingSpec("gleich", to_bool, C.DEFAULT_GLEICH, options=(False, True)),
    "vd_p_select": SettingSpec("vdp", int, C.DEFAULT_VD_P, C.VD_P_OPTIONS[0], C.VD_P_OPTIONS[-1], options=C.VD_P_OPTIONS),
    "vd_m_slider": SettingSpec("vdm", int, C.DEFAULT_VD_M, C.VD_M_MIN, C.VD_M_MAX, step=1),
    "vd_vergleich_radio": SettingSpec("vdv", str, C.DEFAULT_VD_VERGLEICH, options=C.VD_VERGLEICHE),
    "vd_summe_check": SettingSpec("vds", to_bool, C.DEFAULT_VD_SUMME, options=(False, True)),
    "r_slider": SettingSpec("r", int, C.DEFAULT_R, C.R_MIN, C.R_MAX, step=C.R_STEP),
}
PRESET_KEYS = {"p": "p_select", "d": "d_select", "m": "m_slider", "karte": "karte_radio", "nu": "nu_slider", "seed": "seed_input",
               "korr_modell": "korr_modell_radio", "rho": "rho_slider", "richtung": "richtung_radio", "sprung": "sprung_slider", "gleich": "gleich_check",
               "vd_p": "vd_p_select", "vd_m": "vd_m_slider", "vd_vergleich": "vd_vergleich_radio", "vd_summe": "vd_summe_check", "r": "r_slider"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(value, spec: SettingSpec):
    """Permalink-Wert auf die Reglergrenzen begrenzen und auf das Raster bzw. die Auswahl einrasten; None, wenn nicht möglich."""
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else None
        return min(spec.options, key=lambda o: abs(o - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None and spec.lo is not None:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = min(spec.hi, value)
        if spec.caster is float:
            value = round(value, 4)
    return spec.caster(value)


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = _snap(value, spec)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt ALLE Regler: was es nicht nennt, kehrt zum Standardwert zurück (sonst bliebe ein früheres Preset halb erhalten)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
