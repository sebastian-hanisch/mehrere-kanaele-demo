"""Gefrorene Tabellen (mkn_cusum_frozen.json, erzeugt von tools/mkn_make_cusum.py): Schwellen und ARL der Karten mit Gedächtnis (CUSUM, EWMA, MEWMA).

Die Gitterrechnungen brauchen Sekunden bis Minuten und werden nicht live gerechnet. Jedes Feld nennt über `meta` Quelle, Gitter und Stand; die MEWMA-ARL₁ sind
Simulationen (SplitMix64) mit Standardfehler. Zähleinheit: Beobachtungsvektoren bis zum ersten Alarm, Maß zero-state (Start im Nullzustand), wo nicht steady-state steht."""

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parent / "mkn_cusum_frozen.json"


@lru_cache(maxsize=1)
def tables() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return tables()["meta"]


def case_key(m: int, d: float) -> str:
    return f"{m}|{d:.6f}"


def cusum_h(p: int) -> float:
    return float(tables()["cusum"][str(p)]["h"])


def cusum_arl0(p: int) -> float:
    return float(tables()["cusum"][str(p)]["arl0"])


def cusum_arl1(p: int, m: int, d: float):
    """ARL₁ der getrennten zweiseitigen CUSUM (m von p Kanälen springen um d) oder None, wenn der Fall nicht gefroren ist."""
    v = tables()["cusum"][str(p)]["arl1"].get(case_key(m, d))
    return None if v is None else float(v)


def cusum_quantiles0(p: int) -> dict:
    """Mittel, Median, 5-%-, 95-%-Quantil und P(RL <= 100) der gemeinsamen Lauflänge ohne Änderung (Gitter m = 48)."""
    return tables()["cusum"][str(p)]["quantile0"]


def cusum_steady(p: int, d: float):
    """(zero-state, steady-state) ARL₁ der getrennten CUSUM, ein Kanal springt; nur für p = 1 und 5 und d = 0,5 / 1 / 2, sonst None."""
    rec = tables()["steady"].get(str(p))
    if rec is None:
        return None
    v = rec.get(f"{d:.1f}")
    return None if v is None else (float(v[0]), float(v[1]))


def ewma_H(p: int) -> float:
    return float(tables()["ewma"][str(p)]["H"])


def ewma_arl0(p: int) -> float:
    return float(tables()["ewma"][str(p)]["arl0"])


def ewma_arl1(p: int, d: float):
    v = tables()["ewma"][str(p)]["arl1"].get(f"{d:.1f}")
    return None if v is None else float(v)


def mewma_H(p: int) -> float:
    return float(tables()["mewma"][str(p)]["H"])


def mewma_arl0(p: int) -> float:
    """ARL₀ der MEWMA mit der gefrorenen Schwelle aus der radialen Kette (n = 400 Ringe)."""
    return float(tables()["mewma"][str(p)]["arl0"]["400"])


def mewma_arl1_sim(p: int, d: float):
    """(Mittel, Standardfehler, Läufe) der Simulation, ein Kanal springt um d, oder None."""
    v = tables()["mewma"][str(p)]["arl1_sim"].get(f"{d:.1f}")
    return None if v is None else (float(v[0]), float(v[1]), int(v[2]))


def mewma_lambda_paar() -> dict:
    return tables()["mewma_lambda_paar"]
