"""Karten ohne Gedächtnis, exakt: Hotelling-T², getrennte Shewhart-Karten mit Šidák-Schranke, Summenkarte (Richtung bekannt).

Modell: Vektoren x_t ~ N(μ, I) (Einheitsvarianzen, unabhängige Kanäle; Korrelation steht in `mkn_corr.py`), vor der Änderung μ = 0, nach der Änderung springen m der
p Kanäle um d σ. Der Alarm hängt nur vom aktuellen Vektor ab: die Lauflänge ist geometrisch mit der Alarmwahrscheinlichkeit je Vektor, und zero-state-, Lorden- und
Pollak-Verzögerung sind dieselbe Zahl (ARL₁). Zähleinheit: Beobachtungsvektoren bis zum ersten Alarm. Alle Schranken sind auf die gemeinsame ARL₀ = 370,398 geeicht."""

import math
from functools import lru_cache

from scipy.special import ndtr
from scipy.stats import chi2, ncx2, norm

import mkn_constants as C


# ------------------------------------------------------------------------------------------------------------------ Schranken
@lru_cache(maxsize=None)
def ucl_t2(p: int) -> float:
    """Schranke der T²-Karte: Alarm bei x'x > chi2-Quantil zur Alarmwahrscheinlichkeit 1/370,398 (Σ bekannt)."""
    return float(chi2.isf(C.P_ALARM0, p))


@lru_cache(maxsize=None)
def z_sidak(p: int) -> tuple[float, float]:
    """(z, q): q = 1 - (1 - 1/370,398)^(1/p) ist die Alarmwahrscheinlichkeit je Kanal, z = Φ⁻¹(1 - q/2) die zweiseitige Schranke je Kanal."""
    q = 1.0 - (1.0 - C.P_ALARM0) ** (1.0 / p)
    return float(norm.isf(q / 2.0)), q


def arl0_per_channel(p: int) -> float:
    """ARL₀ je Kanal 1/q: so groß muss sie sein, damit die gemeinsame ARL₀ 370,398 ist (unabhängige Kanäle)."""
    return 1.0 / z_sidak(p)[1]


# ------------------------------------------------------------------------------------------------------------------ Alarmwahrscheinlichkeit je Vektor
def pa_t2(p: int, lam: float) -> float:
    """Alarmwahrscheinlichkeit je Vektor der T²-Karte bei Nichtzentralität λ = μ'Σ⁻¹μ (scipy.stats.ncx2)."""
    return float(ncx2.sf(ucl_t2(p), p, lam)) if lam > 0 else float(chi2.sf(ucl_t2(p), p))


def pa_sep(p: int, m: int, d: float, z: float | None = None) -> float:
    """Getrennte Karten (|x_i| > z), unabhängige Kanäle, m Kanäle um d verschoben (geschlossen); z Standard: Šidák."""
    if z is None:
        z = z_sidak(p)[0]
    q = 2.0 * norm.sf(z)
    ps = norm.sf(z - d) + ndtr(-z - d)
    return 1.0 - (1.0 - ps) ** m * (1.0 - q) ** (p - m)


def pa_sum(p: int, m: int, d: float) -> float:
    """Summenkarte: s = Σ x_i / √p ~ N(m·d/√p, 1), Alarm bei |s| > 3 (die Richtung „alle Kanäle gleichsinnig“ ist bekannt)."""
    a = m * d / math.sqrt(p)
    return float(norm.sf(3.0 - a) + ndtr(-3.0 - a))


def arl_t2(p: int, lam: float) -> float:
    return 1.0 / pa_t2(p, lam)


def arl_sep(p: int, m: int, d: float, z: float | None = None) -> float:
    return 1.0 / pa_sep(p, m, d, z)


def arl_sum(p: int, m: int, d: float) -> float:
    return 1.0 / pa_sum(p, m, d)


# ------------------------------------------------------------------------------------------------------------------ Verteilung der Lauflänge (geometrisch)
def geo_quantile(pa: float, q: float) -> int:
    """Kleinstes n mit P(RL <= n) >= q für die geometrische Lauflänge mit Erfolgswahrscheinlichkeit pa."""
    return int(math.ceil(math.log1p(-q) / math.log1p(-pa)))


def geo_summary(pa: float) -> dict:
    """Mittel, Median, 5-%- und 95-%-Quantil, P(Alarm bis einschließlich Beobachtung 100) der geometrischen Lauflänge (exakt)."""
    return dict(mean=1.0 / pa, median=geo_quantile(pa, 0.5), q05=geo_quantile(pa, 0.05), q95=geo_quantile(pa, 0.95), p100=1.0 - (1.0 - pa) ** 100)


# ------------------------------------------------------------------------------------------------------------------ Tabellen
def penalty_row(p: int, d: float) -> dict:
    """ARL₁ der drei Karten ohne Gedächtnis bei einem springenden Kanal der Höhe d (Zeile der Tabelle „Strafe der Kanalzahl“)."""
    z, q = z_sidak(p)
    return dict(p=p, d=d, ucl=ucl_t2(p), z=z, arl0_kanal=1.0 / q, t2=arl_t2(p, d * d), sep=arl_sep(p, 1, d), sum=arl_sum(p, 1, d))


def t2_faster_cells(ps=(2, 5, 10, 20), ds=C.D_OPTIONS) -> list[tuple[int, float]]:
    """Zellen (p, d), in denen T² bei einem springenden Kanal schneller ist als p getrennte Šidák-Karten (unabhängige Kanäle)."""
    return [(p, d) for p in ps for d in ds if arl_t2(p, d * d) < arl_sep(p, 1, d)]


def dilution_row(p: int, m: int, d: float) -> dict:
    """ARL₁ von T², getrennten Karten und Summenkarte, wenn m von p Kanälen um je d springen (λ = m·d²)."""
    return dict(p=p, m=m, d=d, lam=m * d * d, t2=arl_t2(p, m * d * d), sep=arl_sep(p, m, d), sum=arl_sum(p, m, d))


def dilution_d(m: int, mode: str) -> float:
    """Sprunghöhe je Kanal: gleiche Höhe d = 1, oder gleiche Energie λ = m·d² = 4 (d = 2/√m)."""
    return 1.0 if mode == C.VD_GLEICH_HOEHE else 2.0 / math.sqrt(m)
