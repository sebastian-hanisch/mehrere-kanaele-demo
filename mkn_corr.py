"""Korrelierte Kanäle bei bekanntem Σ (Einheitsvarianzen, Korrelation ρ): T² mit Σ⁻¹ gegen getrennte Karten; exakt, ohne Simulation.

T²: ARL₀ = 370,398 für jedes Σ (Whitening), ARL₁ hängt nur an λ = μ'Σ⁻¹μ. Getrennte Karten (|x_i| > z): die Wahrscheinlichkeit, dass kein Kanal alarmiert,
ist eine Rechteckwahrscheinlichkeit der Normalverteilung N(μ, Σ). Wege: p = 2 mit einem eindimensionalen Integral über x₁ (adaptive Quadratur, jedes |ρ| < 1);
p kanäle gleichkorreliert (ρ ≥ 0) mit der Ein-Faktor-Darstellung x_i = μ_i + √ρ·f + √(1-ρ)·e_i und Gauß-Hermite über f. Das Orakel (tests/) rechnet dieselben Größen mit
`scipy.stats.multivariate_normal.cdf` (Genz-Verfahren). Zähleinheit: Vektoren bis zum Alarm."""

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

import mkn_cards as K
import mkn_constants as C

_HG_X, _HG_W = np.polynomial.hermite_e.hermegauss(240)
_HG_W = _HG_W / math.sqrt(2.0 * math.pi)


def cov_equi(p: int, rho: float) -> np.ndarray:
    return (1.0 - rho) * np.eye(p) + rho * np.ones((p, p))


def cov_p2(rho: float) -> np.ndarray:
    return np.array([[1.0, rho], [rho, 1.0]])


def noalarm_p2(z: float, mu1: float, mu2: float, rho: float) -> float:
    """P(|x1| <= z, |x2| <= z) für (x1, x2) ~ N(μ, [[1, ρ], [ρ, 1]]): Integral über x1 (adaptive Quadratur, jedes |ρ| < 1)."""
    s = math.sqrt(1.0 - rho * rho)

    def f(x):
        c = mu2 + rho * (x - mu1)
        return norm.pdf(x - mu1) * (ndtr((z - c) / s) - ndtr((-z - c) / s))
    pts = [mu1] if abs(mu1) < z else None
    return float(quad(f, -z, z, epsabs=1e-14, epsrel=1e-13, limit=400, points=pts)[0])


def noalarm_equi(z: float, mus, rho: float) -> float:
    """P(alle |x_i| <= z) für gleichkorrelierte Kanäle (ρ ≥ 0) mit Mittelwerten μ: Ein-Faktor-Darstellung, Gauß-Hermite mit 240 Knoten."""
    mus = np.asarray(mus, float)
    s = math.sqrt(1.0 - rho)
    sr = math.sqrt(rho)
    a = (z - mus[None, :] - sr * _HG_X[:, None]) / s
    b = (-z - mus[None, :] - sr * _HG_X[:, None]) / s
    return float((_HG_W * (ndtr(a) - ndtr(b)).prod(axis=1)).sum())


def noalarm(model: str, z: float, mu, rho: float) -> float:
    return noalarm_p2(z, float(mu[0]), float(mu[1]), rho) if model == C.KORR_P2 else noalarm_equi(z, mu, rho)


def dim(model: str) -> int:
    return 2 if model == C.KORR_P2 else 5


def sigma(model: str, rho: float) -> np.ndarray:
    return cov_p2(rho) if model == C.KORR_P2 else cov_equi(5, rho)


def arl0_sidak(model: str, rho: float) -> float:
    """ARL₀ der getrennten Karten mit der Šidák-Schranke (setzt Unabhängigkeit voraus) bei Korrelation ρ."""
    p = dim(model)
    z = K.z_sidak(p)[0]
    return 1.0 / (1.0 - noalarm(model, z, np.zeros(p), rho))


@lru_cache(maxsize=None)
def z_calibrated(model: str, rho: float) -> float:
    """Schranke z, bei der die getrennten Karten bei Korrelation ρ die ARL₀ 370,398 haben (neu geeicht)."""
    p = dim(model)
    def f(z):
        return 1.0 / (1.0 - noalarm(model, z, np.zeros(p), rho)) - C.GAMMA0

    return float(brentq(f, 2.0, 5.0, xtol=1e-12))


def direction(model: str, name: str) -> np.ndarray:
    """Sprungrichtung v (μ = d · v): entlang (alle Kanäle gleichsinnig), gegen (die ersten zwei Kanäle gegeneinander), ein Kanal, schief."""
    p = dim(model)
    v = np.zeros(p)
    if name == C.RICHTUNG_ENTLANG:
        v[:] = 1.0
    elif name == C.RICHTUNG_GEGEN:
        v[0], v[1] = 1.0, -1.0
    elif name == C.RICHTUNG_EIN:
        v[0] = 1.0
    elif name == C.RICHTUNG_SCHIEF:
        v[0], v[1] = C.SCHIEF_V
    else:
        raise ValueError(f"unbekannte Richtung: {name!r}")
    return v


def noncentrality(model: str, rho: float, mu) -> float:
    S = sigma(model, rho)
    mu = np.asarray(mu, float)
    return float(mu @ np.linalg.solve(S, mu))


def jump_vector(model: str, rho: float, name: str, d: float, equal_lambda: bool) -> np.ndarray:
    """Sprungvektor μ = d · v; bei „gleiche Nichtzentralität“ so skaliert, dass μ'Σ⁻¹μ = 4."""
    v = direction(model, name)
    if equal_lambda:
        return v * math.sqrt(C.LAMBDA_GLEICH / noncentrality(model, rho, v))
    return v * d


def arl_row(model: str, rho: float, mu) -> dict:
    """ARL₁ je Karte bei Sprung μ: T² (nur λ), getrennt neu auf 370,398 geeicht, getrennt mit Šidák-Schranke; dazu ARL₀ der Šidák-Karte."""
    p = dim(model)
    mu = np.asarray(mu, float)
    lam = noncentrality(model, rho, mu)
    z_n = z_calibrated(model, rho)
    z_s = K.z_sidak(p)[0]
    return dict(lam=lam, t2=K.arl_t2(p, lam), sep_new=1.0 / (1.0 - noalarm(model, z_n, mu, rho)), sep_sidak=1.0 / (1.0 - noalarm(model, z_s, mu, rho)),
                z_new=z_n, z_sidak=z_s, arl0_sidak=arl0_sidak(model, rho))


def ellipse_points(rho: float, ucl: float, n: int = 200):
    """Rand der T²-Ellipse x'Σ⁻¹x = ucl (p = 2) in der (x1, x2)-Ebene: x = √ucl · L u, u auf dem Einheitskreis, L Cholesky-Faktor."""
    L = np.linalg.cholesky(cov_p2(rho))
    th = np.linspace(0.0, 2.0 * math.pi, n)
    pts = math.sqrt(ucl) * (L @ np.vstack([np.cos(th), np.sin(th)]))
    return pts[0], pts[1]
