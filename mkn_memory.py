"""Karten mit Gedächtnis je Kanal (zweiseitiger CUSUM, EWMA) und MEWMA: Markov-Ketten, Produktweg über unabhängige Kanäle, Eichung.

Bei UNABHÄNGIGEN Kanälen ist die Lauflänge der gemeinsamen Karte das Minimum der Kanal-Lauflängen: P(RL > n) = Produkt der Überlebensfunktionen der Kanäle,
ARL = Σ_n Produkt. Die Überlebensfunktion eines Kanals kommt aus einer Markov-Kette (Brook/Evans-Gitter; für den zweiseitigen CUSUM ein 2D-Gitter in (S⁺, S⁻));
der Gitterfehler wird per Richardson-Extrapolation in 1/m² ausgeglichen (Gitter m = 32 und 48 beim CUSUM, m_up = 80 und 160 beim EWMA). Das sind Gitterwerte, nicht
geschlossene Formeln: auf etwa 0,4 % genau (die README nennt die Messung gegen Simulation). Maß: zero-state (Start im Nullzustand, Änderung ab Vektor 1);
`cusum_steady` rechnet das steady-state-Maß (quasistationär, bedingt auf „bis dahin kein Alarm“). Lorden- und Pollak-Maß sind nicht gerechnet.

Quellen im Code: `cusum_two_R` ist `cusum2_grid` aus der Vormessung `ae_v1_lib.py` (ohne die Lösung des Gleichungssystems), `mewma_chain` die radiale Kette aus
`ae_v2_annahmen.py` (Abschnitt 5.3), beide abgeschrieben, nicht importiert. Die Rechenzeit der Gitter ist so, dass die App sie nicht live rechnet: sie liest
`mkn_cusum_frozen.json` (erzeugt von `tools/mkn_make_cusum.py`)."""

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import ncx2

import mkn_constants as C

MG = (32, 48)          # Gitter des CUSUM (Richardson)
MGE = (80, 160)        # Gitter des EWMA (Richardson)
MG_STEADY = (24, 32)   # Gitter für das steady-state-Maß


# ------------------------------------------------------------------------------------------------------------------ Ketten
def cusum_two_R(h: float, delta: float, m: int, k: float = C.CUSUM_K):
    """Übergangsmatrix (lebende Zustände) des zweiseitigen CUSUM (S⁺, S⁻) auf m×m Zellen, Kanal ~ N(delta, 1); Startzustand (0, 0) hat den Index 0.
    Beide Statistiken hängen vom selben x ab: je Zustand wird die x-Achse an allen Zellgrenzen zerlegt."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    q = np.arange(1, m + 1)
    rows, cols, data = [], [], []
    for i in range(m):
        tplus = (q - 0.5) * w - v[i] + k          # S⁺' >= (q-1/2) w  <=>  x >= tplus_q (q = m: Alarm)
        for j in range(m):
            s_q = v[j] - k - (q - 0.5) * w        # S⁻' >= (q-1/2) w  <=>  x <= s_q (fallend in q)
            B = np.sort(np.concatenate([tplus, s_q]))
            lo_e = np.concatenate([[-np.inf], B])
            hi_e = np.concatenate([B, [np.inf]])
            mid = np.where(np.isfinite(lo_e) & np.isfinite(hi_e), 0.5 * (lo_e + hi_e), np.where(np.isfinite(lo_e), lo_e + 1.0, hi_e - 1.0))
            pr = ndtr(hi_e - delta) - ndtr(lo_e - delta)
            ip = np.searchsorted(tplus, mid, side="right")
            jp = m - np.searchsorted(s_q[::-1], mid, side="left")
            ok = (ip < m) & (jp < m) & (pr > 0)
            rows.append(np.full(int(ok.sum()), i * m + j))
            cols.append(ip[ok] * m + jp[ok])
            data.append(pr[ok])
    return sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(m * m, m * m))


def ewma_two_R(lam: float, H: float, delta: float, m_up: int):
    """Übergangsmatrix des zweiseitigen EWMA (Grenzen ±H, Zellmitten j·w, j = -m_up ... m_up) und Index des Startzustands (Mitte 0)."""
    w = H / (m_up + 0.5)
    cent = w * np.arange(-m_up, m_up + 1)
    edges = np.concatenate([[-H], (cent[:-1] + cent[1:]) / 2.0, [H]])
    cdf = ndtr((edges[None, :] - (1.0 - lam) * cent[:, None]) / lam - delta)
    return np.diff(cdf, axis=1), m_up


def mewma_chain(p: int, lam: float, Hc: float, n: int):
    """Radiale Markov-Kette für |Z|² der MEWMA (Σ_Z asymptotisch λ/(2-λ)·I): ARL₀ exakt (Mittelwert 0); Zustand 0 = Start, Zustände 1..n = Ringe bis zur Grenze."""
    rc = Hc * lam / (2.0 - lam)
    edges = np.linspace(0.0, rc, n + 1)
    reps = np.concatenate([[0.0], 0.5 * (edges[:-1] + edges[1:])])
    Q = np.zeros((n + 1, n + 1))
    for i, r in enumerate(reps):
        F = ncx2.cdf(edges / lam ** 2, p, (1.0 - lam) ** 2 * r / lam ** 2)
        Q[i, 1:] = np.diff(F)
    return Q


def chain_arl(Q) -> float:
    L = np.linalg.solve(np.eye(Q.shape[0]) - Q, np.ones(Q.shape[0]))
    return float(L[0])


# ------------------------------------------------------------------------------------------------------------------ Überlebensfunktion, Produktweg
def surv_iter(R, start: int, N: int, p_stop: int | None = None, tol: float = 1e-13) -> np.ndarray:
    """P(RL > n), n = 0..N, eines Kanals ab Zustand `start`; mit p_stop: abbrechen, wenn S^p_stop < tol (die gemeinsame Überlebensfunktion ist dann null)."""
    v = np.ones(R.shape[0])
    out = [1.0]
    for _ in range(N):
        v = R @ v
        out.append(float(v[start]))
        if p_stop is not None and out[-1] ** p_stop < tol:
            break
        if p_stop is None and out[-1] < 1e-18:
            break
    return np.array(out)


def prod_arl(S0: np.ndarray, Sd: np.ndarray, p: int, m: int) -> float:
    """ARL der gemeinsamen Karte: Σ_n S0(n)^(p-m) · Sd(n)^m (m Kanäle geändert, p-m ungeändert), Teilsumme bis zum kürzeren Feld."""
    n = min(len(S0), len(Sd))
    return float((S0[:n] ** (p - m) * Sd[:n] ** m).sum())


def richardson(a1: float, m1: int, a2: float, m2: int) -> float:
    """Extrapolation in 1/m²: der Gitterfehler fällt wie 1/m²."""
    return (m2 ** 2 * a2 - m1 ** 2 * a1) / (m2 ** 2 - m1 ** 2)


def joint_quantiles(S0: np.ndarray, p: int) -> dict:
    """Mittel, Median, 5-%-, 95-%-Quantil und P(RL <= 100) der gemeinsamen Lauflänge ohne Änderung aus der Überlebensfunktion eines Kanals."""
    Sj = S0 ** p
    cdf = 1.0 - Sj
    return dict(mean=float(Sj.sum()), median=int(np.searchsorted(cdf, 0.5)), q05=int(np.searchsorted(cdf, 0.05)), q95=int(np.searchsorted(cdf, 0.95)), p100=float(cdf[100]))


# ------------------------------------------------------------------------------------------------------------------ CUSUM
def cusum_arl0(h: float, p: int, mg=MG) -> float:
    vals = [float((surv_iter(cusum_two_R(h, 0.0, g), 0, 60000, p_stop=p) ** p).sum()) for g in mg]
    return richardson(vals[0], mg[0], vals[1], mg[1])


def cusum_calibrate(p: int, mg=MG) -> float:
    """Schwelle h* je Kanal, bei der die gemeinsame ARL₀ von p zweiseitigen CUSUM 370,398 ist (Sekantenverfahren auf log ARL₀)."""
    def f(h):
        return math.log(cusum_arl0(h, p, mg)) - math.log(C.GAMMA0)

    h0 = 4.77 + math.log(p)
    h1 = h0 + 0.05
    f0, f1 = f(h0), f(h1)
    for _ in range(8):
        h2 = h1 - f1 * (h1 - h0) / (f1 - f0)
        h0, f0 = h1, f1
        h1, f1 = h2, f(h2)
        if abs(f1) < 3e-5:
            break
    return h1


def cusum_table(h: float, p: int, cases, mg=MG):
    """ARL₁ für Fälle (m, d) (m von p Kanälen springen um d), Richardson über mg. Rückgabe: ({(m, d): ARL₁}, Überlebensfunktion S0 des Gitters mg[-1])."""
    raw = {c: [] for c in cases}
    last = None
    for g in mg:
        S0 = surv_iter(cusum_two_R(h, 0.0, g), 0, 60000, p_stop=p)
        last = S0
        by_d = {}
        for (m, d) in cases:
            if d not in by_d:
                by_d[d] = surv_iter(cusum_two_R(h, d, g), 0, len(S0) - 1)
            raw[(m, d)].append(prod_arl(S0, by_d[d], p, m))
    return {c: richardson(v[0], mg[0], v[1], mg[1]) for c, v in raw.items()}, last


def cusum_grid_diff(h: float, p: int, m: int, d: float, mg=MG) -> float:
    """Unterschied der beiden Gitter (vor der Extrapolation) bei ARL₁: Maß für den Gitterfehler."""
    vals = []
    for g in mg:
        S0 = surv_iter(cusum_two_R(h, 0.0, g), 0, 60000, p_stop=p)
        vals.append(prod_arl(S0, surv_iter(cusum_two_R(h, d, g), 0, len(S0) - 1), p, m))
    return abs(vals[0] - vals[1])


def _qsd_left(R):
    """Quasistationäre Verteilung (linker Eigenvektor zum größten Eigenwert der Kette ohne Alarm) und Eigenwert."""
    ev, V = spl.eigs(R.T.tocsc(), k=1, which="LR")
    q = np.abs(V[:, 0].real)
    return q / q.sum(), float(ev[0].real)


def cusum_steady(h: float, p: int, deltas, mg=MG_STEADY) -> dict:
    """zero-state und steady-state ARL₁ (ein Kanal springt) für jedes d in deltas, Richardson über mg. Bei unabhängigen Kanälen ist die gemeinsame
    quasistationäre Verteilung das Produkt der Kanalverteilungen: ARL_ss = Σ_n [π₀' R_d^n 1] · λ₀^(n (p-1))."""
    zero = {d: [] for d in deltas}
    steady = {d: [] for d in deltas}
    for g in mg:
        R0 = cusum_two_R(h, 0.0, g)
        pi0, lam0 = _qsd_left(R0)
        S0 = surv_iter(R0, 0, 60000, p_stop=p)
        for d in deltas:
            Rd = cusum_two_R(h, d, g)
            v = np.ones(Rd.shape[0])
            a_ss = float(pi0 @ v)
            a_z = float(v[0]) * S0[0] ** (p - 1)
            n = 0
            while True:
                n += 1
                v = Rd @ v
                t_ss = float(pi0 @ v) * lam0 ** (n * (p - 1))
                a_ss += t_ss
                if n < len(S0):
                    a_z += float(v[0]) * S0[n] ** (p - 1)
                if (t_ss < 1e-15 and n > 5) or n > 60000:
                    break
            steady[d].append(a_ss)
            zero[d].append(a_z)
    return {d: (richardson(zero[d][0], mg[0], zero[d][1], mg[1]), richardson(steady[d][0], mg[0], steady[d][1], mg[1])) for d in deltas}


# ------------------------------------------------------------------------------------------------------------------ EWMA
def ewma_sigz(lam: float) -> float:
    return math.sqrt(lam / (2.0 - lam))


def ewma_arl_pair(lam: float, H: float, p: int, deltas, mge=MGE):
    """gemeinsame ARL₀ (p Kanäle) und ARL₁ (ein Kanal um d) des zweiseitigen EWMA, Richardson über mge."""
    a0 = []
    a1 = {d: [] for d in deltas}
    for g in mge:
        R, st = ewma_two_R(lam, H, 0.0, g)
        S0 = surv_iter(R, st, 80000, p_stop=p)
        a0.append(float((S0 ** p).sum()))
        for d in deltas:
            R1, st1 = ewma_two_R(lam, H, d, g)
            a1[d].append(prod_arl(S0, surv_iter(R1, st1, len(S0) - 1), p, 1))
    return richardson(a0[0], mge[0], a0[1], mge[1]), {d: richardson(v[0], mge[0], v[1], mge[1]) for d, v in a1.items()}


def ewma_calibrate(lam: float, p: int) -> float:
    def f(H):
        return math.log(ewma_arl_pair(lam, H, p, ())[0]) - math.log(C.GAMMA0)

    sz = ewma_sigz(lam)
    return float(brentq(f, 2.0 * sz, 6.5 * sz, xtol=2e-5))


# ------------------------------------------------------------------------------------------------------------------ MEWMA
def mewma_calibrate(p: int, lam: float = C.EWMA_LAMBDA, n: int = 400) -> float:
    """MEWMA-Schwelle H, bei der die ARL₀ der radialen Kette mit n Ringen 370,398 ist (n = 400; bei p = 1 liegt n = 150 um 0,7 % daneben, die Kette konvergiert dort langsam)."""
    return float(brentq(lambda Hc: chain_arl(mewma_chain(p, lam, Hc, n)) - C.GAMMA0, 2.0, 60.0, xtol=1e-6))
