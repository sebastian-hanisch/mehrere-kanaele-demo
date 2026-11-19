"""Szenario: ein Verlauf aus p unabhängigen Standardnormal-Kanälen mit Sprung in m Kanälen, und die MEWMA-Simulation.

Alle Zufallszahlen kommen aus dem SplitMix64 der Demo (`mkn_normal.Stream`, bitgleich mit `mkn_rng.SplitMix64`); gleicher Zufalls-Seed, gleiche Zahlen, auf jeder
Plattform (Normalzahlen auf etwa 1e-15). ν = Zahl der Vektoren vor der Änderung: die Vektoren mit Nummer t > ν haben im ersten Kanal bis zum m-ten Kanal den
Mittelwert d. Bei ν = 0 gilt „Änderung ab Vektor 1“ (Nullzustand, das Maß der exakten ARL₁)."""

import math
import time
from dataclasses import dataclass, field

import numpy as np

import mkn_cards as K
import mkn_constants as C
import mkn_reference as REF
from mkn_normal import Stream


@dataclass
class Trajectory:
    p: int
    m: int
    d: float
    nu: int
    seed: int
    karte: str
    stat: np.ndarray            # Statistik je Vektor (T²; max |x|; max(S⁺, S⁻) über Kanäle; max |Z| über Kanäle)
    threshold: float
    alarm_t: int | None         # Nummer des ersten Vektors mit Alarm (1-basiert), sonst None
    alarm_channel: int | None   # 1-basiert; nur getrennte Karten nennen den Kanal, T² nicht
    x: np.ndarray = field(repr=False, default=None)

    @property
    def false_alarm(self) -> bool:
        """Alarm vor der Änderung: Vektor t <= ν."""
        return self.alarm_t is not None and self.alarm_t <= self.nu

    @property
    def delay(self) -> int | None:
        """Vektoren nach der Änderung bis zum Alarm (Vektor ν + 1 ist der erste geänderte); None, wenn kein Alarm nach der Änderung."""
        if self.alarm_t is None or self.alarm_t <= self.nu:
            return None
        return self.alarm_t - self.nu


def make_data(p: int, m: int, d: float, nu: int, seed: int, n: int = C.N_VERLAUF) -> np.ndarray:
    """n Vektoren mit p Kanälen, ab Vektor ν + 1 sind die ersten m Kanäle um d verschoben."""
    if not (1 <= m <= p):
        raise ValueError("m muss zwischen 1 und p liegen")
    x = Stream(seed).normal((n, p))
    x[nu:, :m] += d
    return x


def _first_alarm(over: np.ndarray):
    idx = np.flatnonzero(over)
    return (int(idx[0]) + 1) if idx.size else None


def run_trajectory(p: int, m: int, d: float, nu: int, seed: int, karte: str, n: int = C.N_VERLAUF) -> Trajectory:
    """Statistik der gewählten Karte über t, Schwelle und erster Alarm; Schwellen sind auf die gemeinsame ARL₀ 370,398 geeicht."""
    x = make_data(p, m, d, nu, seed, n)
    if karte == C.KARTE_T2:
        stat, thr = (x * x).sum(axis=1), K.ucl_t2(p)
        t = _first_alarm(stat > thr)
        ch = None
    elif karte == C.KARTE_SHEW:
        a = np.abs(x)
        stat, thr = a.max(axis=1), K.z_sidak(p)[0]
        t = _first_alarm(stat > thr)
        ch = int(np.argmax(a[t - 1])) + 1 if t else None
    elif karte == C.KARTE_CUSUM:
        h = REF.cusum_h(p)
        sp_, sn = np.zeros(p), np.zeros(p)
        stat = np.empty(n)
        t = None
        ch = None
        for i in range(n):
            sp_ = np.maximum(0.0, sp_ + x[i] - C.CUSUM_K)
            sn = np.maximum(0.0, sn - x[i] - C.CUSUM_K)
            both = np.maximum(sp_, sn)
            stat[i] = both.max()
            if t is None and stat[i] > h:
                t = i + 1
                ch = int(np.argmax(both)) + 1
        thr = h
    elif karte == C.KARTE_EWMA:
        H = REF.ewma_H(p)
        z = np.zeros(p)
        stat = np.empty(n)
        t = None
        ch = None
        for i in range(n):
            z = C.EWMA_LAMBDA * x[i] + (1.0 - C.EWMA_LAMBDA) * z
            a = np.abs(z)
            stat[i] = a.max()
            if t is None and stat[i] > H:
                t = i + 1
                ch = int(np.argmax(a)) + 1
        thr = H
    else:
        raise ValueError(f"unbekannte Karte: {karte!r}")
    return Trajectory(p, m, d, nu, seed, karte, np.asarray(stat, float), float(thr), t, ch, x)


# ------------------------------------------------------------------------------------------------------------------ MEWMA-Simulation
def mewma_sim(R: int, p: int, shift, Hc: float, seed: int, lam: float = C.EWMA_LAMBDA, tmax: int = 100000):
    """Lauflängen (Vektoren bis zum Alarm, Start Z = 0) von R unabhängigen Läufen der MEWMA mit Verschiebungsvektor `shift`; Rückgabe (Lauflängen, Zahl nicht alarmierter Läufe nach tmax).
    Statistik T = (2-λ)/λ · |Z|², Alarm bei T > Hc."""
    shift = np.asarray(shift, float)
    st = Stream(seed)
    Z = np.zeros((R, p))
    rl = np.full(R, tmax + 1, dtype=np.int64)
    idx = np.arange(R)
    for t in range(1, tmax + 1):
        Z = lam * (st.normal((idx.size, p)) + shift) + (1.0 - lam) * Z
        hit = (Z * Z).sum(axis=1) * (2.0 - lam) / lam > Hc
        rl[idx[hit]] = t
        idx, Z = idx[~hit], Z[~hit]
        if idx.size == 0:
            break
    return rl, int(idx.size)


@dataclass
class MewmaLive:
    mean: float
    se: float
    R: int
    seconds: float
    censored: int


def mewma_live(p: int, d: float, R: int, seed: int) -> MewmaLive:
    """ARL₁ der MEWMA (ein Kanal springt um d) aus R Live-Läufen (Simulation, Zufalls-Seed `seed`); mit Standardfehler und Rechenzeit."""
    if not (1 <= R <= C.R_MAX):
        raise ValueError(f"R muss zwischen 1 und {C.R_MAX} liegen")
    shift = np.zeros(p)
    shift[0] = d
    t0 = time.perf_counter()
    rl, cens = mewma_sim(R, p, shift, REF.mewma_H(p), seed)
    secs = time.perf_counter() - t0
    return MewmaLive(float(rl.mean()), float(rl.std(ddof=1) / math.sqrt(R)), R, secs, cens)
