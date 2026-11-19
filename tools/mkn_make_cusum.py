"""Erzeugt mkn_cusum_frozen.json: Schwellen und ARL der Karten mit Gedächtnis je Kanal (CUSUM, EWMA) und der MEWMA bei gemeinsamer ARL₀ 370,398.

    python tools/mkn_make_cusum.py --voll     # eicht und rechnet alles neu (einige Minuten) und schreibt die Datei
    python tools/mkn_make_cusum.py --check    # rechnet ARL₀ an den gespeicherten Schwellen und einige ARL₁ neu und vergleicht (rund eine Minute), schreibt nichts
    python tools/mkn_make_cusum.py            # zeigt die gespeicherten Werte

Quellen: zweiseitiger CUSUM (k = 0,5) als 2D-Markov-Kette, EWMA (λ = 0,1) als 1D-Kette, beide über das Minimum unabhängiger Kanal-Lauflängen (`mkn_memory.py`),
Richardson in 1/m² (CUSUM m = 32 und 48, EWMA m_up = 80 und 160); MEWMA-Schwelle aus der radialen Kette, MEWMA-ARL₁ per Simulation (SplitMix64, 20 000 Läufe).
Zähleinheit Beobachtungsvektoren bis zum ersten Alarm, zero-state; steady-state für p = 1 und 5."""

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import scipy  # noqa: E402

import mkn_constants as C  # noqa: E402
import mkn_memory as M  # noqa: E402
import mkn_reference as REF  # noqa: E402
import mkn_scenario as SC  # noqa: E402

OUT = ROOT / "mkn_cusum_frozen.json"
MEWMA_R = 20000
MEWMA_SEED0 = 61000
ENERGIE_PAAR = ((1, 2.0), (4, 1.0), (1, 1.0), (5, 1.0))     # p = 5: (m, d): gleiche Nichtzentralität λ = 4 und Vergleichsfälle


def cusum_cases(p: int) -> list[tuple[int, float]]:
    """Gefrorene Fälle (m, d): ein Kanal bei allen d; für p = 5 und 10 zusätzlich die Fälle der Verdünnung (gleiche Höhe d = 1, gleiche Energie d = 2/√m)."""
    cases = [(1, d) for d in C.D_OPTIONS]
    if p in C.VD_M_OPTIONS:
        for m in C.VD_M_OPTIONS[p]:
            for d in (1.0, 2.0 / math.sqrt(m)):
                if (m, d) not in cases:
                    cases.append((m, d))
    return cases


def build() -> dict:
    t0 = time.time()
    out = {"meta": {}, "cusum": {}, "ewma": {}, "mewma": {}, "steady": {}}
    for p in C.P_OPTIONS:
        h = M.cusum_calibrate(p)
        cases = cusum_cases(p)
        arl1, S0 = M.cusum_table(h, p, cases)
        out["cusum"][str(p)] = dict(
            h=h, arl0=M.cusum_arl0(h, p), grid_diff=M.cusum_grid_diff(h, p, 1, 1.0),
            arl1={REF.case_key(m, d): v for (m, d), v in arl1.items()}, quantile0=M.joint_quantiles(S0, p))
        print(f"CUSUM p = {p}: h* = {h:.4f} ({time.time() - t0:.0f} s)", flush=True)
    for p in (1, 5):
        h = out["cusum"][str(p)]["h"]
        out["steady"][str(p)] = {f"{d:.1f}": list(v) for d, v in M.cusum_steady(h, p, (0.5, 1.0, 2.0)).items()}
        print(f"steady-state p = {p} ({time.time() - t0:.0f} s)", flush=True)
    for p in C.P_OPTIONS:
        H = M.ewma_calibrate(C.EWMA_LAMBDA, p)
        a0, a1 = M.ewma_arl_pair(C.EWMA_LAMBDA, H, p, C.D_OPTIONS)
        a0k, a1k = M.ewma_arl_pair(C.EWMA_LAMBDA, H, p, (1.0,), mge=(160, 320))
        out["ewma"][str(p)] = dict(H=H, arl0=a0, arl0_kontrolle=a0k, arl1={f"{d:.1f}": v for d, v in a1.items()}, arl1_kontrolle_d1=a1k[1.0])
        print(f"EWMA p = {p}: H* = {H:.4f} ({time.time() - t0:.0f} s)", flush=True)
    for i, p in enumerate(C.P_OPTIONS):
        Hc = M.mewma_calibrate(p)
        arl0 = {str(n): M.chain_arl(M.mewma_chain(p, C.EWMA_LAMBDA, Hc, n)) for n in (100, 200, 400, 800)}
        sims = {}
        for j, d in enumerate(C.D_OPTIONS):
            shift = np.zeros(p)
            shift[0] = d
            rl, cens = SC.mewma_sim(MEWMA_R, p, shift, Hc, MEWMA_SEED0 + 10 * i + j)
            assert cens == 0
            sims[f"{d:.1f}"] = [float(rl.mean()), float(rl.std(ddof=1) / math.sqrt(MEWMA_R)), MEWMA_R]
        out["mewma"][str(p)] = dict(H=Hc, arl0=arl0, arl1_sim=sims)
        print(f"MEWMA p = {p}: H* = {Hc:.4f} ({time.time() - t0:.0f} s)", flush=True)
    paar = {}
    Hc5 = out["mewma"]["5"]["H"]
    for j, (m, d) in enumerate(ENERGIE_PAAR):
        shift = np.zeros(5)
        shift[:m] = d
        rl, cens = SC.mewma_sim(MEWMA_R, 5, shift, Hc5, MEWMA_SEED0 + 100 + j)
        paar[REF.case_key(m, d)] = [float(rl.mean()), float(rl.std(ddof=1) / math.sqrt(MEWMA_R)), MEWMA_R]
    out["mewma_lambda_paar"] = paar
    out["meta"] = dict(
        quelle="tools/mkn_make_cusum.py --voll", stand=C.FROZEN_STAND, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        massstab="gemeinsame ARL0 = 370,398 (Vektoren bis zum Fehlalarm); zero-state, wo nicht steady-state steht; Zaehleinheit Beobachtungsvektoren bis zum ersten Alarm",
        cusum="zweiseitig, k = 0,5, 2D-Gitter m = 32 und 48, Richardson in 1/m^2; Schwelle h je Kanal",
        ewma="zweiseitig, lambda = 0,1, Gitter m_up = 80 und 160, Richardson in 1/m^2; Grenze H je Kanal",
        mewma=f"lambda = 0,1; H aus der radialen Kette (n = 400 Ringe, Pruefung 100/200/400/800); ARL1 Simulation, {MEWMA_R} Laeufe, SplitMix64, Zufalls-Seeds ab {MEWMA_SEED0}",
        steady="Gitter m = 24 und 32, Richardson; quasistationaer",
        laufzeit_sekunden=round(time.time() - t0))
    return out


def check(tol_rel: float = 2e-3) -> int:
    """Rechnet ARL₀ an den gespeicherten Schwellen neu (CUSUM, EWMA, MEWMA) und je einen ARL₁; Abweichung über tol_rel meldet einen Fehler."""
    T = REF.tables()
    bad = 0

    def cmp(name, ist, soll, tol=tol_rel):
        nonlocal bad
        ok = abs(ist - soll) <= tol * abs(soll)
        bad += (not ok)
        print(f"  [{'OK' if ok else 'FEHLER'}] {name}: neu {ist:.4f}, Soll {soll:.4f}", flush=True)
    for p in C.P_OPTIONS:
        h = REF.cusum_h(p)
        cmp(f"CUSUM p = {p}: ARL0 an h* = {h:.4f}", M.cusum_arl0(h, p), C.GAMMA0, 1e-3)
        cmp(f"CUSUM p = {p}: in der Datei gespeicherte ARL0", REF.cusum_arl0(p), C.GAMMA0, 1e-3)
        S0 = M.surv_iter(M.cusum_two_R(h, 0.0, 32), 0, 60000, p_stop=p)
        S1 = M.surv_iter(M.cusum_two_R(h, 1.0, 32), 0, len(S0) - 1)
        cmp(f"CUSUM p = {p}: ARL1 (m = 1, d = 1), Gitter 32 gegen die gespeicherte Richardson-Zahl", M.prod_arl(S0, S1, p, 1), REF.cusum_arl1(p, 1, 1.0), 2e-3)
    for p in C.P_OPTIONS:
        a0, a1 = M.ewma_arl_pair(C.EWMA_LAMBDA, REF.ewma_H(p), p, (1.0,))
        cmp(f"EWMA p = {p}: ARL0 an H* = {REF.ewma_H(p):.4f}", a0, C.GAMMA0, 1e-3)
        cmp(f"EWMA p = {p}: ARL1 (d = 1)", a1[1.0], REF.ewma_arl1(p, 1.0), 1e-3)
        cmp(f"MEWMA p = {p}: ARL0 (n = 400)", M.chain_arl(M.mewma_chain(p, C.EWMA_LAMBDA, REF.mewma_H(p), 400)), C.GAMMA0, 2e-3)
    print(f"{'ALLES OK' if not bad else str(bad) + ' FEHLER'} (Stand {T['meta']['stand']})")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--voll" in sys.argv:
        data = build()
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"geschrieben: {OUT} ({data['meta']['laufzeit_sekunden']} s)")
    elif "--check" in sys.argv:
        sys.exit(check())
    else:
        print(json.dumps(REF.meta(), ensure_ascii=False, indent=1))
        for p in C.P_OPTIONS:
            print(f"p = {p}: CUSUM h* {REF.cusum_h(p):.4f}, EWMA H* {REF.ewma_H(p):.4f}, MEWMA H* {REF.mewma_H(p):.4f}")
