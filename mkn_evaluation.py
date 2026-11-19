"""Einstellungen, Zahlenformat, Tabellenzeilen und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Maß in jedem Satz: Beobachtungsvektoren bis zum ersten Alarm bei gemeinsamer ARL₀ = 370,398 (nicht dem gesetzten 370), unabhängige standardnormale Kanäle (außer in
Experiment 3), bekannte Parameter, Mittelwertsprung ab Vektor 1 (Nullzustand). Karten ohne Gedächtnis: zero-state, Lorden und Pollak sind dieselbe Zahl; Karten mit
Gedächtnis: zero-state, wo nicht ausdrücklich steady-state steht; Lorden- und Pollak-Maß sind nicht gemessen."""

from dataclasses import dataclass

import mkn_cards as K
import mkn_constants as C
import mkn_corr as R
import mkn_reference as REF


@dataclass(frozen=True)
class Settings:
    p: int = C.DEFAULT_P
    d: float = C.DEFAULT_D
    m: int = C.DEFAULT_M
    karte: str = C.DEFAULT_KARTE
    nu: int = C.DEFAULT_NU
    seed: int = C.DEFAULT_SEED
    korr_modell: str = C.DEFAULT_KORR_MODELL
    rho: float = C.DEFAULT_RHO
    richtung: str = C.DEFAULT_RICHTUNG
    sprung: float = C.DEFAULT_SPRUNG
    gleich: bool = C.DEFAULT_GLEICH
    vd_p: int = C.DEFAULT_VD_P
    vd_m: int = C.DEFAULT_VD_M
    vd_vergleich: str = C.DEFAULT_VD_VERGLEICH
    vd_summe: bool = C.DEFAULT_VD_SUMME
    r: int = C.DEFAULT_R

    @property
    def m_eff(self) -> int:
        """Zahl der springenden Kanäle, auf p begrenzt."""
        return min(self.m, self.p)

    @property
    def vd_m_eff(self) -> int:
        return min(self.vd_m, self.vd_p)


# ------------------------------------------------------------------------------------------------------------------ Zahlenformat (Dezimalkomma)
def de(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def de_s(x: float, nd: int = 1) -> str:
    """Mit Vorzeichen (ASCII-Minus, damit st.metric die Farbe des Deltas erkennt): mein Wert minus Referenz."""
    return ("+" if x >= 0 else "-") + de(abs(x), nd)


def pct(share: float, nd: int = 1) -> str:
    return de(100.0 * share, nd) + " %"


# ------------------------------------------------------------------------------------------------------------------ Tabellenzeilen
def arl1_of_card(karte: str, p: int, m: int, d: float):
    """ARL₁ (zero-state) der Karte bei m von p springenden Kanälen um d, oder None, wenn der Fall für die Karte nicht gefroren ist (CUSUM, EWMA)."""
    if karte == C.KARTE_T2:
        return K.arl_t2(p, m * d * d)
    if karte == C.KARTE_SHEW:
        return K.arl_sep(p, m, d)
    if karte == C.KARTE_CUSUM:
        return REF.cusum_arl1(p, m, d)
    if karte == C.KARTE_EWMA:
        return REF.ewma_arl1(p, d) if m == 1 else None
    raise ValueError(karte)


def arl0_of_card(karte: str, p: int) -> float:
    """ARL₀ der Karte bei der gemeinsamen Eichung (Shewhart-Typ exakt 370,398, Karten mit Gedächtnis Gitter und Richardson)."""
    if karte == C.KARTE_CUSUM:
        return REF.cusum_arl0(p)
    if karte == C.KARTE_EWMA:
        return REF.ewma_arl0(p)
    return C.GAMMA0


def run_length_summary(karte: str, p: int, m: int, d: float):
    """Mittel, Median, 5-%-, 95-%-Quantil, P(Alarm bis Beobachtung 100) der Lauflänge: Shewhart-Typ exakt geometrisch, CUSUM ohne Änderung aus dem Gitter (gefroren); sonst None."""
    if karte == C.KARTE_T2:
        return K.geo_summary(K.pa_t2(p, m * d * d))
    if karte == C.KARTE_SHEW:
        return K.geo_summary(K.pa_sep(p, m, d))
    return None


def null_summary(karte: str, p: int):
    """Verteilung der Lauflänge OHNE Änderung (ARL₀ 370,398): Shewhart-Typ geometrisch (jedes p gleich), CUSUM aus dem Gitter (gefroren), sonst None."""
    if karte in (C.KARTE_T2, C.KARTE_SHEW):
        return K.geo_summary(C.P_ALARM0)
    if karte == C.KARTE_CUSUM:
        return REF.cusum_quantiles0(p)
    return None


def memory_rows(p: int, d: float) -> list[dict]:
    """Tabelle „Was bringt das Gedächtnis?“: ARL₁ (ein Kanal springt um d, zero-state) je Karte; steady-state nur für CUSUM bei p = 1, 5 und d = 0,5 / 1 / 2."""
    rows = [dict(karte=C.KARTE_T2, arl1=K.arl_t2(p, d * d), steady=None, art="exakt"),
            dict(karte=C.KARTE_SHEW, arl1=K.arl_sep(p, 1, d), steady=None, art="exakt"),
            dict(karte=C.KARTE_CUSUM, arl1=REF.cusum_arl1(p, 1, d), steady=(REF.cusum_steady(p, d) or (None, None))[1], art="Gitter, auf etwa 0,4 % genau"),
            dict(karte=C.KARTE_EWMA, arl1=REF.ewma_arl1(p, d), steady=None, art="Gitter, auf etwa 0,4 % genau")]
    sim = REF.mewma_arl1_sim(p, d)
    rows.append(dict(karte="MEWMA (Simulation)", arl1=sim[0], steady=None, art=f"Simulation, {sim[2]} Läufe, ± {de(sim[1], 2)}"))
    return rows


def _kanaele(m: int) -> str:
    return "1 Kanal" if m == 1 else f"{m} Kanäle"


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 1
def verdict_exp1(karte: str, p: int, m: int, d: float, nu: int, alarm_t, alarm_channel, delay, arl1, median) -> tuple[str, str]:
    """Der gezeichnete Verlauf ist eine Stichprobe: der Satz nennt den Alarm dieses Verlaufs und daneben die exakte ARL₁ (ν = 0, zero-state) der Karte."""
    arl_txt = "" if arl1 is None else f" Exakt: ARL₁ dieser Karte {de(arl1, 1)} Vektoren" + (f", Median {median}" if median is not None else "") + " (Änderung ab Vektor 1, zero-state)."
    if arl1 is None:
        arl_txt = " Für diese Kombination aus Karte, p, m und d ist keine ARL₁ gefroren (Experiment 5 und 4 nennen die gemessenen Fälle)."
    stichprobe = " Ein Verlauf ist eine Stichprobe: ein früher oder später Alarm widerlegt die ARL₁ nicht."
    kanal = "" if alarm_channel is None else f" (Kanal {alarm_channel})"
    head = f"{karte}, p = {p}, {_kanaele(m)} um {de(d, 1)} σ ab Vektor {nu + 1}: "
    if alarm_t is None:
        return "info", head + f"in diesen {C.N_VERLAUF} Vektoren kein Alarm." + arl_txt + stichprobe
    if alarm_t <= nu:
        return "warning", head + f"Fehlalarm bei Vektor {alarm_t}{kanal}, noch vor der Änderung (gemeinsame ARL₀ 370,4)." + arl_txt + stichprobe
    return "success", head + f"erster Alarm bei Vektor {alarm_t}{kanal}, {delay} Vektor{'' if delay == 1 else 'en'} nach dem Beginn der Änderung." + arl_txt + stichprobe


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 2
def verdict_exp2(d: float, p_lo: int, p_hi: int, t2_lo: float, t2_hi: float, sep_hi: float, cusum_lo, cusum_hi, n_faster: int, n_cells: int) -> tuple[str, str]:
    """Strafe der Kanalzahl: ARL₁ bei einem springenden Kanal der Höhe d von p_lo auf p_hi Kanäle, bei gleicher gemeinsamer ARL₀ 370,4."""
    cu = ""
    if cusum_lo is not None and cusum_hi is not None:
        cu = f", bei getrennten CUSUM nur von {de(cusum_lo, 1)} auf {de(cusum_hi, 1)}"
    return "info", (f"Bei gemeinsamer ARL₀ 370,4 steigt ARL₁ (d = {de(d, 1)}, ein Kanal springt) von p = {p_lo} auf p = {p_hi} bei T² von {de(t2_lo, 1)} auf {de(t2_hi, 1)}, "
                    f"bei getrennten Shewhart-Karten auf {de(sep_hi, 1)}{cu}. T² ist in {n_faster} von {n_cells} gemessenen Zellen (p = 2, 5, 10, 20; d = 0,5 / 1 / 2 / 3) schneller als "
                    "getrennte Shewhart-Karten (alle bei d = 0,5), sonst langsamer, bei d = 1 nur wenig, bei d ≥ 2 und p ≥ 5 deutlich; das gilt für unabhängige Kanäle und einen springenden Kanal.")


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 3
def compare_words(t2: float, sep: float, tol: float = 0.10) -> str:
    """Welche Karte ist schneller: „t2“, „sep“ oder „gleich“ (innerhalb von tol, relativ zur größeren Zahl)."""
    if sep >= t2 * (1 + tol):
        return "t2"
    if t2 >= sep * (1 + tol):
        return "sep"
    return "gleich"


def verdict_exp3(model: str, rho: float, richtung: str, lam: float, t2: float, sep_new: float, sep_sidak: float, arl0_sidak: float) -> tuple[str, str]:
    """Wer schneller ist, folgt aus den beiden ARL₁; „T² gewinnt“ steht nie ohne Richtung und Korrelation. Die Šidák-Zeile nennt die ARL₀ dieser Schranke bei dieser Korrelation."""
    wer = compare_words(t2, sep_new)
    kanaele = "p = 2" if model == C.KORR_P2 else "p = 5 gleichkorreliert"
    head = f"{kanaele}, ρ = {de(rho, 2)}, Sprung „{richtung}“ (λ = {de(lam, 2)}): "
    if wer == "t2":
        body = f"T² braucht {de(t2, 2)} Vektoren, getrennte Karten (auf ARL₀ 370,4 geeicht) {de(sep_new, 2)}: T² ist {de(sep_new / t2, 1)}-mal schneller."
        kind = "success"
    elif wer == "sep":
        body = f"T² braucht {de(t2, 2)} Vektoren, getrennte Karten (auf ARL₀ 370,4 geeicht) nur {de(sep_new, 2)}: getrennte Karten sind {de(t2 / sep_new, 1)}-mal schneller."
        kind = "warning"
    else:
        body = f"T² braucht {de(t2, 2)} Vektoren, getrennte Karten (auf ARL₀ 370,4 geeicht) {de(sep_new, 2)}: beide liegen innerhalb von 10 %."
        kind = "info"
    if abs(arl0_sidak - C.GAMMA0) < 0.05:
        sidak = f" Die Šidák-Schranke (setzt Unabhängigkeit voraus) hat bei dieser Korrelation die ARL₀ 370,4, wie vorgesehen (ARL₁ dann {de(sep_sidak, 2)})."
    else:
        sidak = (f" Die Šidák-Schranke (setzt Unabhängigkeit voraus) hat bei dieser Korrelation die ARL₀ {de(arl0_sidak, 1)} statt 370,4 "
                 f"(ARL₁ dann {de(sep_sidak, 2)}).")
    return kind, head + body + sidak


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 4
def verdict_exp4(p: int, m: int, mode: str, d: float, lam: float, t2: float, sep: float, sm, cusum) -> tuple[str, str]:
    """Verdünnung: wer bei m von p springenden Kanälen am schnellsten ist; die Summenkarte kennt die Richtung („alle gleichsinnig“)."""
    cands = {"T²": t2, "getrennte Shewhart-Karten": sep}
    if sm is not None:
        cands["Summenkarte (Richtung bekannt)"] = sm
    if cusum is not None:
        cands["getrennte CUSUM"] = cusum
    best = min(cands, key=cands.get)
    zeile = ", ".join(f"{k} {de(v, 2)}" for k, v in cands.items())
    cu = "" if cusum is not None else " Für diese Zahl springender Kanäle ist die CUSUM nicht gefroren."
    return "info", (f"p = {p}, {_kanaele(m)} {'springt' if m == 1 else 'springen'} um je {de(d, 2)} σ (Vergleich: {mode}; λ = {de(lam, 2)}): ARL₁ {zeile}; am schnellsten ist {best}. "
                    f"T² hängt nur an λ, getrennte Karten und Summenkarte nicht.{cu} Die Summe von CUSUMs ist nicht gemessen.")


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 5
def verdict_exp5(p: int, d: float, t2: float, sep: float, cusum, ewma, mewma, mewma_se: float, steady) -> tuple[str, str]:
    """Was bringt das Gedächtnis: ARL₁ bei einem springenden Kanal (zero-state), die Karten ohne Gedächtnis gegen die mit Gedächtnis."""
    st = "" if steady is None else f" (steady-state {de(steady, 2)})"
    ohne = f"T² {de(t2, 2)}, getrennte Shewhart-Karten {de(sep, 2)}"
    mit = f"CUSUM {de(cusum, 2)}{st}, EWMA {de(ewma, 2)}, MEWMA {de(mewma, 2)} ± {de(mewma_se, 2)} (Simulation)"
    best_mem = min(x for x in (cusum, ewma, mewma) if x is not None)
    best_wo = min(t2, sep)
    if best_mem < best_wo:
        urteil = (f"Das Gedächtnis bringt hier (zero-state, ein springender Kanal, unabhängige Kanäle) mehr als die Wahl zwischen T² und getrennten Karten: "
                  f"die schnellste Karte mit Gedächtnis braucht {de(best_mem, 2)}, die schnellere der beiden ohne Gedächtnis {de(best_wo, 2)} Vektoren.")
        kind = "success"
    else:
        urteil = (f"Das Gedächtnis bringt hier nichts: die schnellere der beiden Karten ohne Gedächtnis braucht {de(best_wo, 2)}, die schnellste mit Gedächtnis {de(best_mem, 2)} Vektoren "
                  "(große Sprünge erkennt schon ein einzelner Vektor).")
        kind = "warning"
    return kind, (f"p = {p}, ein Kanal springt um {de(d, 1)} σ, gemeinsame ARL₀ 370,4: ARL₁ {ohne}; {mit}. {urteil} "
                  "Zero-state und steady-state unterscheiden sich beim CUSUM um 4 bis 8 % (gemessen für p = 1 und 5, d = 0,5 / 1 / 2); Lorden- und Pollak-Maß sind nicht gemessen.")


# ------------------------------------------------------------------------------------------------------------------ Experiment 3: Zeile aus den Reglern
def exp3_row(S: Settings):
    """(Sprungvektor, Zeile der ARL) aus den Reglern von Experiment 3; None, wenn die Einstellung nicht gerechnet wird (p = 5 nur für ρ ≥ 0)."""
    if S.korr_modell == C.KORR_P5 and S.rho < 0:
        return None
    mu = R.jump_vector(S.korr_modell, S.rho, S.richtung, S.sprung, S.gleich)
    return mu, R.arl_row(S.korr_modell, S.rho, mu)
