"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Zähleinheit überall: Beobachtungsvektoren bis zum ersten Alarm (Läufe nur, wo eine Simulation gezählt wird). Gemeinsame ARL₀ = 1 / (2 (1 - Φ(3))) = 370,398
für jede Karte und jedes p (das Ziel ist 370,398, nicht das gesetzte 370)."""

import math

# Ziel der gemeinsamen ARL₀ (Vektoren bis zum Fehlalarm): 1 / P(|N(0,1)| > 3) = 1 / erfc(3 / sqrt 2)
GAMMA0 = 1.0 / math.erfc(3.0 / math.sqrt(2.0))
P_ALARM0 = 1.0 / GAMMA0

P_OPTIONS, DEFAULT_P = (1, 2, 5, 10, 20), 5            # Zahl der Kanäle (die Karten mit Gedächtnis sind für diese fünf Werte geeicht und gefroren)
D_OPTIONS, DEFAULT_D = (0.5, 1.0, 2.0, 3.0), 1.0       # Sprunghöhe je springendem Kanal in σ
CUSUM_K = 0.5                                          # Referenzwert des CUSUM
EWMA_LAMBDA = 0.1                                      # Glättung von EWMA und MEWMA

# --- Experiment 1: ein Verlauf -------------------------------------------------------------------------------------------------------------
M_MIN, M_MAX, DEFAULT_M = 1, 20, 1                       # Zahl der springenden Kanäle (begrenzt auf p)
N_VERLAUF = 400                                         # Länge des gezeichneten Verlaufs (Vektoren)
NU_MIN, NU_MAX, NU_STEP, DEFAULT_NU = 0, 300, 10, 50   # ν = Zahl der Vektoren vor der Änderung
KARTE_T2, KARTE_SHEW, KARTE_CUSUM, KARTE_EWMA = "T²-Karte", "Getrennte Shewhart-Karten (Šidák)", "Getrennte CUSUM", "Getrennte EWMA"
KARTEN = (KARTE_T2, KARTE_SHEW, KARTE_CUSUM, KARTE_EWMA)
DEFAULT_KARTE = KARTE_T2

# --- Experiment 3: Korrelation und Richtung -------------------------------------------------------------------------------------------------
KORR_P2, KORR_P5 = "p = 2, Korrelation ρ", "p = 5, gleichkorreliert (ρ ≥ 0)"
KORR_MODELLE = (KORR_P2, KORR_P5)
DEFAULT_KORR_MODELL = KORR_P2
RHO_MIN, RHO_MAX, RHO_STEP, DEFAULT_RHO = -0.95, 0.95, 0.05, 0.0   # |ρ| höchstens 0,95: unterhalb von 1 - |ρ| = 1e-7 verfehlt die Quadratur die Randschicht
RICHTUNG_ENTLANG, RICHTUNG_GEGEN, RICHTUNG_EIN, RICHTUNG_SCHIEF = "entlang", "gegen", "ein Kanal", "schief"
RICHTUNGEN = (RICHTUNG_ENTLANG, RICHTUNG_GEGEN, RICHTUNG_EIN, RICHTUNG_SCHIEF)
DEFAULT_RICHTUNG = RICHTUNG_EIN
SPRUNG_MIN, SPRUNG_MAX, SPRUNG_STEP, DEFAULT_SPRUNG = 0.25, 3.0, 0.25, 1.0       # Skalierung d des Sprungvektors (μ = d · v)
DEFAULT_GLEICH = False                                  # True: der Sprung wird auf gleiche Nichtzentralität skaliert (der Regler d ist dann ohne Wirkung und gesperrt)
LAMBDA_GLEICH = 4.0                                     # „gleiche Nichtzentralität“: der Sprung wird so skaliert, dass μ'Σ⁻¹μ = 4
SCHIEF_V = (1.0, -0.3)                                  # Richtung „schief“ (p = 2)

# --- Experiment 4: Verdünnung --------------------------------------------------------------------------------------------------------------
VD_P_OPTIONS, DEFAULT_VD_P = (5, 10), 5
VD_M_MIN, VD_M_MAX, DEFAULT_VD_M = 1, 10, 1               # springende Kanäle (begrenzt auf p)
VD_M_OPTIONS = {5: (1, 2, 3, 4, 5), 10: (1, 2, 4, 5, 10)}   # nur für diese m ist die CUSUM gefroren; T², Shewhart und Summe sind für jedes m exakt
DEFAULT_VD_SUMME = True
VD_GLEICH_HOEHE, VD_GLEICH_ENERGIE = "gleiche Höhe (d = 1 je springendem Kanal)", "gleiche Energie (λ = m·d² = 4)"
VD_VERGLEICHE = (VD_GLEICH_HOEHE, VD_GLEICH_ENERGIE)
DEFAULT_VD_VERGLEICH = VD_GLEICH_ENERGIE

# --- Experiment 5: Gedächtnis --------------------------------------------------------------------------------------------------------------
R_MIN, R_MAX, R_STEP, DEFAULT_R = 200, 2000, 200, 400    # Zahl der Live-Läufe der MEWMA-Simulation

SEED_MAX, DEFAULT_SEED = 999999, 15

FROZEN_STAND = "2026-11-19"

COLOR_T2 = "#c0392b"
COLOR_SHEW = "#d68910"
COLOR_SUM = "#8e44ad"
COLOR_CUSUM = "#2F6B65"
COLOR_EWMA = "#4c78a8"
COLOR_MEWMA = "#7f8c8d"
COLOR_LINE = "#4c78a8"
COLOR_THRESH = "#555555"
COLOR_CHANGE = "#999999"

PRESETS = {
    "Ein Kanal springt, p = 5": {},
    "p = 20 Kanäle": {"p": 20},
    "Korreliert, Sprung gegen die Korrelation (ρ = 0,8)": {"korr_modell": KORR_P2, "rho": 0.8, "richtung": RICHTUNG_GEGEN},
    "Korreliert, Sprung entlang (ρ = 0,8)": {"korr_modell": KORR_P2, "rho": 0.8, "richtung": RICHTUNG_ENTLANG},
    "Alle fünf springen": {"m": 5},
    "Gleiche Energie, verteilt auf vier Kanäle": {"vd_m": 4},
    "Mit Gedächtnis: getrennte CUSUM": {"karte": KARTE_CUSUM},
}
PRESET_HELP = {
    "Ein Kanal springt, p = 5": "Voreinstellung: fünf unabhängige Kanäle, einer springt um 1 σ. Gemeinsame ARL₀ 370,4 für jede Karte; wie viele Vektoren braucht jede Karte bis zum Alarm?",
    "p = 20 Kanäle": "Zwanzig Kanäle, einer springt um 1 σ: bei gleicher gemeinsamer ARL₀ braucht T² dann 206,9 statt 114,4 Vektoren (Experiment 2).",
    "Korreliert, Sprung gegen die Korrelation (ρ = 0,8)": "Zwei Kanäle mit Korrelation 0,8, die sich gegeneinander bewegen (1, −1): T² braucht 2,22 Vektoren, getrennte Karten 32,99 (Experiment 3).",
    "Korreliert, Sprung entlang (ρ = 0,8)": "Zwei Kanäle mit Korrelation 0,8, die sich gemeinsam bewegen (1, 1), wie sie ohnehin schwanken: T² braucht 59,71 Vektoren, getrennte Karten 41,27 (Experiment 3).",
    "Alle fünf springen": "Alle fünf Kanäle springen um 1 σ (Experiment 1): der Sprung verteilt sich auf alle Kanäle, T² braucht dann 12,1 Vektoren.",
    "Gleiche Energie, verteilt auf vier Kanäle": "Vier von fünf Kanälen springen um je 1 σ (gleiche Energie λ = 4, Experiment 4): getrennte Shewhart-Karten sind langsam, Summenkarte und CUSUM schnell.",
    "Mit Gedächtnis: getrennte CUSUM": "Experiment 1 mit der getrennten CUSUM-Karte: bei p = 5 und d = 1 braucht sie 13,0 Vektoren statt 114,4 bei T² (Experiment 5).",
}
