# Viele Kanäle: was eine gemeinsame Alarmregel kostet – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-mehrere-kanaele-demo.streamlit.app/)**

Stück 6 der Linie **Änderungserkennung** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Der Ast „online“ der Linie (Stücke 1 bis 6) misst überall dasselbe: **Beobachtungen bis zum Alarm**. Dieses Stück hebt die Annahme auf, dass die Karte **eine** Reihe überwacht: p Reihen („Kanäle“) laufen gleichzeitig, es gibt **einen** gemeinsamen Alarm, und die Fehlalarmrate (ARL₀ 370,4) soll für den gemeinsamen Alarm gelten. Vorgänger ist Stück 5 („Wenn die Werte nicht unabhängig sind“: Abhängigkeit **in der Zeit**), Nachfolger Stück 7 („Brüche im Rückblick finden“: anderes Maß). Wie Mittel und Kovarianz aus Phase I geschätzt werden, steht in `anfangsparameter-abhaengig-demo` (Stück 4b); hier sind sie bekannt.

Die Kanäle sind ausgedacht: standardnormale Messreihen mit bekannter Korrelation und einem Sprung im Mittelwert. Es gibt in den gelesenen Quellen keine reale Mehrkanal-Datenquelle für diesen Zweck. Die Karten kennen Mittel und Kovarianz. Gezählt werden Beobachtungsvektoren bis zum ersten Alarm. Was die Zahlen für einen echten Betrieb in Logistik, Lager oder Fertigung heißen, wird hier nicht behauptet; ein Nutzen einer Alarmregel dort ist nicht belegt.

## Warum dieses Problem

Eine Karte, die für eine Reihe „im Mittel alle 370,4 Beobachtungen ein Fehlalarm“ verspricht, verspricht für p Reihen mit derselben Schwelle in jedem Kanal nur noch etwa 370,4 / p (unabhängige Kanäle: 1 / (1 − (1 − 1/370,4)^p)). Soll die **gemeinsame** ARL₀ 370,4 bleiben, muss jede Karte je Kanal vorsichtiger werden, und das kostet Verzögerung. **Kernfrage:** Was kostet eine gemeinsame Alarmregel, und wann gewinnt eine gemeinsame Statistik (Hotelling-T²) gegen getrennte Karten?

**Kernbefund (bekannte Parameter, Mittelwertsprung ab Vektor 1, Zähleinheit Beobachtungsvektoren bis zum Alarm, gemeinsame ARL₀ 370,4).** Mit p = 5 unabhängigen Kanälen, von denen einer um 1 σ springt, braucht eine gemeinsame Alarmregel im Mittel 114,4 Vektoren (T²) bzw. 109,9 (fünf Šidák-Shewhart-Karten) gegen 43,9 bei einer Karte für den bekannten Kanal (p = 1); die Zahl wächst mit p (T²: 206,9 bei p = 20). T² ist dabei in 4 von 16 Zellen (p = 2, 5, 10, 20; d = 0,5 / 1 / 2 / 3) etwas schneller als getrennte Karten, alle bei d = 0,5; bei d = 1 ist es wenig langsamer, bei d ≥ 2 und p ≥ 5 deutlich (31,26 gegen 19,02 bei d = 2, p = 10). Der größere Unterschied ist das Gedächtnis: getrennte zweiseitige CUSUM brauchen 13,0 Vektoren (zero-state, Gitter, auf etwa 0,4 % genau), nicht 114. Ob sich T² gegen getrennte Karten lohnt, hängt an der **Richtung** des Sprungs gegen die Korrelation: bei ρ = 0,8 und p = 2 braucht T² gegen die Korrelationsstruktur 2,22 Vektoren gegen 32,99 bei getrennten Karten (etwa 15-mal schneller), bei einem Sprung in einem Kanal 16,92 gegen 62,22 (3,7-mal schneller), entlang der Korrelation aber 59,71 gegen 41,27 (langsamer). Dieser Verlust entlang der Korrelation beginnt erst bei ρ ≈ 0,24 (p = 2, d = 1); bei ρ = 0 ist T² entlang mit 27,73 gegen 36,64 schneller. Getrennte Karten mit der Šidák-Schranke haben bei Korrelation eine ARL₀ über 370,4 (422,7 bei ρ = 0,8, p = 2). Die Aussage über den Nutzen in Logistik, Lager oder Fertigung ist nicht belegt.

## Befunde und Korrekturen gegenüber Plan und Vormessung

Alle Zahlen des Plans und der Vormessung (V15) sind Erwartungen; im Bau wurden sie neu gerechnet. Die exakten Zahlen (Karten ohne Gedächtnis, korrelierte Karten, Kleinstmodell) stimmen mit der Vormessung auf die gedruckten Stellen überein. Abweichungen:

- **Zufallsquelle.** Der Plan nannte `normal()` der Vorlage (Box-Muller mit `decimal`, rund 82 µs je Zahl). Die App nimmt die Stapel-Fassung desselben SplitMix64 (`mkn_normal.py`, Inversionsverfahren, bitgleich mit der Klasse bei der Gleichverteilung): 10⁶ Normalzahlen in rund 0,04 s. Damit laufen Verlauf und MEWMA-Simulation live. Die Referenzsimulationen der Tests (Millionen Vektoren) nehmen den numpy-Generator als Gegenprobe.
- **MEWMA-ARL₁ (Simulation, 20 000 Läufe, SplitMix64):** p = 5, ein Kanal um 1 σ: 14,61 ± 0,05 (Vormessung 14,68 ± 0,05, V2 14,67 ± 0,05: innerhalb von zwei kombinierten Standardfehlern). Gleiche Nichtzentralität λ = 4 (p = 5): 5,88 ± 0,01 (m = 1, d = 2) und 5,90 ± 0,01 (m = 4, d = 1); die Vormessung nennt 5,92 und 5,90.
- **MEWMA-Schwelle mit n = 400 Ringen** der radialen Kette statt 150: p = 2 / 5 gibt 10,0750 / 16,2895 statt 10,0764 / 16,2902 (Abweichung unter 0,02 %). Bei p = 1 lag n = 150 um 0,7 % daneben (ARL₀ 372,9 bei n = 400; die Kette konvergiert dort langsam, H* = 7,3162); mit n = 400 ist H* = 7,3021 (ARL₀ bei n = 800: 370,9). p = 1, 10 und 20 sind neu gerechnet (V15 nennt nur p = 2 und 5). Bei p = 1 ist die MEWMA ein EWMA: ihre Schwelle gibt dieselbe Grenze wie der EWMA (Test).
- **Rundung:** CUSUM bei p = 10, d = 3 gibt 3,395 (die Vormessung druckt 3,40, diese README 3,39).
- **Genauigkeit der Gitter.** Die Prüfung des Plans fand Simulationen 0,3 bis 0,4 % unter den Gitterwerten und ließ offen, welche Seite näher an der Wahrheit liegt. Im Bau gemessen (Scratch-Läufe, nicht im Test): p = 5, d = 1: Gitter 13,002, Simulation mit 1,2 Mio. Läufen 13,007 ± 0,006 (Abstand 0,04 %); p = 1, d = 1: Gitter 9,9267 (Richardson aus m = 32 und 48; m = 64 und 96 geben 9,9267), Simulation mit 2 Mio. Läufen 9,9263 ± 0,0037. Die 0,3 bis 0,4 % bestätigen sich in diesen zwei Zellen nicht; README und App nennen weiter „auf etwa 0,4 % genau“ als vorsichtige Obergrenze. Die Tests prüfen Gitter gegen Simulation im Band von 4 Standardfehlern (20 000 bis 40 000 Läufe, Standardfehler 0,2 bis 0,7 %).
- **Dichtes Raster für die Šidák-Ungleichung** (neu, nicht im Plan): ARL₀ der Šidák-Karte liegt für p = 2 auf 39 Werten von ρ (−0,95 bis 0,95 in Schritten von 0,05) und für p = 5 gleichkorreliert auf 20 Werten (0 bis 0,95) nicht unter 370,398 (bei ρ = 0 gleich): in 59 von 59 Zellen, zusätzlich zu den 10 von 10 Zellen der Vormessung. Gemessen in diesem Modell (gleiche Varianz, Mittel 0), die Ungleichung selbst ist nicht aus der Quelle gelesen.
- **Verdünnung, p = 5, m = 3** (im Plan nur m = 1, 2, 4, 5) neu: gleiche Höhe T² 28,86 / getrennt 45,92 / Summe 20,56 / CUSUM 8,42; gleiche Energie 17,93 / 30,81 / 13,62 / 7,02.
- **Regler.** Die Zahl der Kanäle ist p = 1, 2, 5, 10, 20 und der Sprung d = 0,5 / 1 / 2 / 3: nur dafür gibt es gefrorene Karten mit Gedächtnis (der Plan sah p bis 20 und d von 0,25 bis 3 vor). Die Kurven in Experiment 2 laufen für T² und getrennte Shewhart-Karten über alle p von 1 bis 20. In Experiment 3 ist die Richtung eine Auswahl (entlang, gegen, ein Kanal, schief) statt eines Winkels: die Zahlen der Vormessung gelten für μ = d · (1, 1), d · (1, −1) und d · (1, 0), nicht für Einheitsvektoren. Das Maß (zero-state oder steady-state) ist in Experiment 5 eine Spalte der Tabelle statt eines Reglers: steady-state ist nur für den CUSUM bei p = 1 und 5 gemessen, ein Regler wäre bei den übrigen Karten wirkungslos.
- **Zufalls-Seed.** Er gilt für den Verlauf und den MEWMA-Live-Lauf (der Plan hatte ihn nur beim Verlauf). Der voreingestellte Zufalls-Seed 15 ist so gewählt, dass bei p = 5, ν = 50 und einem Kanal um 1 σ keine der vier Karten vor der Änderung alarmiert; andere Zufalls-Seeds zeigen Fehlalarme oder späte Alarme, und ein Verlauf ist eine Stichprobe.
- **Fehler im Bau:** (1) Mein Orakel für den EWMA-Produktweg (Integralgleichung) schnitt die Summe beim kürzeren der beiden Felder ab und meldete ARL₀ 42 statt 370; der Fehler lag im Test (die Überlebensfunktion des geänderten Kanals muss so lang laufen wie die des ungeänderten), nicht in der Demo. (2) Die erste MEWMA-Eichung mit n = 150 (siehe oben).

## Modell

Je Zeitpunkt ein Vektor x_t aus p Kanälen, vor der Änderung N(0, Σ) mit Einheitsvarianzen (σ-Einheiten), Σ bekannt (Einheitsmatrix oder Korrelation ρ). Änderung: Mittelwertvektor μ ≠ 0 ab dem ersten Vektor nach der Änderung. Vollständiges Modell mit Formeln: der 📐-Expander der App. Konventionen (in der App als Tabelle):

| Konvention | Festlegung in dieser Demo |
|---|---|
| ν | Zahl der Vektoren **vor** der Änderung; die Zahlen der Tabellen gelten für ν = 0 („Änderung ab Vektor 1“, Nullzustand) |
| Maß | Karten ohne Gedächtnis: zero-state, Lorden und Pollak sind dieselbe Zahl (ARL₁). Karten mit Gedächtnis: **zero-state**, steady-state (quasistationär, bedingt auf „bis dahin kein Alarm“) nur für den CUSUM bei p = 1 und 5; Lorden- und Pollak-Maß nicht gemessen |
| Zähleinheit | Beobachtungsvektoren bis zum ersten Alarm (Läufe nur, wo eine Simulation gezählt wird, und dann so genannt) |
| Fehlalarm | Alarm vor der Änderung, je Zeit gezählt (ARL₀), nicht der Anteil falscher unter den Alarmen |
| Ziel | gemeinsame ARL₀ = 1 / (2 (1 − Φ(3))) = 370,398 für jede Karte und jedes p, zweiseitig (Betrag) |
| Šidák | q = 1 − (1 − 1/370,398)^(1/p) je Kanal, z = Φ⁻¹(1 − q/2); setzt unabhängige Kanäle voraus |
| geeicht | Schranke so gewählt, dass die gemeinsame ARL₀ 370,398 ist (CUSUM: h je Kanal, EWMA: H je Kanal, MEWMA: H, T²: χ²-Quantil) |
| Kanal | eine Messreihe, nicht ein Lkw-Tor |

Das Ziel ist überall 370,398 (= 1 / P(|N(0,1)| > 3)), nicht das gesetzte 370; Schwellen zu verschiedenen Zielen werden nicht vermischt. „Hotelling“ kommt nur als Hotelling-T² vor.

## Methodik

- **Karten ohne Gedächtnis (exakt).** T² (Alarm bei x'Σ⁻¹x > χ²-Quantil), getrennte Shewhart-Karten (irgendein |x_i| > z, Šidák), Summenkarte (|Σ x_i| / √p > 3, setzt voraus, dass die Richtung „alle gleichsinnig“ bekannt ist). Der Alarm hängt nur vom aktuellen Vektor ab: die Lauflänge ist **geometrisch**, ARL₁ = 1 / Alarmwahrscheinlichkeit je Vektor, aus der nichtzentralen χ²-Verteilung bzw. aus Normalverteilungsfunktionen.
- **Karten mit Gedächtnis je Kanal** (zweiseitiger CUSUM mit k = 0,5, EWMA mit λ = 0,1; gemeinsamer Alarm beim ersten Kanalalarm). Bei unabhängigen Kanälen ist die Lauflänge der gemeinsamen Karte das **Minimum der Kanal-Lauflängen**: P(RL > n) ist das Produkt der Überlebensfunktionen, ARL = Σ_n Produkt. Die Überlebensfunktion eines Kanals kommt aus einer Markov-Kette (2D-Gitter für den zweiseitigen CUSUM, 1D-Gitter für den EWMA), der Gitterfehler wird per Richardson-Extrapolation in 1/m² ausgeglichen (CUSUM m = 32 und 48, EWMA m_up = 80 und 160). **Das sind Gitterwerte, auf etwa 0,4 % genau, nicht geschlossene Formeln**; die Simulation liegt in den Tests innerhalb von 4 Standardfehlern der Gitterzahl. Die Gitter brauchen Sekunden bis Minuten und sind **gefroren** (`mkn_cusum_frozen.json`, Skript `tools/mkn_make_cusum.py`, `--check` rechnet nach).
- **MEWMA** (λ = 0,1, Σ_Z asymptotisch): Schwelle und ARL₀ aus der radialen Markov-Kette (Kette mit endlicher Ringzahl, Konvergenz in n geprüft), ARL₁ nur **Simulation** (gelb).
- **Korrelierte Kanäle** (Σ bekannt): T² mit Σ⁻¹ (ARL₀ für jedes Σ 370,398, ARL₁ nur an λ = μ'Σ⁻¹μ); getrennte Karten über eine Rechteckwahrscheinlichkeit der Normalverteilung (p = 2: Integral über x₁; p = 5 gleichkorreliert: Ein-Faktor-Darstellung mit Gauß-Hermite), jeweils neu auf ARL₀ 370,398 geeicht und daneben mit der Šidák-Schranke.
- **Verlauf.** Ein Verlauf aus 400 Vektoren (SplitMix64) mit Statistik, Schwelle, Änderungszeitpunkt, erstem Alarm und dem alarmierenden Kanal (nur getrennte Karten nennen ihn).

### Eichung (gemeinsame ARL₀ 370,398)

| p | Schranke T² (χ²) | z je Kanal (Šidák) | ARL₀ je Kanal | h* CUSUM | H* EWMA | H* MEWMA |
|---|---|---|---|---|---|---|
| 1 | 9,000 | 3,0000 | 370,4 | 4,7749 | 0,6198 | 7,3021 |
| 2 | 11,829 | 3,2050 | 740,3 | 5,4498 | 0,6771 | 10,0750 |
| 5 | 18,205 | 3,4598 | 1850,0 | 6,3507 | 0,7462 | 16,2895 |
| 10 | 26,901 | 3,6422 | 3699,5 | 7,0360 | 0,7943 | 24,7605 |
| 20 | 42,080 | 3,8169 | 7398,5 | 7,7230 | 0,8394 | 39,5885 |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Beobachtungsvektoren bis zum ersten Alarm, gemeinsame ARL₀ 370,398, bekannte Parameter, Mittelwertsprung ab Vektor 1. Die Spalte „Test“ der Tabelle „Zahl → Test“ weiter unten nennt, wo jede Zahl geprüft wird.

### A. Die Strafe der Kanalzahl (unabhängige Kanäle, ein Kanal springt um d; exakt)

ARL₁ von T² / von p getrennten Šidák-Shewhart-Karten:

| p | d = 0,5: T² / getrennt | d = 1,0: T² / getrennt | d = 2,0: T² / getrennt | d = 3,0: T² / getrennt |
|---|---|---|---|---|
| 1 | 155,22 / 155,22 | 43,89 / 43,89 | 6,30 / 6,30 | 2,00 / 2,00 |
| 2 | 202,24 / 205,45 | 67,32 / 66,34 | 9,41 / 8,67 | 2,57 / 2,38 |
| 5 | 259,43 / 267,84 | 114,38 / 109,89 | 17,93 / 13,48 | 4,16 / 3,08 |
| 10 | 292,99 / 304,54 | 159,91 / 152,90 | 31,26 / 19,02 | 6,91 / 3,81 |
| 20 | 317,33 / 330,37 | 206,87 / 200,63 | 54,61 / 26,96 | 12,83 / 4,78 |

T² ist in **4 von 16 Zellen** (p = 2, 5, 10, 20 bei d = 0,5) etwas schneller als getrennte Karten, sonst langsamer: bei d = 1 nur wenig (p = 2: 67,32 gegen 66,34), bei d ≥ 2 und p ≥ 5 deutlich (bei d = 2 und p = 10: 31,26 gegen 19,02). „T² ist nicht besser als p getrennte Karten“ gilt also nur bei **unabhängigen** Kanälen und einem springenden Kanal, und dort in 12 von 16 Zellen. **Verteilung der Lauflänge** (Shewhart-Typ: geometrisch, exakt; CUSUM: aus dem Gitter m = 48 ohne Extrapolation, daher das Mittel unter 370,4, bei p = 20 um 1,2 %; die Angabe „auf etwa 0,4 % genau“ gilt für diese Tabelle nicht), ohne Änderung:

| Karte | Mittel | Median | 5 %-Quantil | 95 %-Quantil | Alarm bis Vektor 100 |
|---|---|---|---|---|---|
| Shewhart-Typ (T², getrennt, Summe), jedes p | 370,4 | 257 | 19 | 1109 | 23,69 % |
| getrennte CUSUM, p = 1 | 369,6 | 258 | 25 | 1095 | 22,9 % |
| getrennte CUSUM, p = 2 | 369,0 | 258 | 26 | 1091 | 22,7 % |
| getrennte CUSUM, p = 5 | 368,1 | 258 | 27 | 1085 | 22,5 % |
| getrennte CUSUM, p = 10 | 367,2 | 258 | 29 | 1079 | 22,3 % |
| getrennte CUSUM, p = 20 | 366,0 | 257 | 30 | 1073 | 22,2 % |

### B. Das Gedächtnis (zero-state, ein Kanal springt; CUSUM und EWMA Gitter, auf etwa 0,4 % genau)

| p | CUSUM je Kanal: d = 0,5 / 1 / 2 / 3 | EWMA je Kanal (λ = 0,1): d = 0,5 / 1 / 2 / 3 |
|---|---|---|
| 1 | 35,27 / 9,93 / 3,86 / 2,49 | 28,23 / 9,74 / 4,18 / 2,76 |
| 2 | 42,03 / 11,24 / 4,31 / 2,75 | 34,72 / 11,07 / 4,59 / 3,00 |
| 5 | 52,07 / 13,00 / 4,91 / 3,12 | 45,64 / 12,93 / 5,10 / 3,29 |
| 10 | 60,35 / 14,35 / 5,37 / 3,39 | 56,18 / 14,41 / 5,48 / 3,50 |
| 20 | 69,13 / 15,71 / 5,82 / 3,67 | 69,17 / 15,98 / 5,84 / 3,69 |

Vergleich bei d = 1, ein Kanal:

| p | T² | getrennt (Šidák) | getrennte CUSUM | Verhältnis CUSUM / T² |
|---|---|---|---|---|
| 1 | 43,89 | 43,89 | 9,93 | 0,226 |
| 2 | 67,32 | 66,34 | 11,24 | 0,167 |
| 5 | 114,38 | 109,89 | 13,00 | 0,114 |
| 10 | 159,91 | 152,90 | 14,35 | 0,090 |
| 20 | 206,87 | 200,63 | 15,71 | 0,076 |

Die Dimensionsstrafe der getrennten CUSUM ist klein (9,93 bei p = 1, 15,71 bei p = 20), die der Karten ohne Gedächtnis groß (43,89 auf 206,87). **MEWMA** (ARL₁ nur Simulation, 20 000 Läufe, ± Standardfehler; ARL₀ aus der radialen Kette mit n = 400 Ringen, gegen n = 800 bei p = 1 um 0,13 % abweichend, sonst um weniger als 0,02 %):

| p | d = 0,5 | d = 1 | d = 2 | d = 3 |
|---|---|---|---|---|
| 1 | 28,39 ± 0,14 | 9,75 ± 0,03 | 4,18 ± 0,01 | 2,77 ± 0,00 |
| 2 | 35,45 ± 0,18 | 11,50 ± 0,04 | 4,80 ± 0,01 | 3,15 ± 0,00 |
| 5 | 49,47 ± 0,27 | 14,61 ± 0,05 | 5,89 ± 0,01 | 3,82 ± 0,01 |
| 10 | 65,55 ± 0,37 | 18,19 ± 0,06 | 7,02 ± 0,01 | 4,52 ± 0,01 |
| 20 | 90,78 ± 0,53 | 23,29 ± 0,08 | 8,51 ± 0,01 | 5,44 ± 0,01 |

| Fall (p = 5, MEWMA) | ARL₁ (Simulation, 20 000 Läufe) |
|---|---|
| m = 1, d = 2 (λ = 4) | 5,88 ± 0,01 |
| m = 4, d = 1 (λ = 4) | 5,90 ± 0,01 |
| m = 1, d = 1 (λ = 1) | 14,64 ± 0,05 |
| m = 5, d = 1 (λ = 5) | 5,20 ± 0,01 |

„± 0,00“ in der ersten Tabelle heißt: Standardfehler unter 0,005, nicht null. Eine unabhängige Nachrechnung mit 200 000 Läufen je Zelle (Prüfung nach dem Bau, nicht im Test) lag in 19 von 20 Zellen innerhalb von zwei Standardfehlern; bei p = 10, d = 0,5 gab sie 66,37 ± 0,05 gegen 65,55 ± 0,37 der Tabelle (2,2 Standardfehler).

Die MEWMA hängt bei dieser Einheitsmatrix an diesem einen Paar (m = 1, d = 2 gegen m = 4, d = 1) nur an λ (Unterschied unter vier kombinierten Standardfehlern); eine allgemeine Aussage ist das nicht. Das Gedächtnis bringt hier (p = 5, d = 1, zero-state, ein springender Kanal, unabhängige Kanäle) mehr als die Wahl zwischen T² und getrennten Shewhart-Karten: CUSUM 13,00, EWMA 12,93, MEWMA 14,61 ± 0,05 gegen 114,38 und 109,89. Es hilft nicht immer: bei p = 1 und d = 3 sind T² und Shewhart-Karte mit 2,00 schneller als CUSUM (2,49) und EWMA (2,76). **Steady-state** (quasistationär; nur CUSUM, p = 1 und 5, Gitter m = 24 und 32 mit Richardson):

| p | d = 0,5: zero / steady (Verhältnis) | d = 1: zero / steady (Verhältnis) | d = 2: zero / steady (Verhältnis) |
|---|---|---|---|
| 1 | 35,27 / 33,76 (0,957) | 9,93 / 9,21 (0,928) | 3,86 / 3,54 (0,918) |
| 5 | 52,07 / 49,72 (0,955) | 13,00 / 12,11 (0,931) | 4,91 / 4,55 (0,927) |

Das steady-state-Maß liegt 4 bis 8 % unter dem zero-state-Maß (in diesen sechs Zellen); Lorden- und Pollak-Maß sind nicht gemessen.

### C. Korrelierte Kanäle (Σ bekannt, Einheitsvarianzen; exakt)

T² mit Σ⁻¹ hat für jedes Σ die ARL₀ 370,398, seine ARL₁ hängt nur an λ = μ'Σ⁻¹μ. Getrennte Karten mit der Šidák-Schranke haben bei Korrelation eine andere ARL₀:

| Kanäle | ρ | ARL₀ der getrennten Karten mit Šidák-Schranke |
|---|---|---|
| p = 2 | -0,90 | 470,4 |
| p = 2 | -0,50 | 379,0 |
| p = 2 | 0,00 | 370,4 |
| p = 2 | 0,50 | 379,0 |
| p = 2 | 0,80 | 422,7 |
| p = 2 | 0,95 | 520,7 |
| p = 5 gleichkorreliert | 0,00 | 370,4 |
| p = 5 gleichkorreliert | 0,30 | 375,2 |
| p = 5 gleichkorreliert | 0,60 | 414,1 |
| p = 5 gleichkorreliert | 0,90 | 669,8 |

Die Šidák-Eichung ist also in den 10 von 10 Zellen der Tabelle und in 59 von 59 Zellen des dichten Rasters (siehe „Befunde und Korrekturen“) bei Korrelation konservativ (ARL₀ nicht unter 370,398, bei ρ = 0 gleich), nie zu optimistisch; gemessen in diesem Modell (gleiche Varianz, Mittelwert 0). Grenzfall von Hand: für ρ → ±1 melden beide Kanäle dasselbe Ereignis, ARL₀ → 1/q = 740,3 (p = 2). **ARL₁ je Richtung** (Sprung μ = d · v mit d = 1, v = (1, 1) „entlang“, (1, −1) „gegen“, (1, 0) „ein Kanal“; p = 2; getrennte Karten neu auf 370,398 geeicht und mit Šidák-Schranke):

| ρ | Richtung (d = 1) | λ | T² | getrennt (neu geeicht) | getrennt (Šidák-Schranke) |
|---|---|---|---|---|---|
| 0,00 | entlang | 2,00 | 27,73 | 36,64 | 36,64 |
| 0,00 | gegen | 2,00 | 27,73 | 36,64 | 36,64 |
| 0,00 | ein Kanal | 1,00 | 67,32 | 66,34 | 66,34 |
| 0,50 | entlang | 1,33 | 47,89 | 38,61 | 39,25 |
| 0,50 | gegen | 4,00 | 9,41 | 35,79 | 36,40 |
| 0,50 | ein Kanal | 1,33 | 47,89 | 65,99 | 67,16 |
| 0,80 | entlang | 1,11 | 59,71 | 41,27 | 45,39 |
| 0,80 | gegen | 10,00 | 2,22 | 32,99 | 36,42 |
| 0,80 | ein Kanal | 2,78 | 16,92 | 62,22 | 68,82 |
| 0,95 | entlang | 1,03 | 65,44 | 43,15 | 55,24 |
| 0,95 | gegen | 40,00 | 1,00 | 28,22 | 36,42 |
| 0,95 | ein Kanal | 10,26 | 2,14 | 53,55 | 69,42 |

**Gleiche Nichtzentralität** (ρ = 0,8, λ = 4): T² gibt in jeder Richtung 9,41, getrennte Karten nicht:

| Richtung (ρ = 0,8) | λ | T² | getrennt (neu geeicht) |
|---|---|---|---|
| gegen | 4,00 | 9,41 | 88,39 |
| schief | 4,00 | 9,41 | 63,92 |
| ein Kanal | 4,00 | 9,41 | 39,20 |
| entlang | 4,00 | 9,41 | 6,81 |

p = 5 gleichkorreliert, d = 1 (alle fünf springen gegen ein Kanal; getrennte Karten neu geeicht):

| ρ | alle fünf: T² | alle fünf: getrennt (neu geeicht) | ein Kanal: T² | ein Kanal: getrennt (neu geeicht) |
|---|---|---|---|---|
| 0,0 | 12,11 | 29,16 | 114,38 | 109,89 |
| 0,6 | 76,70 | 35,07 | 50,25 | 106,51 |
| 0,9 | 105,70 | 41,30 | 5,09 | 81,19 |

**Lesart:** T² vergleicht den Sprung mit der Form der Korrelation. Sprünge **gegen** die Korrelation oder in **einem** Kanal bei hoher Korrelation sind für T² groß (großes λ); Sprünge **entlang** der Korrelation (alle Kanäle gemeinsam, wie sie ohnehin schwanken) sind klein. **Das präzisiert die Vermutung „Vorteil bei Sprüngen entlang der Korrelationsrichtung“ aus Zuschnitt und V2:** gemessen ist der Vorteil gegen die Korrelationsstruktur; entlang verliert T² gegen getrennte Karten (ρ = 0,8: 59,71 gegen 41,27), aber erst ab ρ ≈ 0,24 (p = 2, d = 1; bei ρ = 0 gewinnt T² mit 27,73 gegen 36,64). Die Zeile „T² nicht besser als getrennte Karten“ gilt nur bei **unabhängigen** Kanälen (p = 5, ρ = 0,6, ein Kanal: 50,25 gegen 106,51).

### D. Verdünnung: wie viele Kanäle springen? (p = 5 unabhängige Kanäle; exakt, CUSUM Gitter)

Gleiche **Energie** (λ = m·d² = 4, der Sprung wird auf m Kanäle verteilt):

| m | d | λ = m·d² | T² | getrennt (Šidák) | Summenkarte | getrennte CUSUM |
|---|---|---|---|---|---|---|
| 1 | 2,000 | 4,00 | 17,93 | 13,48 | 56,59 | 4,91 |
| 2 | 1,414 | 4,00 | 17,93 | 23,85 | 24,17 | 6,16 |
| 3 | 1,155 | 4,00 | 17,93 | 30,81 | 13,62 | 7,02 |
| 4 | 1,000 | 4,00 | 17,93 | 35,64 | 8,86 | 7,69 |
| 5 | 0,894 | 4,00 | 17,93 | 39,16 | 6,30 | 8,24 |

Gleiche **Höhe** (jeder springende Kanal um d = 1):

| m | d | λ = m·d² | T² | getrennt (Šidák) | Summenkarte | getrennte CUSUM |
|---|---|---|---|---|---|---|
| 1 | 1,000 | 1,00 | 114,38 | 109,89 | 177,73 | 13,00 |
| 2 | 1,000 | 2,00 | 52,26 | 64,69 | 56,59 | 9,71 |
| 3 | 1,000 | 3,00 | 28,86 | 45,92 | 20,56 | 8,42 |
| 4 | 1,000 | 4,00 | 17,93 | 35,64 | 8,86 | 7,69 |
| 5 | 1,000 | 5,00 | 12,11 | 29,16 | 4,50 | 7,21 |

T² hängt nur an λ (17,93 für alle Zeilen der Energie-Tabelle). Bei **gleicher Energie** gilt: je mehr Kanäle den Sprung teilen, desto besser die **Summenkarte** (sie kennt die Richtung: alle gleich) und desto schlechter die getrennten Shewhart-Karten (p = 5 und p = 10); bei einem springenden Kanal ist die Summenkarte die langsamste der vier Karten (p = 5 und p = 10; 177,73 gegen 114,38 bei T²). Unter den vier gemessenen Karten ist keine für jede Verteilung des Sprungs das schnellste. Für p = 10 gilt dasselbe (T² bei λ = 4: 31,26; getrennt 19,02 / 35,95 / 55,64 / 61,52 / 76,39 bei m = 1 / 2 / 4 / 5 / 10 mit d = 2/√m). Die **Summe von CUSUMs** (Mei; Xie und Siegmund) ist **nicht gemessen**.

### E. Kleinstmodell von Hand (Brüche)

Jeder Kanal liefert ξ ∈ {−1, 0, +1}, ohne Änderung mit (1/4, 1/2, 1/4), nach der Änderung mit (1/8, 3/8, 1/2); einseitiger CUSUM S' = max(0, S + ξ), Alarm bei S' ≥ 2; gemeinsame Karte: Alarm beim ersten Kanalalarm. Ein Kanal: ARL₀ 12, ARL₁ 9/2, Median der Lauflänge ohne Änderung 9 (gegen Mittel 12: rechtsschief). Zwei Kanäle bei gleicher Schwelle: ARL₀ = 3152/451 = 6,9889 (um 41,8 % kleiner als 12), ARL₁ (ein Kanal geändert) = 14308/3629 = 3,9427 (um 12,4 % kleiner als 4,5). Ein Kanal mehr bei gleicher Schwelle senkt die ARL₀ stärker als die ARL₁. Aufzählung aller Pfade, Minimum unabhängiger Lauflängen und ein lineares Gleichungssystem in Brüchen geben dieselben Zahlen.

## 🚧 Bewusst nicht enthalten / nicht gemessen

- **Σ geschätzt (Phase I)** und Šidák-Karten mit geschätzten Parametern: nicht hier. Die Messung zu T² mit geschätztem Mittel und Σ̂ steht in `anfangsparameter-abhaengig-demo` (Stück 4b).
- **Karten mit Gedächtnis bei korrelierten Kanälen**: nicht gemessen; der Produktweg gilt nur für unabhängige Kanäle. Die MEWMA ist bei bekanntem Σ nach Whitening invariant gegen die Richtung; mit Korrelation ist sie nicht gemessen.
- **Summe von CUSUMs und Mischungsverfahren** (Xie und Siegmund): nicht gemessen; **Lorden- und Pollak-Maß** für CUSUM und EWMA: nicht gemessen; steady-state nur für den CUSUM bei p = 1 und 5.
- **MEWMA-ARL₁ nur Simulation** (kein exakter Weg gebaut), nur λ = 0,1.
- **Korrelation** nur für p = 2 (jedes ρ bis ±0,95) und p = 5 gleichkorreliert (ρ ≥ 0); andere Strukturen und negative Gleichkorrelation bei p ≥ 3 nicht.
- **Nicht gemessen:** Sprung in Streuung oder Korrelation (nur Mittelwertsprung), Ausreißer und Schiefe je Kanal, Autokorrelation je Kanal (Stück 5), fehlende Werte, ungleiche Abtastzeiten, p > 20, d > 3. **Welcher Kanal** gesprungen ist: nur als Definition (getrennte Karten nennen den alarmierenden Kanal, T² nicht), keine Messung der Trefferquote.
- **Nutzen** einer Mehrkanalregel in Logistik, Lager oder Fertigung: nicht belegt; es gibt keine reale Datenquelle.
- **Kein Nachbau** von `niveausprung-demo` (Detektor im Prognosekontext; die dortige README stellt „mehrere Reihen gleichzeitig“ als nicht gebaut ab), `elliptic-envelope-demo` (Mahalanobis-Abstand mit χ²-Schwelle, aber offline ohne Zeitachse und ohne Lauflänge) und `multiple-testing-demo` (Familienfehler über Tests je Experiment, Zähleinheit Experimente; hier Šidák als Eichung der Schwelle auf eine gemeinsame ARL₀ je Zeit, Zähleinheit Vektoren): andere Fragen, andere Zähleinheit.
- **Kein Erklärseiten-Expander** („In einfachen Worten“), keine Grundlagen-Begriffe, keine Website-Karte: die Website-Einbindung der Linie ist nicht Teil dieser Demo.
- **Plattform:** Zahlen unter Windows, Python 3.14.6, numpy 2.5.3, scipy 1.18.1 gerechnet; Neu-Rechnen-Tests sind Bänder oder relative Toleranzen. Der Lauf unter Linux (`demo_linux_check.py`, Docker) wurde **nicht** ausgeführt; Rechenzeiten auf Streamlit Cloud sind nicht gemessen.

## Quellen (Stufe: W = gelesen, S = nur Suchtreffer, sonst nicht gelesen)

- **Xie und Siegmund**, Annals of Statistics 41 (2) (2013), arXiv 1207.2386 (**W**, Abstract und Abschnitte 1 und 5): bei Änderung in einer unbekannten Teilmenge der Ströme lassen Summenregeln das Rauschen der unbetroffenen Ströme in die Statistik, die Verzögerung wird lang. Nur qualitativ benutzt, keine Zahl.
- **NIST pmc543** (**W**): T²-Statistik und F-Verteilung nur für den Test, nicht für Kartenlauflängen; die Grenzen der Karte folgen aus der Verteilungstheorie (hier: bekanntes Σ, also χ²).
- **Vanhatalo und Kulahci**, Quality and Reliability Engineering International 31 (2015) 1779–1796 (**W**, Abschnitte 1, 5, 6): Autokorrelation und T² (Stück 5).
- **Lowry u. a. 1992** (MEWMA): **S**, keine Zahl daraus. **Nicht gelesen** (keine Aussage stützt sich darauf): Šidák 1967 (die Ungleichung wird hier nur an Zellen gezählt), Bersimis, Psarakis und Panaretos 2007, Montgomery, Basseville und Nikiforov, Mei, Hotelling 1947.
- Alles übrige sind eigene Nachrechnungen in diesem Modell. **Keine Neuheit:** alles Lehrbuch; neu ist die Vorführbarkeit mit Zähleinheit, exakten Orakeln (auch für das Minimum unabhängiger Karten mit Gedächtnis) und der Messung der Richtung.

## Zahl → Test

| Zahl bzw. Aussage der README | Test (Band oder exakt) |
|---|---|
| Alle Tabellen (Eichung, A bis D, Verteilung, steady-state, MEWMA) | `tests/test_claims.py::test_readme_tabellen_stehen_woertlich_in_der_readme` (die Tabellen stammen aus `tools/mkn_readme_tables.py`; die Rechnung dahinter: exakt für Karten ohne Gedächtnis und korrelierte Karten, Gitter für CUSUM und EWMA, Simulation für MEWMA-ARL₁) |
| 114,4 / 109,9 / 43,9 / 206,9; 4 von 16 Zellen, 12 von 16 | `test_claims.py::test_kernbotschaft_ohne_gedaechtnis`, `test_t2_schneller_in_4_von_16_zellen` |
| CUSUM 13,0; EWMA 12,93; MEWMA 14,61 ± 0,05; steady-state 12,11; 4 bis 8 %; p = 1, d = 3 | `test_claims.py::test_gedaechtnis_zahlen`; Gitter gegen Simulation: `tests/test_oracle_memory.py` (Band, 4 Standardfehler) |
| ρ = 0,8: 2,22 / 32,99 / 16,92 / 62,22 / 59,71 / 41,27, etwa 15-mal, 3,7-mal, ARL₀ 422,7 | `test_claims.py::test_korrelation_zahlen`; zweiter Weg `tests/test_oracle_corr.py` (`multivariate_normal`, Simulation) |
| Šidák: 10 von 10 und 59 von 59 Zellen | `test_claims.py::test_sidak_zaehlung`; `test_oracle_corr.py::test_sidak_ungleichung_auf_einem_dichten_raster_p2_und_p5` |
| Verdünnung (Tabellen, 17,93 konstant, 177,73 gegen 114,38) | `test_claims.py::test_verduennung_zahlen`; `test_oracle_cards.py` |
| Kleinstmodell: 12, 9/2, 3152/451, 14308/3629, Median 9, 41,8 %, 12,4 % | `tests/test_handbeispiele.py` (Brüche, Aufzählung aller Pfade, Produktweg) |
| Median 257, 5-%-Quantil 19, 95-%-Quantil 1109, 23,69 % | `test_handbeispiele.py::test_h7_*`, `test_oracle_cards.py` (Simulation der Lauflängen) |
| Abweichungen (MEWMA 14,61, 5,88 / 5,90, H* 10,0750 / 16,2895 / 7,3021, 3,395, p = 5 m = 3) | `test_claims.py::test_befunde_und_korrekturen_zahlen`; `tests/test_tables.py` |
| Genauigkeit der Gitter: 13,007 ± 0,006, 9,9263 ± 0,0037, 9,9267 | Messung dieser README (Scratch-Läufe mit 1,2 bzw. 2 Mio. Läufen), nicht im Test; im Test nur das Band von 4 Standardfehlern bei 20 000 bis 40 000 Läufen (`tests/test_oracle_memory.py`) |
| 82 µs je Zahl, 0,04 s für 10⁶ Normalzahlen | Messung dieser README, nicht im Test (`test_claims.py::test_zufallsquelle_ist_schnell` prüft nur die Größenordnung: unter 1 s) |
| Texte der App stehen in der README | `test_claims.py::test_app_texte_stehen_in_der_readme` |
| Zahlen in Presets, Texten und Verdikten | `tests/test_verdicts.py`, `tests/test_preset_verdicts.py`, `test_claims.py::test_zahlen_in_texten_und_presets` |

## Verifikation

**271 Tests**, seriell 70 bis 120 s je nach Last der Maschine (Windows, Python 3.14, ein BLAS-Thread; `pytest tests/ -q`); die Zahlen in Klammern sind die Tests je Art. Zusammensetzung:

- **Handbeispiele** (17; `tests/test_handbeispiele.py`): das Kleinstmodell in Brüchen (lineares Gleichungssystem, Aufzählung aller Pfade, Produktfunktionen der Demo auf der diskreten Kette), T² bei p = 2 in geschlossener Form, Summenkarte, Grenzfall der Korrelation, geometrische Lauflänge. Der Code der Demo wird dort nur benutzt, wo die Handrechnung selbst keinen Code hat.
- **Orakel** (47 + 22 + 20; `tests/test_oracle_cards.py`, `test_oracle_corr.py`, `test_oracle_memory.py`): ncx2 gegen Poisson-Mischung, Kreisscheiben-Integral und χ²-Dichte-Integral; getrennte Karten gegen Produkte von Quadraturen; korrelierte Karten gegen `multivariate_normal` (Genz-Verfahren, Toleranz 1e-4); CUSUM und EWMA je Kanal gegen Simulation mit dem numpy-Generator; EWMA gegen die Integralgleichung (Nyström) für einen Kanal und für fünf Kanäle über den Produktweg; MEWMA-Kette gegen Gitterkonvergenz und gegen den EWMA bei p = 1.
- **Eigenschaften** (in den Orakeldateien): T²-ARL₀ für jedes zufällige Σ, ARL₁ nur an λ, Umnummerieren ändert getrennte Karten nicht, Monotonie in d und p, ARL₀ steigt mit h.
- **Claims** (13), **gefrorene Tabellen** (17), **Verdikte** (21 an handgesetzten Werten, 14 je Schnellstart über die App), **Presets** (8), **Randfälle** (15), **Szenario** (12), **Zufallsquelle** (11), **Abbildungen** (5), **App** (AppTest, 19), **Layout** (27), **Footer** (2), **Laufzeit-Importe** (1).
- **Gefrorene Tabelle:** `python tools/mkn_make_cusum.py --check` rechnet ARL₀ aller Karten an den gespeicherten Schwellen neu (rund 12 s); `--voll` eicht und rechnet alles neu (rund 2 Minuten).
- **Live-Rechenzeiten** (diese Maschine, ein Thread, kein Befund): ein Verlauf mit 400 Vektoren unter 0,01 s, die exakten Zahlen der Experimente 2 bis 4 unter 0,1 s, die Eichung der getrennten Karten bei Korrelation unter 0,05 s, der MEWMA-Live-Lauf mit 2000 Läufen bei p = 20 und d = 0,5 (der teuerste Fall) rund 0,1 s, ein ganzer App-Lauf rund 1 bis 2 s.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-App: fünf Experimente, Grenzen, Formeln |
| `mkn_cards.py` | Karten ohne Gedächtnis: Schranken, ARL, geometrische Lauflänge, Verdünnung |
| `mkn_corr.py` | korrelierte Kanäle: Integrale, Eichung, Richtungen, Ellipse |
| `mkn_memory.py` | CUSUM, EWMA, MEWMA: Markov-Ketten, Produktweg, Eichung, steady-state |
| `mkn_scenario.py` | Verlauf (SplitMix64) und MEWMA-Simulation |
| `mkn_normal.py`, `mkn_rng.py` | Normalzahlen im Stapel; Vorlage SplitMix64 (unverändert) |
| `mkn_reference.py`, `mkn_cusum_frozen.json` | gefrorene Tabellen (Schwellen und ARL der Karten mit Gedächtnis) |
| `mkn_evaluation.py` | Einstellungen, Tabellenzeilen, Verdikte als reine Funktionen |
| `mkn_constants.py`, `mkn_presets.py`, `mkn_layout.py`, `mkn_visualization.py`, `mkn_texts.py` | Konstanten und Presets, Permalink und Schnellstart, fester Kopf, Abbildungen, Texte |
| `tools/mkn_make_cusum.py`, `tools/mkn_readme_tables.py` | gefrorene Tabelle erzeugen und prüfen; Tabellen der README erzeugen |
| `tests/` | Handbeispiele, drei Orakeldateien, Claims, Tabellen, Verdikte, Presets, Randfälle, Szenario, Zufallsquelle, Abbildungen, App, Layout, Footer, Laufzeit-Importe |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml`, `.streamlit/config.toml`, `.gitignore` | Laufzeit und Entwicklung, CI, Streamlit-Einstellungen |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
