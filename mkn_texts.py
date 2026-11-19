"""Texte der App (Einleitung, Fragen, Negativbefunde, Grenzen, Formeln). Jede Zahl in diesen Texten steht in der README und in tests/test_claims.py.

Begriffe: ν = Zahl der Vektoren vor der Änderung; „Fehlalarm“ = Alarm vor der Änderung, je Zeit gezählt (ARL₀), nicht Anteil falscher Alarme; „Kanal“ = eine Messreihe,
nicht ein Lkw-Tor; „Hotelling“ kommt nur als Hotelling-T² vor."""

INTRO = """
Mehrere Messreihen („**Kanäle**“) laufen gleichzeitig, und es gibt **einen** gemeinsamen Alarm. Die Zahl, die eine Karte verspricht (im Mittel alle 370,4 Vektoren ein Fehlalarm, die
**ARL₀**), soll für den gemeinsamen Alarm gelten, nicht für jeden Kanal einzeln. Das kostet **Verzögerung**: wie viel, hängt an der Zahl der Kanäle, an der Karte (mit oder ohne
**Gedächtnis**), an der **Korrelation** zwischen den Kanälen und daran, **wie viele Kanäle springen**. Gezählt werden Beobachtungsvektoren bis zum ersten Alarm; die Kanäle sind
ausgedacht (standardnormal, Mittel und Kovarianz bekannt), ein Nutzen in Logistik, Lager oder Fertigung ist nicht belegt.

Stück 6 der Reihe „Änderungserkennung“; Parameter und Kovarianz sind hier bekannt, wie sie aus Phase I geschätzt werden, steht in der Demo `anfangsparameter-abhaengig-demo`.
Weiter unten: fünf Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Das Modell.** Je Zeitpunkt ein Vektor aus p Kanälen, vor der Änderung standardnormal (Einheitsvarianz, Mittel 0); ab Vektor ν + 1 haben m Kanäle den Mittelwert d (in σ).
2. **Karten ohne Gedächtnis** entscheiden mit dem aktuellen Vektor allein: **T²** (Abstand des Vektors vom Nullpunkt, in Einheiten der Kovarianz), **getrennte Karten** (Alarm,
   sobald irgendein Kanal die Schranke z überschreitet; z nach Šidák) und die **Summenkarte** (summiert die Kanäle, setzt voraus, dass die Richtung bekannt ist).
3. **Karten mit Gedächtnis** sammeln Abweichungen: **CUSUM** und **EWMA** je Kanal (gemeinsamer Alarm beim ersten Kanalalarm) und die **MEWMA** (ein Vektor-EWMA).
4. **Gleiche Fehlalarmrate für alle.** Jede Schranke ist so eingestellt, dass die gemeinsame ARL₀ 370,398 ist (das Ziel ist 370,398, nicht das gesetzte 370). Erst dann sind
   die Verzögerungen (ARL₁: Vektoren bis zum Alarm nach dem Sprung) vergleichbar.
5. **Woher die Zahlen kommen.** Karten ohne Gedächtnis: exakt (nichtzentrale χ²-Verteilung, Integrale). CUSUM und EWMA: Markov-Kette je Kanal, die gemeinsame Lauflänge ist das Minimum
   der Kanal-Lauflängen (Gitterwerte mit Richardson-Extrapolation, auf etwa 0,4 % genau, gefroren in `mkn_cusum_frozen.json`). MEWMA-ARL₁: Simulation mit Standardfehler.
"""

CONVENTIONS_MARKDOWN = """
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
"""

# ------------------------------------------------------------------------------------------------------------------ Experimente
EXP1_FRAGE = """
**Frage: Wann schlägt welche Karte bei einem Sprung an?** Ein Verlauf aus 400 Vektoren; die Karte, die Zahl der Kanäle, der Sprung und der Zeitpunkt der Änderung stehen links.
Der Verlauf ist **eine Stichprobe**: die exakte ARL₁ daneben ist der Mittelwert über unendlich viele Verläufe (Änderung ab Vektor 1).
"""
EXP1_HINWEIS = ("Die exakte ARL₁ gilt für den Sprung ab Vektor 1. Bei ν > 0 haben CUSUM und EWMA beim Sprung schon Gedächtnis aus der Zeit davor; ihre Verzögerung im Verlauf ist dann eine "
                "Stichprobe aus einem anderen Maß (steady-state), das hier nur für den CUSUM bei p = 1 und 5 gerechnet ist.")

EXP2_FRAGE = """
**Frage: Was kostet jeder weitere Kanal?** Mit mehr Kanälen steigt die Zahl der Fehlalarme; soll die **gemeinsame** ARL₀ 370,4 bleiben, muss jede Karte je Kanal vorsichtiger werden
(die ARL₀ je Kanal steigt von 370,4 bei p = 1 auf 7 398 bei p = 20), und das kostet Verzögerung. Ein Kanal springt um d, die übrigen nicht.
"""
EXP2_NEGATIV = ("**Negativbefund.** „T² ist nicht besser als p getrennte Karten“ gilt **nur bei unabhängigen Kanälen und einem springenden Kanal**, und auch dort nur in 12 von 16 gemessenen Zellen: "
                "bei d = 0,5 ist T² etwas schneller, bei d = 1 nur wenig langsamer. Bei korrelierten Kanälen und anderer Sprungrichtung dreht sich das um (Experiment 3).")

EXP3_FRAGE = """
**Frage: Was ändert die Korrelation, und in welche Richtung springt es?** T² vergleicht den Sprung mit der Form der Korrelation; getrennte Karten sehen jeden Kanal für sich.
„entlang“ heißt: die Kanäle bewegen sich gemeinsam (1, 1, …), wie sie ohnehin schwanken; „gegen“: die ersten beiden Kanäle gegeneinander (1, −1); „ein Kanal“: nur Kanal 1; „schief“: (1, −0,3).
Die getrennten Karten stehen hier **neu auf ARL₀ 370,4 geeicht** (die Šidák-Schranke ist bei Korrelation konservativ) und daneben mit der Šidák-Schranke.
"""
EXP3_FALLE = ("**Falle des Stücks: Zahl der Kanäle und Richtung entscheiden, nicht die Karte allein.** T² gewinnt nicht allgemein: gegen die Korrelationsstruktur und in einem Kanal bei hoher Korrelation "
              "ist es sehr schnell, **entlang** der Korrelation verliert es gegen getrennte Karten (p = 2, d = 1: ab ρ ≈ 0,24). Der Satz „T² ist besser bei Korrelation“ gilt nur mit Richtung.")

EXP4_FRAGE = """
**Frage: Wie viele Kanäle springen?** Zwei Vergleiche: **gleiche Höhe** (jeder springende Kanal um 1 σ) und **gleiche Energie** (die Gesamtstärke λ = m·d² = 4 wird auf m Kanäle verteilt).
Die **Summenkarte** kennt die Richtung (alle springenden Kanäle gleichsinnig); getrennte Karten und T² kennen sie nicht.
"""
EXP4_NEGATIV = ("**Negativbefund.** Unter den vier gemessenen Karten ist keine für jede Verteilung des Sprungs auf die Kanäle das schnellste: bei gleicher Energie gilt: je mehr Kanäle den Sprung teilen, desto besser die Summenkarte und desto schlechter "
                "getrennte Shewhart-Karten; bei einem springenden Kanal ist die Summenkarte die langsamste der vier (gemessen für p = 5 und p = 10). Die Summe von CUSUMs und Mischungsverfahren (Xie und Siegmund) sind nicht gemessen.")

EXP5_FRAGE = """
**Frage: Was bringt das Gedächtnis?** Ein Kanal springt um d. Zero-state heißt: Start im Nullzustand, Änderung ab Vektor 1. MEWMA-ARL₁ ist eine **Simulation** (gelb, mit Standardfehler); der
Live-Lauf unten rechnet sie mit den Reglern neu.
"""
EXP5_NEGATIV = ("**Negativbefund.** Das Gedächtnis hilft nicht immer: bei einem einzigen Kanal und großem Sprung (p = 1, d = 3) ist die Shewhart-Karte schneller als CUSUM und EWMA. "
                "Und: diese Zahlen gelten für unabhängige Kanäle; mit Korrelation sind Karten mit Gedächtnis nicht gemessen.")

# ------------------------------------------------------------------------------------------------------------------ Grenzen
ASSUMPTIONS_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Mittel und Kovarianz sind bekannt** | Die Schranken kennen sie genau. Mit aus Phase I geschätztem Mittel und Σ̂ ändert sich die ARL₀; hier nicht gemessen. | `anfangsparameter-abhaengig-demo` (Stück 4b) |
| **Die Kanäle sind unabhängig** (Experiment 2, 4, 5) | Bei Korrelation sind nur T² und getrennte Shewhart-Karten gerechnet (Experiment 3: p = 2 mit jedem ρ bis ±0,95, p = 5 gleichkorreliert mit ρ ≥ 0). **Karten mit Gedächtnis bei korrelierten Kanälen sind nicht gemessen**; der Produktweg gilt nur für unabhängige Kanäle. | – |
| **Die Werte sind in der Zeit unabhängig** | Autokorrelation je Kanal ändert die ARL₀ aller Karten; hier nicht wieder aufgenommen. | Stück 5 der Reihe |
| **Mittelwertsprung ab Vektor 1** | Sprung in Streuung oder Korrelation, Ausreißer, fehlende Werte, ungleiche Abtastzeiten, p > 20 und d > 3 sind nicht gemessen. | – |
| **Maß: zero-state** | Steady-state ist nur für den CUSUM bei p = 1 und 5 gerechnet; Lorden- und Pollak-Maß für CUSUM und EWMA sind nicht gemessen. | – |
| **Welche Richtung?** | Die Summenkarte setzt „alle gleichsinnig“ voraus. Die Summe von CUSUMs und Mischungsverfahren sind nicht gemessen. | – |
| **Welcher Kanal ist gesprungen?** | Getrennte Karten nennen den alarmierenden Kanal, T² nicht; eine Trefferquote dieser Diagnose ist nicht gemessen. | – |
| **Ausgedachte Kanäle** | Es gibt in den gelesenen Quellen keine reale Mehrkanal-Datenquelle für diesen Zweck. Was die Zahlen für einen echten Betrieb heißen, wird nicht behauptet; ein Nutzen ist nicht belegt. | – |
"""

# ------------------------------------------------------------------------------------------------------------------ Formeln
MATH_INTRO = """
**Gegeben und gesucht.** Gegeben: Vektoren $x_1, x_2, \\dots \\in \\mathbb{R}^p$. Gesucht: bei gemeinsamer ARL₀ von 370,398 die Zahl der Vektoren bis zum ersten Alarm nach einem Sprung (ARL₁).
"""

MATH_MODEL = r"""
**Modell.** Die Vektoren sind unabhängig, $x_t \sim N(\mu_t, \Sigma)$ mit Einheitsvarianzen $\Sigma_{ii} = 1$ (Experiment 2, 4, 5: $\Sigma = I$; Experiment 3: Korrelation $\rho$). Bis $t = \nu$ gilt
$\mu_t = 0$, danach $\mu_t = \mu$. Die Karte kennt $\Sigma$ und das Mittel vor der Änderung, nicht $\nu$ und nicht $\mu$. Die gemeinsame ARL₀ ist das Ziel:
"""
MATH_TARGET = r"$$\mathrm{ARL}_0 = \frac{1}{2\,(1-\Phi(3))} = 370{,}398$$"
MATH_T2_1 = r"**T².** Alarm, wenn $x_t^{\top}\Sigma^{-1}x_t > c_p$ mit dem $\chi^2$-Quantil zur Alarmwahrscheinlichkeit $1/370{,}398$ je Vektor; nach dem Sprung ist die Statistik nichtzentral $\chi^2$-verteilt mit Nichtzentralität $\lambda = \mu^{\top}\Sigma^{-1}\mu$:"
MATH_T2_2 = r"$$\mathrm{ARL}_1 = \frac{1}{P\!\left(\chi'^{2}_{p}(\lambda) > c_p\right)}$$"
MATH_SEP_1 = r"**Getrennte Karten.** Alarm, wenn $\max_i |x_{t,i}| > z_p$ mit $z_p = \Phi^{-1}(1 - q_p/2)$ und $q_p = 1 - (1 - 1/370{,}398)^{1/p}$ (Šidák). Springen $m$ von $p$ unabhängigen Kanälen um $d$, ist die Alarmwahrscheinlichkeit je Vektor"
MATH_SEP_2 = r"$$P_{\mathrm{Alarm}} = 1 - \bigl(1 - P(|N(d,1)| > z_p)\bigr)^{m}\,(1 - q_p)^{p-m}$$"
MATH_SUM = r"**Summenkarte.** $s_t = \sum_i x_{t,i}/\sqrt{p}$, Alarm bei $|s_t| > 3$; bei $m$ springenden Kanälen ist $s_t \sim N(m d/\sqrt{p},\,1)$."
MATH_GEO_1 = r"**Lauflänge ohne Gedächtnis.** Der Alarm hängt nur vom aktuellen Vektor ab, die Lauflänge $\mathrm{RL}$ ist geometrisch mit der Alarmwahrscheinlichkeit $P$ je Vektor:"
MATH_GEO_2 = r"$$P(\mathrm{RL} > n) = (1-P)^n,\qquad \mathrm{ARL} = 1/P$$"
MATH_MEM_1 = r"**CUSUM und EWMA je Kanal** ($k = 0{,}5$, $\lambda = 0{,}1$, Start $S^{\pm}_0 = 0$, $Z_0 = 0$); der CUSUM alarmiert, wenn $\max(S^+_t, S^-_t) \ge h$, der EWMA, wenn $|Z_t| > H$:"
MATH_CUSUM = r"$$S^{+}_t = \max(0,\, S^{+}_{t-1} + x_t - k),\qquad S^{-}_t = \max(0,\, S^{-}_{t-1} - x_t - k)$$"
MATH_EWMA = r"$$Z_t = \lambda\, x_t + (1-\lambda)\, Z_{t-1}$$"
MATH_MEWMA = r"**MEWMA.** $Z_t = \lambda x_t + (1-\lambda) Z_{t-1}$ mit Vektoren $Z_t$; Alarm, wenn $\frac{2-\lambda}{\lambda}\,|Z_t|^2 > H$ (Kovarianz von $Z$ asymptotisch)."
MATH_MIN_1 = r"**Minimum unabhängiger Lauflängen.** Bei unabhängigen Kanälen ist die gemeinsame Lauflänge das Minimum der Kanal-Lauflängen; die Überlebensfunktion eines Kanals steht in einer Markov-Kette mit Übergangsmatrix $Q$ der Zustände ohne Alarm (Startzustand $e_0$):"
MATH_MIN_2 = r"$$P(\mathrm{RL} > n) = \prod_{i=1}^{p} P(\mathrm{RL}_i > n),\qquad P(\mathrm{RL}_i > n) = e_0^{\top} Q_i^{\,n}\,\mathbf{1},\qquad \mathrm{ARL} = \sum_{n \ge 0} \prod_{i=1}^{p} P(\mathrm{RL}_i > n)$$"
MATH_RICH = r"Der Gitterfehler der Kette fällt wie $1/m^2$; die Zahlen sind $(m_2^2\,a_2 - m_1^2\,a_1)/(m_2^2 - m_1^2)$ aus zwei Gittern $m_1 < m_2$ (Richardson)."
MATH_CORR_1 = r"**Korrelation.** T² mit $\Sigma^{-1}$ hat für jedes $\Sigma$ die ARL₀ 370,398 (mit $x = L\varepsilon$, $\Sigma = LL^{\top}$, ist $x^{\top}\Sigma^{-1}x = \varepsilon^{\top}\varepsilon$) und eine ARL₁, die nur an $\lambda$ hängt. Die getrennten Karten brauchen die Wahrscheinlichkeit $P(|x_i| \le z\ \forall i)$ unter $N(\mu, \Sigma)$; für $p = 2$ ist sie mit $c = \mu_2 + \rho\,(x - \mu_1)$ ein eindimensionales Integral:"
MATH_CORR_2 = r"$$P(|x_1| \le z,\, |x_2| \le z) = \int_{-z}^{z} \varphi(x - \mu_1)\left[\Phi\!\left(\frac{z - c}{\sqrt{1-\rho^2}}\right) - \Phi\!\left(\frac{-z - c}{\sqrt{1-\rho^2}}\right)\right] dx$$"
MATH_CORR_3 = r"Für $p = 5$ gleichkorreliert ($\rho \ge 0$) gilt $x_i = \mu_i + \sqrt{\rho}\,f + \sqrt{1-\rho}\,e_i$ mit unabhängigen $f, e_i \sim N(0,1)$; das Produkt über die Kanäle wird bei festem $f$ gebildet und über $f$ mit Gauß-Hermite integriert."
MATH_LIT = """
**Literatur** (Stufe: W = gelesen, S = nur Suchtreffer, sonst nicht gelesen). **Xie und Siegmund**, Annals of Statistics 41 (2) (2013), arXiv 1207.2386 (**W**, Abstract und Abschnitte 1 und 5): bei Änderung in einer
unbekannten Teilmenge der Ströme lassen Summenregeln das Rauschen der unbetroffenen Ströme in die Statistik, die Verzögerung wird lang; hier nur qualitativ benutzt. **NIST pmc543** (**W**): T²-Statistik und F-Verteilung gelten für den
Test, nicht für Kartenlauflängen; die Grenzen der Karte folgen aus der Verteilungstheorie (hier: bekanntes Σ, also χ²). **Vanhatalo und Kulahci**, Quality and Reliability Engineering International 31 (2015) 1779–1796 (**W**): Autokorrelation und T² (Stück 5). **Lowry u. a. 1992** (MEWMA): **S**,
keine Zahl daraus. **Nicht gelesen** (keine Aussage stützt sich darauf): Šidák 1967 (die Ungleichung wird hier nur an Zellen gezählt), Bersimis, Psarakis und Panaretos 2007, Montgomery, Mei, Hotelling 1947. Alles übrige sind eigene
Nachrechnungen in diesem Modell; neu ist die Vorführbarkeit mit Zähleinheit, exakten Orakeln und der Messung der Richtung, nicht die Verfahren.
"""
