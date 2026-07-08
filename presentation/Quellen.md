# CRSS — Quellen & Methodik (Quellenregister)

**Stand: 2026-06-29.** Kanonische Quelle für *jede* belastbare Zahl/Aussage im Pitch-Deck (PDF) **und** im Ideenpapier. Regel: in beiden Dokumenten identisch zitieren — Zahlen und Quellen dürfen nicht auseinanderlaufen.

**Quellengüte:** 🟢 Primär/offiziell · 🟡 Branchenrichtwert (Beratungs-Fee-Schedules, nicht amtlich) · 🔵 intern/methodisch.

---

## 1 · Marktzahlen
| Aussage (Folie) | Wert | Güte | Quelle / Fundstelle |
|---|---|---|---|
| EU-MedTech-Unternehmen (F5, F9) | 38.000; ~90 % KMU | 🟢 | MedTech Europe, *Facts & Figures 2025* — https://www.medtecheurope.org/about-the-industry/facts-figures/ |
| NRW-MedTech (F5, F9) | ~2.200 medizintechnische Unternehmen; >9 Mrd € Umsatz | 🟢 | **NRW.Global Business** (Landeswirtschaftsförderung NRW) — https://nrwglobalbusiness.com/de/zukunftsthemen/medtech · Zitat: „rund 2.200 medizintechnische Unternehmen erwirtschaften einen Umsatz von über neun Milliarden Euro". Definition: *medizintechnische Unternehmen* → Deck-Label „MedTech-Unternehmen/-Firmen" passt. |
| Abgleich NRW (Hersteller) | ~285 Hersteller; ~2,4 Mrd € | 🟢 | SPECTARIS, *Medizintechnik – Zahlen & Fakten 2024/25* — https://www.spectaris.de/ ·  (zeigt: Zahl 2.200 nur mit breiterer Definition haltbar) |

## 2 · Kosten & Engpass (Folie 2)
| Aussage | Wert | Güte | Quelle |
|---|---|---|---|
| MDR-Konformität Klasse IIb — **Gesamtkosten** | bis in den sechsstelligen Bereich (inkl. klin. Bewertung, QMS, Doku, Tests) | 🟡 | EuroDev — https://www.eurodev.com/blog/cost-of-obtaining-the-european-medical-device-regulations ; MedEnvoy — https://medenvoyglobal.com/blog/whats-the-cost-of-medical-device-approval-in-europe/ |
| — Benannte-Stelle-/Zertifizierungsgebühren | ~€25–80k | 🟡→🟢 | i3CGLOBAL — https://www.i3cglobal.com/notified-body-fee-mdr-ivdr/ ; veröffentlichte NB-Gebühren (Pflicht nach **MDR Art. 50** / MDCG 2022-14) — https://www.qualitiso.com/en/comparison-of-notified-body-fees/ |
| — Beratung (reines Consulting) | ~€10–20k | 🟡 | wie oben (i3CGLOBAL/MedEnvoy) |
| RA-Beratersätze | ~200–500 €/h (Senior/Spezialist) | 🟡 | MedEnvoy — https://medenvoyglobal.com/blog/how-much-do-medical-device-consultants-charge/ ; OMC Medical Fee Schedule — https://omcmedical.com/omc-medical-regulatory-consulting-hourly-rates-fee-schedule/ |
| Kontrast: angestellte/r RA-Spezialist/in DE | ~€28/h (Gehalt) | 🟢 | PayScale/Glassdoor DE — stützt Argument *„Beratung ist teuer & nicht skalierbar"* |
| Benannte Stellen = Kapazitätsengpass | qualitativ | 🟢 | MDCG 2022-14; NB-Kapazitäts-/Survey-Berichte (z. B. PMC11830701) |

## 3 · Regulatorik — Timing & Scope (Folien 2, 9)
| Aussage | Güte | Instrument / Fundstelle |
|---|---|---|
| MDR-Übergangsfristen (2027/2028) | 🟢 | VO (EU) 2017/745 (ELI: https://eur-lex.europa.eu/eli/reg/2017/745/oj ) + Änderungs-VO (EU) **2023/607** ( https://eur-lex.europa.eu/eli/reg/2023/607/oj ) |
| IVDR-Fristen | 🟢 | VO (EU) 2017/746 ( /eli/reg/2017/746/oj ) + Änderungs-VO (EU) **2024/1860** ( /eli/reg/2024/1860/oj ) |
| EUDAMED — gestufte Pflichtnutzung | 🟢 | VO (EU) **2024/1860** (s. o.) |
| KI-Verordnung | 🟢 | VO (EU) **2024/1689** ( https://eur-lex.europa.eu/eli/reg/2024/1689/oj ) |
| KI-Omnibus (Novellierung, „im Wandel") | 🟢 | **COM(2025) 836**; EP-Zustimmung 16.06.2026; polit. Einigung 07.05.2026 — EP Legislative Train: https://www.europarl.europa.eu/legislative-train/package-digital-package/file-digital-omnibus-on-ai ; Rat: https://www.consilium.europa.eu/en/press/press-releases/2026/05/07/artificial-intelligence-council-and-parliament-agree-to-simplify-and-streamline-rules/ |
| DSGVO | 🟢 | VO (EU) 2016/679 ( https://eur-lex.europa.eu/eli/reg/2016/679/oj ) |

## 4 · Produkt- & Methodik-Kennzahlen (Folien 3, 4)
| Aussage | Güte | Grundlage |
|---|---|---|
| „8,5/10 interne Eval (komplexe Fälle)" | 🔵 | LLM-as-Judge gegen 10-Punkte-Rubrik, **32 komplexe Fälle; Ø 8,53, Max 9,0** (`eval/quality_postnudge.json`, `eval/rubric_prompt.txt`). **Intern, nicht extern validiert** — externe RA-Experten-Validierung im Pilot. Nicht als externen Benchmark darstellen. |
| „7.000+ verknüpfte Rechtsnormen" | 🔵 | Neo4j-Graph: `:Provision` + `:Guidance` Knoten über MDR/IVDR/KI-VO/DSGVO + MDCG. Präziser: *verknüpfte Norm-/Leitlinienknoten* (nicht alle = „Rechtsnorm" i. e. S.). |
| „4 Regelwerke + MDCG" | 🔵🟢 | MDR 2017/745 · IVDR 2017/746 · KI-VO 2024/1689 · DSGVO 2016/679 · + MDCG-Leitlinien. (CIR 2026/977 liegt vor, aber Retrieval-Anbindung separat zu prüfen.) |

---

## 5 · Preise (kanonisch — Stand 2026-06-29)
Identisch in **Deck (Folie 6)** und **Ideenpapier (§5)** halten.
| Tier | Preis |
|---|---|
| Einzel | 150–250 €/Monat |
| Teams | 800–1.500 €/Monat |
| Enterprise | 8.000–15.000 €/Jahr |

## 6 · Namenskonvention
**Lexorix** = Unternehmen · **CRSS** (Compliance Readiness Support System) = Produkt. In beiden Dokumenten konsistent; Deck-Titelfolie sollte die Beziehung einmal explizit nennen.

### Offene Punkte
1. ✅ **NRW-Zahl 2.200 / >9 Mrd €** — geklärt: NRW.Global Business (offiziell), Definition „medizintechnische Unternehmen". Bei Rückfrage zur Differenz: SPECTARIS zählt enger nur ~285 *Hersteller*.
2. Kosten-/Satz-Richtwerte sind 🟡 **Branchenschätzungen** (Beratungsblogs/Fee-Schedules) — im Ideenpapier als „branchenübliche Richtwerte" kennzeichnen, nicht als amtliche Statistik.