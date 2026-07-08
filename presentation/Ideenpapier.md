# Ideenpapier — Lexorix / CRSS (korrigiert, Stand 2026-06-29)

*Konsistent mit Pitch-Deck und `Quellen.md`. Terminologie: KI-Verordnung (nicht KI-Gesetz); NRW-Zahl 2.200 (NRW.Global Business); Pricing kanonisch.*

## 1. Kurzzusammenfassung
Unternehmen, die KI-Systeme in regulierten Märkten entwickeln oder einsetzen, müssen die KI-Verordnung (EU) 2024/1689 parallel zu sektorspezifischen Vorschriften wie der MDR 2017/745 für Medizinprodukte anwenden. Diese regulatorische Überlappung betrifft gemäß Anhang I nahezu alle EU-Schlüsselindustrien.

Lexorix entwickelt das Compliance Readiness Support System (CRSS), eine KI-gestützte Regulatory-Intelligence-Plattform auf Basis eines vorschriftenbasierten Wissensgraphen. Das System liefert zitierfähige, quellenverankerte Analysen und unterstützt bei der Identifikation regulatorischer Anforderungen; perspektivisch auch potenzieller Lücken in technischen Dokumenten.

Der Start erfolgt in der Medizintechnik als stark reguliertem Kernmarkt. Durch Erweiterung um weitere EU-Harmonisierungsrechtsvorschriften aus Anhang I ist die Plattform ohne grundlegende Architekturänderungen auf weitere regulierte Branchen skalierbar.

## 2. Geschäftsidee
**1) Problem/Bedarf:** Regulierte Unternehmen müssen die KI-Verordnung und sektorspezifische Regelwerke parallel anwenden; Überschneidungen führen zu hohem manuellen Analyseaufwand, einfache Compliance-Fragen benötigen oft mehrere Stunden Expertenarbeit.

**2) Lösung:** Das von Lexorix entwickelte Compliance Readiness Support System (CRSS) verarbeitet EU-Regelwerke als maschinenlesbaren Wissensgraphen und verknüpft regulatorische Anforderungen automatisiert über Dokumentgrenzen hinweg; das System liefert durchgängig quellenverankerte, rückverfolgbare Analysen und unterstützt bei der Identifikation von Anforderungen und Zusammenhängen; perspektivisch auch potenzieller Lücken; die Architektur ist sektorübergreifend ausgelegt und auf weitere EU-Regelwerke erweiterbar.

**3) Alleinstellungsmerkmal:** Kombination aus EU-weitem Regulierungs-Ingestionssystem, vernetztem Wissensgraphen und durchgängig quellenverankerter, strukturierter Compliance-Analyse.

**4) Innovationsgrad:** Hoch; neuartige Anwendung von GraphRAG auf den EU-Regulierungsraum als sektoragnostische Plattform; Markteintritt über Medizintechnik-Startups (MDR/IVDR/GDPR) aufgrund hoher regulatorischer Komplexität, Architektur von Beginn an skalierbar auf weitere EU-regulierte Branchen.

## 3. Adressierter Markt, Branche und Wettbewerbssituation
**1) Markt:** Primärziel sind Regulatory-Affairs-Teams, MedTech-Startups und KMU in NRW (>1.000 von rund 2.200 Medizintechnik-Unternehmen, NRW.Global Business), insbesondere im Umfeld von TÜV Rheinland, RWTH Aachen und Uniklinik Köln. Mittelfristig Expansion in den DACH-Raum sowie alle regulierten Branchen gemäß Anhang I der KI-Verordnung (z. B. Medizintechnik, Maschinenbau, Aufzüge, PSA).

**2) Partnerinnen/Partner:** TÜV Rheinland, BVMed / MedTech NRW, RWTH Aachen, Uniklinik Köln, Fraunhofer IAIS, TU Dortmund, TOPRA sowie spezialisierte MedTech-Kanzleien in Köln und Düsseldorf.

**3) Wettbewerb:** Allgemeine KI-Tools (z. B. ChatGPT, Copilot), juristische Datenbanken (EUR-Lex), QMS-Systeme (z. B. Greenlight Guru) sowie redaktionelle Compliance-Plattformen (z. B. Johner Institut).

## 4. Kundennutzen und Bedarf
**1)** Lexorix richtet sich an Unternehmen, die KI-Systeme in regulierten Märkten entwickeln und die KI-Verordnung sowie sektorspezifische Regelwerke (MDR/IVDR/GDPR) parallel anwenden müssen. Primäre Zielgruppe sind MedTech-Startups, KMU und SaMD-Hersteller mit hoher regulatorischer Komplexität. Darüber hinaus adressiert die Plattform Regulatory-Affairs-Berater als Multiplikatoren sowie interne Compliance-Teams — perspektivisch in allen Branchen unter Anhang I der KI-Verordnung.

**2)** Der zentrale Nutzen liegt in der drastischen Reduktion regulatorischer Recherchezeit: Aufgaben, die heute Stunden dauern, werden durch KI-gestützte Auswertung auf Minuten verkürzt — als zitierfähige Antworten auf Basis von EU-Rechtstexten. Alle Ergebnisse sind direkt auf Gesetzes- und Richtlinientexte rückführbar, was das Risiko fehlerhafter KI-Ausgaben minimiert; die Ergebnisse dienen der Entscheidungsunterstützung, nicht als Rechtsberatung. Die Plattform skaliert mit neuen Regelwerken und Sektoren ohne grundlegende Anpassungen.

## 5. Machbarkeit und Perspektive der Gründungsidee
**1) Entwicklungsschritte:** Der Kern der Plattform (Wissensgraph, Retrieval-System und KI-Assistent für MDR, IVDR, GDPR und KI-Verordnung) ist als funktionierender Prototyp vorhanden; bis zur Marktreife folgen Ausbau zu einem Dokumentanalyse-Modul für technische Dossiers mit Lücken- und Konsistenzprüfung, Entwicklung einer mandantenfähigen SaaS-Infrastruktur mit DSGVO-konformer Verarbeitung, Erweiterung der Wissensbasis um Leitlinien und Normen sowie Aufbau eines Regulatory-Affairs-Teams zur fachlichen Validierung und Markteinführung.

**2) Geld verdienen:** SaaS-Modell mit gestaffelten Lizenzen (Einzel 150–250 €/Monat, Teams 800–1.500 €/Monat, Enterprise 8.000–15.000 €/Jahr) inkl. Dokumentanalyse und SLA; vor dem SaaS-Launch Monetarisierung über Pilot- und Validierungssessions mit realen Anwendungsfällen.

**3) Risiken:** Hohe regulatorische Komplexität (adressiert durch striktes Quellengrounding), langsamere Adoption in stark regulierten Industrien durch bestehende Nutzung generischer KI-Tools in frühen Workflows, die den Bedarf an auditierbarer, nachvollziehbarer Systemunterstützung zunächst verdecken können, sowie technische Skalierungsanforderungen bei Erweiterung auf komplexe Dokumenttypen und zusätzliche Standards.

## 6. Gründungspersönlichkeit oder Gründungsteam
**1) Ausbildung, Erfahrungen und Kompetenzen:** Diego Barra (Lead) vereint Systemkompetenz, Erfahrung in regulierten biomedizinischen Kontexten (M.Sc. Bionik & Biomimetik, Hochschule Rhein-Waal) und treibt das Projekt in Vollzeit voran. Raman Sheshka (PhD, Co-Founder) bringt als Senior Data Scientist (M.Sc. Theoretische Physik, Dr. rer. nat. Mechanik) tiefgehende wissenschaftliche und praktische Kernexpertise ein. Er begleitet die Core-Tech-Entwicklung (Wissensgraphen/RAG) parallel in einem festen Teilzeitrahmen. (Keine Stipendienbeantragung für Raman aufgrund von Wohnsitz.)

**2) Realisierung der Gründungsidee:** Als ehemaliger ML Research Engineer bei Vibraspex (MedTech-Startup für Brustkrebsdiagnostik) bringt Diego praktische Erfahrung in der Entwicklung von Machine-Learning-Lösungen auf Basis pseudonymisierter biomedizinischer Daten im MedTech-Umfeld ein. Diese Erfahrung fließt in die nutzerorientierte Gestaltung und technische Umsetzung der Lösung ein.

**3) Weitere Unterstützerinnen und Unterstützer:** Das Team nutzt das Netzwerk des lokalen Kölner Startup-Ökosystems und steht im Austausch mit Branchenexperten zur kontinuierlichen Validierung.

## Gründungsstipendium
**1)** Der technische Prototyp von Lexorix ist vorhanden und demonstrierbar. Nächster Schritt ist die Marktvalidierung und Überführung in ein produktives SaaS-System durch Pilotanwendungen mit MedTech- und Regulatory-Affairs-Teams im NRW-Umfeld.

**2)** Das Stipendium ermöglicht volle Fokussierung auf diese Phase durch Absicherung des Lebensunterhalts sowie Priorisierung von Produktentwicklung, Kundenvalidierung und Partneraufbau — mit realen Anwendungsfällen aus technischen Dossiers und gezieltem Kooperationsaufbau.