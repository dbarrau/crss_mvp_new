# Lexorix / CRSS — Verified Evidence Base

*Verification pass: 7 July 2026. Status key: ✅ confirmed · ⚠️ usable with caveat · ✍️ verify exact wording before public use*

---

## 1. The dual compliance burden is confirmed — and now permanent

**✅ MedTech Europe (7 May 2026).** After the Digital Omnibus agreement:
> "We had advocated clearly and consistently for a single, sector-specific compliance pathway, whereby high-risk AI requirements for medical technologies are implemented through the existing MDR and IVDR, rather than through parallel obligations under both frameworks."

- Source: [MedTech Europe — "AI Act deal lands"](https://www.medtecheurope.org/2026/05/07/joint-industry-voice-calls-for-one-coherent-framework-for-ai-enabled-medical-technologies/)
- Corroboration: [Council of the EU press release, 7 May 2026](https://www.consilium.europa.eu/en/press/press-releases/2026/05/07/artificial-intelligence-council-and-parliament-agree-to-simplify-and-streamline-rules/)
- **Decisive fact:** industrial/machinery AI won the double-conformity exemption; **medical devices did not.** The Council's parallel-obligations position prevailed.

**✅ European Commission / Petrie-Flom, Harvard (corrected characterization).**
The Commission's **December 2025** proposal (DG SANTE-led) went further than "making dual compliance workable" — it proposed to **remove AI medical devices from the AI Act's high-risk (HRAIS) requirements**. The May 2026 deal **did not adopt that**, so the dual burden stands.

- Source: [Petrie-Flom Center — "Simplification or Back to Square One?" (5 Mar 2026)](https://petrieflom.law.harvard.edu/2026/03/05/simplification-or-back-to-square-one-the-future-of-eu-medical-ai-regulation/)
- Note: prior draft mischaracterized this proposal — corrected here.

**⚠️ RAPS framing (secondary attribution).**
Experts describe the AI Act and MDR as an *"arranged marriage"* and *"conjoined twins."* The phrases appear in an academic paper *citing* RAPS, not a RAPS publication. Use as color, not as a hard source.

- Source: [arXiv 2406.08695 — "Global AI Governance in Healthcare"](https://arxiv.org/html/2406.08695v1)

---

## 2. Compliance teams are structurally overwhelmed

**✍️ Reed Smith (June 2025).** Advises MedTech firms to
> "organize a cross-functional team consisting of legal, regulatory affairs, quality, engineering, privacy"

and to
> "perform a GAP analysis of current MDAI compliance under MDR/IVDR versus AI Act requirements."

- Source: Reed Smith — "The EU AI Act and Medical Devices: Navigating High-Risk Compliance" — [reedsmith.com](https://www.reedsmith.com/)
- Attribution fix: source is Reed Smith (original draft mis-tagged "Spellbook"). Re-confirm exact wording on the page.

**✍️ Johner Institute (Oct 2025).**
> "These changes require resources, money, trained personnel, and cross-departmental planning. Therefore, the executives in charge should set up a change project and not just assign regulatory affairs with a 'your job.'"

- Source: Johner Institute — "What the AI Act means for Medical Device and IVD Manufacturers" — [blog.johner-institute.com](https://www.johner-institut.de/blog/)
- Attribution fix: source is Johner (original draft mis-tagged "arXiv"). Re-confirm exact wording.

**✍️ IntuitionLabs (July 2026).** New hybrid roles emerging:
> "We foresee growth in 'AI Quality Assurance managers,' 'AI ethics officers', and other hybrid governance positions within pharma and medtech firms. Specialist consultancies in AI regulatory affairs are emerging."

- Source: [IntuitionLabs — "EU AI Act High-Risk Compliance: Pharma & Medical Devices"](https://intuitionlabs.ai/articles/eu-ai-act-pharma-medical-device-compliance)
- Attribution fix: source is IntuitionLabs (original draft mis-tagged "RAPS").

---

## 3. General-purpose LLMs are adopted — but not trustworthy on legal citation

**✅ Moody's (2025).** AI adoption in risk & compliance rose from 9% → 24% among active users (up to 53% incl. pilots), yet only **30% report significant impact** and **34% measure success** — a clear implementation gap.

- Source: [Moody's — "From reactive to proactive: How AI is transforming risk and compliance"](https://www.moodys.com/web/en/us/site-assets/ma-kyc-from-reactive-to-proactive-how-ai-is-transforming-risk-and-compliance.pdf)
- Caveat: finance-leaning risk & compliance population, not MedTech-specific.

**✅ Stanford RegLab — headline reliability stat.** General LLMs hallucinate on **58%–88%** of specific legal queries (GPT-4 58%, GPT-3.5 69%, Llama 2 88%). The legal/regulatory domain is *structurally* the worst for fabrication.

- Source: [Dahl, Magesh et al., "Large Legal Fictions," *Journal of Legal Analysis* 16(1), 2024](https://academic.oup.com/jla/article/16/1/64/7699227)
- Summary: [Stanford Law — "Hallucinating Law"](https://law.stanford.edu/2024/01/11/hallucinating-law-legal-mistakes-with-large-language-models-are-pervasive/)

**✅ Even purpose-built legal RAG tools still hallucinate.** Commercial AI legal-research tools (Lexis+ AI, Westlaw) **still hallucinate 17–33%** of the time — RAG grounding *reduces* but does not eliminate fabrication.

- Source: [Magesh et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools," *J. Empirical Legal Studies*, 2025](https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413)
- **Sharpest wedge:** even the best legal RAG tools fabricate — which is why CRSS *verifies every quote against source law* rather than trusting the model.

**🔄 EQS AI Benchmark (use carefully — flipped from original).** The old "stringing tasks together is still iffy" line is **outdated**. Volume 2 (May 2026), built with the Berufsverband der Compliance Manager (BCM), finds frontier models now handle agentic compliance workflows well (GPT-5.4 87.6%, Claude Opus 4.6 86.1%).

- Source: [EQS AI Benchmark — AI Performance in Compliance & Ethics](https://www.eqs.com/compliance-wpapers/ai-performance-compliance-ethics-eqs/)
- **Correct framing:** models are strong on *discrete* compliance tasks; the unsolved problem is **grounding answers to the actual legal text** — not raw capability.

**❌ DROPPED — TrustArc "53% manual / 62% behind."** Exact pairing could not be verified against the [primary report](https://trustarc.com/wp-content/uploads/2025/06/2025-trustarc-global-privacy-benchmarks-report.pdf). Pull exact numbers from source or omit.

**❌ DROPPED — generic "3%–27% hallucination."** Replaced by the Stanford RegLab figures above (domain-specific and authoritative).

---

## 4. The market is moving — demand is validated

**✅ RegASK (July 2025)** launched "the industry's first agentic AI architecture for regulatory affairs" (vertical LLM + specialized retrieval/translation/assessment agents).

**✅ Clarivate (Aug 2025)** launched the AI-powered Regulatory Assistant within Cortellis Regulatory Intelligence.

- Source: [Clarivate press release](https://clarivate.com/news/clarivate-presents-cortellis-regulatory-ai-assistant/)
- Market context: [Grand View Research — AI in Regulatory Affairs Market](https://www.grandviewresearch.com/industry-analysis/artificial-intelligence-ai-regulatory-affairs-market-report)
- Framing: these are general-purpose assistants — none built on the "verify-against-source-law" principle, and none EU-sovereign. That is the Lexorix gap.

---

## Loose ends to close before public use

1. Re-confirm exact wording of the three ✍️ consultancy quotes (Reed Smith, Johner, IntuitionLabs) on their own pages — original source tags were scrambled.
2. If reusing TrustArc, pull exact figures from the primary PDF.
