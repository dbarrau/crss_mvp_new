---
marp: true
theme: gaia
class: invert
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Helvetica, sans-serif;
    font-size: 1.1rem;
  }
  section.lead h1 {
    font-size: 2.4rem;
  }
  section.lead p {
    font-size: 1.1rem;
    opacity: 0.85;
  }
  h2 { color: #7ecfff; }
  strong { color: #ffffff; }
  table { font-size: 0.85rem; }
  code { font-size: 0.8rem; }
  blockquote {
    border-left: 4px solid #7ecfff;
    padding-left: 1rem;
    font-style: italic;
    opacity: 0.9;
  }
---

<!-- _class: lead -->

# CRSS
## Compliance Readiness Support System

AI-powered regulatory intelligence for medtech startups navigating EU law

**Diego Barra** · Gründungstipendium NRW · June 2026

---

## The Problem

EU medical device companies face **3+ major overlapping regulations**:

| Regulation | Applies to | In force since |
|---|---|---|
| MDR 2017/745 | All medical devices | May 2021 / 2024 |
| IVDR 2017/746 | In vitro diagnostics | May 2022 / 2025 |
| EU AI Act 2024/1689 | AI-enabled devices | Aug 2026 → |
| GDPR 2016/679 | Patient data processing | May 2018 |

> A startup building an AI-powered diagnostic tool must simultaneously comply with **all four** — each referencing the others.

---

## The Problem I Lived

I didn't start with regulations. I started with a cancer detection model.

At a medtech startup building AI for **early breast cancer detection** using Raman Spectroscopy, I found a critical flaw in our model validation pipeline:

- Patient samples were present in **both training and test sets**
- Model accuracy was **artificially inflated** — severe overfitting disguised as performance
- I raised it internally. It wasn't resolved. I left.

Then I investigated the literature — the same error was **widespread across published Raman Spectroscopy cancer detection studies**. Medical AI reporting guidelines existed. They were being ignored.

> **When these research-stage models hit EU AI Act conformity assessment: the gap between how they were validated and what the law requires will be enormous. That is the market.**

---

## Why This Is Getting Worse

- **MDR/IVDR transition** is still ongoing — thousands of legacy devices still seeking CE marks under stricter rules
- **EU AI Act** obligations for high-risk AI (medical devices = automatic high-risk) take effect **August 2026**
- **Notified Body bottleneck** — only ~22 NBs designated under MDR; wait times of 18–36 months
- **Regulatory affairs talent** is scarce and expensive

> The average cost of CE marking for a Class II medical device:
> **€100,000 – €500,000** in regulatory consulting alone

**Getting it wrong means market exclusion.**

---

## Who Needs This

**Primary customer: early-stage medtech & digital health startups**

- Pre-CE-marking phase, figuring out what applies to their device
- No dedicated regulatory affairs team yet
- Founders spending weeks reading legislation to answer basic questions

**Secondary: regulatory affairs professionals**

- Mid-size medtech companies cross-referencing obligations across regulations
- Contract regulatory consultants serving multiple clients

**Market context:** 27,000+ medical device companies in the EU.
NRW alone hosts major clusters in **Düsseldorf, Cologne, and Aachen**.

---

## The Solution

**CRSS is a regulatory intelligence layer built on the actual law.**

A question like:

> *"What does the EU AI Act require from a manufacturer of a Class IIb medical device that incorporates an AI model?"*

Returns a **precise, cited answer** tracing obligations across MDR Article 10, AI Act Article 16, and the relevant MDCG guidance — in seconds.

**Not a chatbot. Not a search engine.**
The system reasons over a structured knowledge graph of the regulations themselves.

---

<!-- _class: lead -->

## Demo

*[Screenshot: question → cited answer across MDR + AI Act]*

> "Article 10(1) MDR requires manufacturers to establish a quality management system. Article 16(a) EU AI Act additionally requires providers of high-risk AI systems to establish a risk management system pursuant to Article 9. MDCG 2025-6 clarifies that these obligations are cumulative and complementary…"

Real article numbers. Real legal text. No hallucination.

---

## Why It Works

Three things that make this different from ChatGPT + a PDF:

1. **The full regulations are parsed and cross-linked** — every article, paragraph, and annex reference is resolved. The system knows that MDR Annex I delegates to Annex II which is referenced in Article 52.

2. **Cross-regulation reasoning** — a single question can pull obligations simultaneously from MDR, AI Act, and GDPR because the graph tracks which articles in one regulation cite which in another.

3. **Grounded answers only** — the system is architecturally prevented from citing anything not in the retrieved legal text. No training-data contamination.

---

## Differentiation

|  | CRSS | Generic LLM | Regulatory Consultant | Static Compliance Tool |
|---|---|---|---|---|
| Grounded in actual law | ✅ | ❌ | ✅ | Partial |
| Cross-regulation reasoning | ✅ | ❌ | ✅ | ❌ |
| AI Act ready | ✅ | ❌ | Depends | ❌ |
| Available 24/7 | ✅ | ✅ | ❌ | ✅ |
| Cost per question | Cents | Cents | €200–500/hr | — |

The gap between a €300/hr regulatory lawyer and a well-grounded AI answer
is **closing fast** — but only if the AI is actually grounded.

---

## Business Model

**SaaS, subscription-based**

| Tier | Target | Price (indicative) |
|---|---|---|
| Startup | ≤10 employees, pre-CE | €149/month |
| Professional | Regulatory affairs teams | €499/month |
| Enterprise | Consultancies, NBs | Custom |

**Additional angles:**
- White-label for regulatory consultancies (they deliver it to their clients)
- Integration into existing QMS tools (Greenlight Guru, Qualio)
- Regulation update notifications when new MDCG guidance is published

*Gross margin potential: >85% (marginal cost = API calls + compute)*

---

## Traction & Validation

**What exists today:**
- Working prototype covering MDR, IVDR, EU AI Act, GDPR + 9 MDCG guidance documents
- Structured knowledge graph: 7,000+ legal provisions, cross-referenced
- Real Q&A capability validated on regulatory test cases

**Next steps with stipend:**
- 5 pilot conversations with NRW-based medtech founders (already 2 warm introductions)
- Refine pricing and onboarding flow based on pilot feedback
- Expand to German-language regulations (MDR/IVDR German text)

*This is a seed-stage project. I am not claiming a validated business — I am claiming a validated problem and a working technical solution.*

---

## Why NRW

NRW is the right place to build this:

- **Düsseldorf**: home to major medtech and pharma companies (Henkel, Evonik, many device companies)
- **Cologne/Bonn**: digital health ecosystem, StartupDock, multiple health-tech accelerators
- **Aachen**: RWTH spin-off culture, strong regulatory research groups

**Personal connection:** Based in NRW, with access to the local medtech network through [relevant connection/university/accelerator].

The Gründungstipendium NRW gives me 12 months to stop splitting focus between building and consulting — and go full-time on customer development and product.

---

<!-- _class: lead -->

## The Ask

**12 months** to validate and launch CRSS commercially

**Milestones:**
- Month 3: 5 paying pilots, pricing confirmed
- Month 6: German-language expansion, first enterprise conversation
- Month 9: 20 active users, first retention data
- Month 12: Seed round or revenue-sustainable

---

<!-- _class: lead -->

## About Me

**Diego Barra** — ML engineer with deep medtech domain experience

- Built AI diagnostics for **early breast cancer detection** (Raman Spectroscopy) at a medtech startup
- Identified systemic validation gaps in medical AI research; investigated medical AI reporting standards and the EU regulatory framework for AI-enabled devices
- Designed and built CRSS end-to-end: knowledge graph (Neo4j), embeddings, LLM reasoning layer, cross-regulation chain retrieval

**Why me specifically:**
The combination of hands-on AI development in a regulated clinical context, regulatory self-study depth, and the software engineering capability to build the full system is rare.
I am not pitching an idea — the system exists.

*[email]* · *[LinkedIn]*
