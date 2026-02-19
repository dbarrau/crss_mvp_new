# **MVP Development Plan**
Compliance Readiness Support System (CRSS)

*Internal Working Document*

---

## Purpose of This Document

This document defines a concrete, step-by-step plan for developing a Minimum Viable Product (MVP) of the Compliance Readiness Support System (CRSS).

The objective of the MVP is **not** to assess regulatory compliance, but to:

- extract regulatory-relevant claims from technical documentation,
- map claims to evidence and regulatory text,
- flag missing or inconsistent documentation linkages,
- produce structured review items for human resolution.

This document is execution-oriented and intended to reduce ambiguity, context switching, and decision fatigue during development.

---

## Non-Goals (Hard Constraints)

The MVP explicitly does **not**:

- judge clinical or scientific sufficiency,
- interpret regulatory obligations,
- predict conformity assessment outcomes,
- automate regulatory decisions.

Any feature that violates these constraints is out of scope for the MVP.

---

## Core MVP Definition

The MVP is considered complete when the system can:

1. Ingest a small set of realistic MDR-style documents.
2. Extract compliance-relevant claims with traceable sources.
3. Link claims to explicitly retrieved regulatory text.
4. Detect missing or inconsistent evidence references.
5. Output human-reviewable findings with full traceability.

Anything beyond this definition belongs to a later iteration.

---

## Architectural Conventions (Mandatory)

The CRSS MVP must follow strict modularity and separation-of-concerns principles.

### Layered Architecture

The system shall be divided into five independent layers:

1. **Regulatory Layer** (deterministic)
2. **Document Layer** (deterministic)
3. **Claim Layer** (LLM-bounded)
4. **Linking & Validation Layer** (deterministic + bounded LLM)
5. **Presentation Layer** (UI or export only)

No layer may directly depend on non-adjacent layers.

### Deterministic Core Principle

All components that:

- parse regulatory text,
- extract document structure,
- manage identifiers,
- store data,
- perform consistency checks,

must be deterministic and testable without LLM calls.

LLMs are only permitted in explicitly defined extraction or similarity tasks.

### Strict Interface Contracts

Each layer must expose a clearly defined input/output schema.

- JSON schemas must be versioned.
- No implicit field inference.
- No cross-layer shared mutable state.

Breaking schema compatibility requires version increment.

### Stable Identifier Convention

All core entities must use stable, opaque identifiers:

- Regulation Provision ID
- Document ID
- Chunk ID
- Claim ID
- Review Item ID

Identifiers must:

- be deterministic,
- remain stable across reruns,
- not encode semantic meaning.

### LLM Boundary Rule

Every LLM call must:

- operate on bounded context,
- return structured JSON,
- include prompt version metadata,
- be reproducible given fixed input.

Free-form generation is prohibited in core processing paths.

---

## Milestone 0: Minimal Realistic Dataset

### Objective

Establish a small but realistic document corpus to anchor development.

### Tasks

- Select excerpts from MDR 2017/745 and/or the EU AI Act.
- Prepare 3–5 technical documents:
  - Technical Description
  - Risk Management File
  - Clinical Evaluation Report

### Deliverable

A local directory structure containing PDFs or DOCX files that resemble a real submission.

### Done When

The dataset can plausibly be described as “a small MDR technical dossier.”

---

## Milestone 1: Regulatory Knowledge Layer

### Objective

Prevent hallucination and ensure legal traceability.

### Tasks

- Parse regulatory text into discrete provisions.
- Assign stable identifiers (Article, Annex, Paragraph).
- Store provisions in a versioned local database.

### Implementation Notes

- Python-only, deterministic.
- No implicit regulatory reasoning.

### Modularity Requirements

- Regulatory parsing logic must be isolated in a dedicated module.
- No claim extraction logic may exist in this layer.
- Storage format must be abstracted behind a repository interface.

### Done When

It is possible to retrieve regulatory provisions by identifier and topic, with full source text.

---

## Milestone 2: Document Ingestion and Chunking

### Objective

Convert unstructured documents into addressable text units.

### Tasks

- Extract text from PDFs or DOCX files.
- Split documents into chunks by section and page.
- Assign stable chunk identifiers.

### Data Model

Each chunk must retain:

- document identifier,
- section title,
- page number,
- raw text.

### Chunking Contract

Chunking must be:

- deterministic,
- idempotent,
- independent of downstream LLM behavior.

Chunk identifiers must be hash-based or structure-based, not incremental counters.

### Done When

Any sentence can be traced back to its document, section, and page.

---

## Milestone 3: Claim Extraction

### Objective

Identify statements that assert regulatory-relevant properties.

### Definition of a Claim

A claim is a sentence or paragraph that implies:

- safety, performance, or risk control,
- data governance or bias mitigation,
- human oversight or lifecycle management.

### Tasks

- Use bounded LLM calls to extract claims.
- Store claims verbatim with source references.

### LLM Isolation Requirement

Claim extraction must:

- run in a dedicated service or module,
- accept only chunk text as input,
- output structured claim objects,
- never modify source content.

Prompt templates must be version-controlled.

### Constraints

- No assessment of correctness.
- No summarization or paraphrasing.

### Done When

A list of claims (10–30) can be enumerated with exact source locations.

---

## Milestone 4: Regulatory Relevance Tagging

### Objective

Associate claims with potentially relevant regulatory provisions.

### Tasks

- Retrieve regulatory provisions by topic.
- Link claims to one or more provisions.
- Record links as tentative, not authoritative.

### Linking Strategy Abstraction

Regulatory linking must be implemented through a strategy interface:

- keyword-based retrieval,
- embedding similarity,
- hybrid retrieval.

The linking mechanism must be swappable without affecting other layers.

### Done When

Claims can be grouped by MDR Annex or AI Act Article without implying compliance.

---

## Milestone 5: Evidence Mapping Checks

### Objective

Determine whether claims reference supporting evidence.

### Tasks

- Detect references to methods, datasets, figures, or tables.
- Flag claims with no identifiable evidence.

### Important Limitation

The system checks for **presence**, not **sufficiency**, of evidence.

### Done When

The system can produce at least several “missing evidence” review items.

---

## Milestone 6: Consistency Checks

### Objective

Detect contradictory or inconsistent descriptions across documents.

### Examples

- Different descriptions of human oversight.
- Conflicting performance metrics.

### Consistency Engine Isolation

Consistency detection must operate solely on structured claim objects.

It must not:

- re-read raw documents,
- re-run claim extraction,
- introduce new semantic interpretation.

### Tasks

- Identify conceptually similar claims.
- Flag materially different descriptions.

### Done When

At least a small number of plausible inconsistencies are detected.

---

## Milestone 7: Human Review Output

### Objective

Present findings as review items, not decisions.

### Output Format

Each review item includes:

- claim text,
- issue type,
- source location,
- suggested human action.

### Done When

A human reviewer can understand and act on each item without additional explanation.

---

## Optional Milestone 8: Minimal Interface

### Objective

Improve usability without increasing scope.

### Tasks

- Simple UI to browse review items.
- Click-through to source text.

### Constraint

No analytics, dashboards, or automated decisions.

---

## Completion Criteria

The MVP is complete when:

- all previous milestones are satisfied,
- outputs are reproducible from identical inputs,
- LLM outputs are traceable to prompt version and input chunk,
- all intermediate artifacts can be reconstructed,
- all findings are traceable to source text,
- human review is clearly supported.
