# What You’re Asking — and Why It’s a Great Instinct

You’re right that definition provisions (e.g., AI Act Article 3, MDR Article 2) are structurally special and deserve first-class treatment in the graph. Here’s the full picture of what’s critical and what it unlocks.

**Prompt:**
```
You are a GraphRAG expert and EU Regulations expert and you will consult me on the following: We have so far our parsed document in Neo4j database, in terms of document parts and structure. But I think that we need to establish somehow nodes for the items that are outline in Definitions. Definitions could be an article, or could be some annex section where important definitions are outlined. Now, keeping in mind our infrastructure, tell me what is critical to achieve for best graph traversal. For example, if we have the EU AI Act, there are "actors" that are defined either as a legal or natural person. I think these nodes would be very well suited for models to understand the definitions of terms and to find benefits of our reasoning agent (which may be very well unknown to me, since I am not an expert). 
```

## Current State

After your fix, Article 3 of the AI Act has 68 Point children, each containing a definition like:

```
(3) ‘provider’ means a natural or legal person, public authority, agency or other body that develops an AI system…
```

These are stored as :Point nodes under `:Article {id: “32024R1689_art_3”}` with `HAS_PART` edges. They get embedded and are findable by vector search. MDR Article 2 has the same structure with its ~71 definitions.

What’s missing: the graph treats ‘provider’ the same as any other Point. There’s no semantic layer that says “this node defines a concept called ‘provider’” and that “Articles 16, 17, 22, etc. impose obligations ON that concept”.

## What’s Critical for Graph Traversal

There are three layers of increasing value, and I’d recommend them in this order:

### Layer 1: Definition Nodes (:DefinedTerm) — High Impact, Simple

Create a new node type that extracts the term name and category from each definition point.

Why it’s critical: Right now, if someone asks “What is a deployer?”, your retriever has to find art_3_pt_4 by vector similarity alone. With a `:DefinedTerm` node, you can do exact-match + graph expansion:

```
(:DefinedTerm {term: “deployer”, category: “actor”})
-[:DEFINED_BY]->
(:Point {id: “32024R1689_art_3_pt_4”})
```
This gives you:

Exact term lookup — no embedding needed, just 

```
MATCH (d:DefinedTerm {term: “deployer”})
Category filtering — “list all actors” → MATCH (d:DefinedTerm {category: “actor”})
```

Cross-regulation term alignment — MDR’s “manufacturer” and AI Act’s “provider” are conceptually related; you can link them later

What to extract from each definition point:

| Property          | Source                          | Example                                      |
|------------------|---------------------------------|----------------------------------------------|
| term             | Text between quotes '…'         | "deployer"                                   |
| term_normalized  | Lowercased, singular            | "deployer"                                   |
| category         | Parsed from definition body     | "actor" (because "natural or legal person")  |
| regulation       | Parent celex                    | "32024R1689"                                 |

Categories you’d want (derivable by simple regex/keyword matching on the definition body):

Category --| Signal in definition body | AI Act examples |
|----------|--------------------------|-----------------|
actor      | “natural or legal person”, “public authority”, “agency or other body” | provider, deployer, importer, distributor, authorised_representative
system | “means a machine-based system”, “means a system” | AI system, high-risk AI system, emotion recognition system
concept | “means the combination of”, “means the process of” | risk, intended purpose, reasonably foreseeable misuse
data | “means data”, “means personal data”, “biometric data” | training data, biometric data, input data
process | “means the placing”, “means the putting” | placing on the market, putting into service
document | “means documentation”, “means a declaration” | technical documentation, EU declaration of conformity
body | “means a body”, “means an authority” | notified body, market surveillance authority

The regex pattern for extracting the term is trivially reliable because EUR-Lex always wraps defined terms in single quotes:

```
term_match = re.match(r”^’([^’]+)’\s+means\b”, point_text)
```

This works for every definition in all three regulations.

### Layer 2: MENTIONS_TERM Edges — High Impact, Moderate Effort

Once you have :DefinedTerm nodes, you can link them to every provision that uses that term:

```
(:Paragraph {id: “32024R1689_016.001”})
-[:MENTIONS_TERM]->
(:DefinedTerm {term: “provider”})
```

Why it’s critical for your agent: When the user asks “What are the obligations of a provider?”, the retriever currently relies on vector similarity to find Article 16 (“Obligations of providers”). But with MENTIONS_TERM edges, you can do:

```
MATCH (d:DefinedTerm {term: “provider”})<-[:MENTIONS_TERM]-(p:Provision)
WHERE p.kind IN [‘article’, ‘paragraph’]
RETURN p
```

This returns every provision that references the concept of “provider” — not just the ones that happen to be semantically close to the query embedding. This is particularly powerful for:

Obligation mapping: “What must a deployer do?” → find all provisions mentioning “deployer”
Scope analysis: “Does this regulation apply to importers?” → check if “importer” is mentioned in Article 2 (Scope)
Definition chain traversal: A definition may reference other defined terms (e.g., “deployer” references “AI system”) — these become navigable

Implementation: A post-parse pass that, for each :DefinedTerm, does a case-insensitive search of its term string across all text_for_analysis fields, and creates `MENTIONS_TERM` edges.

### Layer 3: Cross-Regulation Term Alignment — Future Value

Once :DefinedTerm nodes exist across MDR, IVDR, and AI Act, you can link equivalent or related terms:

```
(:DefinedTerm {term: “manufacturer”, regulation: “MDR”})
-[:EQUIVALENT_TERM]->
(:DefinedTerm {term: “provider”, regulation: “AI Act”})
```

**What This Unlocks for the Agent**

- Definition-grounded answers: When the agent cites “provider”, it can always include the exact legal definition alongside the obligation.
- Actor-obligation matrix: With category: “actor” + `MENTIONS_TERM` edges, you can programmatically build a complete matrix of obligations per actor.
- Disambiguation: Terms like “operator” differ across regulations. DefinedTerm nodes prevent conflation.
- Retrieval boost: DefinedTerm nodes become anchor results for queries about concepts.
- Structured reasoning chains: The agent can traverse:
- `DefinedTerm` → `MENTIONS_TERM` → `Article` → `HAS_PART` → `Paragraph` → CITES → other Article

**Recommended Implementation Approach**

New module: `ingestion/parse/semantic_layer/definitions.py` — extracts term and category, creates :DefinedTerm nodes and DEFINED_BY edges

Post-parse pass: scan all text_for_analysis fields and create MENTIONS_TERM edges

Loader update: support :DefinedTerm and new edge types

Retriever update: include defined_term in parent kinds and expand traversal over MENTIONS_TERM

The term extraction regex `(’([^’]+)’\s+means)` and category classification (keyword matching) are reliable enough to be rule-based — no LLM needed.