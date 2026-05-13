# From Pipeline to Agent: A Sequential Evolution Roadmap for CRSS

*A GraphRAG Engineering & Teaching Document*

---

## Preface: How to Read This Document

This document is both a technical roadmap and a learning guide. Each phase is self-contained: you can ship the system at the end of any phase and it will be meaningfully better than the one before it. No phase requires you to throw away what you already have — each one builds directly on top of the previous architecture.

The key teaching principle throughout: **every improvement is about making context more reliable, not about making the LLM smarter**. The LLM is already capable. What we are engineering is the *information environment* it reasons within.

---

## Current Baseline: What You Have Today

Before describing where to go, it is worth being precise about where you are. Your system is a **deterministic retrieval pipeline** with an LLM at the end. It is not yet an agent, but it is a strong foundation.

```
Question
  │
  ├─ [1] Term detection           → DefinedTerm nodes (Neo4j)
  ├─ [2] Regulation detection     → CELEX filter
  ├─ [3] Direct reference lookup  → Cypher (by display_ref)
  ├─ [4] HyDE generation          → LLM call #1 (100 tokens)
  ├─ [5] Vector retrieval         → In-memory cosine (numpy)
  ├─ [6] Graph expansion          → Cypher (HAS_PART, CITES)
  ├─ [7] Pointer expansion        → Cypher (inline refs)
  ├─ [8] Context assembly         → _format_context()
  └─ [9] LLM answer               → LLM call #2 (streaming)
```

**What is good:** The retrieval is grounded, structured, and multi-hop. The system understands legal hierarchy, cross-regulation links, and defined terms. It never hallucinates provision numbers because it can only cite what is in the context it assembled.

**What is missing:** The pipeline is fixed. It cannot observe its own results, decide it needs more, route differently based on question type, or remember anything across turns. These are the gaps we will close — in order.

---

## The Core Mental Model: From Pipeline to Loop

A pipeline runs once and returns. An agent runs in a loop and stops when it decides it is done.

```
PIPELINE                          AGENT
─────────────────────             ─────────────────────────────────────
Question → Steps → Answer         Question
                                    │
                                    ▼
                                  Analyze
                                    │
                                    ▼
                                  Act (tool call)
                                    │
                                    ▼
                                  Observe (result)
                                    │
                                    ▼
                                  Done? ──No──► Analyze
                                    │
                                   Yes
                                    │
                                    ▼
                                  Answer
```

The pipeline you have today is essentially one pass through that loop. Each phase below adds one dimension of the loop's intelligence.

---

## Phase 0 (Current State): Solid GraphRAG Pipeline

**What it is:** A hardcoded sequence of retrieval steps assembled into a context block, then sent to the LLM once.

**What it gets right:**
- Vector + graph hybrid retrieval is already implemented and effective.
- Context is grounded: only what the graph returns reaches the LLM.
- Multi-regulation awareness: CELEX filtering, per-regulation slot allocation.
- Legal definitions are injected automatically.
- The HyDE step places the query in the same embedding space as stored provisions.

**What it cannot do:**
- It cannot decide mid-flight that the retrieved provisions are insufficient.
- It cannot route differently based on question type (definitional vs. procedural vs. relational).
- It has no memory of prior questions or prior decisions.
- It has no way to check whether its answer is actually grounded.

**The lesson:** A good GraphRAG pipeline already eliminates most hallucination risk. The improvements from here are about *coverage* (not missing relevant provisions), *adaptability* (handling diverse question types), and *trust* (being able to verify and explain what happened).

---

## Phase 1: Role-Aware Graph Layer

**Concept:** Teach the graph who bears each obligation.

### The Problem It Solves

Today the graph stores provisions but does not encode which regulated entity is the *subject* of each obligation. When a hospital asks a question, the system has no way to distinguish "obligations the hospital bears" from "obligations the manufacturer bears." It retrieves by semantic similarity alone, which systematically under-retrieves provisions whose subject is implicit (e.g., Article 26 AI Act starts with "Deployers of high-risk AI systems shall..." — but the word "hospital" never appears in the article text).

### What Changes in the Graph

Three new structural elements are added:

1. **`ActorRole` nodes** — one per regulated entity per regulation. Each node represents a legal identity: `deployer` in the AI Act, `manufacturer` in the MDR, `importer` in both, etc. These are derived from the formal definitions already in the graph (`DefinedTerm` nodes where `category = 'actor'`).

2. **`OBLIGATION_OF` edges** — connecting each provision to the actor role that bears the obligation. Created by detecting the pattern: actor term appears in the first sentence of a provision *and* the provision contains "shall". This is a deterministic, pattern-based extraction — no LLM needed.

3. **`EQUIVALENT_ROLE` edges** — connecting actor roles across regulations when they describe the same real-world entity in different regulatory frameworks. For example: `deployer (AI Act)` ↔ `user (MDR)` in a healthcare procurement context.

### What Changes in Retrieval

A new retrieval path is added alongside the existing vector path: given a detected real-world entity in the question (e.g., "hospital"), resolve it to its regulatory roles, then fetch all provisions connected to those roles via `OBLIGATION_OF`. This path is deterministic and graph-native — it does not depend on embedding similarity.

### What Changes in the Agent

A new pre-retrieval step detects the questioner's role from the question text and triggers the role-based retrieval path. The role detection is a simple dictionary lookup (not an LLM call) using a static mapping: `"hospital" → deployer (AI Act) + user (MDR)`.

### The Teaching Point

This phase illustrates the most important principle in GraphRAG engineering: **encode domain knowledge as graph structure, not as retrieval logic**. Once the graph knows who bears each obligation, every question from any type of actor becomes correctly answerable without changing the retrieval code. The knowledge is in the graph, not in the prompt.

---

## Phase 2: Question Routing (Classification Before Retrieval)

**Concept:** Different question types need different retrieval strategies. Route before you retrieve.

### The Problem It Solves

Today every question goes through the same pipeline: HyDE → vector → graph expansion. This works well for definitional and provision-lookup questions, but it is suboptimal for:

- **Relational questions**: "How does Article 43 of the AI Act relate to Annex IX of the MDR?" — these need pathfinding, not vector similarity.
- **Comparative questions**: "What is the difference between a 'provider' and a 'manufacturer'?" — these need definition retrieval from multiple regulations, not a passage embedding.
- **Obligation enumeration**: "What must a deployer do under the AI Act?" — these need role-based graph traversal, not cosine similarity.
- **Structural questions**: "What does Annex I contain?" — these need direct hierarchical traversal, not embedding search.

### What Changes

A lightweight **question classifier** runs as the first step in the pipeline (before HyDE, before vector search). It categorises the question into one of a small set of types:

| Question Type | Primary Retrieval Strategy |
|---|---|
| Definitional | Direct DefinedTerm lookup |
| Provision-structural | Direct `display_ref` lookup + children |
| Obligation-by-role | Role-based graph traversal |
| Cross-regulation relational | Pathfinding via CITES + EQUIVALENT_ROLE |
| General compliance | HyDE + vector + graph expansion (current default) |

The classifier itself can be a simple rule-based function for most categories (keyword patterns are sufficient), with the LLM only involved for ambiguous cases.

### What Changes in Context Assembly

Different routes produce different context shapes. A relational question should present the two provisions side by side with the path between them explicitly labelled. An obligation-enumeration question should present provisions grouped by obligation type. The context formatter becomes route-aware.

### The Teaching Point

Routing is the first step toward an agent loop: instead of "always do X," the system now says "observe what kind of question this is, then decide what to do." This is the cognitive structure of an agent — even though it is still a single-pass pipeline. Routing is the simplest possible version of "Analyze → Act" from the agent loop.

---

## Phase 3: Retrieval Self-Assessment (Observe Your Own Results)

**Concept:** After retrieving, check whether what you retrieved is sufficient before sending it to the LLM.

### The Problem It Solves

The current pipeline assembles context and sends it to the LLM regardless of quality. It has no way to detect:
- "I retrieved 6 provisions but none of them are about this topic."
- "The question asks about Article 26 but I only retrieved articles from MDR."
- "The question mentions both MDR and the AI Act but all retrieved provisions are from one regulation."

When retrieval is poor, the LLM either hallucinates or correctly says "the context does not include this" — but either way, the user gets a bad answer. The system should detect this *before* the expensive LLM call.

### What Changes: A Retrieval Sufficiency Check

After context assembly, a fast deterministic check evaluates:

1. **Regulation coverage**: If the question targets 2 regulations, are both represented in the retrieved provisions?
2. **Role coverage**: If a questioner role was detected, does at least one retrieved provision bear an `OBLIGATION_OF` edge to that role?
3. **Direct reference coverage**: If the question named a specific article, is that article in the context?

If coverage is insufficient, the system takes a correction action *before* calling the LLM:
- Missing regulation: add a targeted vector retrieval pass scoped to the missing CELEX.
- Missing role: trigger the role-based retrieval path.
- Missing named provision: force-add it via direct lookup.

### The Teaching Point

This is the **Observe** step of the agent loop applied to retrieval quality. The system is no longer blind to what it retrieved — it inspects its own output and acts on what it finds. This is qualitatively different from the pipeline, which treats retrieval as a black box that produces context. The agent treats retrieval as a *tool whose output can be evaluated*.

This phase also introduces an important software principle: **the agent should not trust its own first action**. It should check.

---

## Phase 4: Short-Term Session Memory

**Concept:** Remember what happened in this conversation so you do not re-derive it every turn.

### The Problem It Solves

Today every question is completely stateless. If a user asks:
1. "What are the obligations of a medical device manufacturer under MDR?"
2. "And what about under the AI Act if the device contains AI?"

The second question has no knowledge that the first was asked. The system re-detects regulations, re-runs HyDE, re-retrieves from scratch. This is wasteful and also loses context: the user has been progressively narrowing a question across turns.

### What Changes: A Session State Object

A lightweight, in-memory session object carries forward across turns within a conversation:

- **Detected regulations** from prior turns (avoid re-detecting what was already confirmed).
- **Questioner role** (once a user identifies as a hospital, that persists for the whole session).
- **Provisions already seen** (avoid re-surfacing the same provisions when a follow-up question is a refinement of the previous one).
- **Terms already defined** (avoid injecting the same definition of "manufacturer" in every turn).

The session state is not stored in Neo4j at this phase — it is in-process memory only. It resets when the server restarts or the session ends.

### The Teaching Point

Short-term memory is not about intelligence — it is about **not repeating work**. The teaching analogy: a good expert does not re-read the entire regulation every time you ask a follow-up question in the same meeting. They remember what you established together. Short-term memory is what makes the difference between a system that feels like a stateless search engine and one that feels like a knowledgeable conversation partner.

---

## Phase 5: Tool Abstraction

**Concept:** Make retrieval capabilities callable by name, not hardcoded by sequence.

### The Problem It Solves

Today the retrieval pipeline is a fixed sequence of Python function calls. The LLM cannot influence which retrieval operations happen — it only sees the assembled result. This means:

- If the LLM's answer reveals that a key provision is missing, the system cannot go back and retrieve it.
- The LLM cannot say "I need the definition of X" and have the system fetch it.
- Different question types cannot receive genuinely different retrieval operations.

### What Changes: A Tool Registry

Each retrieval capability is wrapped as a named tool with a clear, documented interface:

| Tool Name | Description | Input | Output |
|---|---|---|---|
| `retrieve_by_vector` | Semantic similarity search | question text, k, celex filter | list of provisions |
| `retrieve_by_role` | Obligation lookup by actor role | role name, regulation | list of provisions |
| `retrieve_by_ref` | Direct provision lookup | article/annex reference | provision with children |
| `retrieve_definition` | Fetch formal definition | term name | definition text |
| `find_path` | Shortest path between two provisions | provision IDs | path of nodes/edges |
| `inspect_schema` | Return node labels and edge types | none | graph schema summary |

The agent loop now works as: LLM receives question + available tools → LLM selects a tool → tool runs → LLM observes result → LLM decides whether to call another tool or answer.

### The Relationship to MCP

Model Context Protocol (MCP) is a standardised way to expose these tools to any LLM client without custom glue code. At this phase, MCP is optional — you can implement the tool registry as a simple Python dispatcher first, then wrap it with MCP for external clients later. The important thing is the *abstraction*, not the protocol.

### The Teaching Point

Tool abstraction is the moment your system transitions from a pipeline to an agent. The key insight: **the LLM should select tools, not be given a pre-assembled context**. When you pre-assemble context, you are making retrieval decisions on behalf of the LLM. When you give the LLM tools, you let it make its own retrieval decisions based on what it needs to answer the question correctly. The system becomes adaptive rather than prescriptive.

---

## Phase 6: Reasoning Trace Storage (Context Graph)

**Concept:** Write the agent's decisions back into Neo4j so they become queryable knowledge.

### The Problem It Solves

Today when the system answers a question, the decision process disappears. You cannot ask:
- "Why did the system retrieve Article 26 for this question?"
- "What provisions were considered and discarded?"
- "How did the system determine the questioner was a deployer?"
- "Has a similar question been asked before, and what answer was given?"

This is not just a debugging problem — it is a *trust* problem. In EU compliance work, auditability is a regulatory requirement. A system that cannot explain its reasoning cannot be used in professional practice.

### What Changes: Reasoning Nodes in Neo4j

For each answered question, a set of nodes and edges is written to Neo4j:

```
(:Query {id, text, timestamp, session_id})
  -[:TRIGGERED]->
(:ReasoningTrace {id, question_type, detected_role, detected_regulations})
  -[:USED_TOOL]->
(:ToolCall {id, tool_name, input, output_count, duration_ms})
  -[:RETRIEVED]->
(:Provision {id, ...})   ← already exists in graph

(:ReasoningTrace)
  -[:PRODUCED]->
(:Answer {id, text, token_count, timestamp, prompt_version})
```

This creates a **context graph** — Neo4j's term for a graph that records decision traces, not just domain knowledge. The context graph is queryable:

```cypher
MATCH (q:Query)-[:TRIGGERED]->(t:ReasoningTrace)-[:RETRIEVED]->(p:Provision)
WHERE p.celex = '32024R1689'
RETURN q.text, collect(p.display_ref)
ORDER BY q.timestamp DESC
LIMIT 10
```

This returns: "what questions about the AI Act have been asked, and which articles were retrieved for each?"

### What This Enables

- **Auditability**: Any answer can be traced back to exactly which provisions were retrieved and which tools were called.
- **Pattern learning**: Frequently retrieved provisions for a given question type can be promoted as default retrieval targets.
- **Consistency checking**: If the same question asked twice produces different provisions, that is a signal of retrieval instability.
- **Human oversight**: A compliance reviewer can inspect the system's reasoning, not just its answer.

### The Teaching Point

This phase closes the loop between the knowledge graph and the agent's own behaviour. The graph is no longer just a static store of regulatory knowledge — it becomes a living record of how the agent used that knowledge. This is the foundation of **explainable AI** in the regulatory domain: not just "what did the LLM say" but "what did the system know, what did it look at, and why."

---

## Phase 7: Evaluation Layer

**Concept:** Measure retrieval quality and answer faithfulness automatically so the system can improve.

### The Problem It Solves

Without evaluation, you cannot know whether a change made the system better or worse. You cannot catch regressions when you update the graph or the prompt. You cannot build confidence with users or partners.

### What Changes: A Benchmark + Evaluation Pipeline

Three evaluation dimensions:

1. **Retrieval recall**: Given a question with a known correct provision, does the system retrieve it? Build a small gold standard of 20–30 question→provision pairs. Measure what fraction of the gold provisions appear in the retrieved context.

2. **Answer faithfulness**: For each answer, check that every cited article number appears in the assembled context. This is fully deterministic — no LLM judge needed. Extract article references from the answer, check them against the context block.

3. **Role coverage**: For questions that identify a questioner role, check that at least one provision with an `OBLIGATION_OF` edge to the detected role is in the context.

These checks are fast, deterministic, and can run automatically after every system change.

### The Teaching Point

Evaluation is the last piece because it requires having something to evaluate — you need the reasoning traces from Phase 6 to run systematic evaluation. The teaching principle: **evaluation is not a quality gate, it is a feedback loop**. You run it continuously, not just before release. Every evaluation failure is a signal that tells you which part of the pipeline to fix next.

---

## Summary: The Sequential Dependency Map

```
Phase 0: GraphRAG Pipeline (current)
  │
  │  Add role-aware graph structure
  ▼
Phase 1: ActorRole Nodes + OBLIGATION_OF Edges
  │
  │  Add routing to use that structure
  ▼
Phase 2: Question Routing
  │
  │  Add self-inspection of retrieved results
  ▼
Phase 3: Retrieval Self-Assessment
  │
  │  Add conversational continuity
  ▼
Phase 4: Short-Term Session Memory
  │
  │  Make retrieval callable by the agent, not hardcoded
  ▼
Phase 5: Tool Abstraction
  │
  │  Record decisions in the graph
  ▼
Phase 6: Reasoning Trace Storage
  │
  │  Measure and improve systematically
  ▼
Phase 7: Evaluation Layer
```

Each phase depends on the one above it. Phases 1–3 strengthen retrieval quality. Phases 4–5 add adaptability. Phases 6–7 add trust and maintainability.

---

## What You Get at Each Stopping Point

| After Phase | What the System Can Do |
|---|---|
| 0 (now) | Answer single-turn questions with grounded, multi-hop retrieval |
| 1 | Correctly surface obligations for any regulatory actor (deployer, importer, etc.) |
| 2 | Route different question types to appropriate retrieval strategies |
| 3 | Detect and self-correct retrieval gaps before calling the LLM |
| 4 | Maintain context across a multi-turn compliance conversation |
| 5 | Let the LLM direct its own retrieval; handle novel question structures |
| 6 | Provide full decision traces; meet auditability requirements |
| 7 | Measure, improve, and demonstrate retrieval and faithfulness quality |

---

## A Note on What Not to Do

Several tempting shortcuts should be avoided:

**Do not expose raw Cypher generation to the LLM.** LLMs produce syntactically invalid or semantically incorrect Cypher. Always wrap retrieval in typed, documented tools. The LLM selects tools; it does not write queries.

**Do not add an orchestration framework (LangChain, LangGraph) before Phase 5.** These frameworks add significant complexity. Until you have tool abstraction, there is nothing for the framework to orchestrate. Phases 1–4 can be implemented with plain Python.

**Do not store reasoning traces in a flat log file.** The value of Phase 6 comes from the traces being *graph-queryable*. A log file gives you debugging; a graph gives you insight.

**Do not skip Phase 1 in favour of Phase 5.** Tool abstraction without a role-aware graph just gives the LLM more ways to retrieve incomplete results. The knowledge must be in the graph first; the tools expose it.

---

## Conclusion

Your system today is already past the hardest part: you have a working knowledge graph, hybrid retrieval, and a grounded generation pipeline. The path to full agentic context engineering is a sequence of seven focused improvements, each one adding a single new capability without requiring you to rebuild what you have.

The underlying principle across all phases is the same: **make context more reliable by encoding more knowledge as graph structure**. The agent loop, the tools, the traces — these are all ways of ensuring that the right knowledge is available, in the right form, at the right moment. The LLM does not need to be smarter. The information environment needs to be better.
