# Neo4j graph — demo visuals setup

Companion to `recorded_demo_playbook.md` (the "optional graph glimpse" beat).
Goal: when you show the Neo4j database, every node shows a **legible reference**
(e.g. "Article 6", "Annex III", "manufacturer") in a **consistent colour** — never
a blank circle you have to guess at. Everything below is verified against your live
graph (Aug 2026).

---

## 1 · One-time setup

### 1a · Backfill chapter/section captions (run once per rebuild)

The EUR-Lex parser captured Chapter/Section headings into `title` for the AI Act but
**not** for the MDR, IVDR or GDPR — so those chapters rendered blank. This script
restores the real Official-Journal headings for all three (extracted from the cached
EUR-Lex source, pinned and verified — e.g. MDR Chapter V → "CLASSIFICATION AND
CONFORMITY ASSESSMENT", GDPR Chapter III → "Rights of the data subject"), then sets a
`display_caption` on every container so nothing is ever blank. It also captions the 148
label-less level-0 communities (e.g. "GDPR cluster 1 (179)") while keeping the
chapter-aligned label on level-1 communities:

```
python scripts/backfill_display_captions.py            # apply
python scripts/backfill_display_captions.py --dry-run  # report only
```

It's display-only and safe (nothing in the answer pipeline reads it). A full
re-ingest (`build_all.py`) wipes it — just re-run this afterwards.

### 1b · Load the stylesheet

Open Neo4j Browser at **http://localhost:7474** (container `crss_neo4j` is already
running), then:

1. *(optional, if the current view looks messy)* type in the command bar:
   ```
   :style reset
   ```
2. **Drag `presentation/crss_graph.grass` onto the Browser window.** It applies
   instantly and persists in the browser's local storage — you only do this once
   per machine.

To go back to Neo4j defaults at any time: `:style reset`.

**Why this is needed:** Neo4j Browser auto-picks a caption per label the first time
it sees it, often landing on an internal id, a `title` that only some regulations
populated, or nothing — that's the "blank coloured node" problem. The stylesheet pins
the caption to the right human property for every label (`display_ref` for provisions,
`display_caption` for chapters/sections/communities, `term` for defined terms & roles,
`ref_text` for external acts, `regulation_id` for documents).

---

## 2 · The demo queries (paste, run, record)

Run one query at a time. After it renders, click the **fullscreen / expand icon** on
the result frame (top-right of the result pane) to hide the editor and sidebar, let
the force-layout settle for ~2 seconds, then record.

### A · "Braxton's question, as a graph" — the hero shot (12 nodes)

```cypher
MATCH path=(a:Article {celex:'32024R1689', display_ref:'Article 6'})
            -[:HAS_PART*1..2]->(child:Provision)
            -[:CITES|DELEGATES_TO]->(annex:Annex {celex:'32024R1689'})
WHERE annex.display_ref IN ['Annex I','Annex III']
RETURN path
```

What renders: the blue **Article 6** node in the middle; its children `Article 6(1)
point (a)/(b)` reaching the green **Annex I**, and `Article 6(2)/(3)/(4)/(6)` reaching
green **Annex III**.

> **Say:** "This is the whole high-risk classification test in one picture. Article 6
> is the gate. It points two ways — through **Annex I**, the product-safety route where
> a device that's a *safety component* lands, and through **Annex III**, the stand-alone
> high-risk list. CRSS answers by walking these edges, not by pattern-matching text."

This is the one to feature — it *is* Braxton's question.

### B · "The legal-reasoning graph" — optional, richer (27 nodes)

```cypher
MATCH (a:Article {celex:'32024R1689', display_ref:'Article 6'})
OPTIONAL MATCH p=(a)-[:TRIGGERS_OBLIGATION_CLUSTER|IS_PREREQUISITE_FOR|REQUIRES_PRIOR_CHECK|DEROGATES_FROM]-(n)
RETURN a, p
```

What renders: Article 6 as a hub with ~26 provisions on **thick orange/purple edges**.

> **Say:** "These coloured edges aren't citations — they're curated legal-reasoning
> links. *This classification triggers that obligation cluster. This step requires a
> prior check first.* That's what makes it a knowledge graph instead of a search index."

### C · "Same actor across regulations" — clean bonus (10 nodes)

```cypher
MATCH (r1:ActorRole)-[e:EQUIVALENT_ROLE]-(r2:ActorRole)
RETURN r1, e, r2
```

What renders: orange **ActorRole** nodes (`manufacturer`, `importer`, …) linked across
the MDR, IVDR and the implementing regulation.

> **Say:** "The graph knows the 'manufacturer' in the MDR is the same actor as in the
> IVDR — so a 'who is responsible?' question resolves to the right role across every
> regulation at once."

### (Optional) wider cross-reg view — AI Act ↔ GDPR (28 nodes, denser)

```cypher
MATCH (a:Provision {celex:'32024R1689'})-[c:CITES]->(b:Provision {celex:'32016R0679'})
RETURN a, c, b
```

Only if you want to show breadth; it's busier, so zoom in on one edge rather than
narrating the whole cloud.

---

## 3 · Making it look clean on camera

- **Fullscreen the frame** (expand icon, top-right of the result pane) — hides the
  editor and left sidebar, giving you an all-canvas shot.
- **Let the layout settle** ~2s before recording; drag any overlapping node apart.
  You can **pin** a node by dragging it where you want it.
- **Don't double-click nodes on camera** unless you mean to — expanding a provision
  can pull in ~100 faint `USES_TERM` edges and turn it into a hairball. The three
  queries above are pre-bounded to stay legible.
- **Zoom** with the scroll wheel or the +/- buttons (bottom-right).
- Clicking a node opens a small inspector showing its real text — nice for a quick
  "and here's the actual legal text" beat, or collapse it for a pure-graph look.
- The default **white canvas** reads best on video; the stylesheet's caption colours
  are tuned for it (dark text on light nodes, white text on the saturated ones).

---

## 4 · Colour legend (for what you say on camera)

| Colour | Node type | In plain words |
|---|---|---|
| **Blue** | Article | the operative rules |
| **Cyan** | Paragraph / Point | the sub-parts of an article |
| **Green** | Annex (+ sub-levels) | the lists & criteria (e.g. Annex I, Annex III) |
| **Gold** | DefinedTerm | legal definitions ("medical device") |
| **Orange** | ActorRole | who's responsible (manufacturer, deployer…) |
| **Purple** | Community | auto-detected topic clusters |
| **Teal** | Guidance | MDCG guidance (non-binding) |
| **Rose** | ExternalAct | other legal acts the text points to |
| **Parchment** | Recital / Citation | the preamble |
| **Slate** | Chapter / Document | structural containers |

Edge thickness/colour tracks meaning too: thin grey = structure (`HAS_PART`), blue =
cross-reference (`CITES`), and the thick orange/purple edges are the curated
legal-reasoning links (`TRIGGERS_OBLIGATION_CLUSTER`, `REQUIRES_PRIOR_CHECK`, …).

---

## 5 · One-line honesty note

If asked "is this the whole database?" — no. These are focused views. The full graph
is ~7,850 provision nodes across the AI Act, MDR, IVDR, GDPR, the implementing
regulation and MDCG guidance, plus ~14k defined-term links and the citation web. The
queries above just isolate the part that answers the question on screen.
