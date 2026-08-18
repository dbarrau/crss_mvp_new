"""Consolidation: apply an amending act's instructions onto the base regulation.

An EU *amending* regulation (e.g. the Digital Omnibus on AI, Reg (EU) 2026/1744)
does not restate the law — it carries a list of surgical instructions ("in
Article 6, the following paragraphs are inserted: '…'", "paragraph 3 is replaced
by the following: '…'", "point 1 is deleted"). To hold *current* law, the base
regulation's graph must have those instructions applied to it.

This package is the isolated home of that feature:

- ``amendment_parser`` — parse an amending act's raw HTML into a structured,
  inspectable list of :class:`AmendmentInstruction`s (Stage 1).
- the applier (Stage 2) consumes that artifact and mutates the base graph.
- the validator (Stage 3) checks the consolidated result against primary text.

It reads the amending act's *raw HTML* directly (the flat ``parsed.json`` folds
nested quoted blocks and cannot be un-folded), and touches only amending acts —
base regulations have no quoted amendment blocks, so blast radius is isolated
here.
"""
