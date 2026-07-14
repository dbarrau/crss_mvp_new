"""Tuning constants for hybrid retrieval, with their rationale.

Single source of truth for the knobs that shape retrieval behaviour.  The
comment blocks are load-bearing documentation: several of these values were
set after observed failures (see each block), so keep the rationale attached
to the value when changing either.
"""
from __future__ import annotations

# E5 asymmetric encoding prefixes.  Provisions are stored with the passage
# prefix (see infrastructure/embeddings/batch_embedder.py); queries must use
# the query prefix.  If either ever changes, all stored embeddings desync and
# must be re-generated.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Cross-encoder reranking: widen the cosine candidate pool before reranking.
# When a reranker is active, retrieve _CANDIDATE_MULTIPLIER × k candidates
# from cosine similarity, then rerank to keep the final k.  The multiplier
# is capped at _CANDIDATE_CAP to bound cross-encoder latency (each candidate
# is one forward pass through a ~568M-param model).
_CANDIDATE_MULTIPLIER = 5
_CANDIDATE_CAP = 48

# Cross-encoder truncation length, in *tokens* (enforced by the tokenizer via
# the CrossEncoder's max_length).  Legal provisions front-load their key
# content, so 320 tokens captures the discriminative text at roughly half the
# inference cost of the model's 512 maximum.
_RERANK_MAX_TOKENS = 320

# Pre-tokenization character slice for reranker passages.  Only a perf guard so
# the tokenizer never chews through a multi-page annex: at ~4 chars/token,
# 4 × _RERANK_MAX_TOKENS comfortably covers the token budget, and the real
# truncation happens in the tokenizer.  (A previous version sliced the passage
# to 320 *characters* — ~80 tokens — silently starving the cross-encoder.)
_RERANK_MAX_CHARS = 4 * _RERANK_MAX_TOKENS

# Blend weight between the (normalised) cross-encoder score and the
# (normalised) cosine score.  0.0 = cosine only, 1.0 = cross-encoder only.
# Kept below 1.0 so a dominant cosine match (e.g. an explicitly-named annex)
# cannot be buried by a vocabulary-similar neighbour the cross-encoder prefers.
_RERANK_WEIGHT = 0.6

# Reciprocal Rank Fusion constant.  RRF score for a document is
# sum over channels of 1/(K + rank).  K=60 is the value from the original
# Cormack et al. RRF paper and is the de-facto standard; larger K flattens
# the contribution of top ranks, smaller K sharpens it.
_RRF_K = 60

# Weight of the lexical (BM25) channel in the fusion, relative to dense=1.0.
# Kept below 1.0 so the lexical channel *breaks ties* and rescues exact
# heading/term matches the dense model smears, without letting a
# mediocre-dense-but-lexically-dense node override a near-perfect dense match
# (e.g. the verbatim "'AI system' means …" definition point).
_LEXICAL_WEIGHT = 0.5

# Upper bound on obligations returned per role-obligation query. Generous
# enough to carry a role's full statutory article set (the largest, MDR
# manufacturer, has ~26) without dumping every annex fragment; the context
# budget trims any low-relevance tail downstream.
_ROLE_OBLIGATION_CAP = 14
