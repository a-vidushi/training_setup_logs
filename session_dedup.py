"""Session deduplication for the OpenAgriNet training pipeline.

Semantic deduplication via embedding + pure numpy radius search
--------------------------------------------------------
Gold (production) and fixable (synthetic/corrected) traces are embedded using
``sentence-transformers/all-MiniLM-L6-v2`` — a small, fully offline-capable
model (~80 MB).  The resulting vectors are clustered using a greedy radius search
(range search) implemented in pure numpy with a strict distance threshold.
This ensures only true semantic paraphrases are collapsed together, and unique 
outlier questions are guaranteed to be preserved. The first occurrence in a local 
neighborhood is kept as the representative seed; neighbors are dropped.

After deduplication a *diversity map* is produced that shows:

- How many unique clusters (seed questions) remain.
- Per-language breakdown of representative seeds.
- Per-complexity-tier / query-type breakdown (simple / moderate / complex).
- Which (language, query-type) cells are underrepresented relative to their
  expected share — these are flagged as coverage gaps to guide future data
  collection.

The function signature is intentionally backwards-compatible with the old
``deduplicate_sessions`` caller in ``build_dataset.py``, but it now returns an
additional third element: the diversity map dict.

Usage (from build_dataset.py)
------------------------------
    from session_dedup import deduplicate_sessions
    sessions, dropped, diversity_map = deduplicate_sessions(sessions)
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types_def import LogSession

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Sentence-transformers model to use for embedding.  The MiniLM variant is
# small (~80 MB) and fast — well suited for an offline / CI environment.
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The distance threshold for collapsing sessions.
# On unit-normalised vectors, an L2 distance of ~0.45 to 0.70 corresponds to
# semantic paraphrases, while >0.85 corresponds to distinct concepts.
_DISTANCE_THRESHOLD = 0.75

# Warn if more than this fraction of sessions are collapsed.
_WARN_THRESHOLD = 0.50

# ── Lazy-loaded singletons ────────────────────────────────────────────────────

_embed_model = None


def _get_embed_model():
    """Return the (cached) SentenceTransformer instance."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic deduplication. "
                "Install it with:  uv add sentence-transformers"
            ) from exc
        logger.info("Loading embedding model %r …", _EMBED_MODEL)
        _embed_model = SentenceTransformer(_EMBED_MODEL)
    return _embed_model


# ── Core deduplication ────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> np.ndarray:
    """Return an (N, D) float32 embedding matrix for *texts*."""
    model = _get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # Normalise to unit length so L2 distance ≈ cosine distance.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = (embeddings / norms).astype(np.float32)
    # FAISS C++ extensions require strict C-contiguous arrays, otherwise they segfault
    return np.ascontiguousarray(normalized)


def _radius_clustering(
    vectors: np.ndarray,
    threshold: float,
) -> tuple[list[int], list[tuple[int, list[int]]]]:
    """Cluster vectors greedily using a strict radius search.
    
    Returns
    -------
    (kept_indices, collapsed_groups)
    where collapsed_groups is a list of (kept_idx, [dropped_idx_1, ...])
    """
    n = vectors.shape[0]
    kept_indices = []
    dropped_set = set()
    collapsed_groups = []
    
    # FAISS is currently fundamentally broken on MacOS + Python 3.14 (segfaults on search).
    # We use pure numpy for this radius search. It is highly vectorized and very fast
    # for offline deduplication pipelines.
    for i in range(n):
        if i in dropped_set:
            continue
        kept_indices.append(i)
        
        group_dropped = []
        for j in range(i + 1, n):
            if j in dropped_set:
                continue
            dist = float(np.linalg.norm(vectors[i] - vectors[j]))
            if dist <= threshold:
                dropped_set.add(j)
                group_dropped.append(j)
                
        if group_dropped:
            collapsed_groups.append((i, group_dropped))
            
    return kept_indices, collapsed_groups


# ── Diversity map ─────────────────────────────────────────────────────────────

def _build_diversity_map(
    representatives: list["LogSession"],
) -> dict:
    """Produce a diversity map over the representative seed set.

    The map contains:
    - ``total_seeds``       : number of unique seeds kept.
    - ``by_language``       : {lang_code: count} for each detected language.
    - ``by_query_type``     : {tier: count}  (simple / moderate / complex).
    - ``coverage_gaps``     : list of (lang, query_type) cells that have zero
                              representation, drawn from the cross-product of
                              observed languages × observed query types.
    - ``underrepresented``  : cells whose share is ≤ 50 % of the uniform share
                              across all observed (lang, type) cells.
    """
    by_lang: dict[str, int] = defaultdict(int)
    by_type: dict[str, int] = defaultdict(int)
    cell_counts: dict[tuple[str, str], int] = defaultdict(int)

    for s in representatives:
        lang = s.language or "unknown"
        # Infer query-type tier from session metadata when available; otherwise
        # derive a cheap heuristic from presence of agent_turns.
        if s.agent_turns:
            n_tools = sum(
                1 for t in s.agent_turns for p in t.parts if p.part_kind == "tool-call"
            )
            if n_tools == 0:
                tier = "simple"
            elif n_tools <= 3:
                tier = "moderate"
            else:
                tier = "complex"
        else:
            tier = "simple"

        by_lang[lang] += 1
        by_type[tier] += 1
        cell_counts[(lang, tier)] += 1

    total = len(representatives)

    # --- Gaps: (lang, type) pairs with zero count ---------------------------
    observed_langs = list(by_lang.keys())
    observed_types = list(by_type.keys())
    coverage_gaps: list[dict] = []
    for lang in observed_langs:
        for typ in observed_types:
            if cell_counts[(lang, typ)] == 0:
                coverage_gaps.append({"language": lang, "query_type": typ})

    # --- Underrepresented cells (share ≤ 50 % of uniform expectation) -------
    n_cells = len(observed_langs) * len(observed_types)
    uniform_share = (1 / n_cells) if n_cells > 0 else 0.0
    underrepresented: list[dict] = []
    for (lang, typ), count in cell_counts.items():
        share = count / total if total > 0 else 0.0
        if share <= 0.5 * uniform_share:
            underrepresented.append(
                {
                    "language": lang,
                    "query_type": typ,
                    "count": count,
                    "share": round(share, 4),
                    "uniform_expected": round(uniform_share, 4),
                }
            )

    return {
        "total_seeds": total,
        "by_language": dict(by_lang),
        "by_query_type": dict(by_type),
        "coverage_gaps": coverage_gaps,
        "underrepresented": underrepresented,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def deduplicate_sessions(
    sessions: list["LogSession"],
) -> tuple[list["LogSession"], int, dict]:
    """Embed, cluster, and deduplicate *sessions* using greedy radius search with pure numpy.

    Parameters
    ----------
    sessions:
        Full list of LogSession objects (gold + fixable traces).

    Returns
    -------
    (unique_sessions, dropped_count, diversity_map)
        ``unique_sessions`` — one representative per semantic cluster.
        ``dropped_count``   — number of near-duplicate sessions removed.
        ``diversity_map``   — dict describing language / query-type coverage
                              and flagging underrepresented cells.
    """
    if not sessions:
        return [], 0, {"total_seeds": 0, "by_language": {}, "by_query_type": {},
                       "coverage_gaps": [], "underrepresented": []}

    n = len(sessions)

    from lang_utils import ensure_language
    from anonymizer import redact_pii

    texts = []
    for s in sessions:
        ensure_language(s)
        # Redact PII so that identical questions with different PII cluster correctly
        redacted_q, _ = redact_pii(s.user_question, language=s.language or "en")
        texts.append(redacted_q)

    logger.info("Embedding %d sessions with %r …", n, _EMBED_MODEL)
    vectors = _embed(texts)  # (N, D) float32

    logger.info("Running pure numpy radius search (threshold=%.2f) on %d sessions …", _DISTANCE_THRESHOLD, n)
    kept_indices, collapsed_indices_map = _radius_clustering(vectors, _DISTANCE_THRESHOLD)

    kept_indices_set = set(kept_indices)
    unique = [sessions[i] for i in kept_indices]
    dropped = n - len(unique)

    # Compile the collapsed groups for auditing
    collapsed_groups = []
    for rep_idx, dropped_indices in collapsed_indices_map:
        collapsed_groups.append({
            "kept": sessions[rep_idx].user_question,
            "dropped": [sessions[d].user_question for d in dropped_indices]
        })

    if dropped / n > _WARN_THRESHOLD:
        logger.warning(
            "High duplication rate: %d/%d sessions (%.0f%%) collapsed into %d seeds.",
            dropped, n, 100 * dropped / n, len(unique),
        )

    diversity_map = _build_diversity_map(unique)
    diversity_map["collapsed_groups"] = collapsed_groups

    logger.info(
        "Deduplication complete: %d → %d seeds (%d dropped). "
        "Languages seen: %s. Query types: %s.",
        n,
        len(unique),
        dropped,
        list(diversity_map["by_language"].keys()),
        list(diversity_map["by_query_type"].keys()),
    )

    return unique, dropped, diversity_map



