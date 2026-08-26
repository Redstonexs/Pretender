"""Deterministic hybrid memory retrieval: strict CJK/ASCII query
normalization, lexical FTS + optional semantic cosine candidates fused with
fixed RRF (rank positions only), and typed recall hits carrying the
source/strength facts a later MemoryService needs.

FTS-only behavior is preserved exactly when semantic is disabled or
unavailable: the lexical path is unchanged, and RRF over a single list
reproduces its order. This module never calls an LLM and never touches the
gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from pretender.budget import BLOCKED, BudgetManager
from pretender.embed import OptionalEmbeddingService
from pretender.types import ChatKey, LexicalHit, MessageRowId
from pretender.vectors import VectorHit, VectorIndex

# Everything that is not a CJK ideograph, an ASCII letter/digit, or
# whitespace is stripped from a query before it reaches the FTS index.
_ALLOWED = re.compile(r"[^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaffA-Za-z0-9 ]+")
_WS = re.compile(r"\s+")

# Fixed RRF constant (standard value).
_RRF_K = 60.0


def normalize_query(query: str, *, max_terms: int = 16) -> str:
    """Strict CJK/ASCII query normalization.

    Strips every character that is not a CJK ideograph, an ASCII
    letter/digit, or whitespace; collapses whitespace runs; and caps the
    number of whitespace-separated terms. Returns "" for an empty result.
    """
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms <= 0:
        raise ValueError(f"max_terms must be a positive integer, got {max_terms!r}")
    cleaned = _ALLOWED.sub(" ", query)
    cleaned = _WS.sub(" ", cleaned).strip()
    if not cleaned:
        return ""
    terms = cleaned.split(" ")
    if len(terms) > max_terms:
        terms = terms[:max_terms]
    return " ".join(terms)


@dataclass(frozen=True)
class MemoryRecallHit:
    """One hybrid memory recall hit with the source/strength facts a later
    MemoryService needs.

    ``source`` is ``"lexical"`` (FTS only), ``"semantic"`` (cosine only), or
    ``"hybrid"`` (both). ``score`` is the fused RRF score; the per-source
    scores are carried alongside when present.
    """

    chat_key: ChatKey
    memory_id: int
    text: str
    score: float
    source: str
    strength: float
    source_first_msg_id: MessageRowId | None = None
    source_last_msg_id: MessageRowId | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.source not in ("lexical", "semantic", "hybrid"):
            raise ValueError(f"invalid source: {self.source!r}")
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be finite and nonnegative")


def rrf_merge(
    lexical: list[LexicalHit],
    semantic: list[VectorHit],
    *,
    limit: int,
    k: float = _RRF_K,
) -> list[tuple[int, float, str]]:
    """Fixed RRF over rank positions only.

    Each list contributes ``1 / (k + rank)`` (1-based rank) to its
    memory_id. Returns ``(memory_id, score, source)`` sorted by score DESC
    then memory_id ASC — deterministic ties. ``source`` is ``"hybrid"`` when
    a memory appears in both lists, else the single source.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit!r}")
    scores: dict[int, float] = {}
    sources: dict[int, set[str]] = {}
    for rank, hit in enumerate(lexical, start=1):
        scores[hit.memory_id] = scores.get(hit.memory_id, 0.0) + 1.0 / (k + rank)
        sources.setdefault(hit.memory_id, set()).add("lexical")
    for rank, hit in enumerate(semantic, start=1):
        scores[hit.owner_id] = scores.get(hit.owner_id, 0.0) + 1.0 / (k + rank)
        sources.setdefault(hit.owner_id, set()).add("semantic")
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[tuple[int, float, str]] = []
    for memory_id, score in ordered[:limit]:
        src = sources[memory_id]
        label = (
            "hybrid"
            if len(src) == 2
            else ("lexical" if "lexical" in src else "semantic")
        )
        out.append((memory_id, score, label))
    return out


class MemorySearch:
    """Hybrid lexical + optional semantic memory retrieval for one chat.

    ``repo`` is a ``KnowledgeRepository``; ``embed`` is the optional
    embedding service (None disables semantic recall); ``vectors`` is the
    chat-scoped cosine index (defaults to one over ``repo``); ``budget_for``
    is an optional per-chat ``BudgetManager`` resolver — when set, a semantic
    QUERY embed reserves exactly one call under the chat's budget (a single
    query text is one provider batch; a cache hit consumes zero), so the
    planner and the embed share atomic physical budget state.
    """

    def __init__(
        self,
        repo: Any,
        *,
        embed: OptionalEmbeddingService | None = None,
        vectors: VectorIndex | None = None,
        budget_for: Callable[[ChatKey], BudgetManager] | None = None,
    ) -> None:
        self._repo = repo
        self._embed = embed
        self._vectors = vectors if vectors is not None else VectorIndex(repo)
        self._budget_for = budget_for

    async def search(
        self,
        chat_key: ChatKey,
        query: str,
        *,
        limit: int = 10,
        max_terms: int = 16,
        semantic_k: int = 20,
    ) -> list[MemoryRecallHit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit!r}")
        norm = normalize_query(query, max_terms=max_terms)
        if not norm:
            return []

        # Lexical path (always).
        lexical = await self._repo.query_memory(chat_key, norm, limit=limit)

        # Optional semantic path: the active generation is selected AND
        # matched to the configured embedding space BEFORE any query
        # embedding call. A missing active generation, a missing embed
        # revision/space, or a space mismatch => FTS-only with ZERO provider
        # calls — never a wasted embed for a space we cannot use. A semantic
        # QUERY embed reserves exactly one call under the chat's budget (a
        # cache hit consumes zero); a blocked budget degrades this query to
        # FTS-only.
        semantic: list[VectorHit] = []
        if self._embed is not None and self._embed.enabled:
            active = await self._active_generation(self._embed.space_id)
            if active is not None:
                if self._budget_for is not None and not self._embed.cached([norm]):
                    decision = await self._budget_for(chat_key).reserve(
                        chat_key, calls=1
                    )
                    if decision.kind == BLOCKED:
                        active = None  # FTS-only for this query
                if active is not None:
                    result = await self._embed.embed([norm])
                    if result.status == "ok":
                        q = result.vectors[0]
                        if q.shape[0] == active.dim:
                            semantic = await self._vectors.search(
                                chat_key, q, active.model, active.id, semantic_k
                            )

        merged = rrf_merge(lexical, semantic, limit=limit)
        if not merged:
            return []

        mems = await self._repo.get_memories(
            chat_key, [memory_id for memory_id, _, _ in merged]
        )
        by_id = {m.id: m for m in mems}
        lex_by_id = {h.memory_id: h for h in lexical}
        sem_by_id = {h.owner_id: h for h in semantic}

        out: list[MemoryRecallHit] = []
        for memory_id, score, source in merged:
            rec = by_id.get(memory_id)
            if rec is None:
                continue  # memory vanished between read and lookup: skip
            lex = lex_by_id.get(memory_id)
            sem = sem_by_id.get(memory_id)
            out.append(
                MemoryRecallHit(
                    chat_key=chat_key,
                    memory_id=memory_id,
                    text=rec.text,
                    score=score,
                    source=source,
                    strength=rec.strength,
                    source_first_msg_id=rec.source_first_msg_id,
                    source_last_msg_id=rec.source_last_msg_id,
                    lexical_score=lex.score if lex is not None else None,
                    semantic_score=sem.score if sem is not None else None,
                )
            )
        return out

    async def _active_generation(self, space_id: str) -> Any:
        """The ACTIVE embedding generation whose space matches the
        configured ``space_id``, or None. A missing active generation or a
        space mismatch yields None — the caller then performs ZERO query
        embedding calls."""
        for gen in await self._repo.list_embedding_generations():
            if gen.state == "active" and space_id and gen.space_id == space_id:
                return gen
        return None
