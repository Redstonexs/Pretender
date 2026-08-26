"""Float32 vector index: ndarray <-> BLOB conversion, a resident per-chat
cache, and deterministic cosine top-k.

Everything here is chat-scoped and local-deterministic: there is NO global
cross-chat matrix. Vectors are rebuildable derived state; the cache is a
resident mirror of the repository's ``vectors`` rows for one
``(chat_key, model, generation)`` and is invalidated on write/delete.

Fail-closed rules:
  - ``blob_to_ndarray`` raises on a wrong-length blob, a dim mismatch, or
    NaN/inf values.
  - ``normalize`` raises on a zero or non-finite norm.
  - ``cosine_top_k`` skips corrupt stored vectors (they never poison the
    result) and normalizes the query internally.
  - ``VectorIndex._load`` skips corrupt rows and never caches them, so a
    corrupt blob degrades to semantic-partial, never a bad result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pretender.types import ChatKey, VectorRow


def ndarray_to_blob(arr: np.ndarray) -> bytes:
    """Explicit little-endian float32 serialization of a 1-D vector."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError("vector must be 1-D")
    return a.astype("<f4", copy=False).tobytes()


def blob_to_ndarray(blob: bytes, dim: int) -> np.ndarray:
    """Parse a float32 BLOB back into a 1-D ndarray.

    Fail closed: a wrong-length blob, a dim mismatch, or NaN/inf values
    raise ``ValueError``.
    """
    if len(blob) != dim * 4:
        raise ValueError(f"blob length {len(blob)} != dim*4 {dim * 4}")
    arr = np.frombuffer(blob, dtype="<f4").copy()
    if arr.shape[0] != dim:
        raise ValueError(f"blob dim {arr.shape[0]} != expected {dim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector contains NaN/inf")
    return arr


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Fail closed on a zero or non-finite norm."""
    a = np.asarray(vec, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError("vector must be 1-D")
    if not np.all(np.isfinite(a)):
        raise ValueError("vector contains NaN/inf")
    norm = float(np.linalg.norm(a))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("zero or non-finite vector norm")
    return a / norm


@dataclass(frozen=True)
class VectorHit:
    """One cosine top-k hit: the owner row id and its cosine similarity."""

    owner_id: int
    score: float
    source_hash: str | None = None
    owner_table: str = "memories"

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


def cosine_top_k(
    query: np.ndarray, vectors: dict[Any, np.ndarray], k: int,
    *, owner_table: str = "memories",
) -> list[VectorHit]:
    """Brute-force-equivalent cosine top-k against a normalized query.

    ``vectors`` maps owner_id -> vector (normalized here). Returns the top
    ``k`` by cosine similarity (the dot product of normalized vectors),
    ties broken deterministically by owner_id ASC. Corrupt entries (wrong
    dim, NaN/inf, zero norm) are skipped and never poison the result.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    q = normalize(query)
    scored: list[VectorHit] = []
    for owner, vec in vectors.items():
        try:
            v = normalize(vec)
        except ValueError:
            continue  # corrupt stored vector: skip, fail closed
        if v.shape != q.shape:
            continue
        if isinstance(owner, tuple):
            table, owner_id = owner
        else:
            table, owner_id = owner_table, owner
        scored.append(VectorHit(
            owner_id=owner_id, owner_table=table,
            score=float(np.dot(q, v)),
        ))
    scored.sort(key=lambda h: (-h.score, h.owner_table, h.owner_id))
    return scored[:k]


class VectorCache:
    """Resident per-``(chat_key, model, generation, vector_revision)`` cache
    of normalized vectors. Each entry maps owner_id -> normalized ndarray.
    The durable ``vector_revision`` is part of the key, so a direct repo
    vector mutation (which bumps the generation's revision) makes the cached
    entry unreachable and forces a reload on the next search. Invalidation
    on write/delete is explicit and never poisons the cache."""

    def __init__(self) -> None:
        self._cache: dict[tuple[Any, str, int, int, str], dict[Any, np.ndarray]] = {}

    def _key(
        self, chat_key: ChatKey, model: str, generation: int, vector_revision: int,
        owner_table: str = "memories",
    ) -> tuple[Any, str, int, int, str]:
        return (chat_key, model, generation, vector_revision, owner_table)

    def get(
        self, chat_key: ChatKey, model: str, generation: int, vector_revision: int,
        *, owner_table: str = "memories",
    ) -> dict[Any, np.ndarray] | None:
        return self._cache.get(self._key(
            chat_key, model, generation, vector_revision, owner_table
        ))

    def set(
        self,
        chat_key: ChatKey,
        model: str,
        generation: int,
        vector_revision: int,
        vectors: dict[Any, np.ndarray],
        *, owner_table: str = "memories",
    ) -> None:
        normalized = {
            owner_id if owner_table == "memories" and not isinstance(owner_id, tuple)
            else ((owner_table, owner_id) if not isinstance(owner_id, tuple) else owner_id): vec
            for owner_id, vec in vectors.items()
        }
        self._cache[self._key(
            chat_key, model, generation, vector_revision, owner_table
        )] = dict(normalized)

    def invalidate(self, chat_key: ChatKey, model: str, generation: int) -> None:
        # Drop every revision of this (chat, model, generation).
        prefix = (chat_key, model, generation)
        for key in [k for k in self._cache if k[:3] == prefix]:
            self._cache.pop(key, None)


class VectorIndex:
    """Chat-scoped cosine index over the repository's vector rows, with a
    resident cache keyed by the durable generation ``vector_revision``.

    ``upsert``/``delete`` write through to the repository and invalidate the
    cache for that ``(chat_key, model, generation)``. ``search`` loads (and
    caches) the chat's vectors for one model/generation and returns the
    deterministic cosine top-k. Because the cache key includes the
    generation's durable ``vector_revision``, a DIRECT repo vector mutation
    (bypassing this index) is visible on the next search. Corrupt rows are
    skipped on load and never cached.
    """

    def __init__(self, repo: Any, cache: VectorCache | None = None) -> None:
        self._repo = repo
        self._cache = cache if cache is not None else VectorCache()

    async def upsert(self, chat_key: ChatKey, row: VectorRow) -> None:
        await self._repo.upsert_vector(chat_key, row)
        self._cache.invalidate(chat_key, row.model, row.generation)

    async def delete(
        self,
        chat_key: ChatKey,
        owner_table: str,
        owner_id: int,
        model: str,
        generation: int,
    ) -> bool:
        ok = await self._repo.delete_vector(
            chat_key, owner_table, owner_id, model, generation
        )
        self._cache.invalidate(chat_key, model, generation)
        return ok

    async def invalidate(self, chat_key: ChatKey, model: str, generation: int) -> None:
        self._cache.invalidate(chat_key, model, generation)

    async def _vector_revision(self, generation: int) -> int:
        gen = await self._repo.get_embedding_generation(generation)
        return gen.vector_revision if gen is not None else 0

    async def _load(
        self, chat_key: ChatKey, model: str, generation: int,
        owner_table: str = "memories",
    ) -> dict[Any, np.ndarray]:
        revision = await self._vector_revision(generation)
        cached = self._cache.get(
            chat_key, model, generation, revision, owner_table=owner_table
        )
        if cached is not None:
            return cached
        try:
            rows = await self._repo.list_vectors(
                chat_key, model, generation, owner_table=owner_table
            )
        except TypeError:
            rows = await self._repo.list_vectors(chat_key, model, generation)
        vectors: dict[Any, np.ndarray] = {}
        gen = await self._repo.get_embedding_generation(generation)
        if gen is None or gen.model != model:
            self._cache.set(
                chat_key, model, generation, revision, {}, owner_table=owner_table
            )
            return {}
        for row in rows:
            if (
                row.owner_table != owner_table
                or gen is None
                or row.model != gen.model
                or row.dim != gen.dim
                or row.generation != generation
            ):
                continue
            try:
                arr = blob_to_ndarray(row.blob, row.dim)
                key = row.owner_id if owner_table == "memories" else (
                    row.owner_table, row.owner_id
                )
                vectors[key] = normalize(arr)
            except ValueError:
                continue  # corrupt row: skip, never cache
        self._cache.set(
            chat_key, model, generation, revision, vectors,
            owner_table=owner_table,
        )
        return vectors

    async def search(
        self,
        chat_key: ChatKey,
        query: np.ndarray,
        model: str,
        generation: int,
        k: int,
        *,
        owner_table: str = "memories",
    ) -> list[VectorHit]:
        if owner_table not in ("memories", "records"):
            raise ValueError(f"unsupported vector owner table: {owner_table!r}")
        vectors = await self._load(chat_key, model, generation, owner_table)
        return cosine_top_k(query, vectors, k, owner_table=owner_table)
