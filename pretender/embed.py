"""Optional embedding service: an injected ``Embedder`` or a disabled state,
a content-addressed SHA1 cache, batching, and fail-closed validation.

Design (fail closed at the service boundary, never the gate):
  - ``embedder=None`` -> permanently disabled: ``embed`` returns an explicit
    ``unavailable`` result and performs ZERO provider calls.
  - Texts are embedded in batches, cached by content SHA1, and validated
    (consistent dimension, finite values, nonzero norm).
  - A provider failure or invalid output degrades the service: it returns a
    ``degraded`` result (never raises) and stops calling the provider, so
    downstream retrieval falls back to FTS-only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pretender.seams import Embedder


@dataclass(frozen=True)
class EmbeddingResult:
    """The explicit outcome of an embed request.

    ``status`` is one of:
      - ``"ok"``: ``vectors`` holds one float32 ndarray per input text, in
        input order.
      - ``"unavailable"``: no embedder is configured (or the service is
        disabled); the caller must fall back to FTS-only.
      - ``"degraded"``: the embedder failed or returned invalid vectors;
        the caller must fall back to FTS-only. ``reason`` explains why.
    """

    status: str
    vectors: tuple[np.ndarray, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("ok", "unavailable", "degraded"):
            raise ValueError(f"invalid status: {self.status!r}")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class EmbeddingCache:
    """Content-addressed SHA1 cache of float32 vectors, keyed by the
    embedding ``space_id`` AND the SHA1 of the source text — so the same
    text under a different model/revision space is a distinct cache miss.
    Optional on-disk persistence under ``path``.

    Every read is validated (finite values, expected dimension); an invalid
    cached vector is treated as a miss and never served. A cache read/write
    failure never breaks embedding: disk errors are swallowed and the
    in-memory map still serves the process.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._mem: dict[tuple[str, str], np.ndarray] = {}

    def _key(self, space_id: str, text: str) -> str:
        return hashlib.sha1(f"{space_id}\x00{text}".encode("utf-8")).hexdigest()

    def _file(self, space_id: str, key: str) -> Path:
        assert self._path is not None
        # Provider model/revision identifiers are configuration data, not safe
        # path components (for example `BAAI/bge-m3`). Keep the on-disk key
        # flat and content-addressed so it cannot escape the cache directory.
        space_key = hashlib.sha1(space_id.encode("utf-8")).hexdigest()
        return self._path / f"{space_key}__{key}.f32"

    def get(self, space_id: str, text: str, dim: int | None = None) -> np.ndarray | None:
        key = self._key(space_id, text)
        arr = self._mem.get((space_id, key))
        if arr is None and self._path is not None:
            f = self._file(space_id, key)
            if f.exists():
                try:
                    arr = np.fromfile(f, dtype="<f4")
                except (OSError, ValueError):
                    arr = None
                if arr is not None and arr.size:
                    self._mem[(space_id, key)] = arr
        if arr is None:
            return None
        # Validate on every read: finite and (when known) the right dim.
        if not np.all(np.isfinite(arr)):
            return None
        if dim is not None and arr.shape[0] != dim:
            return None
        return arr

    def put(self, space_id: str, text: str, vec: np.ndarray) -> None:
        key = self._key(space_id, text)
        arr = np.asarray(vec, dtype=np.float32)
        self._mem[(space_id, key)] = arr
        if self._path is not None:
            try:
                self._path.mkdir(parents=True, exist_ok=True)
                arr.astype("<f4", copy=False).tofile(self._file(space_id, key))
            except OSError:
                pass  # a cache write failure never breaks embedding

    def clear(self) -> None:
        self._mem.clear()


class OptionalEmbeddingService:
    """Embed texts through an injected ``Embedder``, or degrade cleanly.

    - ``embedder=None`` -> permanently disabled: ``embed`` returns
      ``unavailable`` and performs ZERO provider calls.
    - ``space_id`` is the canonical embedding space identity (model +
      revision) the service belongs to; the cache is keyed by it so the same
      text under a different space is a distinct miss.
    - Otherwise texts are embedded in batches, cached by (space_id, content
      SHA1), and validated (consistent dimension, finite values, nonzero
      norm). A provider failure or invalid output degrades the service to
      ``degraded`` (never raises).
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        cache: EmbeddingCache | None = None,
        cache_path: str | Path | None = None,
        batch_size: int = 64,
        dim: int | None = None,
        space_id: str = "",
    ) -> None:
        self._embedder = embedder
        self._cache = cache if cache is not None else EmbeddingCache(cache_path)
        self._batch_size = max(1, batch_size)
        self._dim = dim
        self._space_id = space_id
        self._disabled = embedder is None
        self._degraded = False
        self._degraded_reason: str | None = None
        # The dimension observed from the first provider batch when no
        # explicit ``dim`` is configured. Once set, every later provider
        # batch AND every cache read must match it — one embedding dimension
        # per service/space, never silently mixed.
        self._observed_dim: int | None = None

    @property
    def enabled(self) -> bool:
        return not self._disabled and not self._degraded

    @property
    def space_id(self) -> str:
        """The canonical embedding space identity this service belongs to."""
        return self._space_id

    @property
    def batch_size(self) -> int:
        """The provider batch size (texts per provider call). The semantic
        backfill reserves exactly ``ceil(cache-miss texts / batch_size)``
        budget calls against a chat, so its reservation matches the number
        of provider calls ``embed`` will actually make."""
        return self._batch_size

    @property
    def _resolved_dim(self) -> int | None:
        """The authoritative dimension: the configured ``dim`` when given,
        else the dimension locked from the first provider batch."""
        return self._dim if self._dim is not None else self._observed_dim

    def cached(self, texts: list[str]) -> list[str]:
        """The subset of ``texts`` already in the cache (zero-budget hits).

        The semantic backfill uses this to decide whether a chat needs a
        budget reservation: a chat whose texts are ALL cached performs ZERO
        provider calls and ZERO budget reservation. A cached vector whose
        dimension does not match the service's resolved dimension is a miss.
        """
        return [
            t
            for t in texts
            if self._cache.get(self._space_id, t, self._resolved_dim) is not None
        ]

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        if self._disabled:
            return EmbeddingResult(
                status="unavailable", reason="no embedder configured"
            )
        if self._degraded:
            return EmbeddingResult(status="degraded", reason=self._degraded_reason)
        if not texts:
            return EmbeddingResult(status="ok", vectors=())
        assert self._embedder is not None  # not disabled, not degraded

        # Dedupe preserving order; serve cache hits without a provider call.
        unique: list[str] = []
        seen: set[str] = set()
        for t in texts:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        cached: dict[str, np.ndarray] = {}
        misses: list[str] = []
        for t in unique:
            v = self._cache.get(self._space_id, t, self._resolved_dim)
            if v is not None:
                cached[t] = v
            else:
                misses.append(t)

        # Batch-embed the misses.
        embedded: dict[str, np.ndarray] = {}
        for i in range(0, len(misses), self._batch_size):
            batch = misses[i : i + self._batch_size]
            try:
                raw = await self._embedder.embed(batch)
            except Exception as e:  # provider failure -> degrade, never raise
                return self._degrade(f"embedder failure: {e!r}")
            if len(raw) != len(batch):
                return self._degrade("embedder returned the wrong batch size")
            for text, vec in zip(batch, raw):
                try:
                    arr = self._validate(vec)
                except ValueError as e:
                    return self._degrade(f"invalid vector for {text!r}: {e}")
                embedded[text] = arr
                self._cache.put(self._space_id, text, arr)

        # Assemble in the original input order.
        out: list[np.ndarray] = []
        for t in texts:
            v = cached.get(t)
            if v is None:
                v = embedded.get(t)
            if v is None:
                return self._degrade(f"missing vector for {t!r}")
            out.append(v)
        return EmbeddingResult(status="ok", vectors=tuple(out))

    def _validate(self, vec: Any) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be 1-D")
        resolved = self._resolved_dim
        if resolved is not None and arr.shape[0] != resolved:
            raise ValueError(f"dim {arr.shape[0]} != expected {resolved}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("vector contains NaN/inf")
        if float(np.linalg.norm(arr)) == 0.0:
            raise ValueError("zero vector")
        # Lock the dimension from the first provider batch when none was
        # configured: every later batch must match it (one dimension per
        # service/space, never silently mixed).
        if self._dim is None and self._observed_dim is None:
            self._observed_dim = arr.shape[0]
        return arr

    def _degrade(self, reason: str) -> EmbeddingResult:
        self._degraded = True
        self._degraded_reason = reason
        return EmbeddingResult(status="degraded", reason=reason)
