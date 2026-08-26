"""Prompt loading: package defaults overlaid with a user directory.

- ``PromptStore(user_dir)`` resolves a name against the user directory
  first, then the package's ``pretender/prompts/`` defaults. A same-named
  file in the user dir shadows the package default — the whole personality
  is editable by a non-programmer (PLAN.md §3.1).
- ``{{var}}`` rendering with a hard error on missing variables: a prompt
  that silently renders with a hole is how a bot starts talking nonsense.
- mtime hot-reload: ``load`` re-reads a file when its mtime changes, so a
  running bot picks up prompt edits without a restart. A user file that
  appears later shadows the package default on the next load.
- Names are confined to the prompt roots: a name that resolves outside the
  user dir or the package dir is a PromptError, not a file read.
"""

from __future__ import annotations

import re
from pathlib import Path

from pretender.errors import PromptError

PACKAGE_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_VAR_REF = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def render_text(text: str, **variables: object) -> str:
    """Render ``{{var}}`` references. Any reference without a matching
    keyword argument raises PromptError naming the missing variables."""

    missing: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            missing.append(key)
            return match.group(0)
        return str(variables[key])

    rendered = _VAR_REF.sub(substitute, text)
    if missing:
        raise PromptError(
            f"missing prompt variable(s): {', '.join(sorted(set(missing)))}"
        )
    return rendered


class PromptStore:
    """Resolves prompt names against user dir → package defaults, with
    mtime-based caching and hot reload."""

    def __init__(self, user_dir: str | Path | None = None) -> None:
        self.user_dir = Path(user_dir).resolve() if user_dir else None
        self._cache: dict[str, tuple[float, str]] = {}

    # ── resolution ──────────────────────────────────────────────────────────

    def _resolve(self, prompt: str) -> Path | None:
        if not prompt or prompt.startswith("/") or ".." in Path(prompt).parts:
            raise PromptError(f"invalid prompt name: {prompt!r}")
        if self.user_dir is not None:
            candidate = (self.user_dir / prompt).resolve()
            if candidate.is_file() and self._inside(candidate, self.user_dir):
                return candidate
        candidate = (PACKAGE_PROMPT_DIR / prompt).resolve()
        if candidate.is_file() and self._inside(candidate, PACKAGE_PROMPT_DIR):
            return candidate
        return None

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        return path == root or root in path.parents

    # ── loading ─────────────────────────────────────────────────────────────

    def load(self, prompt: str) -> str:
        """Return the prompt text, re-reading when the file's mtime changed."""
        path = self._resolve(prompt)
        if path is None:
            raise PromptError(
                f"prompt {prompt!r} not found"
                + (f" (user dir: {self.user_dir})" if self.user_dir else "")
                + f" (package dir: {PACKAGE_PROMPT_DIR})"
            )
        mtime = path.stat().st_mtime
        cached = self._cache.get(prompt)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        content = path.read_text(encoding="utf-8")
        self._cache[prompt] = (mtime, content)
        return content

    def render(self, prompt: str, **variables: object) -> str:
        """Load and render in one step."""
        return render_text(self.load(prompt), **variables)

    def load_identity(self, identity_file: str | Path) -> str:
        """Load the bot identity text from ``identity_file``.

        ``identity_file`` is resolved through the prompt infrastructure: a
        bare name (or a path whose basename is a prompt name) resolves
        against the user ``prompt_dir`` overlay over the package defaults,
        so the user's ``prompts/identity.txt`` shadows the shipped default.
        An absolute path is read directly. A missing, unreadable, or empty
        (dead) identity file raises ``PromptError``.
        """
        path = Path(identity_file)
        if path.is_absolute():
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as e:
                raise PromptError(f"cannot read identity file {path}: {e}") from e
        else:
            try:
                text = self.load(str(identity_file))
            except PromptError:
                try:
                    text = self.load(path.name)
                except PromptError:
                    raise PromptError(
                        f"identity file {identity_file!r} not found"
                    ) from None
        if not text.strip():
            raise PromptError(f"identity file {identity_file!r} is empty")
        return text

    def invalidate(self, prompt: str | None = None) -> None:
        """Drop cached entries (all, or one). Useful after bulk edits."""
        if prompt is None:
            self._cache.clear()
        else:
            self._cache.pop(prompt, None)

    def __repr__(self) -> str:
        return f"PromptStore(user_dir={self.user_dir!r})"