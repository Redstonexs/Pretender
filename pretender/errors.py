"""Error taxonomy for Pretender.

The only distinction that matters at runtime is Transient vs Permanent:
every retry decision in the codebase keys off it. Transient errors may
succeed if retried (network blips, 429/5xx, timeouts); Permanent errors
will never succeed without a change of input (bad config, bad request,
schema violations).

Everything else derives from one of the two so callers can catch a single
base class when they do not care which kind it is.
"""

from __future__ import annotations


class PretenderError(Exception):
    """Base class for all Pretender errors."""


class TransientError(PretenderError):
    """Retryable: the operation may succeed if attempted again later."""


class PermanentError(PretenderError):
    """Not retryable: the operation will not succeed without a change of input."""


class ConfigError(PermanentError):
    """Invalid configuration: unknown keys, bad TOML, missing env vars, bad values."""


class PromptError(PermanentError):
    """Prompt loading or rendering failure: missing file, missing {{var}}."""


class RegistryError(PermanentError):
    """Registry misuse: duplicate registration, shape violation, unknown name."""


class AdapterError(PretenderError):
    """Adapter-level failure. Subclasses may be Transient or Permanent;
    adapters should raise TransientError/PermanentError directly where the
    distinction is known (e.g. connection refused vs. bad action name)."""


class AdapterNotReady(TransientError, AdapterError):
    """The adapter proved no write could have started (no ready connection).

    Outbox may safely return this one narrow pre-write failure from in-flight
    to pending. Connection loss after ``send`` begins remains ambiguous and
    must stay in-flight for at-most-once delivery.
    """


class ToolError(PermanentError):
    """Tool definition or dispatch failure (tool layer lands in a later phase)."""


class RepoError(PermanentError):
    """Repository precondition violation: missing chat identity, conflicting
    idempotency key, cross-chat outbox item, missing outbox row, ..."""


class ClaimError(PermanentError):
    """A cycle tried to renew or finish a claim it does not own, whose lease
    has expired, or whose start boundary no longer matches the chat cursor.
    The operation changed nothing."""


class LLMError(PretenderError):
    """Base class for all LLM-layer failures (network, provider, parsing).

    The retry decision keys off ``TransientError`` exactly, never off
    ``LLMError`` itself: ``LLMTransientError`` and ``LLMPermanentError`` are
    the only two kinds this layer raises.
    """


class LLMTransientError(TransientError, LLMError):
    """Retryable LLM failure: network blip, timeout, HTTP 429/5xx, or an
    already-expired deadline. Retrying (with a fresh deadline) may succeed."""


class LLMPermanentError(PermanentError, LLMError):
    """Not retryable: unknown profile, malformed transcript/request, malformed
    provider payload, or HTTP 4xx. Retrying cannot succeed without a change
    of input."""
