"""Typed registration, deterministic ordering, shape validation, and the
three hooks (on_event / pre_send / on_cycle_end).

Design rules from PLAN.md §3:
  - Registration validates shape at boot, because structural typing is
    compile-time only. A ``Registry`` built with a Protocol checks every
    registered item against it and raises RegistryError on a mismatch.
  - Ordering is deterministic: insertion order, and ``replace=True`` keeps
    the original slot rather than moving to the end.
  - The score is nothing but a sum of registered GateFeatures; the five
    built-ins register exactly like a third-party one.
  - Only three hook points exist. pre_gate/post_gate would duplicate
    @gate_feature; post_reply/pre_send would duplicate @stage.

Phase 6 P6.6 (explicit-trust plugins):
  - ``HookBus`` is BOUNDED (every hook invocation is bounded by
    ``hook_timeout_s``) and FREEZABLE: ``on_event`` is observational and
    fail-open, ``pre_send`` is fail-closed (a timeout/error suppresses the
    output), ``on_cycle_end`` is contained. After ``freeze()`` no hook can
    be registered — no runtime mutation, no hot reload.
  - ``PluginLoader`` deterministically resolves ONLY the configured
    ``plugins.paths`` module files and the explicit ``plugins.entry_points``
    names (no auto-discovery), calls each plugin's ``setup`` ONCE with a
    disposable ``PluginAPI`` (staging registries + frozen Config — never the
    App/repo/adapter/raw clients), then validates, dedupes, and freezes the
    staging registries. Any failure raises RegistryError and leaves no
    usable partial registry.
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, cast

from pretender.errors import RegistryError
from pretender.log import get_logger
from pretender.types import AdapterEvent, AdapterSpec, ChatKey, DecisionTrace, LearnerSpec, Outgoing

log = get_logger("registry")

T = TypeVar("T")

Hook = Callable[..., Any]


class Registry(Generic[T]):
    """A typed, ordered collection of extension items.

    ``protocol`` is a runtime_checkable Protocol (see seams.py); when given,
    every registered item is shape-validated against it. Items are keyed by
    their ``name`` attribute, or by the explicit ``name=`` argument.
    """

    def __init__(self, name: str, protocol: type[Any] | None = None) -> None:
        self.name = name
        self._protocol = protocol
        self._items: dict[str, T] = {}
        self._order: list[str] = []

    # ── registration ────────────────────────────────────────────────────────

    def register(
        self,
        item: T | None = None,
        *,
        replace: bool = False,
        name: str | None = None,
    ) -> T | Callable[[T], T]:
        """Register ``item``; usable directly or as a decorator.

        ``@reg.register`` works when the item has a ``name`` attribute.
        Duplicate names raise RegistryError unless ``replace=True``, which
        swaps the item in place (keeping its original position).
        """
        if item is None:
            def decorator(x: T) -> T:
                self.register(x, replace=replace, name=name)
                return x

            return decorator

        key = name or getattr(item, "name", None)
        if key is None:
            raise RegistryError(
                f"{self.name}: item has no 'name' attribute; pass name= explicitly"
            )
        if self._protocol is not None:
            self._validate_shape(item)
        if key in self._items and not replace:
            raise RegistryError(
                f"{self.name}: {key!r} is already registered (use replace=True to shadow)"
            )
        if key not in self._items:
            self._order.append(key)
        self._items[key] = item
        return item

    def unregister(self, name: str) -> None:
        if name not in self._items:
            raise RegistryError(f"{self.name}: {name!r} is not registered")
        del self._items[name]
        self._order.remove(name)

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()

    # ── shape validation ────────────────────────────────────────────────────

    def _validate_shape(self, item: T) -> None:
        protocol = self._protocol
        assert protocol is not None
        missing_data: list[str] = []
        missing_methods: list[str] = []
        for attr, annotation in getattr(protocol, "__annotations__", {}).items():
            if attr.startswith("_"):
                continue
            if not hasattr(item, attr):
                missing_data.append(attr)
        for attr, member in vars(protocol).items():
            if attr.startswith("_"):
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if callable(member) and not callable(getattr(item, attr, None)):
                missing_methods.append(attr)
        if missing_data or missing_methods:
            raise RegistryError(
                f"{self.name}: item {getattr(item, 'name', item)!r} does not satisfy "
                f"protocol {protocol.__name__}: "
                f"missing attributes {missing_data}, missing methods {missing_methods}"
            )

    # ── lookup ──────────────────────────────────────────────────────────────

    def get(self, name: str) -> T | None:
        return self._items.get(name)

    def require(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise RegistryError(
                f"{self.name}: {name!r} is not registered (registered: {self.names()})"
            ) from None

    def all(self) -> tuple[T, ...]:
        """All items in deterministic insertion order."""
        return tuple(self._items[k] for k in self._order)

    def names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def __iter__(self):
        return iter(self.all())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


class StagingRegistry(Generic[T]):
    """A frozen staging registry for later explicit trusted plugin
    discovery (Phase 6 foundation).

    Registration is deterministic (insertion order), duplicates are
    rejected, and ``freeze()`` permanently prevents any further mutation —
    after freeze, register/unregister/clear all raise RegistryError.
    Protected core names are metadata the runtime sets at construction: a
    protected name can only be replaced by an item whose name is on the
    replacement allowlist, so a third-party item can never silently shadow
    a core extension. This class does NOT import or discover plugins — it
    is the primitive the discovery lane will build on.
    """

    def __init__(
        self,
        name: str,
        protocol: type[Any] | None = None,
        *,
        protected: tuple[str, ...] = (),
        allow_replace: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self._protocol = protocol
        self._protected = frozenset(protected)
        self._allow_replace = frozenset(allow_replace)
        self._items: dict[str, T] = {}
        self._order: list[str] = []
        self._frozen = False

    # ── registration ────────────────────────────────────────────────────────

    def register(
        self,
        item: T | None = None,
        *,
        replace: bool = False,
        name: str | None = None,
    ) -> T | Callable[[T], T]:
        """Register ``item``; usable directly or as a decorator.

        Duplicate names raise RegistryError unless ``replace=True``, which
        swaps the item in place (keeping its original position). A
        protected core name is replaceable ONLY when it is on the
        replacement allowlist. A frozen registry rejects every mutation.
        """
        if self._frozen:
            raise RegistryError(f"{self.name}: registry is frozen")
        if item is None:
            def decorator(x: T) -> T:
                self.register(x, replace=replace, name=name)
                return x

            return decorator

        key = name or getattr(item, "name", None)
        if key is None:
            raise RegistryError(
                f"{self.name}: item has no 'name' attribute; pass name= explicitly"
            )
        if self._protocol is not None:
            self._validate_shape(item)
        if key in self._items and not replace:
            raise RegistryError(
                f"{self.name}: {key!r} is already registered (use replace=True to shadow)"
            )
        if key in self._items and replace:
            if self.name == "output" and key == "sanitize":
                raise RegistryError(
                    "output: 'sanitize' is the mandatory core stage and is never replaceable"
                )
            if key in self._protected and key not in self._allow_replace:
                raise RegistryError(
                    f"{self.name}: {key!r} is a protected core name and is not on"
                    " the replacement allowlist"
                )
        if key not in self._items:
            self._order.append(key)
        self._items[key] = item
        return item

    def unregister(self, name: str) -> None:
        if self._frozen:
            raise RegistryError(f"{self.name}: registry is frozen")
        if name not in self._items:
            raise RegistryError(f"{self.name}: {name!r} is not registered")
        del self._items[name]
        self._order.remove(name)

    def clear(self) -> None:
        if self._frozen:
            raise RegistryError(f"{self.name}: registry is frozen")
        self._items.clear()
        self._order.clear()

    # ── freeze ──────────────────────────────────────────────────────────────

    def freeze(self) -> None:
        """Permanently prevent any further mutation. Idempotent."""
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ── protected core names / replacement allowlist metadata ───────────────

    def protected_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._protected))

    def replacement_allowlist(self) -> tuple[str, ...]:
        return tuple(sorted(self._allow_replace))

    # ── shape validation ────────────────────────────────────────────────────

    def _validate_shape(self, item: T) -> None:
        protocol = self._protocol
        assert protocol is not None
        missing_data: list[str] = []
        missing_methods: list[str] = []
        for attr, annotation in getattr(protocol, "__annotations__", {}).items():
            if attr.startswith("_"):
                continue
            if not hasattr(item, attr):
                missing_data.append(attr)
        for attr, member in vars(protocol).items():
            if attr.startswith("_"):
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if callable(member) and not callable(getattr(item, attr, None)):
                missing_methods.append(attr)
        if missing_data or missing_methods:
            raise RegistryError(
                f"{self.name}: item {getattr(item, 'name', item)!r} does not satisfy "
                f"protocol {protocol.__name__}: "
                f"missing attributes {missing_data}, missing methods {missing_methods}"
            )

    # ── lookup ──────────────────────────────────────────────────────────────

    def get(self, name: str) -> T | None:
        return self._items.get(name)

    def require(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise RegistryError(
                f"{self.name}: {name!r} is not registered (registered: {self.names()})"
            ) from None

    def all(self) -> tuple[T, ...]:
        """All items in deterministic insertion order."""
        return tuple(self._items[k] for k in self._order)

    def names(self) -> tuple[str, ...]:
        return tuple(self._order)

    def __iter__(self):
        return iter(self.all())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


class PluginRegistry(StagingRegistry[Any]):
    """The staging registry for later explicit trusted plugin discovery.

    Validates every registered item against the ``Plugin`` protocol shape
    (``name`` + optional ``setup``) and freezes before the runtime hands it
    to the discovery lane. No plugin is imported or discovered here.
    """

    def __init__(
        self,
        *,
        protected: tuple[str, ...] = (),
        allow_replace: tuple[str, ...] = (),
    ) -> None:
        from pretender.seams import Plugin

        super().__init__(
            "plugins",
            protocol=Plugin,
            protected=protected,
            allow_replace=allow_replace,
        )


class HookBus:
    """The three hook points, dispatched in registration order.

    Hooks may be sync or async; the bus awaits when needed. Every hook
    invocation is BOUNDED by ``timeout_s`` (a timed-out hook is treated as
    a hook failure). ``pre_send`` hooks chain: each may return a modified
    Outgoing (replacing the one passed in) or None (keeping it).

    Failure semantics (Phase 6 P6.6):
      - ``on_event`` is OBSERVATIONAL and FAIL-OPEN: an error or timeout is
        contained and logged, never propagated — the event flow continues.
      - ``pre_send`` is FAIL-CLOSED: an error or timeout returns None, which
        the caller must treat as "no output" (the message is suppressed).
      - ``on_cycle_end`` is CONTAINED: an error or timeout is logged and
        never propagates.

    ``freeze()`` permanently prevents further registration — the runtime
    freezes the bus after plugin setup, so there is no runtime mutation and
    no hot reload.
    """

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self._timeout_s = float(timeout_s)
        self._on_event: list[Hook] = []
        self._pre_send: list[Hook] = []
        self._on_cycle_end: list[Hook] = []
        self._frozen = False

    # ── registration (usable as decorators) ─────────────────────────────────

    def _check_frozen(self) -> None:
        if self._frozen:
            raise RegistryError("hook bus is frozen")

    def on_event(self, fn: Hook) -> Hook:
        self._check_frozen()
        self._on_event.append(fn)
        return fn

    def pre_send(self, fn: Hook) -> Hook:
        self._check_frozen()
        self._pre_send.append(fn)
        return fn

    def on_cycle_end(self, fn: Hook) -> Hook:
        self._check_frozen()
        self._on_cycle_end.append(fn)
        return fn

    def unregister_on_event(self, fn: Hook) -> None:
        self._check_frozen()
        self._on_event.remove(fn)

    def unregister_pre_send(self, fn: Hook) -> None:
        self._check_frozen()
        self._pre_send.remove(fn)

    def unregister_on_cycle_end(self, fn: Hook) -> None:
        self._check_frozen()
        self._on_cycle_end.remove(fn)

    def clear(self) -> None:
        self._check_frozen()
        self._on_event.clear()
        self._pre_send.clear()
        self._on_cycle_end.clear()

    # ── freeze ──────────────────────────────────────────────────────────────

    def freeze(self) -> None:
        """Permanently prevent any further registration. Idempotent."""
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ── dispatch ────────────────────────────────────────────────────────────

    async def _invoke_bounded(self, fn: Hook, *args: Any) -> Any:
        """Run sync hooks off-loop and bound both sync and async phases.

        Calling a synchronous plugin before creating an awaitable would stall
        the event loop before ``wait_for`` can help. A daemon worker preserves
        the timeout contract without making loop shutdown wait for a timed-out
        plugin thread.
        """
        if inspect.iscoroutinefunction(fn):
            return await asyncio.wait_for(fn(*args), timeout=self._timeout_s)

        loop = asyncio.get_running_loop()
        result: asyncio.Future[Any] = loop.create_future()

        def finish(value: Any = None, error: BaseException | None = None) -> None:
            if result.done():
                return
            if error is not None:
                result.set_exception(error)
            else:
                result.set_result(value)

        def worker() -> None:
            try:
                value = fn(*args)
            except Exception as exc:
                try:
                    loop.call_soon_threadsafe(finish, None, exc)
                except RuntimeError:
                    # The bounded caller may have already torn down its loop
                    # after a timeout; daemon workers must not leak an
                    # unhandled callback during shutdown.
                    pass
            else:
                try:
                    loop.call_soon_threadsafe(finish, value, None)
                except RuntimeError:
                    pass

        threading.Thread(
            target=worker, name="pretender-plugin-hook", daemon=True
        ).start()
        value = await asyncio.wait_for(result, timeout=self._timeout_s)
        if inspect.isawaitable(value):
            return await asyncio.wait_for(value, timeout=self._timeout_s)
        return value

    async def emit_event(self, event: AdapterEvent) -> None:
        """Observational, fail-open: every hook runs in order; an error or
        timeout is contained and logged, never propagated."""
        for fn in list(self._on_event):
            try:
                await self._invoke_bounded(fn, event)
            except asyncio.TimeoutError:
                log.warning(
                    "on_event hook %r timed out after %gs (contained)",
                    getattr(fn, "__name__", fn),
                    self._timeout_s,
                )
            except Exception:
                log.warning(
                    "on_event hook %r failed (contained)",
                    getattr(fn, "__name__", fn),
                    exc_info=True,
                )

    async def emit_pre_send(self, out: Outgoing) -> Outgoing | None:
        """Fail-closed: hooks chain in order; a timeout or error returns
        None, which the caller must treat as "no output" (the message is
        suppressed and never persisted)."""
        for fn in list(self._pre_send):
            try:
                result = await self._invoke_bounded(fn, out)
            except asyncio.TimeoutError:
                log.warning(
                    "pre_send hook %r timed out after %gs; suppressing output",
                    getattr(fn, "__name__", fn),
                    self._timeout_s,
                )
                return None
            except Exception:
                log.warning(
                    "pre_send hook %r failed; suppressing output",
                    getattr(fn, "__name__", fn),
                    exc_info=True,
                )
                return None
            if result is not None:
                out = result
        return out

    async def emit_cycle_end(
        self, chat_key: ChatKey, trace: DecisionTrace, end_reason: str
    ) -> None:
        """Contained: every hook runs in order; an error or timeout is
        logged and never propagated."""
        for fn in list(self._on_cycle_end):
            try:
                await self._invoke_bounded(fn, chat_key, trace, end_reason)
            except asyncio.TimeoutError:
                log.warning(
                    "on_cycle_end hook %r timed out after %gs (contained)",
                    getattr(fn, "__name__", fn),
                    self._timeout_s,
                )
            except Exception:
                log.warning(
                    "on_cycle_end hook %r failed (contained)",
                    getattr(fn, "__name__", fn),
                    exc_info=True,
                )

    def __len__(self) -> int:
        return len(self._on_event) + len(self._pre_send) + len(self._on_cycle_end)


# ── Explicit-trust plugin loading (Phase 6 P6.6) ─────────────────────────────

class PluginAPI:
    """The disposable staging API handed to each plugin's ``setup``.

    Exposes ONLY the staging registries (gate features, output stages,
    declarative learners/adapters, tools, hooks) and the frozen Config — never the App, the repository,
    the adapter, or any raw client. A plugin registers its extensions here;
    the loader validates, dedupes, and freezes them before the runtime uses
    them. The API is single-use: after the loader finishes, the registries
    are frozen and the API is discarded.
    """

    def __init__(
        self,
        *,
        gate_features: StagingRegistry[Any],
        output_stages: StagingRegistry[Any],
        tools: StagingRegistry[Any],
        learners: StagingRegistry[LearnerSpec],
        adapters: StagingRegistry[AdapterSpec],
        hooks: HookBus,
        config: Any,
    ) -> None:
        self.gate_features = gate_features
        self.output_stages = output_stages
        self.tools = tools
        # These registries contain frozen declarative data only.  In
        # particular they do not expose an adapter instance, repository, LLM,
        # clock, or factory capable of obtaining one.
        self.learners = learners
        self.adapters = adapters
        self.hooks = hooks
        self.config = config


@dataclass(frozen=True)
class PluginLoadResult:
    """The frozen outcome of plugin loading.

    ``plugin_names`` is the deterministic ordered fingerprint of the loaded
    plugins; the staging registries are frozen and ready to seed the live
    Gate / tool registry / output pipeline, and ``hooks`` is the frozen
    bounded HookBus.
    """

    plugin_names: tuple[str, ...]
    gate_features: StagingRegistry[Any]
    output_stages: StagingRegistry[Any]
    tools: StagingRegistry[Any]
    learners: StagingRegistry[LearnerSpec]
    adapters: StagingRegistry[AdapterSpec]
    hooks: HookBus
    plugin_manifest: tuple["PluginManifest", ...] = ()


@dataclass(frozen=True)
class PluginManifest:
    """Import-free identity of one explicitly configured plugin.

    ``content_hash`` is over the plugin source for module paths, and over the
    immutable entry-point descriptor/source files for entry points.  The manifest is kept
    deliberately declarative so replay and dry-run can calculate it without
    importing plugin code.
    """

    name: str
    version: str
    source: str
    content_hash: str

    @property
    def identity(self) -> str:
        """Stable human-facing plugin identity (the declared name)."""
        return self.name

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "hash": self.content_hash,
        }


def _literal_assignments(source: str) -> dict[str, Any]:
    """Read simple manifest assignments without executing a module."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RegistryError(f"cannot parse plugin manifest: {exc}") from exc
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            values[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    manifest = values.get("manifest", values.get("PLUGIN_MANIFEST"))
    if isinstance(manifest, dict):
        values.update(manifest)
    return values


def _path_plugin_manifest(path: str) -> PluginManifest:
    p = Path(path).resolve()
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise RegistryError(f"cannot read plugin module {p}: {exc}") from exc
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegistryError(f"cannot decode plugin manifest for {p!s}: {exc}") from exc
    values = _literal_assignments(source)
    name = values.get("name", values.get("PLUGIN_NAME", values.get("identity")))
    version = values.get("version", values.get("PLUGIN_VERSION", values.get("__version__")))
    if not isinstance(name, str) or not name.strip():
        raise RegistryError(f"plugin {p!s} must expose a 'name' string")
    if not isinstance(version, str) or not version.strip():
        raise RegistryError(f"plugin {name!r} must expose an explicit 'version' string")
    return PluginManifest(name.strip(), version.strip(), str(p), hashlib.sha256(raw).hexdigest())


def configured_plugin_manifest(cfg: Any) -> tuple[PluginManifest, ...]:
    """Return the explicit plugin manifest without importing plugin code."""
    manifests = [_path_plugin_manifest(path) for path in cfg.plugins.paths]
    if cfg.plugins.entry_points:
        from importlib.metadata import entry_points

        eps = entry_points(group="pretender.plugins")
        for requested in cfg.plugins.entry_points:
            matches = [ep for ep in eps if ep.name == requested]
            if not matches:
                raise RegistryError(
                    f"no entry point named {requested!r} in group 'pretender.plugins'"
                )
            if len(matches) != 1:
                raise RegistryError(f"ambiguous entry point {requested!r}")
            ep = matches[0]
            dist = getattr(ep, "dist", None)
            dist_name = getattr(dist, "name", "") or ""
            version = getattr(dist, "version", None) or "0"
            source = f"entry_point:{ep.name}:{ep.value}:{dist_name}"
            hasher = hashlib.sha256()
            hasher.update(source.encode("utf-8"))
            files = getattr(dist, "files", None) or ()
            for file in sorted(
                (file for file in files if str(file).endswith(".py")),
                key=str,
            ):
                try:
                    locate_file = getattr(dist, "locate_file", None)
                    if not callable(locate_file):
                        raise AttributeError("distribution has no locate_file")
                    hasher.update(str(file).encode("utf-8"))
                    hasher.update(Path(cast(Any, locate_file)(file)).read_bytes())
                except (AttributeError, OSError):
                    # Distribution metadata can be incomplete (notably in
                    # zipped/test distributions); the immutable entry-point
                    # descriptor still gives a deterministic identity.
                    hasher.update(str(file).encode("utf-8"))
            digest = hasher.hexdigest()
            manifests.append(PluginManifest(ep.name, str(version), source, digest))
    return tuple(manifests)


def _implementation_identity(item: Any) -> dict[str, str]:
    fn = getattr(item, "contribute", None)
    owner = type(item)
    module = getattr(owner, "__module__", "")
    qualname = getattr(owner, "__qualname__", owner.__name__)
    if fn is not None:
        code = getattr(fn, "__code__", None)
        source = ""
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            source = repr(code.co_code if code is not None else fn)
    else:
        source = repr(item)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {"name": str(getattr(item, "name", "")), "module": module,
            "qualname": qualname, "hash": digest}


def feature_implementation_fingerprint(gate: Any) -> tuple[dict[str, str], ...]:
    current = getattr(gate, "_current_features", None)
    features = cast(tuple[Any, ...], current()) if callable(current) else ()
    return tuple(_implementation_identity(feature) for feature in features)


def load_plugin_module(path: str) -> Any:
    """Import ONE configured plugin module file (deterministic, explicit).

    The module is loaded under a unique synthetic name derived from the
    resolved path, so two plugin files with the same basename never collide
    and the name is stable across processes. Any import failure raises
    ``RegistryError``.
    """
    import hashlib
    import importlib.util

    p = Path(path)
    digest = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:12]
    module_name = f"pretender_plugin_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise RegistryError(f"cannot load plugin module {p}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RegistryError(f"cannot import plugin module {p}: {e}") from e
    return module


def load_plugin_entry_point(name: str) -> Any:
    """Resolve ONE explicitly named entry point in the ``pretender.plugins``
    group. A missing or ambiguous name raises ``RegistryError``."""
    from importlib.metadata import entry_points

    eps = entry_points(group="pretender.plugins")
    matches = [ep for ep in eps if ep.name == name]
    if not matches:
        raise RegistryError(
            f"no entry point named {name!r} in group 'pretender.plugins'"
        )
    if len(matches) > 1:
        raise RegistryError(f"ambiguous entry point {name!r}")
    try:
        return matches[0].load()
    except Exception as e:
        raise RegistryError(f"cannot load entry point {name!r}: {e}") from e


class PluginLoader:
    """Deterministic explicit-trust plugin loading (Phase 6 P6.6).

    Resolves ONLY the configured ``plugins.paths`` module files and the
    explicit ``plugins.entry_points`` names — no auto-discovery, no hot
    reload, and no import before the replacement allowlist is in place (the
    staging registries are constructed with the protected names and the
    operator allowlist BEFORE any plugin is imported). Each plugin's
    ``setup`` is called ONCE with a disposable ``PluginAPI``; every
    registration is shape-validated and duplicate-rejected by the staging
    registries, and the registries are frozen before the runtime uses them.
    Any resolve/import/setup/validation failure raises ``RegistryError``
    and leaves no usable partial registry (the caller aborts startup).
    """

    #: Protected core gate-feature names (the five built-ins).
    _PROTECTED_GATE_FEATURES = (
        "relevance",
        "content",
        "pressure",
        "presence",
        "frequency",
    )
    #: Protected core output-stage names.
    _PROTECTED_OUTPUT_STAGES = ("sanitize",)
    #: Protected terminal-intent tool names (the core verdicts + media sends).
    _PROTECTED_TOOLS = (
        "reply",
        "wait",
        "no_action",
        "send_emoji",
        "send_image",
    )

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg

    @staticmethod
    def manifest(cfg: Any) -> tuple[PluginManifest, ...]:
        """Return the configured manifest without importing any plugin."""
        return configured_plugin_manifest(cfg)

    def load(self) -> PluginLoadResult:
        from pretender.gate import default_features
        from pretender.learn.specs import SPECS
        from pretender.output.pipeline import OutputPipeline
        from pretender.seams import GateFeature, OutputStage
        from pretender.tools.core import core_tool_specs

        allow = self._cfg.plugins.allow_replace
        gate_features: StagingRegistry[Any] = StagingRegistry(
            "gate_features",
            protocol=GateFeature,
            protected=self._PROTECTED_GATE_FEATURES,
            allow_replace=allow,
        )
        for feature in default_features():
            gate_features.register(feature)
        output_stages: StagingRegistry[Any] = StagingRegistry(
            "output",
            protocol=OutputStage,
            protected=self._PROTECTED_OUTPUT_STAGES,
            # The core sanitizer is a mandatory final boundary and is never
            # replaceable, even when other core names are allowlisted.
            allow_replace=(),
        )
        builtin_output = OutputPipeline()
        for name in ("sanitize", "split", "typo"):
            stage = builtin_output.get(name)
            assert stage is not None
            output_stages.register(stage)
        tools: StagingRegistry[Any] = StagingRegistry(
            "tools",
            protocol=None,
            protected=self._PROTECTED_TOOLS,
            allow_replace=allow,
        )
        for spec in core_tool_specs():
            tools.register(spec)
        learners: StagingRegistry[LearnerSpec] = StagingRegistry("learners")
        for spec in SPECS.values():
            learners.register(spec)
        adapters: StagingRegistry[AdapterSpec] = StagingRegistry("adapters")
        adapters.register(AdapterSpec("console"))
        adapters.register(AdapterSpec(
            "onebot", frozenset(("quote", "at", "image", "sticker", "history", "forward"))
        ))
        hooks = HookBus(timeout_s=self._cfg.plugins.hook_timeout_s)

        api = PluginAPI(
            gate_features=gate_features,
            output_stages=output_stages,
            tools=tools,
            learners=learners,
            adapters=adapters,
            hooks=hooks,
            config=self._cfg,
        )
        names: list[str] = []
        # Import/setup errors retain their precise loader diagnostics.  The
        # manifest is still calculated before setup, but after resolution so a
        # broken module cannot be mistaken for a malformed static manifest.
        plugins = tuple(self._iter_plugins())
        manifests = configured_plugin_manifest(self._cfg)
        if len(plugins) != len(manifests):
            raise RegistryError("plugin manifest count mismatch")
        for plugin, manifest in zip(plugins, manifests):
            declared = getattr(plugin, "manifest", getattr(plugin, "PLUGIN_MANIFEST", None))
            name = getattr(plugin, "name", getattr(plugin, "PLUGIN_NAME", None))
            if isinstance(declared, dict):
                name = declared.get("name", declared.get("identity", name))
            if not isinstance(name, str) or not name.strip():
                raise RegistryError(
                    f"plugin {getattr(plugin, '__name__', plugin)!r} must"
                    " expose a 'name' string"
                )
            if name in names:
                raise RegistryError(f"duplicate plugin name: {name!r}")
            version = getattr(
                plugin, "version", getattr(plugin, "PLUGIN_VERSION", getattr(plugin, "__version__", None))
            )
            if isinstance(declared, dict):
                version = declared.get("version", version)
            if name != manifest.name or str(version).strip() != manifest.version:
                raise RegistryError(
                    f"plugin manifest mismatch for {name!r}: expected"
                    f" {manifest.name!r} version {manifest.version!r}"
                )
            names.append(name)
            setup = getattr(plugin, "setup", None)
            if setup is not None and not callable(setup):
                raise RegistryError(f"plugin {name!r}: setup must be callable")
            if setup is not None:
                try:
                    setup(api)
                except Exception as e:
                    raise RegistryError(
                        f"plugin {name!r} setup failed: {e}"
                    ) from e
        self._validate_staged(
            gate_features, output_stages, tools, learners, adapters
        )
        gate_features.freeze()
        output_stages.freeze()
        tools.freeze()
        learners.freeze()
        adapters.freeze()
        hooks.freeze()
        return PluginLoadResult(
            plugin_names=tuple(names),
            gate_features=gate_features,
            output_stages=output_stages,
            tools=tools,
            learners=learners,
            adapters=adapters,
            hooks=hooks,
            plugin_manifest=manifests,
        )

    def _iter_plugins(self) -> Any:
        for path in self._cfg.plugins.paths:
            yield load_plugin_module(path)
        for name in self._cfg.plugins.entry_points:
            yield load_plugin_entry_point(name)

    def _validate_staged(
        self,
        gate_features: StagingRegistry[Any],
        output_stages: StagingRegistry[Any],
        tools: StagingRegistry[Any],
        learners: StagingRegistry[LearnerSpec],
        adapters: StagingRegistry[AdapterSpec],
    ) -> None:
        """Post-setup validation: every staged item must be a usable
        extension. Gate features and output stages were shape-validated at
        registration; tools must be real ``ToolSpec`` instances (the staging
        registry is protocol-free so a plugin cannot smuggle a raw callable
        past the type boundary)."""
        from pretender.tools.base import ToolSpec

        for spec in tools.all():
            if not isinstance(spec, ToolSpec):
                raise RegistryError(
                    f"tools: staged item {getattr(spec, 'name', spec)!r} is"
                    f" not a ToolSpec ({type(spec).__name__})"
                )
        # Gate features / output stages: shape validation already ran at
        # registration; re-validate the frozen set for defense in depth.
        for item in gate_features.all():
            if not callable(getattr(item, "contribute", None)):
                raise RegistryError(
                    f"gate_features: {getattr(item, 'name', item)!r} does not"
                    " implement contribute(ctx)"
                )
        for item in output_stages.all():
            if not callable(getattr(item, "apply", None)):
                raise RegistryError(
                    f"output: {getattr(item, 'name', item)!r} does not"
                    " implement apply(out)"
                )
        for item in learners.all():
            if not isinstance(item, LearnerSpec):
                raise RegistryError(
                    f"learners: staged item {getattr(item, 'name', item)!r} is"
                    " not a declarative LearnerSpec"
                )
        for item in adapters.all():
            if not isinstance(item, AdapterSpec):
                raise RegistryError(
                    f"adapters: staged item {getattr(item, 'name', item)!r} is"
                    " not a declarative AdapterSpec"
                )
