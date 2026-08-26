"""Registry: typed registration, ordering, replace, shape validation;
HookBus: the three hooks with deterministic order and sync/async support."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

import pytest

from pretender.errors import RegistryError
from pretender.registry import HookBus, PluginRegistry, Registry, StagingRegistry
from pretender.types import AdapterEvent, ChatKey, DecisionTrace, Outgoing


@runtime_checkable
class _Named(Protocol):
    name: str

    def ping(self) -> str: ...


class _GoodItem:
    name = "good"

    def ping(self) -> str:
        return "pong"


class _NoPing:
    name = "noping"


class _NoName:
    def ping(self) -> str:
        return "pong"


def test_register_get_all_in_insertion_order():
    reg: Registry[_Named] = Registry("test", protocol=_Named)
    reg.register(_GoodItem())
    other = _GoodItem()
    other.name = "good2"
    reg.register(other)
    assert reg.names() == ("good", "good2")
    item = reg.get("good")
    assert item is not None
    assert item.ping() == "pong"
    assert reg.require("good2").name == "good2"
    assert [i.name for i in reg.all()] == ["good", "good2"]
    assert len(reg) == 2
    assert "good" in reg


def test_duplicate_registration_rejected():
    reg: Registry[_Named] = Registry("test", protocol=_Named)
    reg.register(_GoodItem())
    with pytest.raises(RegistryError, match="already registered"):
        reg.register(_GoodItem())


def test_replace_keeps_original_position():
    reg: Registry[_Named] = Registry("test", protocol=_Named)
    reg.register(_GoodItem())
    other = _GoodItem()
    other.name = "good2"
    reg.register(other)
    replacement = _GoodItem()
    replacement.name = "good"
    replacement.ping = lambda: "replaced"  # type: ignore[method-assign]
    reg.register(replacement, replace=True)
    assert reg.names() == ("good", "good2")  # position preserved
    item = reg.get("good")
    assert item is not None
    assert item.ping() == "replaced"


def test_require_missing_raises():
    reg: Registry[_Named] = Registry("test", protocol=_Named)
    with pytest.raises(RegistryError, match="not registered"):
        reg.require("nope")


def test_item_without_name_rejected():
    reg: Registry[Any] = Registry("test", protocol=_Named)
    with pytest.raises(RegistryError, match="no 'name'"):
        reg.register(_NoName())


def test_shape_validation_rejects_missing_method():
    reg: Registry[Any] = Registry("test", protocol=_Named)
    with pytest.raises(RegistryError, match="missing methods.*ping"):
        reg.register(_NoPing())


def test_shape_validation_rejects_missing_attribute():
    @runtime_checkable
    class _HasCapabilities(Protocol):
        name: str
        capabilities: frozenset[str]

    class _NoCaps:
        name = "x"

    reg: Registry[Any] = Registry("test", protocol=_HasCapabilities)
    with pytest.raises(RegistryError, match="missing attributes.*capabilities"):
        reg.register(_NoCaps())


def test_register_as_decorator():
    reg: Registry[Any] = Registry("test", protocol=_Named)

    @reg.register
    class _Decorated:
        name = "deco"

        def ping(self) -> str:
            return "pong"

    cls = reg.require("deco")
    assert cls().ping() == "pong"  # type: ignore[union-attr]


def test_unregister_and_clear():
    reg: Registry[_Named] = Registry("test", protocol=_Named)
    reg.register(_GoodItem())
    reg.unregister("good")
    assert reg.get("good") is None
    with pytest.raises(RegistryError):
        reg.unregister("good")
    reg.register(_GoodItem())
    reg.clear()
    assert len(reg) == 0


def test_registry_without_protocol_skips_validation():
    reg: Registry[object] = Registry("loose")
    reg.register(object(), name="anything")  # no name attr → explicit name
    assert reg.require("anything") is not None


# ── HookBus ─────────────────────────────────────────────────────────────────

def _event():
    return AdapterEvent(type="message", payload=None)


def test_hooks_run_in_registration_order():
    bus = HookBus()
    order: list[str] = []

    @bus.on_event
    def first(event: AdapterEvent) -> None:
        order.append("first")

    @bus.on_event
    def second(event: AdapterEvent) -> None:
        order.append("second")

    async def scenario():
        await bus.emit_event(_event())

    asyncio.run(scenario())
    assert order == ["first", "second"]


def test_async_and_sync_hooks_both_work():
    bus = HookBus()
    order: list[str] = []

    @bus.on_event
    def sync_hook(event: AdapterEvent) -> None:
        order.append("sync")

    @bus.on_event
    async def async_hook(event: AdapterEvent) -> None:
        order.append("async")

    async def scenario():
        await bus.emit_event(_event())

    asyncio.run(scenario())
    assert order == ["sync", "async"]


def test_pre_send_chaining():
    bus = HookBus()
    chat = ChatKey("qq:group:1")

    @bus.pre_send
    def append_a(out: Outgoing) -> Outgoing:
        out.text += "a"
        return out

    @bus.pre_send
    def keep_as_is(out: Outgoing) -> None:
        return None

    @bus.pre_send
    def replace(out: Outgoing) -> Outgoing:
        return Outgoing(chat_key=chat, text="replaced")

    async def scenario():
        return await bus.emit_pre_send(Outgoing(chat_key=chat, text=""))

    result = asyncio.run(scenario())
    assert result.text == "replaced"


def test_cycle_end_hook_receives_trace():
    bus = HookBus()
    seen: list[tuple] = []

    @bus.on_cycle_end
    def record(chat_key: ChatKey, trace: DecisionTrace, end_reason: str) -> None:
        seen.append((chat_key, trace, end_reason))

    trace = DecisionTrace(
        chat_key=ChatKey("qq:group:1"), mode="reply_necessity",
        threshold=8, trigger_score=80, pending=3,
    )

    async def scenario():
        await bus.emit_cycle_end(ChatKey("qq:group:1"), trace, "no_action")

    asyncio.run(scenario())
    assert seen == [(ChatKey("qq:group:1"), trace, "no_action")]


def test_unregister_hook():
    bus = HookBus()
    order: list[str] = []

    @bus.on_event
    def first(event: AdapterEvent) -> None:
        order.append("first")

    @bus.on_event
    def second(event: AdapterEvent) -> None:
        order.append("second")

    bus.unregister_on_event(first)

    async def scenario():
        await bus.emit_event(_event())

    asyncio.run(scenario())
    assert order == ["second"]


def test_hookbus_clear():
    bus = HookBus()

    @bus.on_event
    def h(event: AdapterEvent) -> None:
        pass

    @bus.pre_send
    def p(out: Outgoing) -> None:
        pass

    assert len(bus) == 2
    bus.clear()
    assert len(bus) == 0


# ── StagingRegistry (frozen staging for later plugin discovery) ─────────────

def test_staging_registry_deterministic_order_and_duplicate_rejection():
    reg: StagingRegistry[_Named] = StagingRegistry("staging", protocol=_Named)
    reg.register(_GoodItem())
    other = _GoodItem()
    other.name = "good2"
    reg.register(other)
    assert reg.names() == ("good", "good2")
    assert [i.name for i in reg.all()] == ["good", "good2"]
    with pytest.raises(RegistryError, match="already registered"):
        reg.register(_GoodItem())
    # replace keeps the original slot.
    replacement = _GoodItem()
    replacement.name = "good"
    reg.register(replacement, replace=True)
    assert reg.names() == ("good", "good2")


def test_staging_registry_freeze_prevents_mutation():
    reg: StagingRegistry[_Named] = StagingRegistry("staging", protocol=_Named)
    reg.register(_GoodItem())
    reg.freeze()
    assert reg.frozen is True
    with pytest.raises(RegistryError, match="frozen"):
        reg.register(_GoodItem())
    with pytest.raises(RegistryError, match="frozen"):
        reg.register(_GoodItem(), replace=True)
    with pytest.raises(RegistryError, match="frozen"):
        reg.unregister("good")
    with pytest.raises(RegistryError, match="frozen"):
        reg.clear()
    # Reads still work after freeze.
    assert reg.require("good").name == "good"
    assert reg.names() == ("good",)
    # Freeze is idempotent.
    reg.freeze()


def test_staging_registry_protected_core_names_and_allowlist():
    reg: StagingRegistry[_Named] = StagingRegistry(
        "staging", protocol=_Named, protected=("core",), allow_replace=("core",)
    )
    core = _GoodItem()
    core.name = "core"
    reg.register(core)
    # A protected name is replaceable ONLY when on the allowlist.
    replacement = _GoodItem()
    replacement.name = "core"
    reg.register(replacement, replace=True)
    assert reg.require("core").ping() == "pong"
    # A protected name NOT on the allowlist is rejected.
    reg2: StagingRegistry[_Named] = StagingRegistry(
        "staging", protocol=_Named, protected=("core",)
    )
    reg2.register(core)
    with pytest.raises(RegistryError, match="protected core name"):
        reg2.register(replacement, replace=True)
    # Metadata is exposed.
    assert reg.protected_names() == ("core",)
    assert reg.replacement_allowlist() == ("core",)
    assert reg2.replacement_allowlist() == ()


def test_staging_registry_shape_validation_and_decorator():
    reg: StagingRegistry[Any] = StagingRegistry("staging", protocol=_Named)
    with pytest.raises(RegistryError, match="missing methods.*ping"):
        reg.register(_NoPing())

    @reg.register
    class _Decorated:
        name = "deco"

        def ping(self) -> str:
            return "pong"

    assert reg.require("deco")().ping() == "pong"  # type: ignore[union-attr]


def test_plugin_registry_validates_plugin_shape_and_freezes():
    reg = PluginRegistry()
    assert reg.name == "plugins"

    class _GoodPlugin:
        name = "good"

        def setup(self, app: Any) -> None:
            pass

    class _SecondPlugin:
        name = "second"

        def setup(self, app: Any) -> None:
            pass

    reg.register(_GoodPlugin())
    reg.register(_SecondPlugin())
    assert reg.names() == ("good", "second")
    # Shape validation rejects a plugin missing the setup method.
    class _NoSetup:
        name = "nosetup"

    with pytest.raises(RegistryError, match="missing methods.*setup"):
        reg.register(_NoSetup())
    reg.freeze()
    with pytest.raises(RegistryError, match="frozen"):
        reg.register(_GoodPlugin())