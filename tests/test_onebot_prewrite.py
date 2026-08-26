from __future__ import annotations

import pytest

from pretender.adapters.onebot import OneBotAdapter
from pretender.config import OneBotConfig
from pretender.errors import AdapterNotReady
from pretender.types import Outgoing
from tests.durable_helpers import run


def test_prewrite_not_ready_discards_provisional_fallback_metadata():
    """A proven no-write failure must not leave a same-payload candidate that
    could bind a later unrelated self echo."""

    async def scenario():
        adapter = OneBotAdapter(
            config=OneBotConfig(host="127.0.0.1", heartbeat_timeout_s=None),
            normalize_media=False,
            self_id="10001",
        )
        async def fail_before_write(*_args, **_kwargs):
            raise AdapterNotReady("generation replaced before write")

        adapter._write_frame = fail_before_write  # type: ignore[method-assign]
        with pytest.raises(AdapterNotReady):
            await adapter.send(
                Outgoing(
                    chat_key="qq:group:111111",
                    text="hello",
                    delivery_key="dispatch:1:0",
                )
            )
        return adapter

    adapter = run(scenario())
    local = "onebot:local:dispatch:1:0"
    assert local not in adapter._delivered
    assert local not in adapter._sent_payload
    assert "dispatch:1:0" not in adapter._early_bound
