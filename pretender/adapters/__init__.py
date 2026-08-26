"""Adapter implementations. Phase 1 ships the console (REPL) adapter; the
OneBot v11 adapter lands in Phase 4. No live adapter may be configured yet —
the app always boots console-only (app adoption of OneBot is a later lane).
"""

from __future__ import annotations

from pretender.adapters.base import Adapter, capability_set
from pretender.adapters.console import ConsoleAdapter
from pretender.adapters.onebot import OneBotAdapter

__all__ = ["Adapter", "ConsoleAdapter", "OneBotAdapter", "capability_set"]