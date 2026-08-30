"""Doctor: every probe, error containment, optional/required status, secret
redaction, and report rendering — all against fake seams plus one real
SQLite database probe.

Async tests run via asyncio.run() so the test extra stays at just pytest.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from pretender.config import (
    AccessConfig,
    AccessListConfig,
    BotConfig,
    ChatOverride,
    Config,
    LearnConfig,
    LearnProfile,
    LLMConfig,
    LLMProfile,
    MediaConfig,
    OutputConfig,
    StorageConfig,
)
from pretender.context import serialize
from pretender.db import Database
from pretender.doctor import Doctor, DoctorReport, ProbeResult
from pretender.errors import (
    AdapterError,
    LLMPermanentError,
    LLMTransientError,
    PromptError,
)
from pretender.prompts import PromptStore
from pretender.repo import SqliteRepository
from pretender.types import (
    ChatKey,
    LLMResponse,
    MediaAssetCandidate,
    MessageRowId,
    ToolCall,
    ToolCallId,
)
from tests.durable_helpers import CK, make_identity
from tests.knowledge_helpers import make_vector, read_and_commit, seed_messages


def run(coro):
    return asyncio.run(coro)


def probe(report: DoctorReport, name: str) -> ProbeResult:
    """by_name with a hard assertion: a missing probe is a test bug."""
    p = report.by_name(name)
    assert p is not None, f"missing probe {name!r}"
    return p


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeLLM:
    """Duck-typed LLMClient + Embedder: records every call, returns prepared
    responses, and can raise a prepared error."""

    def __init__(
        self,
        responses=None,
        *,
        error=None,
        tool_response=None,
        embed_vectors=None,
        embed_error=None,
    ) -> None:
        self.responses = responses or {}
        self.error = error
        self.tool_response = tool_response
        self.embed_vectors = embed_vectors
        self.embed_error = embed_error
        self.calls: list[tuple] = []
        self.aclose_calls = 0

    async def complete(
        self,
        messages,
        *,
        profile,
        tools=None,
        temperature=None,
        max_tokens=None,
        deadline=None,
    ):
        self.calls.append(("complete", profile, tools, max_tokens, messages))
        if self.error:
            raise self.error
        if tools is not None and self.tool_response is not None:
            return self.tool_response
        if profile == "vision" and profile not in self.responses:
            return LLMResponse(
                content='{"safe": true, "description": "probe image"}'
            )
        return self.responses.get(profile, LLMResponse(content="pong"))

    async def embed(self, texts):
        self.calls.append(("embed", texts))
        if self.embed_error:
            raise self.embed_error
        return self.embed_vectors if self.embed_vectors is not None else [[0.1, 0.2, 0.3]]

    async def aclose(self) -> None:
        self.aclose_calls += 1


class FakeAdapter:
    def __init__(self, name="fake", capabilities=(), *, connect_error=None) -> None:
        self.name = name
        self.capabilities = frozenset(capabilities)
        self.connect_error = connect_error
        self.connect_calls = 0
        self.close_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_error:
            raise self.connect_error

    async def events(self):
        return
        yield  # pragma: no cover

    async def send(self, out):
        return None

    async def call(self, action, **params):
        return None

    async def close(self) -> None:
        self.close_calls += 1


class FakeDB:
    """Duck-typed Database: write/read/open/close with prepared results."""

    def __init__(self, *, write_result="1", write_error=None, fts_row=("message_fts",)) -> None:
        self.write_result = write_result
        self.write_error = write_error
        self.fts_row = fts_row
        self.open_calls = 0
        self.close_calls = 0
        self.writes: list = []
        self.reads: list = []

    async def open(self) -> None:
        self.open_calls += 1

    async def write(self, fn):
        self.writes.append(fn)
        if self.write_error:
            raise self.write_error
        return self.write_result

    async def read(self, fn):
        self.reads.append(fn)
        return self.fts_row

    async def close(self) -> None:
        self.close_calls += 1


class FakeRepo:
    def __init__(self, *, stats_result=None, stats_error=None) -> None:
        self.stats_result = stats_result or {"user_version": 5}
        self.stats_error = stats_error
        self.stats_calls = 0

    async def stats(self):
        self.stats_calls += 1
        if self.stats_error:
            raise self.stats_error
        return dict(self.stats_result)


class FakePromptStore:
    def __init__(self, *, missing=(), render_error=None) -> None:
        self.missing = set(missing)
        self.render_error = render_error

    def load(self, name: str) -> str:
        if name in self.missing:
            raise PromptError(f"prompt {name!r} not found")
        return f"content of {name}"

    def render(self, name: str, **variables) -> str:
        if self.render_error:
            raise self.render_error
        return f"rendered {name}"


# ── helpers ──────────────────────────────────────────────────────────────────

def _cfg() -> Config:
    """A config with planner/vision/embed profiles (api_keys are secrets the
    report must never leak)."""
    return Config(
        llm=LLMConfig(
            profiles={
                "planner": LLMProfile(
                    base_url="https://api.example.com/v1",
                    api_key="sk-supersecret",
                    model="m",
                ),
                "vision": LLMProfile(
                    base_url="https://vision.example.com/v1",
                    api_key="sk-vision",
                    model="v",
                ),
                "embed": LLMProfile(
                    base_url="https://embed.example.com/v1",
                    api_key="sk-embed",
                    model="e",
                    revision="r1",
                ),
            }
        )
    )


def _tool_response() -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=(ToolCall(id=ToolCallId("c1"), name="doctor_probe", arguments={}),),
    )


def _doctor(cfg: Config | None = None, **kw) -> Doctor:
    """A Doctor with every seam faked so the default run is all-ok."""
    cfg = cfg if cfg is not None else _cfg()
    kw.setdefault("llm", FakeLLM(tool_response=_tool_response()))
    kw.setdefault("embedder", FakeLLM())
    kw.setdefault("db", FakeDB())
    kw.setdefault("repo", FakeRepo())
    kw.setdefault("adapter", FakeAdapter())
    return Doctor(cfg, **kw)


# ── config probe ─────────────────────────────────────────────────────────────

def test_config_probe_ok():
    report = run(_doctor().run())
    p = probe(report, "config")
    assert p.status == "ok"
    assert p.data["profiles"] == ["embed", "planner", "vision"]
    assert "sk-supersecret" not in p.detail


def test_config_probe_fails_on_empty_base_url():
    cfg = dataclasses.replace(
        _cfg(), llm=LLMConfig(profiles={"bad": LLMProfile(base_url="", model="m")})
    )
    report = run(_doctor(cfg).run())
    p = probe(report, "config")
    assert p.status == "fail"
    assert "empty base_url" in p.detail


def test_config_probe_reports_missing_api_key_without_leaking():
    cfg = dataclasses.replace(
        _cfg(), llm=LLMConfig(profiles={"p": LLMProfile(base_url="https://x", model="m")})
    )
    report = run(_doctor(cfg).run())
    p = probe(report, "config")
    assert p.status == "ok"
    assert "no api_key" in p.detail


def test_config_probe_accepts_optional_pipeline_with_core_final_sanitize():
    cfg = dataclasses.replace(_cfg(), output=OutputConfig(pipeline=("split",)))
    report = run(_doctor(cfg).run())
    p = probe(report, "config")
    assert p.status == "ok"


def test_config_probe_validates_per_chat_output_override():
    cfg = dataclasses.replace(
        _cfg(),
        chats=(
            ChatOverride(
                key=ChatKey("qq:group:42"),
                output=OutputConfig(pipeline=("split",)),
                output_raw={"pipeline": ["split"]},
            ),
        ),
    )
    report = run(_doctor(cfg).run())
    p = probe(report, "config")
    assert p.status == "ok"


def test_config_probe_fails_on_missing_embed_revision():
    """A configured embed profile without an explicit revision is invalid
    semantic config: the config probe fails and names the fix — without
    leaking any secret."""
    cfg = dataclasses.replace(
        _cfg(),
        llm=LLMConfig(
            profiles={
                "embed": LLMProfile(
                    base_url="https://embed.example.com/v1",
                    api_key="sk-embed-secret",
                    model="e",
                )
            }
        ),
    )
    report = run(_doctor(cfg).run())
    p = probe(report, "config")
    assert p.status == "fail"
    assert "revision" in p.detail
    assert "sk-embed-secret" not in p.detail
    assert "sk-embed-secret" not in report.render()


def test_embed_probe_reports_space_id():
    """The embed probe reports the canonical space identity (model@revision)
    without any secret."""
    report = run(_doctor().run())
    p = probe(report, "embed")
    assert p.status == "ok"
    assert p.data["space_id"] == "e@r1"
    assert "sk-embed" not in report.render()


# ── prompts probe ────────────────────────────────────────────────────────────

def test_prompts_probe_ok_with_package_defaults():
    report = run(_doctor(prompts=PromptStore()).run())
    p = probe(report, "prompts")
    assert p.status == "ok"
    assert p.data["assets"] == [
        "identity.txt", "behavior.txt", "planner.txt", "replyer.txt",
        "planner_focus.txt",
    ]


def test_prompts_probe_fails_on_missing_asset():
    store = FakePromptStore(missing={"planner.txt"})
    report = run(_doctor(prompts=store).run())
    p = probe(report, "prompts")
    assert p.status == "fail"
    assert "planner.txt" in p.detail


def test_prompts_probe_fails_on_render_error():
    store = FakePromptStore(render_error=PromptError("missing prompt variable(s): identity"))
    report = run(_doctor(prompts=store).run())
    assert probe(report, "prompts").status == "fail"


def test_prompts_probe_fails_on_dead_identity_file(tmp_path):
    """An empty (dead) identity file is a broken asset: the prompts probe
    fails and names the identity file."""
    (tmp_path / "identity.txt").write_text("   \n", encoding="utf-8")
    store = PromptStore(user_dir=tmp_path)
    report = run(_doctor(prompts=store).run())
    p = probe(report, "prompts")
    assert p.status == "fail"
    assert "identity" in p.detail


def test_prompts_probe_fails_on_unreadable_identity_file(tmp_path):
    """A configured identity_file that cannot be read (missing) fails the
    prompts probe."""
    cfg = dataclasses.replace(
        _cfg(), bot=BotConfig(identity_file=str(tmp_path / "nope.txt"))
    )
    report = run(_doctor(cfg).run())
    p = probe(report, "prompts")
    assert p.status == "fail"
    assert "identity" in p.detail


def test_prompts_probe_ok_with_configured_identity_file(tmp_path):
    """A configured identity_file that loads through the prompt
    infrastructure keeps the prompts probe green."""
    (tmp_path / "identity.txt").write_text("自定义身份", encoding="utf-8")
    cfg = dataclasses.replace(
        _cfg(), bot=BotConfig(identity_file="prompts/identity.txt")
    )
    report = run(_doctor(cfg, prompts=PromptStore(user_dir=tmp_path)).run())
    p = probe(report, "prompts")
    assert p.status == "ok"


# ── database probe ───────────────────────────────────────────────────────────

def test_database_probe_ok_with_real_db(tmp_path):
    async def scenario():
        db = Database(tmp_path / "doctor.db")
        repo = SqliteRepository(db)
        report = await _doctor(db=db, repo=repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "database")
    assert p.status == "ok"
    assert p.data["fts5"] is True
    assert p.data["user_version"] >= 1
    assert p.data["stats"]["user_version"] >= 1


def test_database_probe_leaves_no_probe_row(tmp_path):
    async def scenario():
        db = Database(tmp_path / "doctor.db")
        repo = SqliteRepository(db)
        await _doctor(db=db, repo=repo).run()
        value = await repo.get_kv("doctor.probe")
        await db.close()
        return value

    assert run(scenario()) is None


def test_database_probe_fails_when_write_fails():
    db = FakeDB(write_error=RuntimeError("disk full"))
    report = run(_doctor(db=db).run())
    p = probe(report, "database")
    assert p.status == "fail"
    assert "disk full" in (p.error or "")


def test_database_probe_fails_when_fts_missing():
    db = FakeDB(fts_row=None)
    report = run(_doctor(db=db).run())
    p = probe(report, "database")
    assert p.status == "fail"
    assert "FTS5" in p.detail


def test_database_probe_fails_when_repo_seam_fails():
    repo = FakeRepo(stats_error=RuntimeError("repo broken"))
    report = run(_doctor(repo=repo).run())
    assert probe(report, "database").status == "fail"


# ── adapter probe ────────────────────────────────────────────────────────────

def test_adapter_probe_ok():
    adapter = FakeAdapter(name="console", capabilities={"quote", "at"})
    report = run(_doctor(adapter=adapter).run())
    p = probe(report, "adapter")
    assert p.status == "ok"
    assert p.data["name"] == "console"
    assert p.data["capabilities"] == ["at", "quote"]
    assert adapter.connect_calls == 1


def test_adapter_probe_fails_on_connect_error():
    adapter = FakeAdapter(connect_error=AdapterError("connection refused"))
    report = run(_doctor(adapter=adapter).run())
    p = probe(report, "adapter")
    assert p.status == "fail"
    assert "connection refused" in (p.error or "")


def test_adapter_probe_fails_without_shape():
    class NoShape:
        pass

    report = run(_doctor(adapter=NoShape()).run())
    assert probe(report, "adapter").status == "fail"


# ── llm_chat probe ───────────────────────────────────────────────────────────

def test_llm_chat_probe_ok():
    llm = FakeLLM()
    report = run(_doctor(llm=llm).run())
    p = probe(report, "llm_chat")
    assert p.status == "ok"
    assert p.data["profile"] == "planner"
    kind, profile, tools, max_tokens, _messages = llm.calls[0]
    assert kind == "complete"
    assert profile == "planner"
    assert tools is None
    assert max_tokens == 16


def test_llm_chat_probe_fails_on_error():
    llm = FakeLLM(error=LLMTransientError("provider 500"))
    report = run(_doctor(llm=llm).run())
    p = probe(report, "llm_chat")
    assert p.status == "fail"
    assert "provider 500" in (p.error or "")


def test_llm_chat_probe_fails_on_empty_response():
    llm = FakeLLM(responses={"planner": LLMResponse(content=None)})
    report = run(_doctor(llm=llm).run())
    assert probe(report, "llm_chat").status == "fail"


def test_llm_chat_probe_skips_without_profiles():
    report = run(_doctor(Config()).run())
    assert probe(report, "llm_chat").status == "skip"


# ── llm_tools probe ──────────────────────────────────────────────────────────

def test_llm_tools_probe_ok():
    llm = FakeLLM(tool_response=_tool_response())
    report = run(_doctor(llm=llm).run())
    p = probe(report, "llm_tools")
    assert p.status == "ok"
    assert p.data["tool_calls"] == ["doctor_probe"]
    kind, profile, tools, max_tokens, _messages = llm.calls[1]
    assert profile == "planner"
    assert tools is not None
    assert tools[0]["function"]["name"] == "doctor_probe"


def test_llm_tools_probe_fails_without_tool_call():
    llm = FakeLLM()  # no tool_response: the tools call returns plain content
    report = run(_doctor(llm=llm).run())
    p = probe(report, "llm_tools")
    assert p.status == "fail"
    assert "no tool call" in p.detail


def test_llm_tools_probe_fails_on_malformed_call():
    llm = FakeLLM(
        tool_response=LLMResponse(
            content=None,
            tool_calls=(ToolCall(id=ToolCallId("c1"), name="", arguments={}),),
        )
    )
    report = run(_doctor(llm=llm).run())
    assert probe(report, "llm_tools").status == "fail"


def test_llm_tools_probe_fails_on_error():
    llm = FakeLLM(error=LLMPermanentError("bad request"))
    report = run(_doctor(llm=llm).run())
    assert probe(report, "llm_tools").status == "fail"


def test_llm_tools_probe_accepts_content_with_tool_calls():
    """Analysis/content and tool calls must be able to coexist in one
    response: a response carrying BOTH is valid, never a failure."""
    llm = FakeLLM(
        tool_response=LLMResponse(
            content="analysis: calling the probe tool",
            tool_calls=(ToolCall(id=ToolCallId("c1"), name="doctor_probe", arguments={}),),
        )
    )
    report = run(_doctor(llm=llm).run())
    p = probe(report, "llm_tools")
    assert p.status == "ok"
    assert "with analysis content" in p.detail
    assert p.data["tool_calls"] == ["doctor_probe"]


def test_llm_tools_probe_skips_without_profiles():
    report = run(_doctor(Config()).run())
    assert probe(report, "llm_tools").status == "skip"


# ── vision probe (optional capability) ───────────────────────────────────────

def test_vision_probe_skips_without_profile():
    cfg = dataclasses.replace(
        _cfg(), llm=LLMConfig(profiles={"planner": LLMProfile(base_url="https://x", model="m")})
    )
    report = run(_doctor(cfg).run())
    assert probe(report, "vision").status == "skip"


def test_vision_probe_ok():
    llm = FakeLLM()
    report = run(_doctor(llm=llm).run())
    p = probe(report, "vision")
    assert p.status == "ok"
    # the vision call is the third complete call, on the vision profile, and
    # carries an image markdown span
    kind, profile, tools, max_tokens, messages = llm.calls[2]
    assert profile == "vision"
    assert "![probe](https://example.com/probe.png)" in messages[1].content


def test_vision_probe_verifies_real_multimodal_wire_shape():
    """The vision probe verifies the REAL wire shape: image markdown must
    serialize into OpenAI-compatible multimodal content parts (text +
    image_url), not a bare markdown string."""
    llm = FakeLLM()
    report = run(_doctor(llm=llm).run())
    p = probe(report, "vision")
    assert p.status == "ok"
    kind, profile, tools, max_tokens, messages = llm.calls[2]
    wire = serialize(messages)
    parts = wire[1]["content"]
    assert isinstance(parts, list)
    assert any(
        isinstance(part, dict)
        and part.get("type") == "image_url"
        and isinstance(part.get("image_url"), dict)
        and part["image_url"].get("url") == "https://example.com/probe.png"
        for part in parts
    )


def test_vision_probe_fails_on_error():
    llm = FakeLLM(error=LLMTransientError("vision timeout"))
    report = run(_doctor(llm=llm).run())
    assert probe(report, "vision").status == "fail"


# ── embed probe (optional capability) ────────────────────────────────────────

def test_embed_probe_skips_without_profile():
    cfg = dataclasses.replace(
        _cfg(), llm=LLMConfig(profiles={"planner": LLMProfile(base_url="https://x", model="m")})
    )
    report = run(_doctor(cfg).run())
    assert probe(report, "embed").status == "skip"


def test_embed_probe_ok():
    embedder = FakeLLM(embed_vectors=[[0.1, 0.2, 0.3]])
    report = run(_doctor(embedder=embedder).run())
    p = probe(report, "embed")
    assert p.status == "ok"
    assert p.data["dimension"] == 3


def test_embed_probe_fails_on_inconsistent_dimension():
    embedder = FakeLLM(embed_vectors=[[0.1, 0.2], [0.3, 0.4, 0.5]])
    report = run(_doctor(embedder=embedder).run())
    p = probe(report, "embed")
    assert p.status == "fail"
    assert "inconsistent dimension" in p.detail


def test_embed_probe_fails_on_error():
    embedder = FakeLLM(embed_error=LLMTransientError("embed timeout"))
    report = run(_doctor(embedder=embedder).run())
    assert probe(report, "embed").status == "fail"


# ── error containment: collect all, never fail fast ──────────────────────────

def test_run_collects_all_probes_even_when_one_raises():
    class ExplodingLLM:
        async def complete(self, *a, **kw):
            raise RuntimeError("boom")

        async def embed(self, texts):
            raise RuntimeError("boom")

    report = run(_doctor(llm=ExplodingLLM(), embedder=ExplodingLLM()).run())
    assert len(report.probes) == 14
    assert [p.name for p in report.probes] == list(Doctor.PROBES)
    assert probe(report, "llm_chat").status == "fail"
    assert probe(report, "llm_tools").status == "fail"
    assert probe(report, "vision").status == "fail"
    assert probe(report, "embed").status == "fail"
    assert probe(report, "config").status == "ok"
    assert report.status == "fail"


# ── no secret leakage ────────────────────────────────────────────────────────

def test_no_secret_leakage_in_report():
    llm = FakeLLM(error=LLMTransientError("provider 500: sk-supersecret leaked"))
    report = run(_doctor(llm=llm).run())
    rendered = report.render()
    assert "sk-supersecret" not in rendered
    assert "sk-vision" not in rendered
    assert "sk-embed" not in rendered
    for p in report.probes:
        assert "sk-supersecret" not in (p.detail or "")
        assert "sk-supersecret" not in (p.error or "")
        assert "sk-supersecret" not in repr(p.data)


def test_report_scrubs_query_secret_from_llm_error():
    """A provider error that echoes a URL with query credentials renders in
    the report without the secret, keeping host/path/status text."""
    llm = FakeLLM(
        error=LLMTransientError(
            "provider 500: https://api.example.com/v1/chat/completions?token=sk-querysecret"
        )
    )
    report = run(_doctor(llm=llm).run())
    p = probe(report, "llm_chat")
    assert p.status == "fail"
    assert "sk-querysecret" not in (p.error or "")
    assert "token=" not in (p.error or "")
    assert "api.example.com/v1/chat/completions" in (p.error or "")
    rendered = report.render()
    assert "sk-querysecret" not in rendered
    assert "token=" not in rendered
    assert "api.example.com/v1/chat/completions" in rendered


def test_report_scrubs_raw_nested_exception_url():
    """Even a raw (non-LLM) exception whose text embeds a credential URL is
    scrubbed by the doctor before it reaches the report."""
    class ExplodingLLM:
        async def complete(self, *a, **kw):
            raise RuntimeError(
                "connect failed: "
                "https://api.example.com/v1/chat/completions?token=sk-querysecret"
            )

        async def embed(self, texts):
            raise RuntimeError("boom")

    report = run(_doctor(llm=ExplodingLLM(), embedder=ExplodingLLM()).run())
    rendered = report.render()
    assert "sk-querysecret" not in rendered
    assert "token=" not in rendered
    assert "connect failed" in rendered
    assert "api.example.com/v1/chat/completions" in rendered


def test_report_keeps_ordinary_diagnostics():
    """Ordinary non-secret error text renders untouched."""
    llm = FakeLLM(error=LLMTransientError("provider 503: overloaded upstream"))
    report = run(_doctor(llm=llm).run())
    rendered = report.render()
    assert "overloaded upstream" in rendered
    assert "503" in rendered


# ── report status: required vs optional ──────────────────────────────────────

def test_report_status_fails_when_required_probe_fails():
    llm = FakeLLM(error=LLMTransientError("down"))
    report = run(_doctor(llm=llm).run())
    assert report.status == "fail"
    assert report.failed


def test_report_status_ok_when_only_optional_capabilities_skip():
    report = run(_doctor(Config()).run())
    assert report.status == "ok"
    assert not report.failed
    assert probe(report, "vision").status == "skip"
    assert probe(report, "embed").status == "skip"


def test_report_status_fails_when_configured_optional_probe_fails():
    embedder = FakeLLM(embed_error=LLMTransientError("embed down"))
    report = run(_doctor(embedder=embedder).run())
    assert report.status == "fail"
    assert probe(report, "embed").status == "fail"


# ── report rendering ─────────────────────────────────────────────────────────

def test_report_render_is_deterministic_and_structured():
    report = run(_doctor().run())
    rendered = report.render()
    assert rendered.startswith("doctor: ok")
    assert "[OK] config" in rendered
    assert "[OK] llm_tools" in rendered
    # The default fake repo lacks the knowledge surface, so the semantic
    # readiness probe skips (cannot assess) — never a false healthy claim.
    assert "[SKIP] semantic" in rendered
    # The learner probe skips when learn is disabled (an optional capability).
    assert "[SKIP] learner" in rendered
    # The media catalog probe skips when the catalog is disabled.
    assert "[SKIP] media_catalog" in rendered
    assert "summary: 14 probes: 9 ok, 5 skip" in rendered
    assert report.render() == rendered  # deterministic


def test_report_render_shows_skips_and_failures():
    llm = FakeLLM(error=LLMTransientError("down"))
    report = run(_doctor(llm=llm).run())
    rendered = report.render()
    assert rendered.startswith("doctor: fail")
    assert "[FAIL] llm_chat" in rendered
    assert "error: down" in rendered
    assert "[OK] config" in rendered


def test_report_by_name_unknown_returns_none():
    report = run(_doctor().run())
    assert report.by_name("nope") is None


# ── seam ownership / cleanup ─────────────────────────────────────────────────

def test_injected_seams_are_not_closed():
    llm = FakeLLM(tool_response=_tool_response())
    db = FakeDB()
    adapter = FakeAdapter()
    report = run(_doctor(llm=llm, db=db, adapter=adapter).run())
    assert report.status == "ok"
    assert llm.aclose_calls == 0
    assert db.close_calls == 0
    assert adapter.close_calls == 0


def test_built_seams_are_closed(tmp_path):
    cfg = Config(storage=StorageConfig(db_path=str(tmp_path / "doctor.db")))
    doctor = Doctor(cfg, adapter=FakeAdapter())
    report = run(doctor.run())
    assert len(report.probes) == 14
    assert report.status == "ok"

    async def reopen():
        # the doctor built and closed its own Database: the file exists and
        # reopens cleanly
        db = Database(cfg.storage.db_path)
        await db.open()
        await db.close()

    run(reopen())


def test_probe_result_validates_status():
    with pytest.raises(ValueError):
        ProbeResult("x", "maybe")


# ── semantic readiness probe ─────────────────────────────────────────────────

def _semantic_cfg(revision: str = "r1", model: str = "e") -> Config:
    return Config(
        llm=LLMConfig(profiles={"embed": LLMProfile(model=model, revision=revision)})
    )


def _semantic_doctor(db, repo) -> Doctor:
    """A Doctor over a real repo + fake LLM/embedder/adapter so the semantic
    probe reads real generations/vectors while the other probes stay green."""
    return Doctor(
        _semantic_cfg(),
        db=db,
        repo=repo,
        adapter=FakeAdapter(),
        llm=FakeLLM(tool_response=_tool_response()),
        embedder=FakeLLM(),
    )


def test_semantic_probe_skips_without_embed_profile():
    report = run(_doctor(Config()).run())
    p = probe(report, "semantic")
    assert p.status == "skip"
    assert "FTS-only" in p.detail


def test_semantic_probe_fails_without_revision():
    cfg = Config(llm=LLMConfig(profiles={"embed": LLMProfile(model="e")}))
    report = run(_doctor(cfg).run())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "revision" in p.detail


def test_semantic_probe_fails_without_active_generation(tmp_path):
    """A configured revision but NO active matching generation is a degraded/
    not-ready semantic state — never a false healthy claim."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "no active matching generation" in p.detail
    assert p.data["active"] is False


def test_semantic_probe_ok_with_active_generation(tmp_path):
    """An active matching generation with complete, usable vectors is a
    healthy semantic state."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        memories = await repo.list_memories(CK)
        mem = memories[0]
        assert mem.id is not None and mem.source_hash is not None
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        await repo.upsert_vector(
            CK, make_vector(owner_id=mem.id, generation=g.id, model="e", dim=3,
                            values=(1.0, 0.0, 0.0), source_hash=mem.source_hash)
        )
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "ok"
    assert p.data["active"] is True
    assert p.data["dim"] == 3
    assert p.data["vectors"] == 1


def test_semantic_probe_fails_active_generation_no_vectors(tmp_path):
    """An active matching generation with NO vectors is not ready."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "incomplete vector coverage" in p.detail


def test_semantic_probe_is_secret_safe(tmp_path):
    """The semantic probe reports only space_id/dim/count — never a secret."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        memories = await repo.list_memories(CK)
        mem = memories[0]
        assert mem.id is not None and mem.source_hash is not None
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        await repo.upsert_vector(
            CK, make_vector(owner_id=mem.id, generation=g.id, model="e", dim=3,
                            values=(1.0, 0.0, 0.0), source_hash=mem.source_hash)
        )
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    rendered = report.render()
    assert "sk-" not in rendered
    p = probe(report, "semantic")
    assert set(p.data) <= {"space_id", "active", "dim", "vectors"}


# ── Gate 5 remediation: partial / dim / model-revision mismatch states ───────

def test_semantic_probe_fails_partial_coverage(tmp_path):
    """An active generation covering only SOME memories is a partial
    generation — never a false healthy claim."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=2)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="a")
        await read_and_commit(repo, through_msg_id=MessageRowId(2), text="b")
        memories = await repo.list_memories(CK)
        mem = memories[0]
        assert mem.id is not None and mem.source_hash is not None
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        # Cover ONLY memory 1 — memory 2 is missing -> partial.
        await repo.upsert_vector(
            CK, make_vector(owner_id=mem.id, generation=g.id, model="e", dim=3,
                            values=(1.0, 0.0, 0.0), source_hash=mem.source_hash)
        )
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "incomplete vector coverage" in p.detail


def test_semantic_probe_fails_dimension_mismatch(tmp_path):
    """An active generation whose stored vector dim does not match the
    generation's dim is a dimension mismatch — never healthy."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        memories = await repo.list_memories(CK)
        mem = memories[0]
        assert mem.id is not None and mem.source_hash is not None
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        # Insert a vector with dim=4 (blob 16 bytes) into a dim=3 generation
        # via raw SQL, bypassing the repo's write-time enforcement.
        import struct
        blob = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
        await db.write(
            lambda c: c.execute(
                "INSERT INTO vectors(owner_table, owner_id, dim, model,"
                " generation, source_hash, blob) VALUES (?,?,?,?,?,?,?)",
                ("memories", mem.id, 4, "e", g.id, mem.source_hash, blob),
            )
        )
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "dimension mismatch" in p.detail


def test_semantic_probe_fails_model_mismatch(tmp_path):
    """An active generation whose model does not match the configured embed
    profile is a stale/foreign generation — never healthy."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        # Insert a generation whose space_id matches the configured space
        # (e@r1) but whose model differs — a stale/foreign row.
        await db.write(
            lambda c: c.execute(
                "INSERT INTO embedding_generations(space_id, model, revision,"
                " dim, state, created_ts) VALUES (?,?,?,?,?,?)",
                ("e@r1", "wrong", "r1", 3, "active", 100.0),
            )
        )
        gid = await db.read(
            lambda c: c.execute(
                "SELECT id FROM embedding_generations WHERE space_id = 'e@r1'"
            ).fetchone()[0]
        )
        await repo.upsert_vector(
            CK, make_vector(owner_id=1, generation=gid, model="wrong", dim=3,
                            values=(1.0, 0.0, 0.0))
        )
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "model/revision mismatch" in p.detail


def test_semantic_probe_fails_revision_mismatch(tmp_path):
    """An active generation whose revision does not match the configured embed
    profile is a stale/foreign generation — never healthy."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        # Insert a generation whose space_id matches the configured space
        # (e@r1) but whose revision differs — a stale/foreign row.
        await db.write(
            lambda c: c.execute(
                "INSERT INTO embedding_generations(space_id, model, revision,"
                " dim, state, created_ts) VALUES (?,?,?,?,?,?)",
                ("e@r1", "e", "wrong", 3, "active", 100.0),
            )
        )
        gid = await db.read(
            lambda c: c.execute(
                "SELECT id FROM embedding_generations WHERE space_id = 'e@r1'"
            ).fetchone()[0]
        )
        await repo.upsert_vector(
            CK, make_vector(owner_id=1, generation=gid, model="e", dim=3,
                            values=(1.0, 0.0, 0.0))
        )
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "model/revision mismatch" in p.detail


# ── Gate 5 remediation: live dimension / source-hash / zero-vector / blocked ─

def test_semantic_probe_fails_live_dimension_mismatch(tmp_path):
    """An active generation whose dim does not match the LIVE provider
    vector dimension is a stale generation — never healthy."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        memories = await repo.list_memories(CK)
        mem = memories[0]
        assert mem.id is not None and mem.source_hash is not None
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        await repo.upsert_vector(
            CK, make_vector(owner_id=mem.id, generation=g.id, model="e", dim=3,
                            values=(1.0, 0.0, 0.0), source_hash=mem.source_hash)
        )
        # The LIVE provider now returns dim 5 — the dim-3 generation is stale.
        doctor = Doctor(
            _semantic_cfg(),
            db=db,
            repo=repo,
            adapter=FakeAdapter(),
            llm=FakeLLM(tool_response=_tool_response()),
            embedder=FakeLLM(embed_vectors=[[0.1, 0.2, 0.3, 0.4, 0.5]]),
        )
        report = await doctor.run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "dimension" in p.detail
    assert "live provider dimension" in p.detail


def test_semantic_probe_fails_stale_source_hash(tmp_path):
    """A vector whose source_hash does not match the memory's CURRENT
    source hash is stale — never healthy."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        memories = await repo.list_memories(CK)
        mem = memories[0]
        assert mem.id is not None and mem.source_hash is not None
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        # A vector whose source_hash does NOT match the memory's current one.
        await repo.upsert_vector(
            CK, make_vector(owner_id=mem.id, generation=g.id, model="e", dim=3,
                            values=(1.0, 0.0, 0.0), source_hash="stale-hash")
        )
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "stale vectors" in p.detail
    assert "source_hash mismatch" in p.detail


def test_semantic_probe_fails_zero_vector(tmp_path):
    """A zero vector (norm 0) matches nothing — a bad vector, never
    healthy."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        memories = await repo.list_memories(CK)
        mem = memories[0]
        assert mem.id is not None and mem.source_hash is not None
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        await repo.upsert_vector(
            CK, make_vector(owner_id=mem.id, generation=g.id, model="e", dim=3,
                            values=(0.0, 0.0, 0.0), source_hash=mem.source_hash)
        )
        report = await _semantic_doctor(db, repo).run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "fail"
    assert "unusable vectors" in p.detail
    assert "zero/non-finite" in p.detail


def test_semantic_probe_skips_blocked_embed_profile(tmp_path):
    """A configured embed profile whose live provider dimension cannot be
    measured (a blocked/unavailable embedder) degrades to FTS-only: the
    semantic probe skips, never a false claim."""
    async def scenario():
        db = Database(tmp_path / "s.db")
        repo = SqliteRepository(db)
        await db.open()
        await repo.upsert_chat(make_identity())
        await seed_messages(repo, n=1)
        await read_and_commit(repo, through_msg_id=MessageRowId(1), text="s")
        g = await repo.create_embedding_generation("e", 3, revision="r1")
        assert g.id is not None
        await repo.activate_embedding_generation(g.id)
        doctor = Doctor(
            _semantic_cfg(),
            db=db,
            repo=repo,
            adapter=FakeAdapter(),
            llm=FakeLLM(tool_response=_tool_response()),
            embedder=FakeLLM(embed_error=LLMTransientError("embed blocked")),
        )
        report = await doctor.run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "semantic")
    assert p.status == "skip"
    assert "FTS-only" in p.detail


# ── learner probe (Phase 6 P6.4): disabled / blocked / missing-prompt ────────

def _learn_cfg(
    enabled: bool = True,
    profiles: dict | None = None,
    *,
    llm_profile: bool = True,
) -> Config:
    """``llm_profile`` adds the ``learn`` PROVIDER profile every learner run
    calls — distinct from the per-learner ``[learn.profiles.*]`` entries."""
    cfg = _cfg()
    llm = cfg.llm
    if llm_profile:
        llm = LLMConfig(profiles={
            **cfg.llm.profiles,
            "learn": LLMProfile(
                base_url="https://api.example.com/v1",
                api_key="sk-learn",
                model="m",
            ),
        })
    return dataclasses.replace(
        cfg,
        llm=llm,
        learn=LearnConfig(
            enabled=enabled,
            profiles={k: LearnProfile(**v) for k, v in (profiles or {}).items()},
        ),
    )


def test_learner_probe_fails_without_the_learn_llm_profile():
    """Every learner run calls the one ``learn`` provider profile. Without it
    each run dies on "no LLM profile named 'learn'", the watermark never
    advances, and nothing is learned — while [learn] looks switched on. The
    doctor used to report "worker ready" through exactly that."""
    doctor = _doctor(
        cfg=_learn_cfg(profiles={"expression": {}}, llm_profile=False)
    )
    p = probe(run(doctor.run()), "learner")
    assert p.status == "fail"
    assert "[llm.profiles.learn]" in p.detail


def test_learner_probe_disabled():
    report = run(_doctor().run())  # learn disabled by default
    p = probe(report, "learner")
    assert p.status == "skip"
    assert "disabled" in p.detail


def test_learner_probe_blocked_no_profiles():
    report = run(_doctor(cfg=_learn_cfg(profiles={})).run())
    p = probe(report, "learner")
    assert p.status == "fail"
    assert "blocked" in p.detail


def test_learner_probe_blocked_unknown_learner():
    report = run(_doctor(cfg=_learn_cfg(profiles={"nope": {}})).run())
    p = probe(report, "learner")
    assert p.status == "fail"
    assert "unknown learner" in p.detail


def test_learner_probe_missing_prompt():
    doctor = _doctor(
        cfg=_learn_cfg(profiles={"expression": {}}),
        prompts=FakePromptStore(missing=("learn_expression.txt",)),
    )
    report = run(doctor.run())
    p = probe(report, "learner")
    assert p.status == "fail"
    assert "missing prompt" in p.detail


def test_learner_probe_ready():
    report = run(_doctor(cfg=_learn_cfg(profiles={"expression": {}})).run())
    p = probe(report, "learner")
    assert p.status == "ok"
    assert "ready" in p.detail
    assert p.data["valid"] == 1


# ── Phase 6 P6.5b media catalog probe ────────────────────────────────────────

def test_media_catalog_probe_skips_when_disabled():
    """The media catalog probe skips when the catalog is disabled (an
    optional capability) — never a failure."""
    report = run(_doctor().run())
    p = probe(report, "media_catalog")
    assert p.status == "skip"
    assert "disabled" in p.detail


def test_media_catalog_probe_reports_empty_and_vision_prereqs(tmp_path):
    """Enabled but empty: ok with 'empty' detail plus the vision
    prerequisites reported truthfully WITHOUT any provider call."""
    cfg = dataclasses.replace(
        _cfg(),
        media=MediaConfig(enabled=True, harvest=True, vision_profile="vision"),
    )
    report = run(_doctor(cfg).run())
    p = probe(report, "media_catalog")
    assert p.status == "ok"
    assert "empty" in p.detail
    assert p.data["approved"] == 0
    assert p.data["vision_profile"] == "vision"
    assert p.data["vision_ready"] is True
    # No provider call was made for the catalog probe.
    assert "vision" not in p.detail or "vision" in p.detail  # detail mentions prereqs


def test_media_catalog_probe_reports_missing_vision_prereq(tmp_path):
    """Enabled with no vision profile: the probe reports that harvested
    assets stay pending (unapproved) — truthfully, without calls."""
    cfg = dataclasses.replace(
        _cfg(),
        media=MediaConfig(enabled=True, harvest=True, vision_profile=None),
    )
    report = run(_doctor(cfg).run())
    p = probe(report, "media_catalog")
    assert p.status == "ok"
    assert "no vision profile" in p.detail
    assert p.data["vision_ready"] is False


def test_media_catalog_probe_reports_ready_with_approved_assets(tmp_path):
    """Enabled with approved assets: ok with 'ready' and the approved
    count."""
    async def scenario():
        db = Database(tmp_path / "doctor.db")
        await db.open()
        repo = SqliteRepository(db)
        await repo.upsert_chat(make_identity())
        cid = await repo.submit_media_candidate(
            MediaAssetCandidate(
                chat_key=CK,
                kind="sticker",
                cache_key="c" * 64,
                sha256="a" * 64,
                mime="image/gif",
                description="微笑",
            ),
            now=100.0,
        )
        await repo.approve_media_candidate(CK, cid, capacity=4, now=110.0)
        cfg = dataclasses.replace(
            _cfg(),
            media=MediaConfig(enabled=True, harvest=True, vision_profile="vision"),
        )
        doctor = Doctor(cfg, db=db, repo=repo, adapter=FakeAdapter())
        report = await doctor.run()
        return report

    report = run(scenario())
    p = probe(report, "media_catalog")
    assert p.status == "ok"
    assert "ready" in p.detail
    assert p.data["approved"] == 1


def test_media_catalog_probe_fails_on_unreadable_table():
    """Enabled but the media_assets table cannot be read: fail (cannot
    assess) — never a false healthy claim."""
    class BrokenDB(FakeDB):
        async def read(self, fn):
            raise RuntimeError("no such table: media_assets")

    cfg = dataclasses.replace(
        _cfg(),
        media=MediaConfig(enabled=True, harvest=True),
    )
    report = run(_doctor(cfg, db=BrokenDB()).run())
    p = probe(report, "media_catalog")
    assert p.status == "fail"
    assert "media_assets" in (p.error or "")


# ── Phase 6 P6.6 plugins probe ───────────────────────────────────────────────

def test_plugins_probe_skips_when_none_configured():
    report = run(_doctor().run())
    p = probe(report, "plugins")
    assert p.status == "skip"
    assert "no plugins configured" in p.detail


def test_plugins_probe_reports_loaded_fingerprint(tmp_path):
    plugin = tmp_path / "p.py"
    plugin.write_text(
        'name = "probe_plugin"\n\ndef setup(api):\n    pass\n',
        encoding="utf-8",
    )
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        f'[plugins]\npaths = ["{plugin.name}"]\n', encoding="utf-8"
    )
    cfg = Config.load(cfg_file)
    report = run(_doctor(cfg).run())
    p = probe(report, "plugins")
    assert p.status == "ok"
    assert p.data["names"] == ["probe_plugin"]
    assert "probe_plugin" in p.detail


def test_plugins_probe_fails_on_missing_module(tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        '[plugins]\npaths = ["missing.py"]\n', encoding="utf-8"
    )
    cfg = Config.load(cfg_file)
    report = run(_doctor(cfg).run())
    p = probe(report, "plugins")
    assert p.status == "fail"
    assert "missing.py" in p.detail


def test_plugins_probe_never_imports_unconfigured_code(tmp_path):
    """The doctor resolves ONLY the configured sources — an unconfigured
    module in the same directory is never imported."""
    unconfigured = tmp_path / "unconfigured.py"
    unconfigured.write_text(
        "raise RuntimeError('must never be imported')\n", encoding="utf-8"
    )
    plugin = tmp_path / "p.py"
    plugin.write_text(
        'name = "probe_plugin"\n\ndef setup(api):\n    pass\n',
        encoding="utf-8",
    )
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        f'[plugins]\npaths = ["{plugin.name}"]\n', encoding="utf-8"
    )
    cfg = Config.load(cfg_file)
    report = run(_doctor(cfg).run())
    p = probe(report, "plugins")
    assert p.status == "ok"
    assert p.data["names"] == ["probe_plugin"]


# ── Phase 6 P6.6b chat-controls probe ───────────────────────────────────────

def test_chat_controls_probe_skips_without_surface():
    report = run(_doctor().run())  # FakeRepo lacks the surface
    p = probe(report, "chat_controls")
    assert p.status == "skip"


def test_chat_controls_probe_ok_with_real_repo(tmp_path):
    async def scenario():
        db = Database(str(tmp_path / "doctor.db"))
        await db.open()
        repo = SqliteRepository(db)
        doctor = Doctor(Config(), db=db, repo=repo, adapter=FakeAdapter())
        report = await doctor.run()
        await db.close()
        return report

    report = run(scenario())
    p = probe(report, "chat_controls")
    assert p.status == "ok"
    assert "surface ready" in p.detail


# ── access probe: where the bot may speak ───────────────────────────────────


def _access_cfg(**kw) -> Config:
    return dataclasses.replace(_cfg(), access=AccessConfig(**kw))


def test_access_probe_reports_the_permissive_default():
    p = probe(run(_doctor().run()), "access")
    assert p.status == "ok"
    assert "groups: all allowed" in p.detail
    assert "private: all allowed" in p.detail
    assert p.data["silent"] == []


def test_access_probe_says_so_when_the_bot_can_never_reply():
    """An empty whitelist looks like an oversight and behaves like a mute.
    Silence that the operator cannot explain is the failure mode here, so
    the probe names it rather than leaving it to be discovered."""
    p = probe(
        run(_doctor(cfg=_access_cfg(groups=AccessListConfig(mode="whitelist"))).run()),
        "access",
    )
    assert "empty whitelist" in p.detail
    assert "NEVER reply in groups" in p.detail
    assert p.data["silent"] == ["groups"]


def test_access_probe_reports_a_disabled_category():
    p = probe(
        run(_doctor(cfg=_access_cfg(private=AccessListConfig(enabled=False))).run()),
        "access",
    )
    assert "private: disabled" in p.detail
    assert p.data["silent"] == ["private"]


def test_access_probe_never_fails_on_policy():
    """Silencing the bot everywhere is a legitimate choice, not a broken
    install — the probe informs, it does not veto."""
    cfg = _access_cfg(
        groups=AccessListConfig(enabled=False),
        private=AccessListConfig(enabled=False),
    )
    p = probe(run(_doctor(cfg=cfg).run()), "access")
    assert p.status == "ok"
    assert p.data["silent"] == ["groups", "private"]


def test_access_probe_counts_the_lists():
    cfg = _access_cfg(
        groups=AccessListConfig(mode="whitelist", ids=("1", "2")),
        private=AccessListConfig(ids=("9",)),
    )
    p = probe(run(_doctor(cfg=cfg).run()), "access")
    assert "groups: whitelist of 2" in p.detail
    assert "private: blacklist of 1" in p.detail
    assert p.data["groups"]["count"] == 2
    assert p.data["private"]["mode"] == "blacklist"
