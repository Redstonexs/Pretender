# Pretender

Pretender is a Python LLM agent for group-chat conversations. It records
incoming events, evaluates whether a reply is needed, and can produce replies
through the local console or an explicit OneBot v11 live adapter. The normal
`run` mode is a non-network dry run, so trying the agent does not send chat
messages.

## Install

Pretender requires Python 3.11 or newer.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

For a development environment with the test dependency, install the test
extra instead:

```sh
python -m pip install ".[test]"
```

## Configure

Start with the documented sample:

```sh
cp config.example.toml config.toml
```

Configuration is optional; an empty TOML file uses the built-in defaults. API
keys and OneBot access tokens are environment-only secrets. In TOML they must
be written as `${ENV_NAME}` references; do not put secret values in the file.
Set every referenced environment variable before using the related provider
or adapter.

## Commands

All commands accept `--config PATH`. Without it, Pretender uses its defaults.

```sh
pretender init --config config.toml
pretender run --config config.toml
pretender run --config config.toml --dry-run
pretender run --config config.toml --live
pretender doctor --config config.toml
pretender replay qq:group:123456 --config config.toml
```

- `init` creates the SQLite database and schema.
- `run` and `--dry-run` are console-only: they record and evaluate input but
  create no outbox work and send nothing. This is the default.
- `--live` is the explicit send-enabled mode. It requires the `planner` and
  `reply` LLM profiles. Set `adapter.name = "onebot"` for an explicit live
  OneBot v11 connection; otherwise the configured console adapter is used.
- `doctor` runs secret-free configuration and dependency preflight checks; it
  never sends chat output.
- `replay CHAT` re-evaluates a complete recorded dispatch ledger without
  adapter or outbox operations and reports what would have been spoken.

For OneBot v11, the connection must negotiate array messages. Configure
NapCat/OneBot with `message_format=array`; string/CQ-style message payloads
are rejected so structured mentions and replies are not lost.

## Safety and durability

- Live delivery is opt-in; plain `run`, dry-run, `doctor`, and `replay` do not
  send messages.
- Incoming events and dispatch state are persisted in the SQLite-backed
  ledger. The outbox provides durable delivery work, and replay uses the
  recorded ledger rather than sending anything.
- Media is normalized into a bounded, content-addressed cache with safety
  checks before use.
- Plugins are explicit configuration inputs rather than auto-discovered or
  hot-reloaded code.
- Adaptive learners are disabled by default; enabling them is an explicit
  configuration choice.

## Test

```sh
python -m pytest
```
