# Pretender

Pretender 是面向群聊的 Python LLM Agent：记录收到的事件，判断是否需要
回复，并通过本地控制台或显式启用的 OneBot v11 适配器生成回复。它把
SQLite ledger、JSONL corpus 和 outbox 作为可恢复的持久状态。

## 安装

需要 Python 3.11 或更高版本：

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

开发和测试环境：

```sh
python -m pip install ".[test]"
```

## 配置与密钥

从示例开始。配置文件为空时使用内置默认值：

```sh
cp config.example.toml config.toml
```

相对的 `data/`、`logs/` 和 `prompts/` 路径相对于当前工作目录。API 密钥
和 OneBot token 只能通过环境变量提供；TOML 中只能写 `${ENV_NAME}`，不能
写入真实密钥。示例配置使用 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、
`SILICONFLOW_API_KEY` 和 `ONEBOT_ACCESS_TOKEN`，使用相应 profile 或适配器
前先导出所引用的变量。用户 prompt 会覆盖包内同名默认 prompt。

## CLI 与安全

所有命令都支持 `--config PATH`：

```sh
pretender init --config config.toml
pretender run --config config.toml
pretender run --config config.toml --dry-run
pretender run --config config.toml --live
pretender doctor --config config.toml
pretender db --config config.toml --stats
pretender replay qq:group:123456 --config config.toml
```

`init` 创建 SQLite schema。普通 `run`/`--dry-run` 只使用控制台，不启动
OneBot，不创建 outbox，也不发送消息；但提供的示例配置选择了 OneBot，执行这两种
命令前需改用 `adapter.name = "console"` 的配置。若配置了 agent，评估阶段仍可能调用
LLM API。只有 `--live` 才启用发送，且需要 `planner`、`reply` profile。
`doctor` 是不发送消息的预检，`replay` 只重演已记录 ledger，不执行适配器
或 outbox 操作。默认适配器是 console；OneBot v11 必须协商
`message_format=array`，否则结构化 @ 和 reply 数据会被拒绝。

## 测试

```sh
python -m pytest
```

## Docker 部署

当前示例配置使用 `reverse_ws`，生产部署需要原生 Linux。准备目录、配置和环境文件：

```sh
mkdir -p config prompts
cp config.example.toml config/config.toml
cp .env.example .env
```

编辑 `config/config.toml` 和 `.env`。示例配置引用以下四个变量，四者都必须在 `.env`
中提供：`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`SILICONFLOW_API_KEY`、
`ONEBOT_ACCESS_TOKEN`。`.env` 未被 Git 跟踪，请勿提交真实密钥。

### docker run

使用明确的 release tag；生产环境不要使用 `latest`：

```sh
IMAGE=ghcr.io/redstonexs/pretender:v1.0.2
docker pull "$IMAGE"
docker volume create pretender-data

docker run -d --name pretender \
  --restart unless-stopped \
  --network host \
  --env-file .env \
  --mount "type=bind,src=$PWD/config/config.toml,dst=/config/config.toml,readonly" \
  --mount "type=bind,src=$PWD/prompts,dst=/config/prompts,readonly" \
  --mount type=volume,source=pretender-data,target=/config/data \
  "$IMAGE" run --live --config /config/config.toml
```

查看日志并优雅停止：

```sh
docker logs -f pretender
docker stop -t 10 pretender
docker rm pretender
```

### Docker Compose

Compose 使用 `.env` 中固定的 `PRETENDER_IMAGE`，配置和 prompt 只读挂载到容器，数据保存
在 `pretender-data` 命名卷中。镜像为预构建发布镜像，不要执行 build：

```sh
docker compose pull
docker compose up -d
```

查看日志或停止服务：

```sh
docker compose logs -f pretender
docker compose stop
docker compose down
```

启动时会自动创建或升级数据库 schema；`init` 仅用于可选的离线预检或升级，例如：

```sh
docker compose run --rm pretender init --config /config/config.toml
```

#### 更新

只在 `.env` 中将 `PRETENDER_IMAGE` 改为明确的 release tag，然后拉取并重启：

```sh
docker compose pull
docker compose up -d
```

升级前请备份 `pretender-data` 命名卷中的数据。SQLite 不支持多个副本同时写入，勿运行
多个 Pretender replicas。

#### 网络与安全

当前 `reverse_ws` 配置刻意只绑定 `127.0.0.1:3001`，因此 Docker run 和 Compose 都使用
host 网络，并要求 NapCat/OneBot 与容器在同一台原生 Linux 主机上；不发布 `ports`。
Docker Desktop 或普通端口映射不适用于此配置。使用 forward `ws`/`wss` 的用户可以在自己
的 Compose override 中移除 host 网络，但不得在没有 TLS 能力的情况下将 `reverse_ws` 暴露
到外部。

#### 本地镜像（仅开发）

如需验证本地代码，可先执行 `docker build -t pretender:dev .`，再将 `docker run` 中的
`IMAGE` 或 `.env` 中的 `PRETENDER_IMAGE` 替换为 `pretender:dev`；这不是生产发布流程。
