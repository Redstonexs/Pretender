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

以下方式都把配置文件挂载到 `/config/config.toml`，并从
`/config/prompts` 读取相对 prompt 文件。先准备目录和配置：

```sh
mkdir -p config prompts data logs
cp config.example.toml config/config.toml
export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...
export SILICONFLOW_API_KEY=...
export ONEBOT_ACCESS_TOKEN=...
```

### Compose：本地源码镜像

构建并启动当前源码：

```sh
docker compose build
docker compose run --rm pretender init
docker compose up -d
```

### Compose：已发布的 GHCR 镜像

使用发布版本时指定明确的 release tag；不要在生产环境默认使用 `latest`：

```sh
export PRETENDER_IMAGE=ghcr.io/redstonexs/pretender:v1.0.2
docker compose pull
docker compose run --rm pretender init
docker compose up -d --no-build
```

此路径不要运行 `docker compose build`，否则会改为构建本地源码镜像。

查看日志或优雅停止 Compose 服务：

```sh
docker compose logs -f pretender
# 保留容器，后续可用 `docker compose start` 恢复
docker compose stop
# 停止并移除容器；data/ 和 logs/ 绑定挂载会保留
docker compose down
```

镜像默认执行 `pretender run --live --config /config/config.toml`。Compose 将
`./config` 和 `./prompts` 以只读方式挂载，将 `./data`（SQLite、JSONL corpus、
embedding cache）和 `./logs` 挂载到容器。当前运行时的普通日志写入 stdout，不会主动
将 JSONL 日志写入该挂载目录。请在首次启动前
由宿主用户创建 `data`、`logs` 目录；若目录属其他 UID，请调整其写权限。
密钥只在运行时从环境变量传入，不会写入镜像。

### Docker：发布镜像生产部署

下面是使用 GHCR 发布镜像的独立流程。将 `IMAGE` 替换为实际发布的 release tag；生产环境
不要默认改用 `latest`：

```sh
IMAGE=ghcr.io/redstonexs/pretender:v1.0.2

docker pull "$IMAGE"

# 仅首次运行：初始化数据目录
docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  --env DEEPSEEK_API_KEY \
  --env DASHSCOPE_API_KEY \
  --env SILICONFLOW_API_KEY \
  --env ONEBOT_ACCESS_TOKEN \
  -v "$PWD/config/config.toml:/config/config.toml:ro" \
  -v "$PWD/prompts:/config/prompts:ro" \
  -v "$PWD/data:/config/data" \
  -v "$PWD/logs:/config/logs" \
  "$IMAGE" init

# 后台运行
docker run -d --name pretender \
  --restart unless-stopped \
  --network host \
  --user "$(id -u):$(id -g)" \
  --env DEEPSEEK_API_KEY \
  --env DASHSCOPE_API_KEY \
  --env SILICONFLOW_API_KEY \
  --env ONEBOT_ACCESS_TOKEN \
  -v "$PWD/config/config.toml:/config/config.toml:ro" \
  -v "$PWD/prompts:/config/prompts:ro" \
  -v "$PWD/data:/config/data" \
  -v "$PWD/logs:/config/logs" \
  "$IMAGE" run --live

docker logs -f pretender
docker stop -t 10 pretender
docker rm pretender
```

`init` 和运行命令都使用宿主用户、host 网络以及相同的配置、prompt、数据和日志挂载。
命令不需要显式添加 `--config`，镜像 entrypoint 会提供 `/config/config.toml`。

示例配置的 `adapter.onebot.mode = "reverse_ws"` 使用仅回环绑定
`127.0.0.1:3001`。因此 Compose 使用 `network_mode: host` 且不发布任何
`ports`；这要求 Linux 宿主机，并要求 NapCat/OneBot 与容器运行在同一台
宿主机上，通过 `ws://127.0.0.1:3001/onebot/v11/ws?message_format=array` 回连。该模式不能用
Docker Desktop 的普通端口映射替代；远程连接应先在宿主机配置本地 TLS 反向
代理。
