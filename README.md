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

从示例开始。`config.example.toml` 是完整的高级参考配置；Docker 快速部署使用
`config.docker.example.toml`。配置文件为空时使用内置默认值：

```sh
cp config.example.toml config.toml
```

相对的 `data/`、`logs/` 和 `prompts/` 路径相对于当前工作目录。API 密钥
和 OneBot token 只能通过环境变量提供；TOML 中只能写 `${ENV_NAME}`，不能
写入真实密钥。完整示例使用 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、
`SILICONFLOW_API_KEY` 和 `ONEBOT_ACCESS_TOKEN`，使用相应 profile 或适配器
前先提供所引用的变量。用户 prompt 会覆盖包内同名默认 prompt。

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

### 部署前提

- 原生 Linux 主机；提供的 `reverse_ws` 拓扑不支持 Docker Desktop 或普通端口映射。
- 已安装 Docker Engine 和 Docker Compose v2。
- 与容器同一台主机上的 NapCat/OneBot 实例，且已登录。
- 一个 LLM 提供商的 API key；默认配置使用在 DeepSeek 控制台创建的 API key。

### 部署文件在哪里 / 从零开始

需要使用包含以下三个文件的分支或 release 检出版本：
`./docker-compose.yml`、`./.env.example` 和 `./config.docker.example.toml`。
先克隆仓库；不要假定本地已经有部署模板：

```sh
git clone https://github.com/Redstonexs/Pretender.git
cd Pretender
```

以下所有命令都在包含上述文件的仓库根目录运行。

### 一键部署向导

在仓库根目录运行：

```sh
python3 scripts/deploy.py
```

向导只使用仓库内受信任的文件，不使用调用者当前工作目录中的同名文件。它会交互选择
Docker Compose、`docker run` 或原生主机部署，选择提供商以及 planner/reply 模型 ID（包括
自定义兼容端点），并选择 split、typo、media 三项功能开关。向导先展示计划并请求确认，
确认后才写入受保护的环境变量和配置文件，并执行离线校验；随后暂停，让操作者把 OneBot
token 复制到 NapCat，再要求**第二次明确确认**，确认后才进行任何 live 启动。Docker 模式
会拒绝与另一种 Docker 模式并发使用共享数据卷；原生主机模式默认前台运行，也可选择
systemd user service。使用 `--dry-run` 可预览流程，不写入文件、不执行子进程，并会遮盖密钥。
向导只支持 split、typo、media 目录；harvesting、vision、embed、learn、plugins 等需手动
使用高级配置，向导不会生成这些配置。

### 准备配置

```sh
mkdir -p config
cp config.docker.example.toml config/config.toml
cp .env.example .env
chmod 600 .env
```

然后编辑 `config/config.toml` 和 `.env`。完整的 `config.example.toml` 是高级/参考配置，
不是 Docker 快速部署模板；快速部署应使用 `config.docker.example.toml`。最小配置使用镜像
内置的 prompts，不需要创建或挂载 `prompts` 目录。

创建一次供两种部署方式共用的外部数据卷：

```sh
docker volume create pretender-data
```

该卷由 `docker run` 和 Docker Compose 共用，用于持久化 SQLite 数据。

### .env 怎么填

`.env.example` 只有以下三项；`.env` 未被 Git 跟踪，请勿提交真实密钥：

| 变量 | 填写方式 |
| --- | --- |
| `PRETENDER_IMAGE` | 填一个已发布的 GHCR release tag，例如 `ghcr.io/redstonexs/pretender:v1.0.2`；不要使用 `latest`。`docker run` 中的 `IMAGE` 必须使用完全相同的 tag。 |
| `PRETENDER_LLM_API_KEY` | 填所选 LLM 提供商的 API key；默认模板以 DeepSeek 为例，但变量名是通用的。不要把真实值写入或粘贴到 README。 |
| `ONEBOT_ACCESS_TOKEN` | 执行 `openssl rand -hex 32` 生成，填入 `.env`，并在 NapCat 中配置相同 token。 |

基础部署不需要 `DASHSCOPE_API_KEY` 或 `SILICONFLOW_API_KEY`。只有在高级配置中手动启用
相应的可选 profile 后，才添加这些变量。

生成并填写 OneBot token：

```sh
openssl rand -hex 32
$EDITOR .env
```

将命令输出的值原样写入 `.env` 的 `ONEBOT_ACCESS_TOKEN`，并在 NapCat 中配置完全相同的
token。

### 配置 NapCat

在启动 Pretender **之前**，让同机已登录的 NapCat/OneBot 连接到：
`ws://127.0.0.1:3001/onebot/v11/ws?message_format=array`。连接选项必须使用
`message_format=array`，并配置与 `.env` 完全相同的 `ONEBOT_ACCESS_TOKEN`。

**二选一：**使用下面的 `docker run` 或 Docker Compose 其中一种，不能同时运行两者；两者
共享 SQLite 数据卷。

### 方式一：docker run

确认 `IMAGE` 与 `.env` 中的 `PRETENDER_IMAGE` 完全一致，然后拉取镜像并启动：

```sh
IMAGE=ghcr.io/redstonexs/pretender:v1.0.2
docker pull "$IMAGE"

docker run -d --name pretender \
  --restart unless-stopped \
  --network host \
  --env-file .env \
  --mount "type=bind,src=$PWD/config/config.toml,dst=/config/config.toml,readonly" \
  --mount type=volume,source=pretender-data,target=/config/data \
  "$IMAGE" run --live --config /config/config.toml
```

查看日志并优雅停止：

```sh
docker logs -f pretender
docker stop -t 30 pretender
docker rm pretender
```

### 方式二：Docker Compose（推荐）

Compose 使用仓库根目录的 `./docker-compose.yml` 和 `.env` 中的
`PRETENDER_IMAGE`，采用预构建 release 镜像、原生 Linux host 网络、只读配置挂载、
`pretender-data` 命名卷、`unless-stopped` 重启策略和停止宽限期。服务命令明确执行
`run --live --config /config/config.toml`，不需要 build：

```sh
docker compose pull
docker compose up -d
```

查看日志：

```sh
docker compose logs -f pretender
```

停止服务时二选一。临时停止并保留容器：

```sh
docker compose stop
```

之后恢复：

```sh
docker compose start
```

或移除容器但保留 `pretender-data` 外部数据卷：

```sh
docker compose down
```

首次启动时会自动创建或升级数据库 schema，不需要运行 `init`。

#### 更新与安全

只在 `.env` 中将 `PRETENDER_IMAGE` 改为明确的 release tag，然后拉取并重启：

```sh
docker compose pull
docker compose up -d
```

升级前请备份 `pretender-data` 外部数据卷中的数据。SQLite 不支持多个副本同时写入，只运行
一个 replica。`.env` 始终保持未跟踪；由于数据卷是外部卷，`docker compose down` 和
`docker compose down -v` 都不会删除它。确认已停止并移除使用该卷的容器后，必须明确执行
`docker volume rm pretender-data` 才会删除数据。生产环境不要使用 `latest`，只更新为明确的
release tag。

#### 网络与安全

当前 `reverse_ws` 配置刻意只绑定 `127.0.0.1:3001`，因此两种方式都需要原生 Linux host
网络，并要求 NapCat/OneBot 与容器在同一台主机上；不要配置或发布 `ports`。Docker Desktop
和普通端口映射不适用于此配置。使用 forward `ws`/`wss` 的用户可以在自己的 Compose
override 中移除 host 网络，但不得在没有 TLS 能力的情况下将 `reverse_ws` 暴露到外部。
