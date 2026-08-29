# Pretender

Pretender 是面向群聊的 Python LLM Agent：记录收到的事件，判断是否需要
回复，并通过本地控制台或显式启用的 OneBot v11 适配器生成回复。它把
SQLite ledger、JSONL corpus 和 outbox 作为可恢复的持久状态。

## 快速开始（三步部署）

前提：一台原生 Linux 主机、已安装 Docker Engine + Compose v2、同机的
NapCat/OneBot 安装、一个 LLM 提供商的 API key。使用 Docker 化 NapCat 时，NapCat
容器也必须使用原生 Linux host 网络；请勿使用 bridge 网络或 `ports:` 映射。

**1. 运行一键部署脚本**

```sh
curl -fsSL https://raw.githubusercontent.com/Redstonexs/Pretender/main/deploy.sh | sh
```

已经克隆过仓库就直接 `./deploy.sh`。向导默认中文（`--lang en` 切英文），只问四个问题：
部署方式、提供商、模型、端口——除 API key 外全部有默认值，一路回车即可。默认端口是 3002；
被占用时会自动探测并改用下一个空闲端口，并告诉你是谁占着它。

**2. 按向导提示配置 NapCat**

向导会打印现成的连接地址，例如：

```
ws://127.0.0.1:3002/onebot/v11/ws?message_format=array
```

访问令牌用 `./deploy.sh token` 查看，填入 NapCat 后保存。`message_format=array`
必须保留。

**3. 保存配置后启动，再验证**

NapCat 反向客户端配置保存后，回到向导确认启动 Pretender；不要要求 NapCat 在 listener
启动前已经连通。Pretender 启动后再运行 `./deploy.sh validate`，然后进行日常操作：

```sh
./deploy.sh logs       # 跟随日志
./deploy.sh status     # 查看容器状态与连接地址
./deploy.sh validate   # 服务归属的在线拓扑/传输检查
./deploy.sh stop       # 停止
./deploy.sh start      # 启动
./deploy.sh update     # 拉取新镜像并重启
./deploy.sh token      # 打印 OneBot token
```

无人值守部署（密钥只从环境变量读取，不进 `ps` 和 shell 历史）：

```sh
PRETENDER_LLM_API_KEY=sk-... ./deploy.sh --non-interactive --port 3002
```

自定义端口时，Pretender 配置和 NapCat 连接地址必须同时改为同一个值；默认应使用 `3002`。

想逐项确认全部选项（回复模型、分段/错别字/媒体开关、镜像 tag）用 `./deploy.sh --advanced`；
只看方案不落盘用 `--dry-run`。完整部署说明见下面的 [Docker 部署](#docker-部署)。

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
`doctor` 虽然不发送消息，但会绑定监听器；不要把 `pretender doctor` 或一次性 doctor
进程配置为 live healthcheck，否则会与正在运行的 listener 冲突。运行中的服务请使用
`./deploy.sh validate` 做安全的在线检查；它不会抢占 listener。`replay` 只重演已记录 ledger，
不执行适配器或 outbox 操作。默认适配器是 console；OneBot v11 必须协商
`message_format=array`，否则结构化 @ 和 reply 数据会被拒绝。

## 测试

```sh
python -m pytest
```

## Docker 部署

### 部署前提

- 原生 Linux 主机；提供的 `reverse_ws` 拓扑要求 Pretender 和 Docker 化 NapCat
  都使用 native Linux host networking，不支持 Docker Desktop、bridge 网络或普通端口映射。
- 已安装 Docker Engine 和 Docker Compose v2。
- 与容器同一台主机上的 NapCat/OneBot 实例，且已登录。NapCat 是独立的、由操作者负责的服务。
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

在仓库根目录运行 `./deploy.sh`（等价于 `python3 scripts/deploy.py`，只需要 Python 3.9+）。

向导只使用仓库内受信任的文件，不使用调用者当前工作目录中的同名文件。默认（快捷）路径
只问四个问题：部署方式（Docker Compose / `docker run` / 本机原生）、提供商、模型 ID、
OneBot 端口，再加一次不回显的 API key 输入。回复模型默认与规划模型相同，OneBot token
自动生成，分段/错别字/媒体开关与镜像 tag 使用默认值——需要逐项确认时加 `--advanced`。

向导先展示计划并请求确认，确认后才写入受保护的环境变量和配置文件，并执行离线校验；随后
打印现成的 NapCat 连接地址并暂停，让操作者先保存 NapCat 反向客户端配置，再要求**第二次明确确认**；
NapCat 不必在 listener 启动前已经连通，第二次确认后才进行 live 启动。Docker 模式会拒绝与另一种
Docker 模式并发使用共享数据卷；
原生主机模式默认前台运行，也可选择 systemd user service。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--lang zh\|en` | 向导语言，默认按 locale 自动判断（无法判断时用中文）。 |
| `--port N` | 指定 OneBot 反向 WebSocket 端口，跳过端口提问。 |
| `--advanced` | 询问全部选项，而不是给可选项填默认值。 |
| `--non-interactive` | 完全由参数和环境变量构建方案，不提任何问题。 |
| `--yes` / `--start` | 分别跳过“写入配置”与“启动 live”两道确认。 |
| `--dry-run` | 只展示方案；不写文件、不执行子进程，密钥全部遮盖。 |
| `--force` | 覆盖向导自己生成过的文件（先备份）。 |

非交互模式的密钥只从环境变量读取（默认 `PRETENDER_LLM_API_KEY` 和
`ONEBOT_ACCESS_TOKEN`，可用 `--api-key-env` / `--token-env` 改名），因为命令行参数会出现在
`ps` 输出和 shell 历史里。`ONEBOT_ACCESS_TOKEN` 未设置时自动生成一个。

向导只支持 split、typo、media 目录；harvesting、vision、embed、learn、plugins 等需手动
使用高级配置，向导不会生成这些配置。

### 端口冲突

3002 是 canonical default（另一个 OneBot 桥接、某个开发服务器可能占用它）。现在三处都会处理：

- 向导启动时探测 3002，被占用就自动改用往上第一个空闲端口，并打印占用它的进程名和 pid；
  手动填一个已被占用的端口会被拒绝并重新询问。
- 手动部署时改 `config/config.toml` 里的 `adapter.onebot.port`，NapCat 的连接地址要改成
  同一个端口。默认连接地址为
  `ws://127.0.0.1:3002/onebot/v11/ws?message_format=array`；因为用的是 host 网络，
  不需要也不要配置 `ports:`。
- 运行时端口无法绑定会报出具体地址、占用原因和修复方式，而不是一个裸的 errno。

用 `./deploy.sh status` 可以随时看到当前配置的连接地址。

### 日常运维

```sh
./deploy.sh logs       # 跟随日志
./deploy.sh status     # 容器状态 + 当前连接地址
./deploy.sh validate   # 在线安全检查，不绑定 OneBot listener
./deploy.sh backup     # 在线备份 SQLite 数据库后再做变更
./deploy.sh start      # 启动（自动创建数据卷）
./deploy.sh stop       # 停止
./deploy.sh restart    # 对齐当前配置并重启
./deploy.sh update     # 拉取新镜像、对齐配置并重启
./deploy.sh token      # 打印 OneBot token
./deploy.sh uninstall  # 停止并移除容器，保留 pretender-data 数据卷
```

`./deploy.sh validate` 是 Linux-only、面向 Compose 的服务归属拓扑/传输检查：它确认运行中的
容器、host 网络、配置挂载、数据卷及由该服务 PID 拥有的 loopback TCP listener。它不绑定
listener，也不检查 OneBot 协议是否 ready：

- 返回码 **0** 仅表示观察到服务拥有的 TCP transport peer，不等于 OneBot 协议已就绪。
- 返回码 **2** 表示核心拓扑有效，但尚未观察到服务拥有的 transport peer；常见原因是 QR/登录
  尚未完成，或 NapCat 还在等待/重试连接。
- 其他失败先修复 `validate` 输出的核心拓扑问题。确认协议 ready 前查看无密日志：
  `docker compose logs --tail 200 pretender`，并等待
  `onebot readiness established: generation=…` lifecycle 记录。

不要让 live `pretender doctor` 或一次性 doctor 与服务并发运行；doctor 会绑定 listener。
修改配置或升级前先执行 `./deploy.sh backup`，确认备份成功，再使用 `restart` 或
`update`，让 Compose 重新协调（reconcile）受 Pretender 管理的配置后启动。不要把独立 NapCat
的配置当作 Pretender 配置；这些文件和卷必须由 NapCat 操作者另行维护和备份。

`start`、`restart` 和 `update` 在启动/重建后都会等待核心服务保持 liveness；服务退出或
重启循环会使命令失败。这仍不是 OneBot 协议 readiness。`update` 不会自动回滚：若新镜像
启动失败，先查看日志，将 `.env` 的 `PRETENDER_IMAGE` 恢复为上一个明确的 tag 或 digest，
再运行 `./deploy.sh restart`。

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

创建一次供两种部署方式共用的外部数据卷（`./deploy.sh` 和 `./deploy.sh start` 会自动创建）：

```sh
docker volume create pretender-data
```

该卷由 `docker run` 和 Docker Compose 共用，用于持久化 SQLite 数据。外部卷意味着
`docker compose down -v` 也不会删除它。

### .env 怎么填

`.env.example` 只有以下三项；`.env` 未被 Git 跟踪，请勿提交真实密钥：

| 变量 | 填写方式 |
| --- | --- |
| `PRETENDER_IMAGE` | 使用带显式 release tag 或 digest 的 GHCR 镜像，例如 `ghcr.io/redstonexs/pretender:v1.0.2`；未标记镜像和 `latest` 会被拒绝。digest 才是不可变引用；`docker run` 中的 `IMAGE` 必须使用完全相同的引用。 |
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

### 配置独立的 NapCat（NapCat-owned setup）

NapCat 是 Pretender 之外的独立安装，NapCat 的文件、登录状态、卷和密钥由 NapCat
操作者负责；Pretender 不会自动修改独立 NapCat 安装。请在启动 Pretender **之前**完成以下
检查（External NapCat checklist）。挂载方式、登录流程和变量含义随 NapCat 镜像/版本而异，
应以该镜像/版本的文档为准：

- 在同一台原生 Linux 主机运行。Docker 化 NapCat 必须使用 `--network host`（或等价的
  `network_mode: host`）；bridge-mode NapCat 无法访问 Pretender 的 `127.0.0.1` listener，
  不要用 `ports:` 绕过这个限制。
- 持久化 NapCat 的 `/app/.config/QQ` 和 `/app/napcat/config`。重建容器时保留并复用这两个
  bind/named volumes；不要为“重新创建”删除它们，否则可能触发重新登录。
- 首次登录按所用 NapCat 镜像/版本的流程扫码（QR login），确认账号在 NapCat 中已在线。
- 若该镜像/版本提供 `ACCOUNT`，它只选择 QQ UIN；它本身不足以支持无人值守重启。无人值守
  所需的登录/认证状态必须由 NapCat 自己持久化并正确配置。
- `NAPCAT_QUICK_PASSWORD` 的含义和配置位置随 NapCat 镜像/版本而异，只能放在 NapCat 自己
  受保护的 secret config 中。绝不要放进 Pretender 的 `.env`、wizard 参数或日志；Pretender
  的 `.env.example` 不包含任何 QQ password 变量。

先保存 NapCat/OneBot 的反向客户端配置；NapCat 在 Pretender listener 启动前显示等待连接是
正常的。Pretender 启动后，才让 NapCat/OneBot 连接到 canonical URL：
`ws://127.0.0.1:3002/onebot/v11/ws?message_format=array`。若使用自定义端口，必须同时修改
`config/config.toml` 的 `adapter.onebot.port` 和 NapCat 的连接地址；连接选项必须使用
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
./deploy.sh start
./deploy.sh validate
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
./deploy.sh start
```

或移除容器但保留 `pretender-data` 外部数据卷：

```sh
docker compose down
```

首次启动时会自动创建或升级数据库 schema，不需要运行 `init`。

#### 更新与安全

`./deploy.sh backup` 只能从当前已经运行的受信任 Pretender 容器执行在线 SQLite
database-only backup；未运行时先执行 `./deploy.sh start`。请使用普通的非 root Docker
操作者运行。备份保存于
`${XDG_STATE_HOME:-$HOME/.local/state}/pretender/deploy-backups`，权限为私有目录和 0600 文件。
它不包含 recorder、cache 或完整数据卷状态；需要完整恢复时，必须另行制作并实际测试卷级/平台级
快照。NapCat 的两个持久化路径和 NapCat 自己的 secure config 也必须另行备份。

升级前先备份，再只在 `.env` 中将 `PRETENDER_IMAGE` 改为明确的 tag 或 digest，然后使用会
重新协调配置的更新流程：

```sh
./deploy.sh backup
./deploy.sh update
```

`update` 会先运行同一个在线数据库备份；备份失败时会在 pull 或重建前中止，不会继续拉取镜像。
SQLite 不支持多个副本同时写入，只运行一个 replica。`.env` 始终保持未跟踪；由于数据卷是外部卷，`docker compose down` 和
`docker compose down -v` 都不会删除它。确认已停止并移除使用该卷的容器后，必须明确执行
`docker volume rm pretender-data` 才会删除数据。生产环境不要使用 `latest`，只更新为明确的
tag 或 digest。

配置变更使用 `./deploy.sh restart`（先校验并协调 Compose 配置，再 force-recreate 并等待
核心 liveness）；不要用手工容器替换掩盖配置差异。

#### NapCat 要求重新扫码（QR-needed recovery）

如果重建或更新后 NapCat 显示需要 QR：

1. 不要删除 `/app/.config/QQ`、`/app/napcat/config` 的卷，也不要删除
   `pretender-data`；先保留现场并确认备份。
2. 用原来的两个 NapCat 持久化挂载、原生 Linux host 网络和受保护 secret config 启动 NapCat，
   按 NapCat 流程重新扫码登录。
3. 确认 NapCat 已在线、反向 WebSocket 地址和 token 仍正确，再启动/重启 Pretender。
4. 用 `./deploy.sh validate` 检查 live 连接；仍失败时查看 NapCat 与 Pretender 日志，
   不要把 QQ 密码写入 Pretender 配置。

#### 网络与安全

当前 `reverse_ws` 配置刻意只绑定 `127.0.0.1:3002`。因此 Pretender 容器和 Docker 化
NapCat 容器都必须使用原生 Linux host 网络，并且 NapCat/OneBot 与 Pretender 在同一台主机上；
不要配置或发布 `ports:`。bridge-mode NapCat 看不到 Pretender 的 loopback listener，端口映射
不是修复方式。Docker Desktop 和普通端口映射不适用于此配置。使用 forward `ws`/`wss` 的用户
可以在自己的 Compose override 中移除 host 网络，但不得在没有 TLS 能力的情况下将
`reverse_ws` 暴露到外部。
