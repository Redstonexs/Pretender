#!/bin/sh
# Pretender 一键部署 / one-line deployment entry point.
#
#   ./deploy.sh                 交互式部署 / interactive deploy
#   ./deploy.sh logs|status|validate|backup|stop|start|restart|update|token|uninstall
#
# 未克隆仓库时也可直接运行 / works before the repo is cloned:
#   curl -fsSL https://raw.githubusercontent.com/Redstonexs/Pretender/main/deploy.sh | sh

set -eu

REPO_URL=${PRETENDER_REPO:-https://github.com/Redstonexs/Pretender.git}
REPO_REF=${PRETENDER_REF:-main}
CLONE_DIR=${PRETENDER_DIR:-Pretender}

# ── output ──────────────────────────────────────────────────────────────────

# Mirrors detect_language() in scripts/deploy.py: the first variable that
# says anything decides, an unset/neutral locale means Chinese.
is_zh() {
    for value in "${PRETENDER_LANG:-}" "${LC_ALL:-}" "${LC_MESSAGES:-}" "${LANG:-}"; do
        [ -n "$value" ] || continue
        case "$value" in
            zh*|*zh_*|*zh-*|ZH*) return 0 ;;
            C|C.*|POSIX|posix|c|c.*) continue ;;
            *) return 1 ;;
        esac
    done
    return 0
}

say() {
    if is_zh; then printf '%s\n' "$1"; else printf '%s\n' "$2"; fi
}

die() {
    if is_zh; then printf '错误：%s\n' "$1" >&2; else printf 'Error: %s\n' "$2" >&2; fi
    exit 1
}

# ── locate the checkout ─────────────────────────────────────────────────────

is_checkout() {
    [ -f "$1/scripts/deploy.py" ] && [ -f "$1/docker-compose.yml" ]
}

find_root() {
    case "${0:-}" in
        */*)
            # shellcheck disable=SC1007  # CDPATH= scopes the empty value to cd
            candidate=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || candidate=""
            if [ -n "$candidate" ] && is_checkout "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
            ;;
    esac
    if is_checkout "$PWD"; then
        printf '%s\n' "$PWD"
        return 0
    fi
    return 1
}

bootstrap() {
    command -v git >/dev/null 2>&1 || die "需要先安装 git" "git is required"
    if is_checkout "$CLONE_DIR"; then
        say "复用已有检出：$CLONE_DIR" "Reusing the existing checkout: $CLONE_DIR"
    else
        if [ -e "$CLONE_DIR" ]; then
            die "$CLONE_DIR 已存在但不是 Pretender 检出；请换一个 PRETENDER_DIR" \
                "$CLONE_DIR exists and is not a Pretender checkout; set PRETENDER_DIR"
        fi
        say "正在克隆 $REPO_URL ($REPO_REF) 到 $CLONE_DIR" \
            "Cloning $REPO_URL ($REPO_REF) into $CLONE_DIR"
        git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$CLONE_DIR"
    fi
    # shellcheck disable=SC1007  # CDPATH= scopes the empty value to cd
    root=$(CDPATH= cd -- "$CLONE_DIR" && pwd)
    [ -f "$root/deploy.sh" ] || die \
        "$REPO_REF 分支里没有 deploy.sh；请用带该脚本的分支或 release（PRETENDER_REF）" \
        "$REPO_REF has no deploy.sh; use a branch or release that ships it (PRETENDER_REF)"
    # Under `curl ... | sh` the script itself occupies stdin, so the wizard
    # would read EOF instead of the operator's answers.  Reattach the
    # terminal when there is one.
    # `[ -r /dev/tty ]` is true even with no controlling terminal, so probe by
    # actually opening it.  The probe runs in a subshell because a failed
    # redirection on a special builtin is fatal to the whole shell in dash.
    if (: < /dev/tty) 2>/dev/null; then
        exec sh "$root/deploy.sh" "$@" < /dev/tty
    fi
    exec sh "$root/deploy.sh" "$@"
}

ROOT=$(find_root) || bootstrap "$@"

# ── prerequisites ───────────────────────────────────────────────────────────

find_python() {
    for candidate in python3 python3.13 python3.12 python3.11 python3.10 python; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

require_docker() {
    command -v docker >/dev/null 2>&1 || die \
        "需要 Docker Engine：https://docs.docker.com/engine/install/" \
        "Docker Engine is required: https://docs.docker.com/engine/install/"
    docker compose version >/dev/null 2>&1 || die \
        "需要 Docker Compose v2（docker compose）" \
        "Docker Compose v2 (docker compose) is required"
    docker info >/dev/null 2>&1 || die \
        "Docker 守护进程不可用；请启动 Docker 或把当前用户加入 docker 组" \
        "the Docker daemon is unreachable; start Docker or join the docker group"
}

compose() {
    require_docker
    docker compose -f "$ROOT/docker-compose.yml" --project-directory "$ROOT" "$@"
}

# The data volume is external so removing Compose resources cannot delete the
# database; creating it here keeps that safety without making it manual.
ensure_volume() {
    require_docker
    docker volume create pretender-data >/dev/null
}

require_env() {
    [ -f "$ROOT/.env" ] || die \
        "未找到 $ROOT/.env；请先运行 ./deploy.sh" \
        "$ROOT/.env is missing; run ./deploy.sh first"
}

# Only the port inside [adapter.onebot]; other sections may have one too.
configured_port() {
    [ -f "$ROOT/config/config.toml" ] || return 0
    awk '
        /^[[:space:]]*\[/ { in_section = ($0 ~ /^[[:space:]]*\[adapter\.onebot\]/) }
        in_section && /^[[:space:]]*port[[:space:]]*=/ {
            if (match($0, /[0-9]+/)) { print substr($0, RSTART, RLENGTH); exit }
        }
    ' "$ROOT/config/config.toml"
}

# Only the host inside [adapter.onebot]; this is deliberately separate from
# configured_port because other sections may contain a host setting too.
configured_host() {
    [ -f "$ROOT/config/config.toml" ] || return 0
    awk '
        /^[[:space:]]*\[/ { in_section = ($0 ~ /^[[:space:]]*\[adapter\.onebot\]/) }
        in_section && /^[[:space:]]*host[[:space:]]*=/ {
            if (match($0, /"[^"]*"/)) {
                print substr($0, RSTART + 1, RLENGTH - 2); exit
            }
        }
    ' "$ROOT/config/config.toml"
}

configured_image() {
    [ -f "$ROOT/.env" ] || return 0
    sed -n 's/^PRETENDER_IMAGE=//p' "$ROOT/.env" | head -n 1
}

validate_image_reference_shell() {
    image_reference=$1
    case "$image_reference" in
        ''|*[!A-Za-z0-9._/@:+-]*)
            return 1
            ;;
    esac
    if [ "${image_reference#*@}" != "$image_reference" ]; then
        image_digest=${image_reference#*@}
        case "$image_digest" in
            *:*)
                image_algorithm=${image_digest%%:*}
                image_hex=${image_digest#*:}
                case "$image_algorithm" in
                    ''|*[!A-Za-z0-9+._-]*) return 1 ;;
                esac
                case "$image_hex" in
                    ''|*[!0-9A-Fa-f]*) return 1 ;;
                esac
                [ "${#image_hex}" -ge 32 ] || return 1
                ;;
            *) return 1 ;;
        esac
        case "${image_reference%@*}" in
            *:[lL][aA][tT][eE][sS][t]) return 1 ;;
        esac
        return 0
    fi
    image_last_component=${image_reference##*/}
    case "$image_last_component" in
        *:*) image_tag=${image_last_component##*:} ;;
        *) return 1 ;;
    esac
    [ -n "$image_tag" ] || return 1
    case "$image_tag" in
        [-.]*|*[!A-Za-z0-9_.-]*) return 1 ;;
    esac
    [ "${#image_tag}" -le 128 ] || return 1
    case "$image_tag" in
        [lL][aA][tT][eE][sS][t]) return 1 ;;
    esac
    return 0
}

require_configured_image() {
    configured_image_value=$(configured_image)
    if ! validate_image_reference_shell "$configured_image_value"; then
        die "PRETENDER_IMAGE 必须是带显式 tag 或 digest 的固定镜像，且不能是 latest" \
            "PRETENDER_IMAGE must be an explicitly tagged or digest-pinned image, not latest"
    fi
}

usage() {
    cat <<'USAGE'
Pretender deploy

  ./deploy.sh [install] [wizard flags]   部署向导 / run the wizard
  ./deploy.sh start | stop | restart     启停容器 / control the container
  ./deploy.sh logs                       跟随日志 / follow logs
  ./deploy.sh status                     查看状态 / show status
  ./deploy.sh validate                   非破坏性检查 / non-disruptive checks
  ./deploy.sh backup                     在线备份数据库 / online database backup
  ./deploy.sh update                     拉取新镜像并重启 / pull and restart
  ./deploy.sh token                      打印 OneBot token / print the token
  ./deploy.sh uninstall                  停止并移除容器（保留数据卷）

以上运维子命令针对 Docker Compose 部署；docker run 与原生部署的对应命令由向导结束时打印。
Those management subcommands target the Compose deployment; the wizard prints the
equivalents for the docker run and native targets when it finishes.

Wizard flags: --advanced --port N --lang zh|en --non-interactive --yes --start
              --dry-run --force --provider deepseek|openai|custom --model ID
USAGE
}

# ── subcommands ─────────────────────────────────────────────────────────────

validation_failures=0

validation_ok() {
    say "通过：$1" "OK: $2"
}

validation_bad() {
    say "失败：$1" "FAIL: $2"
    validation_failures=$((validation_failures + 1))
}

validation_waiting() {
    say "等待：$1" "WAITING: $2"
}

validate_core_deployment() {
    validation_failures=0
    validation_container=$(docker compose -f "$ROOT/docker-compose.yml" \
        --project-directory "$ROOT" ps -q --all pretender 2>/dev/null) || validation_container=""
    if [ -z "$validation_container" ]; then
        validation_bad "Compose 服务 pretender 不存在" "Compose service pretender does not exist"
        validation_waiting "请先运行 ./deploy.sh start" "run ./deploy.sh start first"
        return 1
    fi
    validation_ok "Compose 服务存在" "Compose service exists"

    validation_state=$(docker inspect --format '{{.State.Status}}:{{.State.Running}}:{{.State.Restarting}}' \
        "$validation_container" 2>/dev/null) || validation_state=""
    validation_status=${validation_state%%:*}
    validation_running=${validation_state#*:}
    validation_running=${validation_running%%:*}
    validation_restarting=${validation_state##*:}
    if [ "$validation_running" = "true" ] && [ "$validation_restarting" = "false" ] && \
        [ "$validation_status" != "restarting" ]; then
        validation_ok "服务正在运行" "service is running"
    elif [ "$validation_status" = "restarting" ] || [ "$validation_restarting" = "true" ]; then
        validation_bad "服务正在重启循环；请检查 ./deploy.sh logs" \
            "service is in a restart loop; inspect ./deploy.sh logs"
    else
        validation_bad "服务未运行；请运行 ./deploy.sh start" \
            "service is not running; run ./deploy.sh start"
    fi

    validation_requested_image=$(configured_image)
    if validate_image_reference_shell "$validation_requested_image"; then
        validation_ok "PRETENDER_IMAGE 是固定镜像引用" "PRETENDER_IMAGE is pinned"
    else
        validation_bad "PRETENDER_IMAGE 必须带显式 tag 或 digest，且不能是 latest" \
            "PRETENDER_IMAGE must be explicitly tagged or digest-pinned, not latest"
    fi
    validation_live_image=$(docker inspect --format '{{.Config.Image}}' \
        "$validation_container" 2>/dev/null) || validation_live_image=""
    if [ -n "$validation_requested_image" ] && \
        [ "$validation_live_image" = "$validation_requested_image" ]; then
        validation_ok "运行镜像与请求的固定镜像一致" "running image matches the requested pin"
    else
        validation_bad "运行镜像与 PRETENDER_IMAGE 不一致；请用固定镜像重建" \
            "running image does not match PRETENDER_IMAGE; recreate with the pinned image"
    fi

    validation_network=$(docker inspect --format '{{.HostConfig.NetworkMode}}' \
        "$validation_container" 2>/dev/null) || validation_network=""
    if [ "$validation_network" = "host" ]; then
        validation_ok "使用 host 网络" "host networking is enabled"
    else
        validation_bad "必须使用 native host 网络（不要使用端口映射）" \
            "native host networking is required (do not use port mappings)"
    fi

    validation_config_mount=$(docker inspect --format \
        '{{range .Mounts}}{{if eq .Destination "/config/config.toml"}}{{.Type}}:{{.RW}}:{{.Source}}{{end}}{{end}}' \
        "$validation_container" 2>/dev/null) || validation_config_mount=""
    validation_expected_config="$ROOT/config/config.toml"
    if [ "$validation_config_mount" = "bind:false:$validation_expected_config" ]; then
        validation_ok "配置文件 bind 为只读" "config bind is read-only"
    else
        validation_bad "配置文件必须从 $validation_expected_config 只读 bind 挂载" \
            "config.toml must be a read-only bind mount from the expected source"
    fi

    validation_data_mount=$(docker inspect --format \
        '{{range .Mounts}}{{if eq .Destination "/config/data"}}{{.Type}}:{{.Name}}{{end}}{{end}}' \
        "$validation_container" 2>/dev/null) || validation_data_mount=""
    if [ "$validation_data_mount" = "volume:pretender-data" ]; then
        validation_ok "使用 pretender-data 数据卷" "pretender-data volume is mounted"
    else
        validation_bad "必须挂载名为 pretender-data 的数据卷到 /config/data" \
            "the pretender-data volume must be mounted at /config/data"
    fi

    # --no-trunc: `docker compose ps -q` yields the full 64-char id while a
    # bare `docker ps -aq` yields the 12-char short id, so the identity
    # comparison below could never match and validate always failed on a
    # correctly configured deployment.
    validation_consumers=$(docker ps -aq --no-trunc --filter volume=pretender-data 2>/dev/null) || validation_consumers=""
    validation_consumer_count=$(printf '%s\n' "$validation_consumers" | awk 'NF { count += 1 } END { print count + 0 }')
    validation_consumer_id=$(printf '%s\n' "$validation_consumers" | awk 'NF { print; exit }')
    if [ "$validation_consumer_count" -eq 1 ] && [ "$validation_consumer_id" = "$validation_container" ]; then
        validation_ok "pretender-data 只有当前服务一个消费者" "pretender-data has exactly one consumer: this service"
    else
        validation_bad "pretender-data 必须恰好由当前服务一个容器使用（当前 $validation_consumer_count 个）" \
            "pretender-data must have exactly one consumer, the current service (found $validation_consumer_count)"
    fi

    validation_topology_script='import ipaddress,sys,tomllib
with open(sys.argv[1], "rb") as stream:
    config=tomllib.load(stream)
adapter=config.get("adapter", {})
onebot=adapter.get("onebot", {})
host=onebot.get("host")
port=onebot.get("port")
try:
    loopback=host == "localhost" or ipaddress.ip_address(host).is_loopback
except (ValueError, TypeError):
    loopback=False
if (adapter.get("name") != "onebot" or onebot.get("mode") != "reverse_ws" or
        not loopback or not isinstance(port, int) or isinstance(port, bool) or
        not 1024 <= port <= 65535 or onebot.get("path") != "/onebot/v11/ws"):
    raise SystemExit(2)
print(port)'
    validation_topology=$(docker exec "$validation_container" python -c \
        "$validation_topology_script" /config/config.toml 2>/dev/null) || validation_topology=""
    validation_listener_valid=0
    case "$validation_topology" in
        ''|*[!0-9]*) ;;
        *)
            if [ "$validation_topology" -ge 1024 ] && [ "$validation_topology" -le 65535 ]; then
                validation_port=$validation_topology
                validation_listener_valid=1
            fi
            ;;
    esac
    if [ "$validation_listener_valid" -eq 1 ]; then
        validation_ok "已配置 onebot/reverse_ws 回环监听器，端口 $validation_port" \
            "configured onebot/reverse_ws loopback listener on port $validation_port"
    else
        validation_bad "运行配置必须使用 onebot、reverse_ws、回环 host、规范路径 /onebot/v11/ws 和有效端口" \
            "live config must use adapter onebot, reverse_ws, a loopback host, canonical path /onebot/v11/ws, and a valid port"
    fi

    validation_peer_established=0
    if [ "$validation_running" = "true" ] && [ "$validation_listener_valid" -eq 1 ]; then
        validation_socket_probe='import ipaddress,os,sys
port=int(sys.argv[1])
proc_root=sys.argv[2] if len(sys.argv) > 2 else "/proc"
inodes=set()
for descriptor in os.listdir(os.path.join(proc_root, "1", "fd")):
    try:
        target=os.readlink(os.path.join(proc_root, "1", "fd", descriptor))
    except OSError:
        continue
    if target.startswith("socket:[") and target.endswith("]"):
        inodes.add(target[8:-1])

def loopback(address, ipv6):
    raw=bytes.fromhex(address)
    if ipv6:
        raw=b"".join(raw[index:index + 4][::-1] for index in range(0, 16, 4))
    else:
        raw=raw[::-1]
    return ipaddress.ip_address(raw).is_loopback

listener=False
peer=False
for table,ipv6 in ((os.path.join(proc_root, "net", "tcp"),False),
                   (os.path.join(proc_root, "net", "tcp6"),True)):
    try:
        rows=open(table, encoding="ascii")
    except OSError:
        continue
    with rows:
        next(rows, None)
        for row in rows:
            fields=row.split()
            if len(fields) < 10:
                continue
            address,hex_port=fields[1].split(":", 1)
            if int(hex_port, 16) != port or fields[9] not in inodes:
                continue
            if not loopback(address, ipv6):
                continue
            listener |= fields[3] == "0A"
            peer |= fields[3] == "01"
print(int(listener), int(peer))'
        validation_socket_probe_result=$(docker exec "$validation_container" python -c \
            "$validation_socket_probe" "$validation_port" 2>/dev/null) || validation_socket_probe_result=""
        case "$validation_socket_probe_result" in
            "1 0") validation_ok "已确认 PID 1 拥有 TCP 监听器" "PID 1 owns the TCP listener" ;;
            "1 1")
                validation_ok "已确认 PID 1 拥有 TCP 监听器" "PID 1 owns the TCP listener"
                validation_peer_established=1
                ;;
            *) validation_bad "未确认运行服务 PID 1 拥有端口 $validation_port 的回环 TCP 监听器" \
                "could not prove that service PID 1 owns the loopback TCP listener on port $validation_port" ;;
        esac
    fi

    if [ "$validation_failures" -ne 0 ]; then
        say "核心验证失败：请修复上述配置后重试。" \
            "Core validation failed: fix the invariants above and retry."
        return 1
    fi
    return 0
}

validate_peer() {
    if [ "$validation_peer_established" -eq 1 ]; then
        validation_ok "已建立 TCP 传输对端（仅表示 transport，不代表协议就绪）" \
            "TCP transport peer established; this does not prove protocol readiness"
        return 0
    fi
    validation_waiting "尚未建立 TCP 传输对端；请等待或手动完成 NapCat 登录（尚未协议就绪）" \
        "no TCP transport peer; wait or complete NapCat login manually (not protocol-ready)"
    return 2
}

validate_deployment() {
    require_env
    require_docker
    if ! compose config --quiet >/dev/null 2>&1; then
        say "Compose 配置无效；请修复 .env 或 docker-compose.yml。" \
            "Compose configuration is invalid; fix .env or docker-compose.yml."
        return 1
    fi
    if validate_core_deployment; then
        validate_peer
        return $?
    fi
    return 1
}

wait_for_core_liveness() {
    validation_attempt=1
    while [ "$validation_attempt" -le 15 ]; do
        if validate_core_deployment >/dev/null 2>&1; then
            validation_first_container=$validation_container
            validation_first_restart_count=$(docker inspect --format '{{.RestartCount}}' \
                "$validation_first_container" 2>/dev/null) || validation_first_restart_count=""
            sleep 1
            if validate_core_deployment >/dev/null 2>&1 && \
                [ "$validation_container" = "$validation_first_container" ] && \
                [ -n "$validation_first_restart_count" ] && \
                [ "$(docker inspect --format '{{.RestartCount}}' "$validation_container" 2>/dev/null)" = \
                    "$validation_first_restart_count" ]; then
                return 0
            fi
        fi
        sleep 1
        validation_attempt=$((validation_attempt + 1))
    done
    validate_core_deployment
    return 1
}

backup_database() {
    require_env
    require_docker
    backup_live_container=$(docker compose -f "$ROOT/docker-compose.yml" \
        --project-directory "$ROOT" ps -q pretender 2>/dev/null) || backup_live_container=""
    if [ -z "$backup_live_container" ]; then
        die "数据库备份需要已经运行的 Pretender 服务；请先运行 ./deploy.sh start" \
            "database backup requires the already-running Pretender service; run ./deploy.sh start first"
    fi

    backup_container_temp=".pretender-database-backup-$$.db"
    backup_host_home=${HOME:-}
    [ -n "$backup_host_home" ] || die "未设置 HOME，无法选择私有备份目录" \
        "HOME is unset; cannot choose a private backup directory"
    backup_state_root=${XDG_STATE_HOME:-$backup_host_home/.local/state}
    case "$backup_state_root" in
        /*) : ;;
        *) die "XDG_STATE_HOME 必须是绝对路径" "XDG_STATE_HOME must be an absolute path" ;;
    esac
    backup_probe="$backup_state_root"
    while [ ! -e "$backup_probe" ] && [ "$backup_probe" != "/" ]; do
        backup_probe=${backup_probe%/*}
        [ -n "$backup_probe" ] || backup_probe=/
    done
    if [ -d "$backup_probe" ]; then
        backup_real_probe=$(CDPATH= cd -- "$backup_probe" && pwd -P) || die \
            "无法解析备份状态目录" "could not resolve the backup state directory"
        case "$backup_real_probe" in
            "$ROOT"|"$ROOT"/*)
                die "数据库备份状态目录不能解析到 Git checkout 内" \
                    "database backup state directory resolves inside the Git checkout"
                ;;
        esac
    fi
    backup_dir="$backup_state_root/pretender/deploy-backups"
    case "$backup_dir" in
        "$ROOT"|"$ROOT"/*)
            die "数据库备份目录不能位于 Git checkout 内：$backup_dir" \
                "database backup directory must be outside the Git checkout: $backup_dir"
            ;;
    esac
    if [ -L "$backup_dir" ] || { [ -e "$backup_dir" ] && [ ! -d "$backup_dir" ]; }; then
        die "备份目录不安全：$backup_dir" "unsafe backup directory: $backup_dir"
    fi
    backup_owner_uid=$(id -u)
    backup_owner_gid=$(id -g)
    if [ "$backup_owner_uid" = "0" ]; then
        die "请以普通 Docker 用户运行数据库备份（不要创建 root 所有的备份文件）" \
            "run database backup as a non-root Docker user (do not create root-owned backup files)"
    fi
    mkdir -p "$backup_dir"
    chmod 700 "$backup_dir"
    backup_real_dir=$(CDPATH= cd -- "$backup_dir" && pwd -P) || die \
        "无法访问私有备份目录" "cannot access the private backup directory"
    case "$backup_real_dir" in
        "$ROOT"|"$ROOT"/*)
            die "数据库备份目录不能解析到 Git checkout 内" \
                "database backup directory resolves inside the Git checkout"
            ;;
    esac

    backup_stamp=$(date -u +%Y%m%dT%H%M%SZ) || die "无法生成备份时间戳" "could not create backup timestamp"
    backup_name="pretender-database-$backup_stamp-$$.db"
    backup_temp_path="$backup_dir/.$backup_name.tmp"
    backup_path="$backup_dir/$backup_name"
    if [ -e "$backup_temp_path" ] || [ -L "$backup_temp_path" ] || \
        [ -e "$backup_path" ] || [ -L "$backup_path" ]; then
        die "备份目标已存在：$backup_path" "backup destination already exists: $backup_path"
    fi
    if ! docker exec "$backup_live_container" test ! -e "/tmp/$backup_container_temp" \
        >/dev/null 2>&1; then
        die "运行容器中的临时备份目标已存在；请稍后重试" \
            "the temporary backup target already exists in the running container; retry later"
    fi

    backup_cleanup() {
        if [ -n "${backup_live_container:-}" ] && [ -n "${backup_container_temp:-}" ]; then
            docker exec "$backup_live_container" rm -f "/tmp/$backup_container_temp" >/dev/null 2>&1 || true
        fi
        if [ -n "${backup_temp_path:-}" ]; then
            rm -f "$backup_temp_path"
        fi
    }
    trap backup_cleanup 0 HUP INT TERM
    backup_config_check='import sys,tomllib
with open(sys.argv[1], "rb") as stream:
    config=tomllib.load(stream)
db_path=config.get("storage", {}).get("db_path")
if db_path != "data/pretender.db":
    raise SystemExit(2)'
    if ! docker exec "$backup_live_container" python -c "$backup_config_check" \
        /config/config.toml >/dev/null 2>&1; then
        die "当前运行配置的数据库路径不是受支持的 /config/data/pretender.db；未执行备份" \
            "the live config does not use the supported /config/data/pretender.db; backup refused"
    fi
    backup_python='import os,sqlite3,sys
source_path=sys.argv[1]
target_path=sys.argv[2]
if not os.path.isfile(source_path):
    raise SystemExit(2)
source=sqlite3.connect("file:" + source_path + "?mode=ro", uri=True)
fd=os.open(target_path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
os.close(fd)
target=sqlite3.connect(target_path)
source.backup(target)
target.close()
source.close()'
    if ! docker exec "$backup_live_container" python -c "$backup_python" \
        /config/data/pretender.db "/tmp/$backup_container_temp" >/dev/null 2>&1; then
        die "SQLite 在线数据库备份失败；运行中的服务未被修改" \
            "SQLite online database backup failed; the running service was not modified"
    fi
    backup_integrity='import sqlite3,sys
connection=sqlite3.connect(sys.argv[1])
result=connection.execute("PRAGMA integrity_check").fetchone()[0]
connection.close()
raise SystemExit(0 if result == "ok" else 1)'
    if ! docker exec "$backup_live_container" python -c "$backup_integrity" \
        "/tmp/$backup_container_temp" >/dev/null 2>&1; then
        die "数据库备份完整性检查失败；备份已丢弃" \
            "database backup integrity check failed; the backup was discarded"
    fi
    if ! docker cp "$backup_live_container:/tmp/$backup_container_temp" "$backup_temp_path" \
        >/dev/null 2>&1; then
        die "无法从运行中的服务复制数据库备份" "could not copy the database backup from the running service"
    fi
    [ -f "$backup_temp_path" ] || die "复制的数据库备份不是普通文件" "copied database backup is not a regular file"
    chmod 600 "$backup_temp_path"
    backup_metadata=$(stat -c '%u:%g:%a:%F' "$backup_temp_path") || die \
        "无法验证数据库备份所有权和权限" "could not verify database backup ownership and mode"
    if [ "$backup_metadata" != "$backup_owner_uid:$backup_owner_gid:600:regular file" ]; then
        die "数据库备份必须归当前用户所有且权限为 0600（实际值已拒绝）" \
            "database backup must be caller-owned with mode 0600 (actual value rejected)"
    fi
    mv "$backup_temp_path" "$backup_path"
    backup_temp_path=""
    backup_live_container=""
    backup_container_temp=""
    trap - 0 HUP INT TERM
    chmod 700 "$backup_dir"
    say "数据库备份已保存（不是完整数据卷备份）：$backup_path" \
        "Database backup saved (not a full volume backup): $backup_path"
}

command_name=${1:-install}
case "$command_name" in
    install|deploy|"") [ $# -gt 0 ] && shift || true ;;
    -h|--help) command_name=help; shift ;;
    -*) command_name=install ;;
    *) shift ;;
esac

case "$command_name" in
    install)
        python=$(find_python) || die \
            "需要 Python 3.9 或更高版本" "Python 3.9 or newer is required"
        exec "$python" "$ROOT/scripts/deploy.py" "$@"
        ;;
    start)
        require_env
        require_configured_image
        compose config --quiet
        ensure_volume
        compose up -d
        compose ps
        wait_for_core_liveness
        ;;
    stop)    require_env; compose stop ;;
    restart)
        require_env
        require_configured_image
        compose config --quiet
        ensure_volume
        compose up -d --force-recreate
        compose ps
        wait_for_core_liveness
        ;;
    logs)    require_env; compose logs -f --tail 200 ;;
    update)
        require_env
        require_configured_image
        compose config --quiet
        ensure_volume
        backup_database
        compose pull
        compose up -d --force-recreate
        compose ps
        wait_for_core_liveness
        ;;
    status)
        require_env
        compose ps
        port=$(configured_port)
        if [ -n "$port" ]; then
            say "OneBot 反向 WebSocket：ws://127.0.0.1:$port/onebot/v11/ws?message_format=array" \
                "OneBot reverse WebSocket: ws://127.0.0.1:$port/onebot/v11/ws?message_format=array"
        fi
        ;;
    validate) validate_deployment ;;
    backup) backup_database ;;
    token)
        require_env
        token=$(sed -n 's/^ONEBOT_ACCESS_TOKEN=//p' "$ROOT/.env" | head -n 1)
        [ -n "$token" ] || die "在 .env 中未找到 ONEBOT_ACCESS_TOKEN" \
            "ONEBOT_ACCESS_TOKEN is not set in .env"
        printf '%s\n' "$token"
        ;;
    uninstall)
        require_env
        compose down
        say "容器已移除；数据卷 pretender-data 仍然保留。彻底删除数据：docker volume rm pretender-data" \
            "Containers removed; the pretender-data volume was kept. Delete it with: docker volume rm pretender-data"
        ;;
    help) usage ;;
    *) usage; exit 2 ;;
esac
