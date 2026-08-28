#!/bin/sh
# Pretender 一键部署 / one-line deployment entry point.
#
#   ./deploy.sh                 交互式部署 / interactive deploy
#   ./deploy.sh logs|status|stop|start|restart|update|token|uninstall
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

# The data volume is external so `compose down -v` cannot delete the database;
# creating it here keeps that safety without making it a manual pre-step.
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

usage() {
    cat <<'USAGE'
Pretender deploy

  ./deploy.sh [install] [wizard flags]   部署向导 / run the wizard
  ./deploy.sh start | stop | restart     启停容器 / control the container
  ./deploy.sh logs                       跟随日志 / follow logs
  ./deploy.sh status                     查看状态 / show status
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
    start)   require_env; ensure_volume; compose up -d; compose ps ;;
    stop)    require_env; compose stop ;;
    restart) require_env; compose restart ;;
    logs)    require_env; compose logs -f --tail 200 ;;
    update)  require_env; ensure_volume; compose pull; compose up -d; compose ps ;;
    status)
        require_env
        compose ps
        port=$(configured_port)
        if [ -n "$port" ]; then
            say "OneBot 反向 WebSocket：ws://127.0.0.1:$port/onebot/v11/ws?message_format=array" \
                "OneBot reverse WebSocket: ws://127.0.0.1:$port/onebot/v11/ws?message_format=array"
        fi
        ;;
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
