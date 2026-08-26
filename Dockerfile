FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY pretender ./pretender

RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin pretender \
    && mkdir -p /config/data /config/logs /config/prompts \
    && chown -R 1000:1000 /config

USER pretender
WORKDIR /config

# Keep the config path fixed while allowing `docker compose run ... init` to
# replace the default command without repeating the config argument.
ENTRYPOINT ["sh", "-c", "exec pretender \"$@\" --config /config/config.toml", "pretender"]
CMD ["run", "--live"]
