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

ENTRYPOINT ["pretender"]
