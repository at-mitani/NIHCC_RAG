# syntax=docker/dockerfile:1
# Ingest Worker（設計書 11.1）。Queue consumer。build context はリポジトリルート。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home /srv --shell /usr/sbin/nologin app

WORKDIR /srv

COPY apps/backend-api/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY packages/common /srv/packages/common
RUN pip install --no-cache-dir /srv/packages/common

COPY apps/ingest-worker/app /srv/app

USER app
EXPOSE 9100

CMD ["python", "-m", "app.main"]
