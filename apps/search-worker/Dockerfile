# syntax=docker/dockerfile:1
# Search Worker（設計書 11.1）。backend-api の検索実装を再利用する。
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

# 同期経路（backend-api）と同一の検索実装を使うため app.services を同梱する
COPY apps/backend-api/app /srv/app
COPY apps/search-worker/app/main.py /srv/worker_main.py

USER app
EXPOSE 9101

CMD ["python", "/srv/worker_main.py"]
