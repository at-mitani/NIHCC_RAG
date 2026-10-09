# syntax=docker/dockerfile:1
# Migration Job（設計書 11.4.2）。全ノード起動後に一度だけ実行する。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home /srv --shell /usr/sbin/nologin app

WORKDIR /srv
RUN pip install --no-cache-dir "psycopg[binary]==3.2.3"

COPY db/migrate/run_migrations.py /srv/run_migrations.py
COPY db/metadata-migrations /migrations/metadata
COPY db/vector-migrations /migrations/vector

USER app
ENTRYPOINT ["python", "/srv/run_migrations.py"]
