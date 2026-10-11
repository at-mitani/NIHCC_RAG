# syntax=docker/dockerfile:1
# Search Aggregator（設計書 11.1）。ベクトル用 Citus Coordinator への検索を担う。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN groupadd --system --gid 10001 app \
 && useradd --no-create-home --uid 10001 --gid app --home /srv --shell /usr/sbin/nologin app

WORKDIR /srv

COPY apps/backend-api/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY packages/common /srv/packages/common
RUN pip install --no-cache-dir /srv/packages/common

COPY apps/search-aggregator/app /srv/app

USER app
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/live', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
