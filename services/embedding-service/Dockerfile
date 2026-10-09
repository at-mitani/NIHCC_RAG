# syntax=docker/dockerfile:1
# Embedding Service（設計書 7.3.4 / 11.1）。
#   WITH_TORCH=true  : BiomedCLIP（採用モデル）を使う。既定。
#   WITH_TORCH=false : deterministic backend のみの軽量イメージ（閉域検証・CI 用）。
FROM python:3.11-slim

ARG WITH_TORCH=true
# torch / torchvision の取得元。CPU: https://download.pytorch.org/whl/cpu
#                              GPU: https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf-cache \
    TRANSFORMERS_OFFLINE=0

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home /srv --shell /usr/sbin/nologin app

WORKDIR /srv

COPY services/embedding-service/requirements.txt /srv/requirements.txt
COPY services/embedding-service/requirements-torch.txt /srv/requirements-torch.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt
# torch / torchvision は TORCH_INDEX_URL から先に入れる（CPU wheel と CUDA wheel の
# 取り違えを防ぐため --index-url で固定する）。残りは PyPI から取得する。
RUN if [ "${WITH_TORCH}" = "true" ]; then pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch==2.5.1 torchvision==0.20.1 && pip install --no-cache-dir -r /srv/requirements-torch.txt; fi

COPY packages/common /srv/packages/common
RUN pip install --no-cache-dir /srv/packages/common

COPY services/embedding-service/app /srv/app

# モデル artifact は read-only volume でマウントする（イメージへは焼き込まない）
RUN mkdir -p /models/hf-cache && chown -R app:app /models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/live', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
