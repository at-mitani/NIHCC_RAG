# syntax=docker/dockerfile:1
# ROI Service（設計書 11.1 / 15.1）。OpenCV + NumPy による rule-based 抽出。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    OPENCV_IO_MAX_IMAGE_PIXELS=67108864

# apt の対話プロンプトを出さない（build 中は端末が無く、debconf が警告を出す）。
# ENV ではなく ARG にして、実行時のイメージには残さない。
ARG DEBIAN_FRONTEND=noninteractive

# opencv-python-headless / scikit-image の実行時共有ライブラリ
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app \
 && useradd --no-create-home --uid 10001 --gid app --home /srv --shell /usr/sbin/nologin app

WORKDIR /srv

COPY services/roi-service/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY packages/common /srv/packages/common
COPY packages/image-pipeline /srv/packages/image-pipeline
RUN pip install --no-cache-dir /srv/packages/common /srv/packages/image-pipeline

COPY services/roi-service/app /srv/app

USER app
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/live', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
