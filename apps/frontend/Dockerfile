# syntax=docker/dockerfile:1
# Frontend（設計書 11.1）。静的配信 + API への reverse proxy。ビルド工程を持たない。
FROM nginxinc/nginx-unprivileged:1.27-alpine

USER root
COPY apps/frontend/nginx.conf /etc/nginx/nginx.conf
COPY apps/frontend/static/ /usr/share/nginx/html/
RUN chown -R nginx:nginx /usr/share/nginx/html
USER nginx

EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=3s --start-period=10s --retries=5 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
