"""RabbitMQ 経由のジョブ配送（設計書 5.1 / 7.1.5 / 10.1）。

- quorum queue + publisher confirm で at-least-once
- dead letter exchange で毒メッセージを退避
- 消費側は冪等 UPSERT 前提（重複配送を許容する）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import pika
from pika.exchange_type import ExchangeType

from . import config

_LOG = logging.getLogger(__name__)

EXCHANGE = "nihcc.jobs"
DLX_EXCHANGE = "nihcc.jobs.dlx"

QUEUE_INGEST = "ingest.items"
QUEUE_SEARCH = "search.requests"
QUEUE_INGEST_DLQ = "ingest.items.dlq"
QUEUE_SEARCH_DLQ = "search.requests.dlq"

ROUTING_INGEST = "ingest.item"
ROUTING_SEARCH = "search.request"

_QUEUE_SPECS = (
    (QUEUE_INGEST, ROUTING_INGEST, QUEUE_INGEST_DLQ),
    (QUEUE_SEARCH, ROUTING_SEARCH, QUEUE_SEARCH_DLQ),
)

MAX_DELIVERY_ATTEMPTS = 5


def connect(*, client_name: str, retries: int = 30, delay: float = 2.0) -> pika.BlockingConnection:
    params = pika.URLParameters(config.amqp_url())
    params.client_properties = {"connection_name": client_name}
    params.heartbeat = 60
    params.blocked_connection_timeout = 30
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return pika.BlockingConnection(params)
        except Exception as exc:  # noqa: BLE001 - 起動順序に対するリトライ
            last = exc
            _LOG.warning(
                "queue connect retry", extra={"fields": {"attempt": attempt + 1}}
            )
            time.sleep(delay)
    raise RuntimeError(f"cannot connect to queue: {last}")


def declare_topology(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    channel.exchange_declare(EXCHANGE, ExchangeType.direct, durable=True)
    channel.exchange_declare(DLX_EXCHANGE, ExchangeType.direct, durable=True)
    for queue, routing_key, dlq in _QUEUE_SPECS:
        channel.queue_declare(dlq, durable=True, arguments={"x-queue-type": "quorum"})
        channel.queue_bind(dlq, DLX_EXCHANGE, routing_key)
        channel.queue_declare(
            queue,
            durable=True,
            arguments={
                "x-queue-type": "quorum",
                "x-dead-letter-exchange": DLX_EXCHANGE,
                "x-dead-letter-routing-key": routing_key,
                "x-delivery-limit": MAX_DELIVERY_ATTEMPTS,
            },
        )
        channel.queue_bind(queue, EXCHANGE, routing_key)


class Publisher:
    """publisher confirm 付きの発行クライアント。"""

    def __init__(self, *, client_name: str) -> None:
        self._client_name = client_name
        self._connection: pika.BlockingConnection | None = None
        self._channel = None

    def _ensure(self):
        if self._connection is None or self._connection.is_closed:
            self._connection = connect(client_name=self._client_name)
            self._channel = self._connection.channel()
            self._channel.confirm_delivery()
            declare_topology(self._channel)
        return self._channel

    def publish(self, routing_key: str, payload: dict[str, Any], *, priority: int = 0) -> None:
        channel = self._ensure()
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        try:
            channel.basic_publish(
                exchange=EXCHANGE,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                    priority=priority,
                ),
                mandatory=True,
            )
        except Exception:
            # 接続断は 1 度だけ張り直して再試行する
            self.close()
            channel = self._ensure()
            channel.basic_publish(
                exchange=EXCHANGE,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    content_type="application/json", delivery_mode=2, priority=priority
                ),
                mandatory=True,
            )

    def publish_many(self, routing_key: str, payloads: list[dict[str, Any]]) -> int:
        for payload in payloads:
            self.publish(routing_key, payload)
        return len(payloads)

    def close(self) -> None:
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._connection = None
            self._channel = None


def consume(
    queue: str,
    handler: Callable[[dict[str, Any]], None],
    *,
    client_name: str,
    prefetch: int = 4,
) -> None:
    """メッセージを 1 件ずつ処理する。例外時は nack（再配送 → 上限で DLQ）。"""
    connection = connect(client_name=client_name)
    channel = connection.channel()
    declare_topology(channel)
    channel.basic_qos(prefetch_count=prefetch)

    for method, _properties, body in channel.consume(queue, inactivity_timeout=1.0):
        if method is None:
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            _LOG.error("malformed message routed to dlq")
            channel.basic_reject(method.delivery_tag, requeue=False)
            continue
        try:
            handler(payload)
        except Exception:  # noqa: BLE001 - 個別メッセージの失敗でワーカーを落とさない
            _LOG.exception("message handling failed")
            channel.basic_nack(method.delivery_tag, requeue=True)
        else:
            channel.basic_ack(method.delivery_tag)


def queue_depth(queue: str, *, client_name: str = "probe") -> int:
    connection = connect(client_name=client_name, retries=1, delay=0.5)
    try:
        channel = connection.channel()
        declared = channel.queue_declare(queue, durable=True, passive=True)
        return int(declared.method.message_count)
    finally:
        connection.close()


def check_ready() -> bool:
    try:
        connection = connect(client_name="readyz", retries=1, delay=0.5)
        connection.close()
        return True
    except Exception:  # noqa: BLE001 - readiness probe
        return False
