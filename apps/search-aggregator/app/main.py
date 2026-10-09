"""Search Aggregator: ベクトル用 Citus クラスタへの 2 段検索（設計書 5.2 / 7.2）。

検索の単位は画像 1 枚の胸部野である。ベクトル用クラスタは 1 つ（1 Coordinator +
5 Worker）で、circuit breaker もこのクラスタに 1 つだけ持つ。

2 段検索（coarse -> fine）
--------------------------
query_vector_fine が指定されたとき、検索を 2 文に分ける。

  1 段目: embedding（BiomedCLIP、HNSW index）の ANN で candidate_k の
          VECTOR_STAGE1_MULTIPLIER 倍まで候補を取る（近似）。
  2 段目: その image_id 集合に対して embedding_fine（胸部野の分類器のプロファイル、
          index 無し）の cosine 距離を厳密に計算し、並べ直す。

所見プロファイルでの全件検索（stage1 = fine）
------------------------------------------------
画像だけの検索では、1 段目を使わず embedding_fine（15 次元）で全行を厳密に検索する
2 段検索では 1 段目（BiomedCLIP）が拾わなかった画像は並べ替えの対象にすらならず、
所見の合う画像を取りこぼすため。index は張らない。15 次元 × 11 万行は shard ごとの順次走査で
足り、HNSW の近似による取りこぼしも無い。ORDER BY ... LIMIT は Citus が shard ごとに
押し下げ、Coordinator は各 shard の上位だけをまとめる。

1 文の CTE にまとめない理由:

  - Citus は CTE を中間結果として Coordinator へ実体化することがある。2 文に
    分ければ距離計算は Worker 側で完結し、ネットワークを渡るのは image_id と
    距離だけになる。
  - 2 文目は分散列（image_id）の = ANY(...) なので shard pruning が効き、
    Citus の実行計画が予測しやすい。

失敗の返し方
------------
**「遅かった」と「落ちている」は対処が正反対である。** 文ごとタイムアウト
（statement_timeout）による打ち切りは 504、それ以外の失敗は 503 で返す
（設計書 9.7）。
"""

from __future__ import annotations

import logging
import threading
import time

import psycopg
from nihcc_rag import config, db, problem, rerank, service
from nihcc_rag.contract import (
    LungShapeRequest,
    LungShapeResponse,
    ShapeCandidate,
    VectorCandidate,
    VectorSearchRequest,
    VectorSearchResponse,
)
from prometheus_client import Counter, Gauge, Histogram

_LOG = logging.getLogger("search-aggregator")

QUERIES = Counter("vector_queries_total", "Vector queries", ["kind", "outcome"])
LATENCY = Histogram(
    "vector_query_seconds",
    "Vector query latency",
    ["kind"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
STAGE_LATENCY = Histogram(
    "vector_stage_seconds",
    "Vector query latency per stage (coarse ANN / exact fine re-scoring)",
    ["stage"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
RERANK_OUTCOMES = Counter(
    "vector_rerank_total", "Second-stage re-scoring outcomes", ["rerank_status"]
)
BREAKER_STATE = Gauge("vector_circuit_breaker_open", "Circuit breaker open")

app = service.create_app("search-aggregator", title="NIHCC RAG Search Aggregator")

_EF_SEARCH = config.env_int("HNSW_EF_SEARCH", 100)
#: pgvector の hnsw.ef_search の上限。1 段目の LIMIT はこれを超えられない。
_EF_SEARCH_MAX = 1000
_MAX_SCAN_TUPLES = config.env_int("HNSW_MAX_SCAN_TUPLES", 20000)
_TIMEOUT_MS = config.env_int("VECTOR_QUERY_TIMEOUT_MS", 2000)
_BREAKER_THRESHOLD = config.env_int("CIRCUIT_BREAKER_FAILURES", 3)
_BREAKER_COOLDOWN = config.env_int("CIRCUIT_BREAKER_COOLDOWN_SECONDS", 30)
_STAGE1_MULTIPLIER = config.rerank_stage1_multiplier()
_STAGE1_MAX = min(config.rerank_stage1_max(), _EF_SEARCH_MAX)
#: 2 段目に残しておく最低の timeout 予算。これを割ったら 2 段目を諦め、
#: 1 段目の結果を rerank_status = BUDGET_EXHAUSTED として返す。
_STAGE2_MIN_BUDGET_MS = config.env_int("VECTOR_STAGE2_MIN_BUDGET_MS", 200)

_iterative_scan_supported = False


class ClusterTimeout(RuntimeError):
    """statement_timeout による打ち切り（504 で返す）。"""


class ClusterUnavailable(RuntimeError):
    """接続失敗・SQL エラー・遮断中（503 で返す）。"""


class CircuitBreaker:
    """クラスタの circuit breaker（設計書 10.2）。連続失敗で一定時間遮断する。"""

    def __init__(self) -> None:
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= _BREAKER_COOLDOWN:
                self._opened_at = None
                self._failures = 0
                BREAKER_STATE.set(0)
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            BREAKER_STATE.set(0)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= _BREAKER_THRESHOLD and self._opened_at is None:
                self._opened_at = time.monotonic()
                BREAKER_STATE.set(1)
                _LOG.warning("circuit breaker opened")


_breaker = CircuitBreaker()


@app.on_event("startup")
def _startup() -> None:
    global _iterative_scan_supported
    db.init_vector_pool(role="app_search", application_name="search-aggregator")
    BREAKER_STATE.set(0)
    _iterative_scan_supported = _probe_iterative_scan()
    _LOG.info(
        "aggregator ready",
        extra={"fields": {"iterative_scan": _iterative_scan_supported, "ef_search": _EF_SEARCH}},
    )


@app.on_event("shutdown")
def _shutdown() -> None:
    db.close_all()


def _probe_iterative_scan() -> bool:
    """pgvector 0.8 以降の iterative scan が使えるか確認する（設計書 11.4.4）。"""
    try:
        with db.vector_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        return True
    except Exception:  # noqa: BLE001 - 未対応バージョンでは無効化して継続
        _LOG.warning("hnsw.iterative_scan is not supported by this pgvector build")
        return False


# 1 段目: BiomedCLIP（胸部野 crop）の HNSW（近似）。
# similarity は距離から Python 側で求めるため SELECT に含めない（`<=>` の評価は 1 回）。
_STAGE1_SQL = """
SELECT image_id,
       (embedding <=> %(query)s::vector) AS coarse_distance,
       roi_confidence
FROM image_embedding
WHERE embedding_model_version = %(model)s
  AND preprocess_version = %(preprocess)s
  AND (%(exclude)s::uuid[] IS NULL OR NOT (image_id = ANY(%(exclude)s::uuid[])))
ORDER BY embedding <=> %(query)s::vector
LIMIT %(limit)s
"""

# 2 段目: 1 段目が返した image_id に限った embedding_fine の厳密距離。
# index を使わず、shard pruning された少数行の順次アクセスで済む。
#
# embedding_fine_model_version の一致確認が要点である。次元が違えば pgvector が
# エラーにするが、次元が同じ別モデルのベクトルは黙って無意味な距離を返す。
_STAGE2_SQL = """
SELECT image_id,
       (embedding_fine <=> %(query_fine)s::vector) AS fine_distance
FROM image_embedding
WHERE embedding_model_version = %(model)s
  AND preprocess_version = %(preprocess)s
  AND embedding_fine_model_version = %(fine_model)s
  AND embedding_fine IS NOT NULL
  AND image_id = ANY(%(ids)s::uuid[])
"""


# 所見プロファイルでの全件検索。embedding_fine_model_version の一致確認は 2 段目と同じ理由。
_FINE_EXACT_SQL = """
SELECT image_id,
       (embedding_fine <=> %(query_fine)s::vector) AS fine_distance,
       roi_confidence
FROM image_embedding
WHERE embedding_model_version = %(model)s
  AND preprocess_version = %(preprocess)s
  AND embedding_fine_model_version = %(fine_model)s
  AND embedding_fine IS NOT NULL
  AND (%(exclude)s::uuid[] IS NULL OR NOT (image_id = ANY(%(exclude)s::uuid[])))
ORDER BY embedding_fine <=> %(query_fine)s::vector
LIMIT %(limit)s
"""


def _failed(exc: Exception, started: float, fields: dict[str, object]) -> RuntimeError:
    """失敗を記録し、上へ投げる例外（ClusterTimeout / ClusterUnavailable）を返す。"""
    timed_out = isinstance(exc, psycopg.errors.QueryCanceled)
    _breaker.record_failure()
    QUERIES.labels("search", "timeout" if timed_out else "error").inc()
    # 種別だけでは障害調査ができない（timeout なのか次元不一致なのか判らない）。
    # DB からのメッセージも記録する。認証情報・画像内容は含まれない（設計書 13）。
    _LOG.warning(
        "vector query failed",
        extra={
            "fields": {
                "error_type": type(exc).__name__,
                "timed_out": timed_out,
                "error": str(exc).splitlines()[0][:300] if str(exc) else "",
                # 「timeout まで使い切った」のか「即座に失敗した」のかで原因が
                # 全く違う（前者は負荷、後者は設定やスキーマの不一致）。
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                **fields,
            }
        },
    )
    return (ClusterTimeout if timed_out else ClusterUnavailable)(type(exc).__name__)


def _run_profile_exact(request: VectorSearchRequest) -> VectorSearchResponse:
    """所見プロファイルで全行を厳密に検索する（stage1 = fine）。"""
    query = request.query
    timeout_ms = request.timeout_ms or _TIMEOUT_MS
    started = time.perf_counter()
    try:
        with db.vector_conn() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                    cur.execute(
                        _FINE_EXACT_SQL,
                        {
                            "query_fine": db.to_pgvector(query.query_vector_fine or []),
                            "model": request.embedding_model_version,
                            "preprocess": request.preprocess_version,
                            "fine_model": request.fine_embedding_model_version,
                            "exclude": [str(v) for v in query.exclude_image_ids] or None,
                            "limit": query.candidate_k,
                        },
                    )
                    rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - 種類を分けて上位へ伝える
        raise _failed(
            exc, started, {"stage1": "fine", "timeout_ms": timeout_ms, "limit": query.candidate_k}
        ) from exc

    elapsed = time.perf_counter() - started
    LATENCY.labels("search").observe(elapsed)
    STAGE_LATENCY.labels("fine_exact").observe(elapsed)
    QUERIES.labels("search", "ok").inc()
    _breaker.record_success()

    ranked, rerank_status = rerank.profile_exact(
        [
            rerank.FineCandidate(
                image_id=str(row["image_id"]),
                fine_distance=float(row["fine_distance"]),
                roi_confidence=float(row["roi_confidence"]),
            )
            for row in rows
        ],
        limit=query.candidate_k,
        min_similarity=query.min_similarity,
    )
    RERANK_OUTCOMES.labels(rerank_status).inc()
    return VectorSearchResponse(
        candidates=[_candidate(item) for item in ranked],
        timing_ms=int(elapsed * 1000),
        stage1_ms=int(elapsed * 1000),
        rerank_status=rerank_status,
        stage1_count=len(rows),
    )


def _candidate(item: rerank.RankedCandidate) -> VectorCandidate:
    return VectorCandidate(
        image_id=item.image_id,
        similarity=item.similarity,
        distance=item.distance,
        coarse_similarity=item.coarse_similarity,
        coarse_distance=item.coarse_distance,
        reranked=item.reranked,
        roi_confidence=item.roi_confidence,
    )


def run_search(request: VectorSearchRequest) -> VectorSearchResponse:
    """検索を実行する。失敗は ClusterTimeout / ClusterUnavailable で上げる。"""
    if not _breaker.allow():
        QUERIES.labels("search", "circuit_open").inc()
        raise ClusterUnavailable("CIRCUIT_OPEN")
    if request.stage1 == "fine":
        return _run_profile_exact(request)

    query = request.query
    ef_search = request.ef_search or _EF_SEARCH
    timeout_ms = request.timeout_ms or _TIMEOUT_MS
    fine_model_version = request.fine_embedding_model_version

    # 2 段目を行うのは「fine のクエリベクトルがある」かつ「突き合わせる fine
    # モデル版が判っている」ときだけ。片方でも欠ければ 1 段検索になる。
    two_stage = bool(query.query_vector_fine) and bool(fine_model_version)
    if two_stage:
        limit = rerank.stage1_limit(
            query.candidate_k, multiplier=_STAGE1_MULTIPLIER, max_limit=_STAGE1_MAX
        )
    else:
        limit = query.candidate_k
    # HNSW は ef_search を超える件数を返せない（iterative_scan 無効時）。
    # 1 段目の LIMIT を増やしたら ef_search も必ず引き上げる。
    effective_ef = min(_EF_SEARCH_MAX, max(ef_search, limit))

    started = time.perf_counter()
    stage1_ms = 0
    stage2_ms = 0
    rerank_status = rerank.RERANK_DISABLED
    try:
        with db.vector_conn() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                    cur.execute(f"SET LOCAL hnsw.ef_search = {int(effective_ef)}")
                    if _iterative_scan_supported:
                        cur.execute("SET LOCAL hnsw.iterative_scan = strict_order")
                        cur.execute(f"SET LOCAL hnsw.max_scan_tuples = {int(_MAX_SCAN_TUPLES)}")

                    stage1_started = time.perf_counter()
                    cur.execute(
                        _STAGE1_SQL,
                        {
                            "query": db.to_pgvector(query.query_vector),
                            "model": request.embedding_model_version,
                            "preprocess": request.preprocess_version,
                            "exclude": [str(v) for v in query.exclude_image_ids] or None,
                            "limit": limit,
                        },
                    )
                    rows = cur.fetchall()
                    stage1_ms = int((time.perf_counter() - stage1_started) * 1000)
                    STAGE_LATENCY.labels("coarse").observe(time.perf_counter() - stage1_started)

                    stage1_candidates = [
                        rerank.Stage1Candidate(
                            image_id=str(row["image_id"]),
                            coarse_distance=float(row["coarse_distance"]),
                            roi_confidence=float(row["roi_confidence"]),
                        )
                        for row in rows
                    ]

                    fine_distances: dict[str, float] = {}
                    if two_stage and stage1_candidates:
                        remaining = timeout_ms - stage1_ms
                        if remaining < _STAGE2_MIN_BUDGET_MS:
                            # 1 段目で予算を使い切った。2 段目を強行すると timeout で
                            # 全滅するため、1 段目の結果を返して縮退する。
                            rerank_status = rerank.RERANK_BUDGET_EXHAUSTED
                            _LOG.warning(
                                "skipped fine re-scoring: timeout budget exhausted",
                                extra={"fields": {"stage1_ms": stage1_ms, "timeout_ms": timeout_ms}},
                            )
                        else:
                            stage2_started = time.perf_counter()
                            cur.execute(f"SET LOCAL statement_timeout = {int(remaining)}")
                            cur.execute(
                                _STAGE2_SQL,
                                {
                                    "query_fine": db.to_pgvector(query.query_vector_fine or []),
                                    "model": request.embedding_model_version,
                                    "preprocess": request.preprocess_version,
                                    "fine_model": fine_model_version,
                                    "ids": [c.image_id for c in stage1_candidates],
                                },
                            )
                            fine_distances = {
                                str(row["image_id"]): float(row["fine_distance"])
                                for row in cur.fetchall()
                            }
                            stage2_ms = int((time.perf_counter() - stage2_started) * 1000)
                            STAGE_LATENCY.labels("fine").observe(
                                time.perf_counter() - stage2_started
                            )
    except Exception as exc:  # noqa: BLE001 - 種類を分けて上位へ伝える
        raise _failed(
            exc,
            started,
            {
                "two_stage": two_stage,
                "timeout_ms": timeout_ms,
                "ef_search": effective_ef,
                "limit": limit,
            },
        ) from exc

    elapsed = time.perf_counter() - started
    LATENCY.labels("search").observe(elapsed)
    QUERIES.labels("search", "ok").inc()
    _breaker.record_success()

    if two_stage and rerank_status != rerank.RERANK_BUDGET_EXHAUSTED:
        ranked, rerank_status = rerank.merge(
            stage1_candidates,
            fine_distances,
            limit=query.candidate_k,
            min_similarity=query.min_similarity,
        )
    else:
        ranked = rerank.coarse_only(
            stage1_candidates, limit=query.candidate_k, min_similarity=query.min_similarity
        )
    RERANK_OUTCOMES.labels(rerank_status).inc()

    candidates = [_candidate(item) for item in ranked]
    return VectorSearchResponse(
        candidates=candidates,
        timing_ms=int(elapsed * 1000),
        stage1_ms=stage1_ms,
        stage2_ms=stage2_ms,
        rerank_status=rerank_status,
        stage1_count=len(stage1_candidates),
        reranked_count=sum(1 for item in candidates if item.reranked),
    )


@app.post("/api/v1/vector/search", response_model=VectorSearchResponse)
def search(request: VectorSearchRequest) -> VectorSearchResponse:
    try:
        return run_search(request)
    except ClusterTimeout as exc:
        # 503（サービス停止）にすると、実際には動いているものを止まっていると
        # 報告することになり、調査が死活確認から始まってしまう。
        raise problem.gateway_timeout("vector cluster") from exc
    except ClusterUnavailable as exc:
        raise problem.dependency_unavailable("vector cluster") from exc


_LUNG_SQL = """
SELECT image_id,
       (embedding_lung <=> %(query)s::vector) AS distance,
       roi_confidence
FROM image_embedding
WHERE preprocess_version = %(preprocess)s
  AND embedding_lung IS NOT NULL
  AND (%(exclude)s::uuid[] IS NULL OR NOT (image_id = ANY(%(exclude)s::uuid[])))
ORDER BY embedding_lung <=> %(query)s::vector
LIMIT %(limit)s
"""


def _query_lung(request: LungShapeRequest, timeout_ms: int, limit: int) -> list[dict]:
    """肺野形状の近い行を引く（DB への問い合わせだけ。失敗はそのまま上げる）。"""
    with db.vector_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                cur.execute(
                    _LUNG_SQL,
                    {
                        "query": db.to_pgvector(request.query.query_vector),
                        "preprocess": request.preprocess_version,
                        "exclude": [str(v) for v in request.query.exclude_image_ids] or None,
                        # 版違いの重複を除いても limit 件残るよう少し多めに取る
                        "limit": limit * 2,
                    },
                )
                return list(cur.fetchall())


@app.post("/api/v1/vector/lung-shape", response_model=LungShapeResponse)
def lung_shape_search(request: LungShapeRequest) -> LungShapeResponse:
    """胸郭の形が似た症例を探す（画像単位）。

    **所見の判別ではない。** 形態の似た症例を並べる道具として提供する。

    同じ画像がモデル版の違いで複数行ある場合に備えて image_id で重複を除く。
    """
    timeout_ms = request.timeout_ms or _TIMEOUT_MS
    limit = request.query.candidate_k
    started = time.perf_counter()
    try:
        rows = _query_lung(request, timeout_ms, limit)
    except Exception as exc:  # noqa: BLE001 - 種類を分けて上位へ伝える
        timed_out = isinstance(exc, psycopg.errors.QueryCanceled)
        QUERIES.labels("lung", "timeout" if timed_out else "error").inc()
        _LOG.warning(
            "lung shape query failed",
            extra={
                "fields": {
                    "error_type": type(exc).__name__,
                    "error": str(exc).splitlines()[0][:300] if str(exc) else "",
                    "timed_out": timed_out,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            },
        )
        if timed_out:
            raise problem.gateway_timeout("vector cluster") from exc
        raise problem.dependency_unavailable("vector cluster") from exc

    QUERIES.labels("lung", "ok").inc()
    best: dict[str, dict] = {}
    for row in rows:
        key = str(row["image_id"])
        if key not in best or float(row["distance"]) < float(best[key]["distance"]):
            best[key] = row
    ordered = sorted(best.values(), key=lambda r: float(r["distance"]))[:limit]
    return LungShapeResponse(
        candidates=[
            ShapeCandidate(
                image_id=str(row["image_id"]),
                distance=float(row["distance"]),
                similarity=1.0 - float(row["distance"]),
                roi_confidence=float(row["roi_confidence"]),
            )
            for row in ordered
        ],
        empty_space=not ordered,
        timing_ms=int((time.perf_counter() - started) * 1000),
    )


@app.get("/api/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/health/ready")
def ready() -> dict[str, object]:
    if not db.check_vector():
        raise problem.dependency_unavailable("vector cluster")
    return {
        "status": "ok",
        "iterative_scan": _iterative_scan_supported,
        "stage1_multiplier": _STAGE1_MULTIPLIER,
        "stage1_max_limit": _STAGE1_MAX,
    }


@app.get("/api/v1/vector/status")
def cluster_status() -> dict[str, object]:
    """管理画面向けに Worker・shard の配置・件数を返す（設計書 6.1 管理画面）。"""
    try:
        with db.vector_conn() as conn:
            # 列名が Citus のバージョンで異なるため pg_dist_node を参照する
            workers = db.fetch_all(
                conn,
                "SELECT nodename, nodeport FROM pg_dist_node "
                "WHERE noderole = 'primary' AND isactive AND groupid <> 0 ORDER BY nodename",
            )
            placement = db.fetch_all(
                conn,
                "SELECT nodename, count(*) AS shards FROM citus_shards "
                "WHERE table_name = 'image_embedding'::regclass "
                "GROUP BY nodename ORDER BY nodename",
            )
            rows = db.fetch_one(
                conn,
                "SELECT count(*) AS row_count, "
                "count(DISTINCT embedding_model_version) AS model_versions, "
                # 2 段目が効く行数。未充填行は 1 段目の順位で返る
                "count(*) FILTER (WHERE embedding_fine IS NOT NULL) AS fine_row_count, "
                "count(*) FILTER (WHERE embedding_lung IS NOT NULL) AS lung_row_count "
                "FROM image_embedding",
            )
            versions = db.fetch_all(
                conn,
                "SELECT embedding_model_version, preprocess_version, "
                "embedding_fine_model_version, count(*) AS rows "
                "FROM image_embedding GROUP BY 1, 2, 3 ORDER BY 4 DESC",
            )
    except Exception as exc:  # noqa: BLE001 - 管理画面は縮退表示する
        return {"cluster": {"available": False, "error": type(exc).__name__}}
    return {
        "cluster": {
            "available": True,
            "workers": [f"{w['nodename']}:{w['nodeport']}" for w in workers],
            "shard_placement": {p["nodename"]: p["shards"] for p in placement},
            "shard_count": sum(p["shards"] for p in placement),
            "row_count": rows["row_count"] if rows else 0,
            "fine_row_count": rows["fine_row_count"] if rows else 0,
            "lung_row_count": rows["lung_row_count"] if rows else 0,
            "model_versions": [dict(v) for v in versions],
        }
    }
