"""Migration Job（設計書 11.4.2）。

全ノードの起動・health check 後に専用ジョブから実行する:
  1. Metadata DB のロール作成とスキーマ適用
  2. ベクトル用 Citus クラスタの全ノード（Coordinator 1 + Worker 5）に extension を作成
  3. Coordinator へ Worker を登録し、分散テーブルと HNSW index を作成
  4. 適用状況を schema_migration へ記録（再実行しても安全）

途中で失敗した場合は非ゼロ終了し、アプリケーションの model/schema version
切替を行わせない。
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import psycopg
from psycopg import sql

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"db-migrate","message":"%(message)s"}',
)
LOG = logging.getLogger("db-migrate")

WORKERS_PER_CLUSTER = int(os.environ.get("WORKERS_PER_CLUSTER", "5"))
#: Worker 5 台に 6 個ずつ均等に載る数（24 だと 5,5,5,5,4 と偏る）
SHARD_COUNT = int(os.environ.get("CITUS_SHARD_COUNT", "30"))
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "512"))
# 2 段検索の 2 段目（fine）に使うベクトルの次元。embedding_fine 列の幅を決めるだけ
# なので、fine モデルを使わない構成でも列は作る（全行 NULL になる）。
EMBEDDING_FINE_DIMENSION = int(os.environ.get("EMBEDDING_FINE_DIMENSION", "15"))
# 肺野全体の形状ベクトルの次元。nihcc_rag.shape.LUNG_DIMENSION と揃えること。
EMBEDDING_LUNG_DIMENSION = int(os.environ.get("EMBEDDING_LUNG_DIMENSION", "16"))

#: 記録済み checksum を再実行なしで更新する版（カンマ区切り）。
#: 置換値の変更で checksum がずれた既存環境を 1 回だけ通すための措置。
REPAIR_CHECKSUMS = {
    v.strip() for v in os.environ.get("MIGRATION_REPAIR_CHECKSUMS", "").split(",") if v.strip()
}


def _repair_allowed(version: str) -> bool:
    """この版の checksum 修復が許可されているか。

    完全一致と、ファイル名だけの一致（"x/V001__y.sql" に対する "V001__y.sql"）を認める。
    """
    if version in REPAIR_CHECKSUMS:
        return True
    return version.rsplit("/", 1)[-1] in REPAIR_CHECKSUMS


SUPERUSER = os.environ.get("POSTGRES_USER", "postgres")
METADATA_MIGRATIONS = Path(os.environ.get("METADATA_MIGRATIONS", "/migrations/metadata"))
VECTOR_MIGRATIONS = Path(os.environ.get("VECTOR_MIGRATIONS", "/migrations/vector"))


def _secret(name: str, default: str | None = None) -> str:
    path = os.environ.get(f"{name}_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"missing secret: {name}")
    return value


DB_PASSWORD = _secret("POSTGRES_PASSWORD")
APP_PASSWORD = _secret("APP_DB_PASSWORD", DB_PASSWORD)


def connect(host: str, dbname: str, *, port: int = 5432, retries: int = 60) -> psycopg.Connection:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            conn = psycopg.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=SUPERUSER,
                password=DB_PASSWORD,
                connect_timeout=5,
                application_name="db-migrate",
            )
            conn.autocommit = True
            return conn
        except psycopg.OperationalError as exc:
            last = exc
            if attempt % 10 == 0:
                LOG.info("waiting for %s/%s (attempt %d)", host, dbname, attempt + 1)
            time.sleep(2)
    raise RuntimeError(f"cannot connect to {host}/{dbname}: {last}")


def ensure_migration_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration (
          version text PRIMARY KEY,
          checksum char(64) NOT NULL,
          applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def applied_versions(conn: psycopg.Connection) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT version, checksum FROM schema_migration")
        return {row[0]: row[1] for row in cur.fetchall()}


def split_statements(body: str) -> list[str]:
    """SQL を文単位へ分割する。

    Citus は DDL を Worker へ伝播する際に文を prepare するため、複数文をまとめて
    送ると "cannot insert multiple commands into a prepared statement" になる。
    そのため 1 文ずつ実行する。文字列リテラル・ドル引用符・コメントを考慮する。
    """
    statements: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(body)
    in_single = in_double = False
    dollar_tag: str | None = None

    while index < length:
        char = body[index]
        pair = body[index : index + 2]

        if dollar_tag:
            buffer.append(char)
            if body.startswith(dollar_tag, index):
                buffer.append(body[index + 1 : index + len(dollar_tag)])
                index += len(dollar_tag)
                dollar_tag = None
                continue
            index += 1
            continue
        if in_single:
            buffer.append(char)
            if char == "'":
                in_single = False
            index += 1
            continue
        if in_double:
            buffer.append(char)
            if char == '"':
                in_double = False
            index += 1
            continue
        if pair == "--":
            end = body.find("\n", index)
            index = length if end < 0 else end
            continue
        if pair == "/*":
            end = body.find("*/", index)
            index = length if end < 0 else end + 2
            continue
        if char == "'":
            in_single = True
            buffer.append(char)
            index += 1
            continue
        if char == '"':
            in_double = True
            buffer.append(char)
            index += 1
            continue
        if char == "$":
            closing = body.find("$", index + 1)
            if closing > 0 and body[index + 1 : closing].replace("_", "").isalnum() or (
                closing == index + 1
            ):
                dollar_tag = body[index : closing + 1]
                buffer.append(dollar_tag)
                index = closing + 1
                continue
        if char == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            index += 1
            continue
        buffer.append(char)
        index += 1

    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def apply_migration(
    conn: psycopg.Connection, version: str, body: str, identity: str | None = None
) -> None:
    """migration を 1 つ適用する。

    checksum は **置換前のファイル本文**（`identity`）で取る。置換後の本文で取ると、
    `${EMBEDDING_FINE_DIMENSION}` のような環境変数の値が migration の同一性に
    混ざってしまう。設定を変えただけで「適用済み migration が編集された」と誤検出し、
    既に適用済みの全環境で起動が止まる。

    修復の手段を 1 つだけ用意する。`MIGRATION_REPAIR_CHECKSUMS` に版名を
    並べると、その版に限り**再実行せずに記録上の checksum だけ**を更新する。
    置換値の変更で記録上の checksum がずれた環境を 1 回だけ通すための措置であり、
    SQL の中身を変えた場合に使ってはならない（その場合は新しい V00x を追加する）。
    """
    identity = body if identity is None else identity
    checksum = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    applied = applied_versions(conn)
    if version in applied:
        if applied[version] != checksum:
            if _repair_allowed(version):
                LOG.warning(
                    "repairing recorded checksum for %s (%s -> %s); "
                    "the migration is NOT re-run",
                    version,
                    applied[version][:12],
                    checksum[:12],
                )
                conn.execute(
                    "UPDATE schema_migration SET checksum = %s WHERE version = %s",
                    (checksum, version),
                )
                return
            raise RuntimeError(
                f"migration {version} was already applied with a different checksum; "
                "create a new migration instead of editing the applied one. "
                "置換値（EMBEDDING_FINE_DIMENSION 等）を変えただけで SQL は同じ場合は、"
                f"MIGRATION_REPAIR_CHECKSUMS='{version}' を指定して 1 回だけ実行する"
            )
        LOG.info("skip %s (already applied)", version)
        return

    statements = split_statements(body)
    LOG.info("applying %s (%d statements)", version, len(statements))
    for statement in statements:
        try:
            conn.execute(statement)
        except psycopg.Error as exc:
            raise RuntimeError(
                f"migration {version} failed on statement: {statement.splitlines()[0][:120]} — {exc}"
            ) from exc
    conn.execute(
        "INSERT INTO schema_migration (version, checksum) VALUES (%s, %s)", (version, checksum)
    )


def ensure_role(conn: psycopg.Connection, role: str, password: str) -> None:
    """ロールが無ければ作成する。

    Citus に登録済みの Worker では DDL が拒否される（"operation is not allowed on
    this node"）ため、存在確認を必ず先に行い、必要なときだけ実行する。
    Coordinator で作成したロールは Citus が Worker へ伝播する。
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
        if cur.fetchone() is not None:
            return
    conn.execute(
        sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {}").format(
            sql.Identifier(role), sql.Literal(password)
        )
    )


def ensure_extension(conn: psycopg.Connection, name: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (name,))
        if cur.fetchone() is not None:
            return
    conn.execute(sql.SQL("CREATE EXTENSION {}").format(sql.Identifier(name)))


def is_registered_worker(conn: psycopg.Connection) -> bool:
    """Citus クラスタへ Worker として登録済みかを判定する。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_dist_node WHERE groupid <> 0 AND noderole = 'primary'"
        )
        workers = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM pg_dist_node WHERE groupid = 0")
        coordinators = cur.fetchone()[0]
    return workers > 0 and coordinators > 0


def grant_schema(conn: psycopg.Connection, role: str, *, write: bool) -> None:
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
    if write:
        conn.execute(
            sql.SQL(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}"
            ).format(sql.Identifier(role))
        )
        conn.execute(
            sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(
                sql.Identifier(role)
            )
        )
        conn.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {}"
            ).format(sql.Identifier(role))
        )
    else:
        conn.execute(
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(
                sql.Identifier(role)
            )
        )
        conn.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}"
            ).format(sql.Identifier(role))
        )


# --- Metadata DB -------------------------------------------------------------

#: metadata の初期スキーマ。統合前の V001〜V012 を当てた結果と同じスキーマになる。
INITIAL_METADATA_VERSION = "V001__initial_schema.sql"

#: 統合前の版。これらがすべて記録済みの DB は、初期スキーマを適用済みとして引き継ぐ。
LEGACY_METADATA_VERSIONS = (
    "V001__metadata_schema.sql",
    "V002__seed_findings.sql",
    "V003__app_user.sql",
    "V004__search_query_region.sql",
    "V005__search_query_region_mask.sql",
    "V006__drop_query_region_mask.sql",
    "V007__cancel_ingest_item.sql",
    "V008__embedding_fine_model_type.sql",
    "V009__region_mil_model_type.sql",
    "V010__search_query_outline.sql",
    "V011__evaluation_run.sql",
    "V012__chest_unit.sql",
)


def adopt_legacy_metadata(conn: psycopg.Connection, body: str) -> None:
    """統合前の版で作った DB を、初期スキーマ適用済みとして記録し直す。

    統合前の V001〜V012 がすべて記録済みなら、スキーマは初期スキーマと同じなので
    SQL は実行せず、記録だけを初期スキーマ 1 行に置き換える。一部だけが記録済みの
    DB は初期スキーマと一致しないため、止める。
    """
    applied = applied_versions(conn)
    legacy = [v for v in LEGACY_METADATA_VERSIONS if v in applied]
    if not legacy or INITIAL_METADATA_VERSION in applied:
        return
    if len(legacy) != len(LEGACY_METADATA_VERSIONS):
        missing = [v for v in LEGACY_METADATA_VERSIONS if v not in applied]
        raise RuntimeError(
            "統合前の metadata migration が一部しか適用されていない（未適用: "
            + ", ".join(missing)
            + "）。統合前の版で V012 まで適用してから更新する"
        )
    checksum = hashlib.sha256(body.encode("utf-8")).hexdigest()
    # 削除と登録を 1 文で行い、途中で止まっても記録が欠けないようにする
    conn.execute(
        "WITH legacy AS (DELETE FROM schema_migration WHERE version = ANY(%s)) "
        "INSERT INTO schema_migration (version, checksum) VALUES (%s, %s)",
        (list(LEGACY_METADATA_VERSIONS), INITIAL_METADATA_VERSION, checksum),
    )
    LOG.warning(
        "adopted %d legacy metadata migrations as %s; the schema is NOT re-run",
        len(legacy),
        INITIAL_METADATA_VERSION,
    )


def migrate_metadata() -> None:
    host = os.environ.get("METADATA_DB_HOST", "metadata-db")
    dbname = os.environ.get("METADATA_DB_NAME", "metadata")
    LOG.info("=== metadata database: %s/%s ===", host, dbname)
    with connect(host, dbname) as conn:
        ensure_migration_table(conn)
        ensure_role(conn, "app_meta", APP_PASSWORD)
        initial = METADATA_MIGRATIONS / INITIAL_METADATA_VERSION
        adopt_legacy_metadata(conn, initial.read_text(encoding="utf-8"))
        for path in sorted(METADATA_MIGRATIONS.glob("V*.sql")):
            apply_migration(conn, path.name, path.read_text(encoding="utf-8"))
        grant_schema(conn, "app_meta", write=True)
        # アプリケーションは superuser を使わない（設計書 8.5.2）
        conn.execute("ALTER ROLE app_meta SET statement_timeout = '30s'")
        conn.execute("ALTER ROLE app_meta SET idle_in_transaction_session_timeout = '60s'")
    LOG.info("metadata database is ready")


# --- ベクトル用 Citus クラスタ --------------------------------------------------


def cluster_nodes() -> tuple[str, list[str]]:
    """(Coordinator, Worker の一覧)。ベクトル用クラスタは 1 つ。"""
    coordinator = os.environ.get("VECTOR_HOST", "vector-coordinator")
    workers = [f"vector-worker-{index}" for index in range(1, WORKERS_PER_CLUSTER + 1)]
    return coordinator, workers


def migrate_cluster() -> None:
    coordinator, workers = cluster_nodes()
    dbname = os.environ.get("VECTOR_DB", "vector")
    LOG.info("=== vector cluster: %s + %s ===", coordinator, ",".join(workers))

    # 1. 全ノードで extension とロールを用意する（設計書 11.4.2）
    #    Worker は登録前のみ DDL 可能なため、登録済みノードでは何もしない。
    for host in [coordinator, *workers]:
        with connect(host, dbname) as conn:
            ensure_extension(conn, "citus")
            ensure_extension(conn, "vector")
            if host == coordinator or not is_registered_worker(conn):
                ensure_role(conn, "app_ingest", APP_PASSWORD)
                ensure_role(conn, "app_search", APP_PASSWORD)
                conn.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO app_ingest, app_search").format(
                        sql.Identifier(dbname)
                    )
                )

    # 2. Coordinator へ Worker を登録する
    with connect(coordinator, dbname) as conn:
        # 既に登録済みの Coordinator へ再度 citus_set_coordinator_host を実行すると
        # "operation is not allowed on this node" になるため、未登録時のみ実行する。
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_dist_node WHERE groupid = 0")
            if cur.fetchone() is None:
                LOG.info("registering coordinator %s", coordinator)
                cur.execute("SELECT citus_set_coordinator_host(%s, 5432)", (coordinator,))
        for worker in workers:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_dist_node WHERE nodename = %s AND noderole = 'primary'",
                    (worker,),
                )
                if cur.fetchone() is None:
                    LOG.info("registering worker %s", worker)
                    cur.execute("SELECT citus_add_node(%s, 5432)", (worker,))
        with conn.cursor() as cur:
            # citus_get_active_worker_nodes() の列名は Citus のバージョンで異なるため
            # （13 系: nodename / 14 系: node_name）、pg_dist_node を直接参照する。
            cur.execute(
                "SELECT nodename FROM pg_dist_node "
                "WHERE noderole = 'primary' AND isactive AND groupid <> 0 ORDER BY nodename"
            )
            active = [row[0] for row in cur.fetchall()]
        if sorted(active) != sorted(workers):
            raise RuntimeError(
                f"vector cluster: active workers {active} do not match expected {workers}"
            )

        # 3. スキーマ適用
        ensure_migration_table(conn)
        conn.execute(f"SET citus.shard_count = {SHARD_COUNT}")
        for path in sorted(VECTOR_MIGRATIONS.glob("V*.sql")):
            raw = path.read_text(encoding="utf-8")
            body = (
                raw.replace("${SHARD_COUNT}", str(SHARD_COUNT))
                .replace("${EMBEDDING_DIMENSION}", str(EMBEDDING_DIMENSION))
                .replace("${EMBEDDING_FINE_DIMENSION}", str(EMBEDDING_FINE_DIMENSION))
                .replace("${EMBEDDING_LUNG_DIMENSION}", str(EMBEDDING_LUNG_DIMENSION))
            )
            # checksum は置換前の raw で取る。環境変数の値を同一性に含めないため。
            apply_migration(conn, path.name, body, identity=raw)

        # 4. 権限。app_ingest は SELECT/INSERT/UPDATE、app_search は SELECT のみ
        conn.execute("GRANT USAGE ON SCHEMA public TO app_ingest, app_search")
        conn.execute("GRANT SELECT, INSERT, UPDATE ON image_embedding TO app_ingest")
        conn.execute("GRANT SELECT ON image_embedding TO app_search")
        conn.execute("GRANT SELECT ON schema_migration TO app_search, app_ingest")
        conn.execute("ALTER ROLE app_search SET statement_timeout = '5s'")
        conn.execute("ALTER ROLE app_ingest SET statement_timeout = '30s'")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT nodename, count(*) FROM citus_shards "
                "WHERE table_name = 'image_embedding'::regclass GROUP BY nodename ORDER BY nodename"
            )
            placement = {row[0]: row[1] for row in cur.fetchall()}
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            vector_version = cur.fetchone()[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'citus'")
            citus_version = cur.fetchone()[0]
    LOG.info(
        "vector cluster ready: shards=%s placement=%s citus=%s pgvector=%s "
        "coarse_dim=%s fine_dim=%s",
        sum(placement.values()), placement, citus_version, vector_version,
        EMBEDDING_DIMENSION, EMBEDDING_FINE_DIMENSION,
    )


def main() -> int:
    started = time.perf_counter()
    try:
        migrate_metadata()
        migrate_cluster()
    except Exception as exc:  # noqa: BLE001 - ジョブとして失敗を明示する
        LOG.error("migration failed: %s", exc)
        return 1
    LOG.info("all migrations completed in %.1fs", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
