"""migration の同一性（checksum）の回帰テスト。

**checksum を置換後の本文で取ってはならない。** ベクトル用クラスタの migration は
`${EMBEDDING_FINE_DIMENSION}` などを実行時に埋めるテンプレートで、置換後の本文で
checksum を取ると *設定値* が migration の同一性に混ざる。`.env` を変えただけで
（たとえば EMBEDDING_FINE_DIMENSION を変えただけで）「適用済み migration が
編集された」と誤検出し、**既に適用済みの全環境で起動が止まる**。

壊れ方が起動時の全面停止なので、ここを固定しておく価値が大きい。
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MIGRATE_PY = REPO / "db" / "migrate" / "run_migrations.py"


def _load_module():
    """run_migrations を読み込む。

    db/migrate/ はパッケージではないのでパス指定で読む。psycopg は dev 環境に
    入っていないため、モジュールが実際に使う分だけスタブを置く（型注釈は
    `from __future__ import annotations` により評価されない）。
    """
    # psycopg が入っている環境では実物を使う。スタブを sys.modules へ入れると、
    # 同じプロセスで動く他のテスト（psycopg.rows を使う search-aggregator など）が
    # 実物を掴めなくなる。
    try:
        import psycopg  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("psycopg")
        stub.Error = type("Error", (Exception,), {})
        stub.Connection = object
        stub.sql = types.ModuleType("psycopg.sql")
        sys.modules["psycopg"] = stub
        sys.modules["psycopg.sql"] = stub.sql
    # モジュール読み込み時に secret を要求するため、ダミーを入れておく。
    # 接続は一切行わないので値は使われない。
    os.environ.setdefault("POSTGRES_PASSWORD", "test-only")
    spec = importlib.util.spec_from_file_location("run_migrations", MIGRATE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rm = _load_module()


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.last = (sql, params)

    def fetchall(self):
        return self._rows


class FakeConn:
    """schema_migration だけを持つ最小の接続。実行された SQL を記録する。"""

    def __init__(self, applied: dict[str, str] | None = None):
        self.applied = dict(applied or {})
        self.executed: list[tuple[str, tuple | None]] = []

    def cursor(self):
        return FakeCursor(list(self.applied.items()))

    def execute(self, sql, params=None):
        self.executed.append((sql, params))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- 同一性は置換前の本文で決まる --------------------------------------------


def test_checksum_ignores_substituted_values() -> None:
    """置換値が変わっても checksum が変わらないこと。

    これが崩れると、.env の変更だけで既存環境の db-migrate が止まる。
    """
    raw = "CREATE TABLE t (v vector(${EMBEDDING_FINE_DIMENSION}));"
    conn_a = FakeConn()
    rm.apply_migration(conn_a, "X/V001.sql", raw.replace("${EMBEDDING_FINE_DIMENSION}", "1024"), identity=raw)
    conn_b = FakeConn()
    rm.apply_migration(conn_b, "X/V001.sql", raw.replace("${EMBEDDING_FINE_DIMENSION}", "15"), identity=raw)

    recorded = [
        params[1]
        for conn in (conn_a, conn_b)
        for sql_text, params in conn.executed
        if "INSERT INTO schema_migration" in sql_text
    ]
    assert len(recorded) == 2
    assert recorded[0] == recorded[1] == sha(raw)


def test_substituted_body_is_what_actually_runs() -> None:
    """記録は置換前でも、実行されるのは置換後の SQL であること。"""
    raw = "CREATE TABLE t (v vector(${EMBEDDING_FINE_DIMENSION}));"
    body = raw.replace("${EMBEDDING_FINE_DIMENSION}", "15")
    conn = FakeConn()
    rm.apply_migration(conn, "X/V001.sql", body, identity=raw)
    assert any("vector(15)" in s for s, _ in conn.executed)


def test_identity_defaults_to_body() -> None:
    """置換の無い migration（metadata 側）は本文そのもので checksum を取ること。"""
    body = "ALTER TABLE t ADD COLUMN c int;"
    conn = FakeConn()
    rm.apply_migration(conn, "V009.sql", body)
    recorded = [params for sql_text, params in conn.executed if "INSERT INTO schema_migration" in sql_text]
    assert recorded[0][1] == sha(body)


# --- 適用済みの扱い ----------------------------------------------------------


def test_matching_checksum_is_skipped() -> None:
    body = "SELECT 1;"
    conn = FakeConn({"V001.sql": sha(body)})
    rm.apply_migration(conn, "V001.sql", body)
    assert conn.executed == []


def test_mismatch_raises_without_repair(monkeypatch) -> None:
    """既定では checksum のずれを黙って通さないこと。"""
    monkeypatch.setattr(rm, "REPAIR_CHECKSUMS", set())
    conn = FakeConn({"V001.sql": sha("old")})
    with pytest.raises(RuntimeError, match="different checksum"):
        rm.apply_migration(conn, "V001.sql", "new")
    assert conn.executed == []


# --- 修復の逃げ道 ------------------------------------------------------------


def test_repair_updates_record_without_rerunning(monkeypatch) -> None:
    """修復は記録だけを更新し、SQL を再実行しないこと。

    再実行すると V003 のような破壊的な migration がデータを消す。
    """
    monkeypatch.setattr(rm, "REPAIR_CHECKSUMS", {"V002.sql"})
    conn = FakeConn({"V002.sql": sha("old")})
    rm.apply_migration(conn, "V002.sql", "DROP TABLE t;", identity="raw")

    assert len(conn.executed) == 1
    sql_text, params = conn.executed[0]
    assert "UPDATE schema_migration SET checksum" in sql_text
    assert params == (sha("raw"), "V002.sql")
    assert not any("DROP TABLE" in s for s, _ in conn.executed)


def test_repair_matches_by_filename(monkeypatch) -> None:
    """ファイル名だけの指定でも、接頭辞付きの版名に効くこと。

    記録の版名が "<接頭辞>/V001__x.sql" の形でも修復できるようにしておく。
    """
    monkeypatch.setattr(rm, "REPAIR_CHECKSUMS", {"V001__image_embedding.sql"})
    for version in ("V001__image_embedding.sql", "old/V001__image_embedding.sql"):
        conn = FakeConn({version: sha("old")})
        rm.apply_migration(conn, version, "SELECT 1;", identity="raw")
        assert "UPDATE schema_migration SET checksum" in conn.executed[0][0], version


def test_repair_does_not_leak_to_other_migrations(monkeypatch) -> None:
    """指定していない版は修復されないこと。"""
    monkeypatch.setattr(rm, "REPAIR_CHECKSUMS", {"V001__image_embedding.sql"})
    conn = FakeConn({"V002__image_embedding_more.sql": sha("old")})
    with pytest.raises(RuntimeError, match="different checksum"):
        rm.apply_migration(conn, "V002__image_embedding_more.sql", "x", identity="raw")


# --- テンプレートの実体 ------------------------------------------------------


def test_vector_migrations_are_templates() -> None:
    """ベクトルクラスタの migration が実際に置換を含むこと。

    守りたいのは**次元を直書きしないこと**である。直書きすると .env の設定と
    schema がずれ、検索時に `different vector dimensions` で落ちる。
    """
    files = sorted((REPO / "db" / "vector-migrations").glob("V00*.sql"))
    assert files, "vector-migrations が見つからない"
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "vector(" in text:
            assert "${" in text, f"{path.name} が次元を直書きしている"
    joined = "".join(f.read_text(encoding="utf-8") for f in files)
    assert "${EMBEDDING_FINE_DIMENSION}" in joined
    assert "${SHARD_COUNT}" in joined


def test_one_cluster_with_five_workers() -> None:
    """ベクトル用クラスタは 1 Coordinator + 5 Worker。"""
    coordinator, workers = rm.cluster_nodes()
    assert coordinator == "vector-coordinator"
    assert workers == [f"vector-worker-{i}" for i in range(1, 6)]
    assert rm.SHARD_COUNT % len(workers) == 0, "shard が Worker に均等に載らない"


# --- 統合前の版の引き継ぎ ----------------------------------------------------


def test_adopt_replaces_legacy_records_without_running_sql() -> None:
    """統合前の版がすべて記録済みなら、記録だけを初期スキーマ 1 行に置き換えること。"""
    conn = FakeConn({v: sha(v) for v in rm.LEGACY_METADATA_VERSIONS})
    rm.adopt_legacy_metadata(conn, "CREATE TABLE t ();")

    assert len(conn.executed) == 1
    sql_text, params = conn.executed[0]
    assert "DELETE FROM schema_migration" in sql_text
    assert "INSERT INTO schema_migration" in sql_text
    assert params == (
        list(rm.LEGACY_METADATA_VERSIONS),
        rm.INITIAL_METADATA_VERSION,
        sha("CREATE TABLE t ();"),
    )
    assert not any("CREATE TABLE" in s for s, _ in conn.executed)


def test_adopt_does_nothing_on_a_new_database() -> None:
    """記録が無い DB では何もしない（初期スキーマは通常どおり適用される）。"""
    conn = FakeConn()
    rm.adopt_legacy_metadata(conn, "x")
    assert conn.executed == []


def test_adopt_does_nothing_once_adopted() -> None:
    """初期スキーマが記録済みなら何もしない（再実行しても安全）。"""
    conn = FakeConn({rm.INITIAL_METADATA_VERSION: sha("x")})
    rm.adopt_legacy_metadata(conn, "x")
    assert conn.executed == []


def test_adopt_refuses_a_partial_legacy_chain() -> None:
    """統合前の版が一部だけ記録された DB は、初期スキーマと一致しないので止めること。"""
    conn = FakeConn({v: sha(v) for v in rm.LEGACY_METADATA_VERSIONS[:5]})
    with pytest.raises(RuntimeError, match="一部しか適用されていない"):
        rm.adopt_legacy_metadata(conn, "x")
    assert conn.executed == []
