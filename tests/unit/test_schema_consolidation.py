"""スキーマの置き場を固定する。

DDL の正本は migration だけである。metadata は初期スキーマ
`db/metadata-migrations/V001__initial_schema.sql` から始まり、変更は新しい V00x を
足して行う。ベクトル用クラスタは `db/vector-migrations/`。

統合前の版（V001__metadata_schema.sql〜V012__chest_unit.sql）が記録された既存の DB は、
db-migrate が初期スキーマ適用済みとして引き継ぐ（`run_migrations.adopt_legacy_metadata`）。
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
METADATA_MIGRATIONS = REPO / "db" / "metadata-migrations"
VECTOR_MIGRATIONS = REPO / "db" / "vector-migrations"
MIGRATE_PY = REPO / "db" / "migrate" / "run_migrations.py"


def _load_run_migrations():
    try:
        import psycopg  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("psycopg")
        stub.Error = type("Error", (Exception,), {})
        stub.Connection = object
        stub.sql = types.ModuleType("psycopg.sql")
        sys.modules["psycopg"] = stub
        sys.modules["psycopg.sql"] = stub.sql
    os.environ.setdefault("POSTGRES_PASSWORD", "test-only")
    spec = importlib.util.spec_from_file_location("run_migrations_layout", MIGRATE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rm = _load_run_migrations()


def chain_text(directory: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(directory.glob("V0*.sql"))
    )


def created_tables(sql: str) -> set[str]:
    return {
        name.lower()
        for name in re.findall(r"CREATE TABLE (?:IF NOT EXISTS )?(?:public\.)?(\w+)", sql)
    }


def test_metadata_starts_from_the_initial_schema() -> None:
    """metadata の最初の版が初期スキーマであること。"""
    files = sorted(path.name for path in METADATA_MIGRATIONS.glob("V*.sql"))
    assert files, "metadata-migrations が空"
    assert files[0] == rm.INITIAL_METADATA_VERSION


def test_legacy_versions_are_not_in_the_directory() -> None:
    """統合前の版をファイルとして戻さないこと。

    戻すと、初期スキーマと同じテーブルを作り直そうとして新規構築が失敗する。
    """
    present = {path.name for path in METADATA_MIGRATIONS.glob("*.sql")}
    assert not present & set(rm.LEGACY_METADATA_VERSIONS)


def test_initial_schema_seeds_the_finding_vocabulary() -> None:
    """初期スキーマが付随ラベルの語彙（14 所見 + No Finding）を入れること。"""
    text = (METADATA_MIGRATIONS / rm.INITIAL_METADATA_VERSION).read_text(encoding="utf-8")
    assert "INSERT INTO finding" in text
    assert len(re.findall(r"^\s+\('[A-Za-z_ ]+',\s+'", text, re.M)) == 15


def test_migration_directories_hold_only_versioned_files() -> None:
    """migration 置き場には V00x__ の版だけを置くこと。"""
    for directory in (METADATA_MIGRATIONS, VECTOR_MIGRATIONS):
        for path in directory.glob("*.sql"):
            assert re.match(r"V\d{3}__", path.name), (
                f"migration 置き場に V00x 以外のファイルがある: {path.name}"
            )


@pytest.mark.parametrize(
    "placeholder",
    ["${EMBEDDING_DIMENSION}", "${EMBEDDING_FINE_DIMENSION}", "${EMBEDDING_LUNG_DIMENSION}"],
)
def test_vector_schema_keeps_the_placeholders(placeholder: str) -> None:
    """**次元を直書きしない。**

    直書きすると `.env` と schema がずれ、取り込みが全件失敗する。
    """
    assert placeholder in chain_text(VECTOR_MIGRATIONS)


def test_vector_schema_is_one_row_per_image() -> None:
    """ベクトルは 1 画像 1 行（image_embedding）。部位ごとの表は作らない。"""
    chain = chain_text(VECTOR_MIGRATIONS)
    assert "image_embedding" in created_tables(chain)
    assert "region_embedding" not in created_tables(chain)
    assert "${REGION_CODE}" not in chain
