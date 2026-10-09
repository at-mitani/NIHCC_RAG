""".env の作り直しで値を取りこぼさないことを固定する。

`.env` は取り込み済みデータとの整合を握っている。PREPROCESS_VERSION や
EMBEDDING_MODEL_REVISION がずれると、検索の 1 段目が
`WHERE embedding_model_version = ...` で 0 件になり、**エラーを出さずに
「検索結果なし」になる**。気づきにくい壊れ方なので、引き継ぎを明示的に守る。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sync_env", REPO / "scripts" / "sync_env.py")
sync_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_env)

EXAMPLE = """# 先頭コメント
A=1
# B の説明。この行が消えてはいけない
B=default-b
C=default-c
EMBEDDING_FINE_BACKEND=none
MIL_ARTIFACT_DIR=./infra/mil
"""


def render(current: dict[str, str], reset: tuple[str, ...] = ()):
    return sync_env.render(EXAMPLE, current, reset)


# --- 引き継ぎ ----------------------------------------------------------------


def test_existing_values_win_over_example_defaults() -> None:
    body, report = render({"B": "keep-me"})
    assert "B=keep-me" in body
    assert report["carried"] == ["B"]


def test_new_keys_take_example_defaults() -> None:
    """pull でキーが増えたとき、書き足し忘れが起きないこと。"""
    body, report = render({"A": "1"})
    assert "C=default-c" in body
    assert "C" in report["added"]


def test_comments_are_preserved() -> None:
    """コメントには運用上の制約が書いてある。値だけの寄せ集めにしない。"""
    body, _ = render({"B": "x"})
    assert "# B の説明。この行が消えてはいけない" in body
    assert "# 先頭コメント" in body


def test_keys_absent_from_example_are_kept() -> None:
    """COMPOSE_FILE のような環境固有の設定を落とさないこと。

    落とすと GPU 構成の compose が読まれず、CPU で動き続ける。
    """
    body, report = render({"COMPOSE_FILE": "a.yml:b.yml"})
    assert "COMPOSE_FILE=a.yml:b.yml" in body
    assert report["dropped"] == ["COMPOSE_FILE"]


# --- reset -------------------------------------------------------------------


def test_reset_restores_example_default() -> None:
    """評価作業で入れた値を既定へ戻せること（今回の 2 段目の設定）。"""
    body, report = render(
        {"EMBEDDING_FINE_BACKEND": "rad-dino", "B": "keep"}, reset=("EMBEDDING_FINE",)
    )
    assert "EMBEDDING_FINE_BACKEND=none" in body
    assert "B=keep" in body
    assert report["defaulted"] == ["EMBEDDING_FINE_BACKEND"]


def test_reset_matches_by_prefix() -> None:
    body, _ = render({"MIL_ARTIFACT_DIR": "./tmp/mil"}, reset=("MIL_",))
    assert "MIL_ARTIFACT_DIR=./infra/mil" in body


def test_reset_does_not_touch_other_keys() -> None:
    body, _ = render({"B": "keep", "EMBEDDING_FINE_BACKEND": "x"}, reset=("EMBEDDING_FINE",))
    assert "B=keep" in body


# --- 解析 --------------------------------------------------------------------


def test_parse_takes_the_first_assignment() -> None:
    """compose と同じく先勝ちであること。

    追記で重複したとき、compose が読む値とこのスクリプトが読む値が食い違うと、
    引き継いだつもりの値が実際には使われていない状態になる。
    """
    assert sync_env.parse_env("K=first\nK=second\n")["K"] == "first"


def test_parse_ignores_comments_and_blanks() -> None:
    parsed = sync_env.parse_env("# K=commented\n\n  \nK=real\n")
    assert parsed == {"K": "real"}


def test_values_with_equals_and_colons_survive() -> None:
    """LOCAL_USERS や COMPOSE_FILE は ':' を含み、base64 風の値は '=' を含む。"""
    raw = "LOCAL_USERS=admin:pw==:admin,viewer:p2:viewer"
    assert sync_env.parse_env(raw)["LOCAL_USERS"] == "admin:pw==:admin,viewer:p2:viewer"
    body, _ = render(sync_env.parse_env(raw))
    assert raw in body


# --- 実ファイルでの往復 ------------------------------------------------------


def test_roundtrip_on_the_real_example_keeps_every_key() -> None:
    """実際の .env.example で、キーが 1 つも欠けないこと。"""
    example = (REPO / ".env.example").read_text(encoding="utf-8")
    current = dict.fromkeys(sync_env.parse_env(example), "carried-value")
    current["COMPOSE_FILE"] = "docker-compose.yml:infra/compose/docker-compose.gpu.yml"
    body, _ = sync_env.render(example, current, ())

    produced = sync_env.parse_env(body)
    assert set(produced) == set(current)
    assert produced["COMPOSE_FILE"] == current["COMPOSE_FILE"]


def test_roundtrip_resets_only_the_fine_family() -> None:
    example = (REPO / ".env.example").read_text(encoding="utf-8")
    defaults = sync_env.parse_env(example)
    current = {k: "stale" for k in defaults}
    body, _ = sync_env.render(example, current, ("EMBEDDING_FINE", "MIL_"))
    produced = sync_env.parse_env(body)

    for key, value in produced.items():
        if key.startswith(("EMBEDDING_FINE", "MIL_")):
            assert value == defaults[key], key
        else:
            assert value == "stale", key


@pytest.mark.parametrize("key", ["PREPROCESS_VERSION", "EMBEDDING_MODEL_REVISION", "LOCAL_USERS"])
def test_data_bound_keys_are_never_reset_by_the_fine_reset(key: str) -> None:
    """取り込み済みデータと結びつく値が、2 段目の reset で巻き込まれないこと。"""
    example = (REPO / ".env.example").read_text(encoding="utf-8")
    body, _ = sync_env.render(example, {key: "must-survive"}, ("EMBEDDING_FINE", "MIL_"))
    assert sync_env.parse_env(body)[key] == "must-survive"


# --- compose との突き合わせ --------------------------------------------------


def test_compose_variables_are_all_documented() -> None:
    """**compose が参照する変数は必ず `.env.example` にあること。**

    無いと、compose の既定値で動いてしまうため気づかない。運用者はその設定を
    調整できるとすら知らない状態になる（EMBEDDING_SHAPE_DIMENSION /
    CITUS_SHARD_COUNT / TZ などは漏れやすい）。
    """
    example_keys = set(sync_env.parse_env((REPO / ".env.example").read_text(encoding="utf-8")))
    gaps = sync_env.undocumented(REPO, example_keys)
    assert gaps == [], f".env.example に無い compose 変数: {gaps}"


def test_escaped_compose_variables_are_ignored() -> None:
    """`$${NAME}` はコンテナ内のシェル変数なので拾わないこと。

    拾うと、compose の変数でないものを .env.example へ足させてしまう。
    """
    assert sync_env._COMPOSE_VAR.findall("${REAL} and $${SHELL_ONLY}") == ["REAL"]


def test_compose_variables_are_found_at_all() -> None:
    """監査の対象が 0 件なら、この検査は何も守っていない。"""
    assert len(sync_env.compose_variables(REPO)) >= 30


# --- 新規キーで挙動を変えないこと --------------------------------------------


def test_new_keys_take_the_compose_default_not_the_example() -> None:
    """**`.env` に無いキーの実効値は compose の既定である。**

    example の値を入れると、書き足しただけのつもりで挙動が変わる。実測では
    TORCH_INDEX_URL が cpu -> cu121 になり、CUDA 版 PyTorch を入れる設定になった
    （この PC に NVIDIA GPU は無い）。
    """
    example = "A=example-value\n"
    body, report = sync_env.render(example, {}, (), {"A": "compose-default"})
    assert "A=compose-default" in body
    assert report["kept_effective"] == ["A"]


def test_new_keys_use_the_example_when_compose_agrees() -> None:
    """compose の既定と example が同じなら、例のまま（＝コメントも残る）。"""
    example = "# 説明\nA=same\n"
    body, report = sync_env.render(example, {}, (), {"A": "same"})
    assert "A=same" in body and "# 説明" in body
    assert report["kept_effective"] == []


def test_new_keys_without_a_compose_default_use_the_example() -> None:
    """compose が既定を持たないキーは example の値を使うしかない。"""
    body, report = sync_env.render("A=only-example\n", {}, (), {})
    assert "A=only-example" in body
    assert report["kept_effective"] == []


def test_existing_values_still_win_over_the_compose_default() -> None:
    """既に `.env` にある値は compose の既定より優先すること。

    ここが崩れると、利用者が設定した値が既定へ戻される。
    """
    body, _ = sync_env.render("A=example\n", {"A": "mine"}, (), {"A": "compose"})
    assert "A=mine" in body


def test_compose_defaults_are_actually_found() -> None:
    """実ファイルから既定値を拾えていること。0 件なら仕組みが効いていない。"""
    defaults = sync_env.compose_defaults(REPO)
    assert len(defaults) >= 30
    assert defaults.get("EMBEDDING_LUNG_DIMENSION") == "16"


def test_escaped_defaults_are_ignored() -> None:
    assert sync_env._COMPOSE_DEFAULT.findall("${A:-1} $${B:-2}") == [("A", "1")]
