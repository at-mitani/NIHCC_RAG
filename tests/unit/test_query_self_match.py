"""検索元そのもの（同じファイル名・類似度 1.0）を結果から外すことを固定する。

取り込み済みの画像で検索すると、先頭に必ず自分自身が類似度 1.0 で並ぶ。
比べる相手として意味が無いので外す。ファイル名が同じでも中身が違えば
（類似度が 1.0 でなければ）残す。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _services import REPO, load  # noqa: E402

# search.py は `from . import chest` を使うので、親パッケージを用意してから読む
_PACKAGE = "nihcc_backend_services"
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(REPO / "apps/backend-api/app/services")]
    sys.modules[_PACKAGE] = package
load(f"{_PACKAGE}.chest", "apps/backend-api/app/services/chest.py")
search = load(f"{_PACKAGE}.search", "apps/backend-api/app/services/search.py")


def test_same_name_and_similarity_one_is_excluded() -> None:
    assert search.is_query_itself("00000001_000.png", "00000001_000.png", 1.0)


def test_similarity_rounding_matches_display() -> None:
    """表示（小数 4 桁）で 1.0 になるものは同じ画像として扱う。"""
    assert search.is_query_itself("00000001_000.png", "00000001_000.png", 0.99999)
    assert not search.is_query_itself("00000001_000.png", "00000001_000.png", 0.9998)


def test_different_name_is_kept() -> None:
    assert not search.is_query_itself("00000001_000.png", "00000001_001.png", 1.0)


def test_missing_query_name_is_kept() -> None:
    """テキスト検索などファイル名が無い場合は何も外さない。"""
    assert not search.is_query_itself(None, "00000001_000.png", 1.0)
    assert not search.is_query_itself("", "", 1.0)


def test_path_and_case_are_ignored() -> None:
    assert search.is_query_itself(r"C:\fakepath\00000001_000.PNG", "00000001_000.png", 1.0)
