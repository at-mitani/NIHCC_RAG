"""サービス実装をテストから読み込むための共有ヘルパー。

各サービスはどれも `app` パッケージなので、sys.path へ足して `import app.main` と
すると、先に読まれた別サービスの `app` を掴む。一意な名前でファイルから直接読む。

**同じモジュールを 2 回読み込まないこと。** Prometheus の指標はモジュール読み込み時に
既定レジストリへ登録されるため、2 度読むと "Duplicated timeseries" で collection が
止まる。ここで 1 つだけ保持して共有する。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_COMMON = str(REPO / "packages" / "common" / "src")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

_loaded: dict[str, object] = {}


def load(name: str, relative: str):
    """サービスのモジュールを一意な名前で読み込む（2 回目以降は同じものを返す）。"""
    if name not in _loaded:
        spec = importlib.util.spec_from_file_location(name, REPO / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _loaded[name] = module
    return _loaded[name]


def aggregator():
    return load("nihcc_search_aggregator", "apps/search-aggregator/app/main.py")
