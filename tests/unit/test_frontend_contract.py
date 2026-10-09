"""画面が読む応答の欄が、contract に実在することを固定する。

型のない JavaScript と Python の contract は、ずれても誰も気づかない。backend が
返さない欄を `app.js` が読むと、検索画面は

    Cannot read properties of undefined (reading 'length')

で落ちる。ここでは、画面が読む欄が contract に実在すること、contract に無い
（部位単位の形状・部位ごとの順位の）欄を画面が読まないことを固定する。

**この試験が防げるのは「`app.js` を直し忘れること」だけである。** 静的ファイルは
image へ焼き込まれるので、image の作り直し漏れは検出できない（配信物はここから
見えない）。そちらは個別のサービスを選んで build せず `scripts/redeploy.sh` を
使うことで防ぐ。
"""

from __future__ import annotations

import re
from pathlib import Path

from nihcc_rag import contract

REPO = Path(__file__).resolve().parents[2]
APP_JS = REPO / "apps" / "frontend" / "static" / "app.js"


def function_body(source: str, name: str) -> str:
    """`function name(...) { ... }` の本体を波括弧の対応で取り出す。

    「次の `function` まで」では切れない。画面の処理は無名関数で書かれていることが
    多く、取り込み画面まで飲み込んで誤検出になる（実測: `valid` / `validation_id`）。
    """
    start = source.index(f"function {name}(")
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace : index + 1]
    raise AssertionError(f"{name} の括弧が閉じていない")


def test_the_retired_region_shape_fields_are_gone() -> None:
    """contract に無い部位単位の形状の欄を、画面が読まないこと。"""
    source = APP_JS.read_text(encoding="utf-8")
    for field in ("shape_results", "shape_status"):
        for hit in re.findall(r"\b\w+\." + field + r"\b", source):
            assert hit.startswith("result.lung_"), f"部位単位の形状を読んでいる: {hit}"


def test_the_contract_has_no_region_shape_fields() -> None:
    """contract 側にも部位単位の形状の欄が無いこと。画面と両側で押さえる。"""
    fields = set(contract.SearchResponse.model_fields)
    assert "shape_results" not in fields
    assert "shape_status" not in fields
    # 胸郭の方は残っている
    assert "lung_shape_results" in fields


#: 検索の単位が胸部野 1 つなので contract に存在しない、部位ごとの欄
RETIRED_REGION_FIELDS = (
    "regions", "overall", "region_code", "region_crop_url", "roi_url",
    "region_finding_state", "query_region_crop_url",
)


def test_the_screen_no_longer_reads_region_fields() -> None:
    """部位ごとの順位の欄（contract に無い）を、画面が読まないこと。"""
    source = APP_JS.read_text(encoding="utf-8")
    for field in RETIRED_REGION_FIELDS:
        hits = re.findall(r"\b(?:result|item|region|status)\." + field + r"\b", source)
        assert not hits, f"廃止した部位の欄を読んでいる: {hits}"


def test_the_contract_has_no_region_fields() -> None:
    for model in (contract.SearchResponse, contract.SearchResultItem, contract.SearchQueryEcho):
        left = set(model.model_fields) & set(RETIRED_REGION_FIELDS)
        assert not left, f"{model.__name__} に部位の欄が残っている: {sorted(left)}"


def test_the_fields_the_search_screen_reads_exist() -> None:
    """検索結果の描画が読む欄が contract に在ること。

    無い欄を読むと、その場では `undefined` になり、`.length` を触った時点で初めて
    落ちる。**落ちる場所と原因が離れる**ので、ここで先に見つける。

    対象は検索結果の描画だけに絞る。取り込み画面や管理画面まで見ると、応答以外の
    由来（DOM・局所変数）を拾って誤検出になる。
    """
    source = APP_JS.read_text(encoding="utf-8")
    checks = {
        # 関数名: (変数名, その変数が指す contract のモデル)
        "renderResults": ("result", contract.SearchResponse),
        "queryChestCard": ("result", contract.SearchResponse),
        "rerankBadge": ("result", contract.SearchResponse),
        "hitCard": ("item", contract.SearchResultItem),
        "labelBlock": ("item", contract.SearchResultItem),
        "lungCard": ("item", contract.SearchResultItem),
        "estimateBlock": ("estimate", contract.FindingEstimate),
    }
    missing = []
    for name, (variable, model) in checks.items():
        body = function_body(source, name)
        used = set(re.findall(rf"\b{variable}\.([a-z][a-z0-9_]*)\b", body))
        # 配列のメソッド（estimates.forEach など）は除く
        used -= {"length", "forEach", "map", "slice"}
        for field in sorted(used - set(model.model_fields)):
            missing.append(f"{name}: {variable}.{field}")
    assert not missing, (
        "検索結果の描画が contract に無い欄を読んでいる: " + ", ".join(missing)
        + "。contract を変えたら app.js も直し、frontend image を作り直すこと"
    )


def test_estimates_are_labelled_as_not_a_diagnosis() -> None:
    """推定には「診断ではない」見出しが必ず付く（推定の表示規則）。

    見出しは応答の estimate_label を使い、無いときも同じ文言へ倒す。
    """
    body = function_body(APP_JS.read_text(encoding="utf-8"), "estimateBlock")
    assert "診断ではありません" in body
