"""画面の辞書（static/i18n/{ja,en,zh}.json）と、それを引く側が食い違わないことを固定する。

画面の文言は辞書にだけ置く。辞書にキーが無いと、その場所にはキー名（`search.run`
など）がそのまま出る。言語を切り替えて初めて気づく類いの不具合なので、ここで
先に見つける。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from nihcc_rag import csvspec, text_norm

REPO = Path(__file__).resolve().parents[2]
STATIC = REPO / "apps" / "frontend" / "static"
LANGUAGES = ("ja", "en", "zh")

#: サーバから英語で受け取る値（所見ラベル・定型所見）の表示名。言語によって持つ範囲が
#: 違う（英語は原文のままでよいので定型所見を持たない）ため、キーの揃いの検査から外し、
#: 個別に検査する。
FINDING_PREFIX = "finding."
PHRASE_PREFIX = "phrase."
VALUE_PREFIXES = (FINDING_PREFIX, PHRASE_PREFIX)


def load(lang: str) -> dict[str, str]:
    return json.loads((STATIC / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))


def strip_js_comments(source: str) -> str:
    """コメントを除く（コメント中の書き方の例を拾わないため）。文字列に // は無い前提。"""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"(?m)^\s*//.*$", "", source)
    return re.sub(r"(?m)\s//[^'`\"]*$", "", source)


DICTIONARIES = {lang: load(lang) for lang in LANGUAGES}
APP_JS = strip_js_comments((STATIC / "app.js").read_text(encoding="utf-8"))
INDEX_HTML = re.sub(r"<!--.*?-->", "", (STATIC / "index.html").read_text(encoding="utf-8"), flags=re.S)


def ui_keys(lang: str) -> set[str]:
    return {key for key in DICTIONARIES[lang] if not key.startswith(VALUE_PREFIXES)}


def prefixed(lang: str, prefix: str) -> dict[str, str]:
    return {key[len(prefix):]: text for key, text in DICTIONARIES[lang].items() if key.startswith(prefix)}


@pytest.mark.parametrize("lang", LANGUAGES[1:])
def test_every_language_has_the_same_keys(lang: str) -> None:
    missing = ui_keys("ja") - ui_keys(lang)
    extra = ui_keys(lang) - ui_keys("ja")
    assert not missing, f"{lang} に無いキー: {sorted(missing)}"
    assert not extra, f"{lang} にだけあるキー: {sorted(extra)}"


@pytest.mark.parametrize("lang", LANGUAGES[1:])
def test_placeholders_match_across_languages(lang: str) -> None:
    """`{count}` などの差し込み位置の名前が言語間で同じであること。"""
    names = lambda text: set(re.findall(r"\{(\w+)\}", text))  # noqa: E731
    wrong = [
        key for key, text in DICTIONARIES["ja"].items()
        if key in DICTIONARIES[lang] and names(text) != names(DICTIONARIES[lang][key])
    ]
    assert not wrong, f"{lang} の差し込み名が日本語と違う: {wrong}"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_finding_names_cover_every_dataset_label(lang: str) -> None:
    assert set(prefixed(lang, FINDING_PREFIX)) == set(csvspec.ALLOWED_FINDINGS)


def test_japanese_finding_names_match_the_server_table() -> None:
    """日本語の所見名は、絞り込みの入力を解決するサーバの表と同じ語であること。

    画面の候補に出ている語を入れても絞り込みが一致しない、を防ぐ。
    """
    assert prefixed("ja", FINDING_PREFIX) == csvspec.FINDING_LABELS_JA


def server_phrases() -> set[str]:
    """サーバが返す定型所見（英語）。"""
    return {phrase["en"] for phrase in text_norm.known_phrases()}


@pytest.mark.parametrize("lang", ("ja", "zh"))
def test_phrase_names_cover_every_server_phrase(lang: str) -> None:
    """英語以外の表示では、定型所見すべてに表示名があること（英語は原文を出す）。"""
    assert set(prefixed(lang, PHRASE_PREFIX)) == server_phrases()


def test_japanese_phrase_names_translate_back() -> None:
    """日本語の定型所見名は、サーバがその英語へ訳せる語であること。

    画面は日本語表示で定型所見を押すと日本語の語を入れる。サーバの表に無い語だと
    検索が TEXT_NOT_TRANSLATABLE で失敗する。
    """
    wrong = {
        english: japanese
        for english, japanese in prefixed("ja", PHRASE_PREFIX).items()
        if text_norm.normalize(japanese, "ja").english != english
    }
    assert not wrong, f"サーバが同じ英語に訳さない日本語名: {wrong}"


def backend_message_ids() -> set[str]:
    """サーバが画面向けに付ける message_id と警告の id。"""
    ids: set[str] = set()
    for root in ("apps", "packages/common/src"):
        for path in (REPO / root).rglob("*.py"):
            if "tests" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            ids |= set(re.findall(r'\bmessage_id="([A-Z_]+)"', source))
            ids |= set(re.findall(r'ServerMessage\(\s*id="([A-Z_]+)"', source))
            ids |= set(re.findall(r'^\s*id="([A-Z_]+)",', source, flags=re.M))
            ids |= set(re.findall(r'^RESEARCH_WARNING_ID = "([A-Z_]+)"', source, flags=re.M))
    return ids


#: 利用者へ返す文言を作るモジュール。ここの文字列（docstring を除く）は英語で書く。
#: csvspec（日本語の入力を解決する表）・text_norm（日本語の判定）・mail（メール本文）は
#: 日本語を持つのが役目なので対象外。
USER_FACING_SOURCES = (
    *sorted((REPO / "apps" / "backend-api" / "app").rglob("*.py")),
    *(REPO / "packages" / "common" / "src" / "nihcc_rag" / name
      for name in ("auth.py", "passwords.py", "problem.py", "contract.py")),
)


def test_user_facing_server_strings_are_english() -> None:
    """API が返す文言に日本語を書き戻さないこと（画面が表示言語に訳せなくなる）。"""
    import ast

    japanese = re.compile(r"[぀-ヿ一-鿿]")
    hits = []
    for path in USER_FACING_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        hits += [
            f"{path.relative_to(REPO)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and japanese.search(node.value) and id(node) not in docstrings
        ]
    assert not hits, f"API の文言に日本語がある: {hits}"


def test_problem_body_carries_message_id_and_params() -> None:
    """エラー応答に、画面が訳すための message_id と params が載ること。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from nihcc_rag import problem

    app = FastAPI()
    problem.install(app)

    @app.get("/login")
    def login() -> None:
        raise problem.unauthorized("The user name or password is incorrect.",
                                   message_id="AUTH_INVALID_CREDENTIALS")

    @app.get("/policy")
    def policy() -> None:
        raise problem.unprocessable("PASSWORD_POLICY_VIOLATION", "too short",
                                    params={"min": 8, "max": 128})

    client = TestClient(app)
    body = client.get("/login").json()
    assert (body["code"], body["message_id"]) == ("UNAUTHENTICATED", "AUTH_INVALID_CREDENTIALS")
    assert "params" not in body
    body = client.get("/policy").json()
    assert body["message_id"] == "PASSWORD_POLICY_VIOLATION"  # 既定は code
    assert body["params"] == {"min": 8, "max": 128}


def test_server_message_ids_have_translations() -> None:
    """サーバが付けた id はすべて辞書で訳せること（訳せないと英語の原文が出る）。"""
    ids = backend_message_ids()
    assert {"AUTH_INVALID_CREDENTIALS", "RESEARCH_USE", "SELF_MATCH_EXCLUDED"} <= ids
    missing = {f"server.{i}" for i in ids} - ui_keys("ja")
    assert not missing, f"辞書に無いサーバの文言: {sorted(missing)}"


def test_html_keys_exist() -> None:
    keys = set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', INDEX_HTML))
    assert keys, "index.html に data-i18n が無い"
    missing = keys - ui_keys("ja")
    assert not missing, f"index.html が辞書に無いキーを指している: {sorted(missing)}"


def test_literal_keys_in_app_js_exist() -> None:
    keys = set(re.findall(r"\bt\('([^']+)'", APP_JS))
    # 表の見出しは headerRow(['admin.queue', ...]) とキーの配列で渡す
    for row in re.findall(r"\bheaderRow\(\[([^\]]+)\]", APP_JS, flags=re.S):
        keys |= set(re.findall(r"'([^']+)'", row))
    missing = keys - ui_keys("ja")
    assert not missing, f"app.js が辞書に無いキーを引いている: {sorted(missing)}"


def object_keys(name: str) -> set[str]:
    """`const NAME = { A: ..., B: ... };` のキーを取り出す。"""
    body = re.search(rf"const {name} = \{{(.*?)\}};", APP_JS, flags=re.S).group(1)
    return set(re.findall(r"\b([A-Z_]+):", body))


def test_status_keys_built_at_runtime_exist() -> None:
    """`t(\\`results.rerank.${status}\\`)` のように組み立てるキーも辞書に在ること。"""
    expected = set()
    expected |= {f"results.quality.{s}" for s in object_keys("QUALITY_CLASSES") - {"OK"}}
    expected |= {f"results.rerank.{s}" for s in object_keys("RERANK_CLASSES")}
    expected |= {f"results.shape.{s}" for s in object_keys("SHAPE_CLASSES")}
    expected |= {f"job.status.{s}" for s in ("PENDING", "RUNNING", "SUCCEEDED", "PARTIAL", "FAILED", "CANCELLED")}
    expected |= {f"job.field.{f}" for f in
                 ("total", "succeeded", "failed", "pending", "running", "quarantined", "cancelled")}
    expected |= {f"roles.{r}" for r in ("viewer", "operator", "admin")}
    missing = expected - ui_keys("ja")
    assert not missing, f"組み立てたキーが辞書に無い: {sorted(missing)}"


def test_no_hard_coded_japanese_in_app_js() -> None:
    """文言を app.js に直接書き戻さないこと。コメントと区切りの「・」は除く。"""
    code = APP_JS.replace("・", "")
    hits = [line.strip() for line in code.splitlines()
            if re.search(r"[぀-ヿ一-鿿]", line)]
    assert not hits, f"app.js に日本語の文言が残っている: {hits}"


@pytest.mark.parametrize(("lang", "phrase"), [
    ("ja", "診断ではありません"),
    ("en", "not a diagnosis"),
    ("zh", "并非诊断"),
])
def test_estimate_heading_says_not_a_diagnosis(lang: str, phrase: str) -> None:
    """所見の推定の見出しは、どの言語でも「診断ではない」と書く（推定の表示規則）。"""
    assert phrase in DICTIONARIES[lang]["results.estimateHeading"]
