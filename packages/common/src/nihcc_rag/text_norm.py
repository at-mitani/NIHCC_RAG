"""日本語所見テキストの英語正規化（設計書 7.3.2）。

BiomedCLIP の text encoder は英語モデルであるため、日本語をそのまま入力しない。
外部翻訳 API へ医療情報を送信しない方針のため、承認済みローカル辞書のみで
正規化し、辞書で処理できない自由文は翻訳せずエラーとして利用者へ返す。
（設計書 7.3.2 の「翻訳モデルの品質を保証できない場合は英語と定型日本語所見に限定」）
"""

from __future__ import annotations

import functools
import json
import re
import unicodedata
from dataclasses import dataclass
from importlib import resources

TRANSLATION_MODEL_VERSION = "medical-ja-en-dict-v1"


@dataclass(frozen=True)
class NormalizedText:
    original: str
    english: str
    language: str
    translation_model_version: str
    matched_terms: list[str]


class UntranslatableText(ValueError):
    """辞書で英語化できない日本語入力。"""

    def __init__(self, message: str, unresolved: list[str]) -> None:
        super().__init__(message)
        self.unresolved = unresolved


@functools.lru_cache(maxsize=1)
def _dictionary() -> dict[str, str]:
    raw = resources.files("nihcc_rag.resources").joinpath("ja_en_medical.json").read_text("utf-8")
    data = json.loads(raw)
    # 長い語から置換するため、長さ降順で保持する
    return dict(sorted(data.items(), key=lambda kv: len(kv[0]), reverse=True))


_JA_RANGES = (
    (0x3040, 0x30FF),  # ひらがな・カタカナ
    (0x3400, 0x4DBF),  # CJK 拡張A
    (0x4E00, 0x9FFF),  # CJK 統合漢字
    (0xFF66, 0xFF9F),  # 半角カタカナ
)


def is_japanese(text: str) -> bool:
    return any(
        any(low <= ord(ch) <= high for low, high in _JA_RANGES) for ch in text
    )


def normalize(text: str, language: str = "en") -> NormalizedText:
    cleaned = unicodedata.normalize("NFKC", text).strip()
    if not cleaned:
        raise UntranslatableText("text is empty", [])

    if language.lower() == "en" and not is_japanese(cleaned):
        return NormalizedText(text, cleaned, "en", TRANSLATION_MODEL_VERSION, [])

    if not is_japanese(cleaned):
        # ja 指定でも英語のみならそのまま使う
        return NormalizedText(text, cleaned, "en", TRANSLATION_MODEL_VERSION, [])

    remaining = cleaned
    matched: list[str] = []
    parts: list[str] = []
    for term, english in _dictionary().items():
        if term in remaining:
            remaining = remaining.replace(term, " ")
            matched.append(term)
            if english.strip():
                parts.append(english.strip())

    unresolved = [
        token for token in re.findall(r"[ぁ-んァ-ヶ一-龥]+", remaining) if len(token) >= 1
    ]
    if not parts or unresolved:
        raise UntranslatableText(
            "Japanese free text cannot be normalised with the approved local dictionary. "
            "Use English, or a registered Japanese finding phrase.",
            unresolved,
        )

    english = " ".join(dict.fromkeys(parts))
    return NormalizedText(text, english, "ja", TRANSLATION_MODEL_VERSION, matched)


def known_phrases() -> list[dict[str, str]]:
    """UI の入力補助に使う定型日本語所見の一覧（助詞などの補助語は除く）。"""
    return [
        {"ja": ja, "en": en}
        for ja, en in _dictionary().items()
        if en.strip() and len(ja) >= 2
    ]
