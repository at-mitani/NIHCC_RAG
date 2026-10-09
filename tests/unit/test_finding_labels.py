"""所見ラベルの日本語表示と、日本語入力での絞り込みを固定する。

画面の所見テキストは日本語で入れられるので、絞り込みのラベルも日本語で表示する。
**片方だけ日本語なのは使う側から見て一貫しない。**

絞り込みは「入力した語が一致しなければ 0 件」という失敗の仕方をするため、
綴りの違いが検索の不具合に見える。ここで対訳と言い換えの受け付けを守る。
"""

from __future__ import annotations

import pytest
from nihcc_rag import csvspec


def test_every_finding_has_a_japanese_label() -> None:
    """15 所見すべてに表示名があること。抜けると画面に英語が混ざる。"""
    assert set(csvspec.FINDING_LABELS_JA) == set(csvspec.ALLOWED_FINDINGS)
    assert all(csvspec.FINDING_LABELS_JA[label] for label in csvspec.ALLOWED_FINDINGS)


def test_japanese_labels_are_distinct() -> None:
    """表示名が重複しないこと。

    Consolidation と Infiltration はどちらも「浸潤」と訳されがちだが、同じ名前に
    すると絞り込みで選び分けられない。
    """
    names = list(csvspec.FINDING_LABELS_JA.values())
    assert len(set(names)) == len(names)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Fibrosis", "Fibrosis"),
        ("fibrosis", "Fibrosis"),
        ("線維化", "Fibrosis"),
        ("繊維化", "Fibrosis"),          # 誤字に近い表記も拾う
        ("肺線維症", "Fibrosis"),
        ("Pleural_Thickening", "Pleural_Thickening"),
        ("pleural thickening", "Pleural_Thickening"),   # `_` と空白の違い
        ("胸膜肥厚", "Pleural_Thickening"),
        ("気胸", "Pneumothorax"),
        ("胸水", "Effusion"),
        ("胸水貯留", "Effusion"),
        ("心拡大", "Cardiomegaly"),
        ("心陰影拡大", "Cardiomegaly"),
        ("所見なし", "No Finding"),
        ("異常なし", "No Finding"),
        ("No Finding", "No Finding"),
        ("ｎｏ　ｆｉｎｄｉｎｇ", "No Finding"),          # 全角
    ],
)
def test_resolve_finding(text: str, expected: str) -> None:
    assert csvspec.resolve_finding(text) == expected


def test_pneumothorax_and_emphysema_are_not_confused() -> None:
    """「気胸」と「肺気腫」を取り違えないこと。字が似ていて紛れやすい。"""
    assert csvspec.resolve_finding("気胸") == "Pneumothorax"
    assert csvspec.resolve_finding("肺気腫") == "Emphysema"
    assert csvspec.resolve_finding("気腫") == "Emphysema"


def test_unknown_text_is_not_guessed() -> None:
    """知らない語を当てずっぽうで寄せないこと。

    近い語へ勝手に寄せると、利用者は別の所見で絞り込まれたことに気づけない。
    """
    assert csvspec.resolve_finding("でたらめな所見") is None
    assert csvspec.resolve_finding("") is None


def test_resolve_findings_separates_the_unresolved() -> None:
    """解決できたものと、できなかったものを分けて返すこと。

    黙って捨てると「絞り込みが効かない」のか「綴りが違う」のかが分からない。
    """
    resolved, unresolved = csvspec.resolve_findings(["線維化", "Mass", "存在しない"])
    assert resolved == ["Fibrosis", "Mass"]
    assert unresolved == ["存在しない"]


def test_resolve_findings_deduplicates() -> None:
    """同じ所見を別の表記で入れても 1 つに寄せること。"""
    resolved, unresolved = csvspec.resolve_findings(["線維化", "Fibrosis", "肺線維症"])
    assert resolved == ["Fibrosis"]
    assert unresolved == []


def test_finding_label_ja_passes_through_unknown_values() -> None:
    """表に無い値はそのまま返すこと。表示が消えるより英語で出る方が良い。"""
    assert csvspec.finding_label_ja("Fibrosis") == "線維化"
    assert csvspec.finding_label_ja("SomethingNew") == "SomethingNew"
