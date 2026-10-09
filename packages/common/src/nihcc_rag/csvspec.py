"""NIHCC ChestX-ray14 `Data_Entry_2017*.csv` の解析（設計書 7.1.1）。

CSV 列は配布版で差異があり得るため、位置固定ではなく列名＋schema version で
マッピングする。`OriginalImage[Width,Height]` と
`OriginalImagePixelSpacing[x,y]` は header 1 列に対して data 2 列となるため、
schema にその展開を持つ。
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = "nihcc-data-entry-2017-v1"

#: 設計書 3.1 の 14 所見 + "No Finding"
ALLOWED_FINDINGS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)

#: 「所見なし」を表すラベル。部位レベルの状態を導ける唯一の手掛かりである。
NO_FINDING = "No Finding"

#: 所見ラベルの日本語表示名。
#:
#: **保存する値は英語のままにする。** データセットの値であり、再現性と外部との
#: 突き合わせのためにそのまま持つ。日本語は表示と入力の受け口だけで使う。
#:
#: Consolidation と Infiltration はどちらも「浸潤」と訳されがちだが、区別が付かないと
#: 絞り込みで選べない。前者をコンソリデーション（硬化像）、後者を浸潤影とする。
FINDING_LABELS_JA: dict[str, str] = {
    "Atelectasis": "無気肺",
    "Cardiomegaly": "心拡大",
    "Consolidation": "コンソリデーション（硬化像）",
    "Edema": "肺水腫",
    "Effusion": "胸水",
    "Emphysema": "肺気腫",
    "Fibrosis": "線維化",
    "Hernia": "ヘルニア",
    "Infiltration": "浸潤影",
    "Mass": "腫瘤",
    "No Finding": "所見なし",
    "Nodule": "結節影",
    "Pleural_Thickening": "胸膜肥厚",
    "Pneumonia": "肺炎",
    "Pneumothorax": "気胸",
}
assert set(FINDING_LABELS_JA) == set(ALLOWED_FINDINGS)

#: 入力で受け付ける言い換え。表示名は 1 つに決めるが、入力は揺れるため広く採る。
#:
#: 絞り込みは「入力した語が所見に一致しなければ 0 件」という失敗の仕方をする。
#: 利用者からは検索が壊れたようにしか見えないので、よくある言い方は拾っておく。
FINDING_ALIASES_JA: dict[str, tuple[str, ...]] = {
    "Atelectasis": ("無気肺", "肺虚脱"),
    "Cardiomegaly": ("心拡大", "心陰影拡大", "心肥大", "心陰影の拡大"),
    "Consolidation": ("コンソリデーション", "硬化像", "硬化影", "浸潤・硬化"),
    "Edema": ("肺水腫", "肺うっ血", "うっ血"),
    "Effusion": ("胸水", "胸水貯留", "胸膜液", "胸水の貯留"),
    "Emphysema": ("肺気腫", "気腫"),
    "Fibrosis": ("線維化", "繊維化", "肺線維症", "線維症"),
    "Hernia": ("ヘルニア", "横隔膜ヘルニア", "裂孔ヘルニア"),
    "Infiltration": ("浸潤影", "浸潤", "浸潤陰影"),
    "Mass": ("腫瘤", "腫瘤影", "腫瘤陰影"),
    "No Finding": ("所見なし", "異常なし", "正常", "特記所見なし"),
    "Nodule": ("結節影", "結節", "結節陰影", "小結節"),
    "Pleural_Thickening": ("胸膜肥厚", "胸膜の肥厚", "胸膜肥厚像"),
    "Pneumonia": ("肺炎", "肺炎像"),
    "Pneumothorax": ("気胸", "気胸像"),
}


def _fold(text: str) -> str:
    """一致判定のための正規化。

    全角と半角、空白、`_`、大文字小文字の違いで外さないようにする。
    `Pleural_Thickening` / `pleural thickening` / 「胸膜肥厚」がすべて同じ鍵になる。
    """
    import unicodedata

    folded = unicodedata.normalize("NFKC", text).strip().lower()
    return "".join(ch for ch in folded if not ch.isspace() and ch not in "_-・")


@functools.lru_cache(maxsize=1)
def _finding_index() -> dict[str, str]:
    """入力の表記 -> 正式な英語ラベル。"""
    index: dict[str, str] = {}
    for label in ALLOWED_FINDINGS:
        index[_fold(label)] = label
        index[_fold(FINDING_LABELS_JA[label])] = label
        for alias in FINDING_ALIASES_JA.get(label, ()):
            if alias:
                index.setdefault(_fold(alias), label)
    return index


def finding_label_ja(label: str) -> str:
    """英語ラベルの日本語表示名。未知の値はそのまま返す。"""
    return FINDING_LABELS_JA.get(label, label)


def resolve_finding(text: str) -> str | None:
    """入力された所見名（日本語・英語・言い換え）を正式な英語ラベルへ寄せる。"""
    return _finding_index().get(_fold(text))


def resolve_findings(values) -> tuple[list[str], list[str]]:
    """複数まとめて寄せる。`(解決できた英語ラベル, 解決できなかった入力)` を返す。

    **解決できなかったものを黙って捨てない。** 捨てると絞り込みが効いていないのか
    綴りが違うのか利用者に分からない。呼び出し側で明示的に扱えるよう分けて返す。
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    for value in values:
        label = resolve_finding(value)
        if label is None:
            unresolved.append(value)
        elif label not in resolved:
            resolved.append(label)
    return resolved, unresolved

#: 部位の所見状態。データセットのラベルは画像単位なので、この 2 値しか導けない。
REGION_STATE_NORMAL = "NORMAL"
REGION_STATE_UNKNOWN = "UNKNOWN"


def region_finding_state(finding_labels) -> str:
    """画像単位の所見ラベルから、各部位の所見状態として**確定できるもの**を返す。

    NIHCC のラベルは画像全体に付いており、部位単位の正解は存在しない。
    そこから論理的に導けるのは次の 2 つだけである。

      `No Finding`  ⇒ 5 部位すべてが正常   -> NORMAL（確定）
      所見 F あり   ⇒ 少なくとも 1 部位が F -> UNKNOWN（どの部位かは不明）

    画像単位のラベルを部位の所見として表示すると、気胸画像の正常な右上葉を
    正しく検索できていても "Pneumothorax" と出てしまい、人の目でも正誤を
    判別できなくなる。表示側はこの関数の値で「断定できる範囲」を区切ること。

    同じ含意は検索品質の評価（scripts/region_signal_eval.py）でも使う。
    規則を 2 箇所に書くと食い違うため、ここを唯一の置き場とする。
    """
    return REGION_STATE_NORMAL if NO_FINDING in set(finding_labels) else REGION_STATE_UNKNOWN


_FINDING_LOOKUP = {name.lower().replace(" ", "_"): name for name in ALLOWED_FINDINGS}
_FINDING_LOOKUP.update({name.lower(): name for name in ALLOWED_FINDINGS})

#: 配布 CSV の header は `OriginalImage[Width,Height]` のように列名自体へカンマを
#: 含むため、CSV としては 11 フィールドに分割される。header 行・データ行とも
#: この 11 フィールド構成を正とする。
_EXPECTED_HEADER = [
    "Image Index",
    "Finding Labels",
    "Follow-up #",
    "Patient ID",
    "Patient Age",
    "Patient Sex",
    "View Position",
    "OriginalImage[Width",
    "Height]",
    "OriginalImagePixelSpacing[x",
    "y]",
]

#: 分割された header 名 → 保存時の論理名
_COLUMN_ALIASES = {
    "OriginalImage[Width": "original_width",
    "Height]": "original_height",
    "OriginalImagePixelSpacing[x": "pixel_spacing_x",
    "y]": "pixel_spacing_y",
}

_IMAGE_NAME_RE = re.compile(r"^[0-9]{8}_[0-9]{3}\.png$", re.IGNORECASE)


class CsvSchemaError(ValueError):
    """CSV の列構成が既知 schema と一致しない。"""


@dataclass
class CsvRow:
    source_image_name: str
    patient_key: str
    finding_labels: list[str]
    unknown_labels: list[str]
    follow_up: int | None
    patient_age: int | None
    patient_sex: str | None
    view_position: str | None
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def patient_key_hash(self) -> str:
        """患者 ID は画面へ出さず hash 化キーで保持する（設計書 13）。"""
        return hashlib.sha256(f"nihcc:{self.patient_key}".encode()).hexdigest()

    def source_metadata(self) -> dict[str, object]:
        return {
            "csv_schema_version": SCHEMA_VERSION,
            "raw": self.raw,
            "follow_up": self.follow_up,
            "unknown_labels": self.unknown_labels,
        }


def normalise_finding(label: str) -> str | None:
    """前後空白・大小文字・区切り文字を正規化し、許可語彙へ写像する。"""
    cleaned = label.strip().replace("　", " ")
    if not cleaned:
        return None
    key = cleaned.lower().replace(" ", "_")
    return _FINDING_LOOKUP.get(key) or _FINDING_LOOKUP.get(cleaned.lower())


def validate_header(header: list[str]) -> None:
    normalised = [h.strip() for h in header]
    if normalised[: len(_EXPECTED_HEADER)] != _EXPECTED_HEADER:
        raise CsvSchemaError(
            f"unexpected CSV header for schema {SCHEMA_VERSION}: "
            f"{normalised[: len(_EXPECTED_HEADER)]}"
        )


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def iter_rows(csv_path: Path) -> Iterator[CsvRow]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from iter_rows_from_stream(handle)


def iter_rows_from_stream(handle: io.TextIOBase) -> Iterator[CsvRow]:
    reader = csv.reader(handle)
    try:
        header = next(reader)
    except StopIteration:
        raise CsvSchemaError("CSV is empty") from None
    validate_header(header)

    for values in reader:
        if not values or not values[0].strip():
            continue
        raw: dict[str, str] = {}
        for index, name in enumerate(_EXPECTED_HEADER):
            key = _COLUMN_ALIASES.get(name, name)
            raw[key] = values[index].strip() if index < len(values) else ""

        labels: list[str] = []
        unknown: list[str] = []
        for token in raw.get("Finding Labels", "").split("|"):
            if not token.strip():
                continue
            mapped = normalise_finding(token)
            if mapped is None:
                unknown.append(token.strip())
            elif mapped not in labels:
                labels.append(mapped)

        yield CsvRow(
            source_image_name=raw.get("Image Index", "").strip(),
            patient_key=raw.get("Patient ID", "").strip(),
            finding_labels=labels,
            unknown_labels=unknown,
            follow_up=_parse_int(raw.get("Follow-up #")),
            patient_age=_parse_int(raw.get("Patient Age")),
            patient_sex=(raw.get("Patient Sex") or "").strip() or None,
            view_position=(raw.get("View Position") or "").strip().upper() or None,
            raw=raw,
        )


def is_valid_image_name(name: str) -> bool:
    return bool(_IMAGE_NAME_RE.match(name))
