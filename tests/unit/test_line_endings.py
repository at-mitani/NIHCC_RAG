"""Linux で実行されるファイルが CRLF になっていないことを固定する。

`set -euo pipefail` の行が CRLF だと、bash は `pipefail\r` を不正なオプション名と
して扱い、**1 行目から動かなくなる**。しかもエラー文の末尾に \r が入るため
`: invalid option nameline 13: set: pipefail` のように表示が壊れ、原因に辿り着き
にくい。

混入経路は 2 つある。

1. Windows の checkout（core.autocrlf=true）。.gitattributes の eol=lf で止める
2. **編集ツールの書き出し**。Python の `Path.write_text` は既定でプラットフォーム
   依存の改行へ変換するため、Windows 上で書き直すと CRLF に戻る

2 は設定では防げないので、ここで検出する。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
#: Linux のシェル・プロセスへ渡るファイル。.gitattributes の対象と揃える。
PATTERNS = ("*.sh", "Dockerfile", "*/Dockerfile", "*.conf")


def tracked(pattern: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", pattern],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return [REPO / name for name in out.split("\0") if name]


def targets() -> list[Path]:
    seen: dict[str, Path] = {}
    for pattern in PATTERNS:
        for path in tracked(pattern):
            if path.is_file():
                seen[str(path)] = path
    return sorted(seen.values())


@pytest.mark.parametrize("path", targets(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_crlf(path: Path) -> None:
    body = path.read_bytes()
    assert b"\r\n" not in body, (
        f"{path.relative_to(REPO)} が CRLF です。Linux で実行すると壊れます。"
        " 直し方: python -c \"p=open(PATH,'rb');d=p.read();p.close();"
        "open(PATH,'wb').write(d.replace(b'\r\n',b'\n'))\""
    )


def test_there_is_something_to_check() -> None:
    """対象が 0 件なら、この検査は何も守っていない。"""
    assert len(targets()) >= 4


def test_gitattributes_pins_shell_scripts() -> None:
    """checkout 側の混入も止まっていること。"""
    text = (REPO / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in text
