"""置いてあるスクリプトが、すべて運用マニュアルに載っていることを固定する。

**使っていないスクリプトは置かない。** 載っていないものは消し忘れか、用途を
書き忘れたかのどちらかである。どちらも、次に触る人が「これは何か」を調べる
手間になる。

逆に、マニュアルに載っているのに実体が無いものも検出する（改名や削除の追従漏れ）。
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
OPERATIONS = REPO / "docs" / "OPERATIONS.md"

#: マニュアルに載せなくてよいもの（実行対象ではない）。
NOT_SCRIPTS = {"initial_users.example.json"}


def script_names() -> set[str]:
    return {
        path.name
        for path in SCRIPTS.iterdir()
        if path.is_file() and path.suffix in (".py", ".sh") and path.name not in NOT_SCRIPTS
    }


def test_every_script_is_documented() -> None:
    text = OPERATIONS.read_text(encoding="utf-8")
    missing = sorted(name for name in script_names() if name not in text)
    assert not missing, (
        "docs/OPERATIONS.md に載っていないスクリプト: " + ", ".join(missing)
        + "。用途を書くか、使っていないなら消すこと"
    )


def test_every_documented_script_exists() -> None:
    import re

    text = OPERATIONS.read_text(encoding="utf-8")
    mentioned = set(re.findall(r"`([\w-]+\.(?:py|sh))`", text))
    names = script_names()
    # scripts/ 以外の場所にあるものは対象外（tests/ の smoke_test.py など）
    missing = sorted(
        name for name in mentioned
        if name not in names and not list(REPO.rglob(name))
    )
    assert not missing, f"マニュアルにあるが実体が無い: {missing}"
