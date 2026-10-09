"""`.env.example` の構成へ `.env` を作り直し、既存の値を引き継ぐ。

`.env` は git 管理外なので、`git pull` でキーが増えても自動では追随しない。
手で書き足すと、増えたキーの取りこぼしと、古い値の残留の両方が起きる（たとえば
評価作業で入れた `EMBEDDING_FINE_BACKEND` の値が残ると、DB の列幅と次元が合わず
2 段目の検索が全件失敗する）。

そこで**キーの集合は `.env.example` を正**とし、**値は既存の `.env` を正**とする。
どちらかだけを正にすると、上のどちらかの失敗に倒れる。

`--reset` を付けたキー（接頭辞可）は引き継がず、`.env.example` の既定へ戻す。
版を上げたときに「この系統だけは既定に戻したい」という場面のためにある。

**新しく増えたキーは compose の既定値を採る。** `.env` に無いキーの実効値は
compose の `${NAME:-既定}` であって `.env.example` の値ではない。example を入れると、
書き足しただけのつもりで挙動が変わる（例: `TORCH_INDEX_URL` が cpu -> cu121 に
なり、NVIDIA GPU の無い PC で CUDA 版 PyTorch を入れる設定になる）。

**compose が参照するのに `.env.example` に無いキーは警告する。** そのままでは
`.env.example` を正とする同期が成立せず、運用者はその設定を調整できるとすら
知らない状態になる。

書き出す前に必ず差分を表示する。`.env` は取り込み済みデータとの整合（
PREPROCESS_VERSION / EMBEDDING_MODEL_REVISION など）を握っており、黙って
書き換えてよいファイルではない。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

#: `KEY=value`。コメント行と空行は対象外。
_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

#: compose の変数参照 `${NAME}` / `${NAME:-既定}`。
#: `$${NAME}` はコンテナ内のシェル変数なので除く（直前が `$` でないものだけ拾う）。
_COMPOSE_VAR = re.compile(r"(?<!\$)\$\{([A-Z_][A-Z0-9_]*)")

#: `${NAME:-既定}` の既定値つき参照。
_COMPOSE_DEFAULT = re.compile(r"(?<!\$)\$\{([A-Z_][A-Z0-9_]*):-([^}]*)\}")

#: 監査の対象にする compose ファイル。
COMPOSE_FILES = ("docker-compose.yml", "infra/compose/*.yml")


def parse_env(text: str) -> dict[str, str]:
    """`KEY=value` を取り出す。最初の 1 件を採用する（compose と同じ挙動）。"""
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _ASSIGN.match(line)
        if match and match.group(1) not in values:
            values[match.group(1)] = match.group(2)
    return values


def compose_variables(root: Path) -> set[str]:
    """compose が参照している環境変数名を集める。"""
    names: set[str] = set()
    for pattern in COMPOSE_FILES:
        for path in sorted(root.glob(pattern)) if "*" in pattern else [root / pattern]:
            if path.is_file():
                names |= set(_COMPOSE_VAR.findall(path.read_text(encoding="utf-8")))
    return names


def compose_defaults(root: Path) -> dict[str, str]:
    """compose の `${NAME:-既定}` から既定値を集める。

    **`.env` に無いキーの実効値はこれである。** `.env.example` の値ではない。
    同じ名前が複数のファイルにあれば最初に見つけたものを採る（compose の合成順では
    なく、あくまで「既定として妥当か」を判断する材料）。
    """
    found: dict[str, str] = {}
    for pattern in COMPOSE_FILES:
        paths = sorted(root.glob(pattern)) if "*" in pattern else [root / pattern]
        for path in paths:
            if not path.is_file():
                continue
            for name, default in _COMPOSE_DEFAULT.findall(path.read_text(encoding="utf-8")):
                found.setdefault(name, default)
    return found


def undocumented(root: Path, example_keys: set[str]) -> list[str]:
    """compose が参照するのに `.env.example` に無いキー。

    **これがあると `.env.example` を正とする同期が成立しない。** compose の既定値で
    動いてしまうため気づきにくく、運用者はその設定を調整できるとすら知らない。
    """
    return sorted(compose_variables(root) - example_keys)


def is_reset(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key == p or key.startswith(p) for p in prefixes)


def render(
    example_text: str,
    current: dict[str, str],
    reset: tuple[str, ...],
    compose: dict[str, str] | None = None,
) -> tuple[str, dict]:
    """`.env.example` の行構成のまま、値だけを差し替えた本文を作る。

    コメントを残すのが要点。`.env.example` のコメントには「この値を変えたら
    migration を追加せよ」といった運用上の制約が書いてある。値だけの寄せ集めを
    出力すると、それが失われる。
    """
    compose = compose or {}
    out: list[str] = []
    carried: list[str] = []
    defaulted: list[str] = []
    #: 新規キーのうち、compose の既定を採って example と変えたもの
    kept_effective: list[str] = []
    for line in example_text.splitlines():
        match = _ASSIGN.match(line)
        if not match:
            out.append(line)
            continue
        key, default = match.group(1), match.group(2)
        if key in current and not is_reset(key, reset):
            out.append(f"{key}={current[key]}")
            if current[key] != default:
                carried.append(key)
        elif key not in current and key in compose and compose[key] != default:
            # **新規キーは compose の既定を採る。** `.env` に無いキーの実効値は
            # compose の既定であって example の値ではない。example を入れると、
            # 書き足しただけのつもりで挙動が変わる（例: TORCH_INDEX_URL が
            # cpu -> cu121 になり、CUDA 版 PyTorch を入れる設定になる）。
            out.append(f"{key}={compose[key]}")
            kept_effective.append(key)
        else:
            out.append(line)
            if key in current and current[key] != default:
                defaulted.append(key)

    example_keys = set(parse_env(example_text))
    dropped = sorted(k for k in current if k not in example_keys)
    report = {
        "carried": sorted(carried),
        "defaulted": sorted(defaulted),
        "kept_effective": sorted(kept_effective),
        "added": sorted(example_keys - set(current)),
        "dropped": dropped,
    }
    # .env.example に無いキーは末尾へ残す。COMPOSE_FILE のように環境固有で
    # 例に載せていない設定があり、落とすと GPU 構成の compose が読まれなくなる。
    if dropped:
        out.append("")
        out.append("# ---- .env.example に無い環境固有の設定（引き継ぎ） ----")
        out.extend(f"{k}={current[k]}" for k in dropped)
    return "\n".join(out) + "\n", report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--example", default=".env.example")
    parser.add_argument(
        "--reset",
        default="",
        help="既定へ戻すキー（カンマ区切り、接頭辞可）。例: EMBEDDING_FINE,MIL_",
    )
    parser.add_argument("--write", action="store_true", help="実際に書き出す（既定は差分表示のみ）")
    args = parser.parse_args(argv)

    env_path, example_path = Path(args.env), Path(args.example)
    if not example_path.is_file():
        print(f"{example_path} がありません", file=sys.stderr)
        return 1
    example_text = example_path.read_text(encoding="utf-8")
    current = parse_env(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
    reset = tuple(v.strip() for v in args.reset.split(",") if v.strip())

    compose = compose_defaults(example_path.resolve().parent)
    body, report = render(example_text, current, reset, compose)

    def show(title: str, keys: list[str], note: str) -> None:
        print(f"\n--- {title} ({len(keys)}) --- {note}")
        for key in keys:
            print(f"  {key}")

    show("既存の値を引き継ぐ", report["carried"], "example の既定とは異なる値")
    show("既定へ戻す", report["defaulted"], "--reset の指定、または example に無かった")
    show("新しく増えたキー", report["added"], "既定が入る（実効値は変わらない）")
    show(
        "うち compose の既定を採った",
        report["kept_effective"],
        "example の値とは違うため、現在の挙動を保つ方を採った",
    )
    show("example に無いキー", report["dropped"], "末尾へそのまま残す")

    # **compose が参照するのに example に無いキーを警告する。**
    # .env.example を正として同期するため、そこに無い設定は運用者から見えなくなる。
    gaps = undocumented(example_path.resolve().parent, set(parse_env(example_text)))
    if gaps:
        print(f"\n--- !! compose が参照するのに {example_path} に無い ({len(gaps)}) ---")
        for key in gaps:
            print(f"  {key}")
        print("  compose の既定値で動くため気づきにくい。example へ追記してください。")

    if not args.write:
        print("\n差分表示のみです。書き出すには --write を付けてください。")
        return 0

    if env_path.is_file():
        backup = env_path.with_suffix(f".bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(env_path, backup)
        print(f"\n退避: {backup}")
    env_path.write_text(body, encoding="utf-8")
    print(f"書き出し: {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
