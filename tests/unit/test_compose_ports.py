"""compose のポート指定と .env.example の既定値が噛み合っていることを固定する。

compose 側が `"127.0.0.1:${X}:8000"` と待ち受けアドレスを自分で付けているのに、
.env.example の X にも `127.0.0.1:8002` と書くと、展開結果が
`127.0.0.1:127.0.0.1:8002:8000` になる。compose は設定全体を解釈するので、
**関係ないサービスの build まで** `invalid IP address` で止まる。

古い .env を持つ環境では表に出ず、bootstrap.sh で .env を作り直した新規構築で
初めて踏んだ（EC2、EMBEDDING_SERVICE_PORT）。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMPOSE_FILES = [REPO / "docker-compose.yml", *sorted((REPO / "infra" / "compose").glob("*.yml"))]
#: 待ち受けアドレスを compose 側で付けている変数参照
BOUND = re.compile(r"\d+\.\d+\.\d+\.\d+:\$\{(\w+)(?::-[^}]*)?\}")


def env_example() -> dict[str, str]:
    values = {}
    for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_bound_port_variables_hold_only_a_port_number() -> None:
    values = env_example()
    bad = []
    for path in COMPOSE_FILES:
        for name in BOUND.findall(path.read_text(encoding="utf-8")):
            value = values.get(name)
            if value is not None and not value.isdigit():
                bad.append(f"{name}={value}（{path.relative_to(REPO)} が待ち受けアドレスを付けている）")
    assert not bad, "番号だけにすること: " + ", ".join(bad)
