"""評価の実行と結果を DB へ残す。

評価結果をファイル（`tmp/` など）に置くと、別の機械で測った結果を手元へ持って
こられず、消えても気づけず、「前の版と比べてどうか」を問い合わせることもできない。

**再生成できない**（同じ実行は二度とできない）わりに**小さい**ので、DB に置く。

## 使い方

    from nihcc_rag import evaluation

    with evaluation.record(
        kind="roi-dice",
        target_version=profile.roi_model_version,
        params={"limit": 1000, "min_rca": 0.7, "seed": 0},
        sample_size=len(names),
    ) as run:
        ...
        run.metric("dice_median", 0.968)
        run.metric("breakdown_rate", 0.005, baseline=0.238)

DB へ繋げないときは**黙って諦める**。評価そのものを止めない。記録は補助であって、
それで測定が失敗しては本末転倒である。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

_LOG = logging.getLogger("evaluation")


@dataclass
class Run:
    """1 回の評価実行。`metric()` で結果を足す。"""

    run_id: uuid.UUID
    #: DB へ書けないときは False。呼び出し側は気にしなくてよい。
    persisted: bool = True
    _metrics: list[dict[str, Any]] = field(default_factory=list)

    def metric(self, name: str, value: float, *, baseline: float | None = None,
               region_code: str | None = None, finding: str | None = None) -> None:
        """指標を 1 つ足す。

        `baseline` は**帰無（null）の値**である。帰無を測らずに「効いた」と言わない
        ために置いてある。比較対象が無い指標では省いてよい。
        """
        self._metrics.append({
            "name": name,
            "value": float(value),
            "baseline": None if baseline is None else float(baseline),
            "region_code": region_code,
            "finding": finding,
        })

    @property
    def metrics(self) -> list[dict[str, Any]]:
        return list(self._metrics)


def git_revision() -> str | None:
    """実行したときの HEAD。あとから何のコードか辿れるようにする。

    **コンテナの中には git が無い。** 評価はコンテナ内で走らせる（DB へ届くのが
    そこだけなので）ため、素直に `git` を呼ぶと必ず空になる。`EVAL_GIT_REVISION`
    で外から渡せるようにし、`scripts/redeploy.sh` や運用手順でそれを使う。
    """
    from_env = os.environ.get("EVAL_GIT_REVISION", "").strip()
    if from_env:
        return from_env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - 記録は補助。git が無くても評価は続ける
        return None


def where_am_i() -> str:
    """どこで走らせたか。同じ条件でも機械で差が出るため残す。"""
    return os.environ.get("EVAL_RAN_ON") or ("ec2" if os.path.exists("/sys/hypervisor/uuid")
                                             else "local")


@contextmanager
def record(*, kind: str, target_version: str, params: dict[str, Any] | None = None,
           sample_size: int | None = None, note: str | None = None) -> Iterator[Run]:
    """評価を記録する。DB へ繋げなければ黙って諦める。

    `with` を抜けるときにまとめて書く。途中で例外が出たら `finished_at` を埋めない
    ので、**中断した実行を完了と取り違えない。**
    """
    run = Run(run_id=uuid.uuid4())
    try:
        yield run
    finally:
        _flush(run, kind=kind, target_version=target_version,
               params=params or {}, sample_size=sample_size, note=note)


def persist(run: Run, *, kind: str, target_version: str,
            params: dict[str, Any] | None = None, sample_size: int | None = None,
            note: str | None = None) -> None:
    """`with` を使わずに書き出す。

    既に指標を集め終えた script（`Run` を先に作って各所で `metric()` を呼ぶ形）の
    ための入口。`record()` と同じことを行う。
    """
    _flush(run, kind=kind, target_version=target_version, params=params or {},
           sample_size=sample_size, note=note)


def _flush(run: Run, *, kind: str, target_version: str, params: dict[str, Any],
           sample_size: int | None, note: str | None) -> None:
    if not run.metrics:
        return
    try:
        from nihcc_rag import db

        # 評価 script は DB を使わずに動くので、プールが立っていないことがある。
        # **ここで面倒を見る。** 呼び出し側に初期化を強いると、記録のためだけに
        # 評価 script が DB の事情を知ることになる。
        # 既に立っていれば何もしない実装なので、毎回呼んでよい。
        db.init_metadata_pool(role="app_meta", application_name="evaluation")

        with db.metadata_conn() as conn:
            with conn.transaction():
                db.execute(
                    conn,
                    """
                    INSERT INTO evaluation_run
                      (run_id, kind, target_version, params, ran_on, git_revision,
                       sample_size, finished_at, note)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, now(), %s)
                    """,
                    (str(run.run_id), kind, target_version,
                     json.dumps(params, ensure_ascii=False, default=str),
                     where_am_i(), git_revision(), sample_size, note),
                )
                for metric in run.metrics:
                    db.execute(
                        conn,
                        """
                        INSERT INTO evaluation_metric
                          (run_id, name, region_code, finding, value, baseline)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (str(run.run_id), metric["name"], metric["region_code"],
                         metric["finding"], metric["value"], metric["baseline"]),
                    )
        _LOG.info("評価結果を記録した: run_id=%s 指標 %d 件", run.run_id, len(run.metrics))
        # **プールを閉じる。** 記録のためだけに立てたので、開いたままだと script の
        # 終了時に psycopg が「スレッドを止められない」と警告を並べる。
        try:
            db.close_all()
        except Exception:  # noqa: BLE001 - 閉じられなくても記録は済んでいる
            pass
    except Exception as exc:  # noqa: BLE001 - 記録は補助。評価そのものは止めない
        run.persisted = False
        _LOG.warning("評価結果を記録できなかった（評価そのものは成功している）: %s: %s",
                     type(exc).__name__, str(exc).splitlines()[0][:200])
