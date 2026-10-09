"""モデル成果物の所在と SHA-256 を `model_version` へ記録する。

    python scripts/register_model_artifact.py --type LUNG_SEGMENTATION \\
        --path infra/lung-unet/lung-unet.onnx --version unet-lung-spine-v2.0.0

**成果物そのものは DB に入れない。** コンテナは read-only mount で読むので、DB に
入れると起動のたびに取り出してファイルへ書く処理が要り、`read_only: true` と
噛み合わない。置き場所はファイルのままにして、**どれで動いているかを DB から
辿れるように**記録だけを持つ。

これが無いと、EC2 と手元で違う成果物が動いていても気づけない。検索結果の
`roi_model_version` は profile の版であって、.onnx の中身までは表さない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "common" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "image-pipeline" / "src"))

from nihcc_rag import db  # noqa: E402
from nihcc_rag import logging as jsonlog  # noqa: E402

#: model_version.model_type の CHECK（db/metadata-migrations/）と揃えること。
TYPES = ("ROI", "EMBEDDING", "EMBEDDING_FINE", "REGION_MIL", "TRANSLATION",
         "LUNG_SEGMENTATION")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_onnx(path: Path) -> dict:
    """ONNX なら入力の形を読む。**モデルは自分の入力の大きさを知っている。**"""
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        return {
            "input_shape": [d if isinstance(d, int) else str(d) for d in shape],
            "output_shape": [d if isinstance(d, int) else str(d)
                             for d in session.get_outputs()[0].shape],
        }
    except Exception:  # noqa: BLE001 - 読めなくても登録はできる
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", required=True, choices=TYPES)
    parser.add_argument("--path", type=Path, required=True, help="成果物のファイル")
    parser.add_argument("--version", required=True, help="版の名前")
    parser.add_argument("--name", default="", help="モデル名（既定はファイル名）")
    parser.add_argument("--source-revision", default="", help="学習元の revision など")
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--activate", action="store_true",
                        help="同じ型の active を落として、これを active にする")
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"成果物が無い: {args.path}", file=sys.stderr)
        return 1

    jsonlog.configure("register-model-artifact")
    db.init_metadata_pool(role="app_meta", application_name="register-model-artifact")

    digest = sha256_of(args.path)
    config = {
        "artifact_path": str(args.path),
        "size_bytes": args.path.stat().st_size,
        **(describe_onnx(args.path) if args.path.suffix == ".onnx" else {}),
    }

    with db.metadata_conn() as conn:
        with conn.transaction():
            if args.activate:
                db.execute(
                    conn,
                    "UPDATE model_version SET active = false WHERE model_type = %s",
                    (args.type,),
                )
            db.execute(
                conn,
                """
                INSERT INTO model_version
                  (model_version, model_type, model_name, source_revision,
                   artifact_sha256, embedding_dimension, config, active)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (model_version) DO UPDATE
                  SET artifact_sha256 = EXCLUDED.artifact_sha256,
                      config = EXCLUDED.config,
                      active = EXCLUDED.active
                """,
                (args.version, args.type, args.name or args.path.name,
                 args.source_revision, digest, args.dimension,
                 json.dumps(config, ensure_ascii=False), args.activate),
            )

    print(f"登録した: {args.version} ({args.type})")
    print(f"  ファイル : {args.path}")
    print(f"  SHA-256  : {digest}")
    if config.get("input_shape"):
        print(f"  入力      : {config['input_shape']}")
    if args.activate:
        print(f"  **active** にした（同じ型の他の版は false）")
    db.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
