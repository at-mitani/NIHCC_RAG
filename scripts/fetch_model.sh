#!/usr/bin/env bash
# 埋め込みモデルの取得と検証（設計書 7.3.4）。
#
# インターネット接続可能な staging 端末で一度実行し、SHA256SUMS とともに
# model volume（または隔離環境への持ち込み用アーカイブ）を作成する。
# 実行後、compose の embedding-service では HF_HUB_OFFLINE=1 に切り替え、
# モデルディレクトリを read-only mount する運用とする。
#
# 取得対象は 2 つある（2 段検索の coarse / fine）。
#   coarse: EMBEDDING_BACKEND / EMBEDDING_MODEL_ID / EMBEDDING_MODEL_REVISION
#   fine  : EMBEDDING_FINE_BACKEND / EMBEDDING_FINE_MODEL_ID /
#           EMBEDDING_FINE_MODEL_REVISION（EMBEDDING_FINE_BACKEND=none なら取得しない）
#
# モデル ID と revision は .env から compose 経由でコンテナへ渡して使う
# （ホストのシェルは .env を読まないため）。
set -euo pipefail

cd "$(dirname "$0")/.."

# 環境変数は必ず -e でサービス名より前に渡す。bash -c "..." の後ろに書くと
# 位置引数（$0, $1）になり、コンテナ内の環境変数には入らない。
# .env が HF_HUB_OFFLINE=1 でも取得できるよう、この実行だけ online にする。
docker compose run --rm --no-deps \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  --entrypoint bash embedding-service -c "
  set -euo pipefail
  python - <<'PY'
import os

from huggingface_hub import snapshot_download

DEFAULT_COARSE = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
#: backend 種別ごとの既定リポジトリ（app/backends.py の _DEFAULT_FINE_MODEL_ID と対応）
DEFAULT_FINE = {'rad-dino': 'microsoft/rad-dino', 'biomedclip': DEFAULT_COARSE}
DISABLED = {'', 'none', 'off', 'disabled', '0', 'false'}


def fetch(role, backend, model_id, revision):
    '''1 モデルを取得し、サービスと同じ経路で一度ロードして検証する。'''
    repo_id = model_id.removeprefix('hf-hub:')
    print()
    print(f'=== {role} ({backend}) ===')
    print('model    :', repo_id)
    print('revision :', revision or '<未固定>')

    # cache_dir は指定しない。HF_HOME=/models/hf-cache から導かれる
    # /models/hf-cache/hub がサービス実行時の参照先と一致する。
    path = snapshot_download(repo_id=repo_id, revision=revision or None)
    print('downloaded to', path)
    commit = os.path.basename(path)
    print('commit   :', commit)

    if backend == 'biomedclip':
        # open_clip は text tower の設定とトークナイザを別リポジトリから取得する
        # ため、snapshot_download だけでは HF_HUB_OFFLINE=1 で起動できない。
        import open_clip

        open_clip.create_model_and_transforms(model_id)
        open_clip.get_tokenizer(model_id)
        print('open_clip load: ok')
    elif backend == 'rad-dino':
        # RAD-DINO は transformers 経由。processor の設定ファイルも cache へ落とす。
        from transformers import AutoImageProcessor, AutoModel

        kwargs = {'revision': revision} if revision else {}
        AutoImageProcessor.from_pretrained(repo_id, **kwargs)
        model = AutoModel.from_pretrained(repo_id, **kwargs)
        dim = int(model.config.hidden_size)
        print('transformers load: ok')
        print(f'出力次元 : {dim}  -> EMBEDDING_FINE_DIMENSION={dim} を設定すること')
    else:
        raise SystemExit(f'unsupported backend for {role}: {backend}')
    return commit


coarse_backend = os.environ.get('EMBEDDING_BACKEND', 'biomedclip').strip().lower()
commits = {}
if coarse_backend == 'deterministic':
    print('EMBEDDING_BACKEND=deterministic のため coarse の取得は不要です')
else:
    commits['EMBEDDING_MODEL_REVISION'] = fetch(
        'coarse',
        coarse_backend,
        os.environ.get('EMBEDDING_MODEL_ID') or DEFAULT_COARSE,
        (os.environ.get('EMBEDDING_MODEL_REVISION') or '').strip(),
    )

fine_backend = os.environ.get('EMBEDDING_FINE_BACKEND', 'none').strip().lower()
if fine_backend == 'chest-profile':
    # 胸部野の分類器そのものは infra/chest の成果物で、取得するのは backbone（RAD-DINO）。
    # revision は分類器を学習したときのものに固定する（EMBEDDING_FINE_MODEL_REVISION）。
    fine_backend = os.environ.get('EMBEDDING_FINE_BACKBONE', 'rad-dino').strip().lower()
    print()
    print(f'EMBEDDING_FINE_BACKEND=chest-profile: backbone の {fine_backend} を取得します')
if fine_backend in DISABLED:
    print()
    print('EMBEDDING_FINE_BACKEND が未設定のため、2 段目モデルは取得しません')
elif fine_backend == 'deterministic':
    print()
    print('EMBEDDING_FINE_BACKEND=deterministic のため取得は不要です')
else:
    commits['EMBEDDING_FINE_MODEL_REVISION'] = fetch(
        'fine',
        fine_backend,
        os.environ.get('EMBEDDING_FINE_MODEL_ID')
        or DEFAULT_FINE.get(fine_backend, DEFAULT_COARSE),
        (os.environ.get('EMBEDDING_FINE_MODEL_REVISION') or '').strip(),
    )

print()
print('--- .env へ設定する revision ---')
for key, value in commits.items():
    print(f'{key}={value}')
PY
  find /models/hf-cache -type f ! -name SHA256SUMS ! -name SHA256SUMS.tmp -exec sha256sum {} + \
    | sort -k2 > /models/hf-cache/SHA256SUMS.tmp
  mv /models/hf-cache/SHA256SUMS.tmp /models/hf-cache/SHA256SUMS
  echo \"files hashed: \$(wc -l < /models/hf-cache/SHA256SUMS)\"
" || {
  echo "取得に失敗しました。ネットワークとプロキシ設定を確認してください。" >&2
  exit 1
}

echo
echo "完了しました。次を実施してください:"
echo "  1) /models/hf-cache/SHA256SUMS を管理台帳へ登録する"
echo "  2) .env の *_MODEL_REVISION へ上に表示された commit を設定する"
echo "  3) chest-profile の場合、EMBEDDING_FINE_MODEL_REVISION は分類器を学習したときの"
echo "     RAD-DINO の版のままにする（上の表示と違えば、取得した版が分類器と合っていない）"
echo "  4) .env の HF_HUB_OFFLINE=1 に変更して再起動する"
