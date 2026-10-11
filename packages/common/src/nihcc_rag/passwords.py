"""パスワードのハッシュ化と入力規則（機能追加: ユーザー管理）。

- 平文パスワードは保存も出力もしない
- KDF は標準ライブラリの scrypt（メモリハード）を使う
- 保存形式: `scrypt$<n>$<r>$<p>$<base64 salt>$<base64 derived key>`
  パラメータを保存形式へ含めるため、後からコスト係数を上げても既存の
  ハッシュを検証できる
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import unicodedata

from . import problem

#: 入力規則（機能要件: 8 文字以上 128 文字以下）
MIN_LENGTH = 8
MAX_LENGTH = 128

#: scrypt パラメータ。RFC 7914 の推奨に沿った対話用途向けの値。
_N = 2**14
_R = 8
_P = 1
_DKLEN = 32
_SALT_BYTES = 16
_ALGORITHM = "scrypt"


def normalize(password: str) -> str:
    """Unicode 正規化のみ行う。空白は利用者の意図として保持する。"""
    return unicodedata.normalize("NFKC", password)


def validate_policy(password: str, *, field: str = "password") -> str:
    """長さ規則を検証し、正規化済みパスワードを返す。"""
    candidate = normalize(password)
    if len(candidate) < MIN_LENGTH or len(candidate) > MAX_LENGTH:
        raise problem.unprocessable(
            "PASSWORD_POLICY_VIOLATION",
            f"Enter a password of {MIN_LENGTH} to {MAX_LENGTH} characters.",
            errors=[{"field": field, "reason": f"length must be {MIN_LENGTH}..{MAX_LENGTH}"}],
            params={"min": MIN_LENGTH, "max": MAX_LENGTH},
        )
    return candidate


def validate_confirmation(password: str, confirmation: str, *, field: str = "password") -> str:
    """入力間違い防止の 2 回入力を検証する。"""
    candidate = validate_policy(password, field=field)
    if not hmac.compare_digest(candidate, normalize(confirmation)):
        raise problem.unprocessable(
            "PASSWORD_CONFIRMATION_MISMATCH",
            "The confirmation password does not match.",
            errors=[{"field": f"{field}_confirmation", "reason": "does not match"}],
        )
    return candidate


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        normalize(password).encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN
    )
    return "$".join(
        [
            _ALGORITHM,
            str(_N),
            str(_R),
            str(_P),
            base64.b64encode(salt).decode(),
            base64.b64encode(derived).decode(),
        ]
    )


def verify_password(password: str, encoded: str) -> bool:
    """定数時間比較で検証する。形式不正は False（例外を投げない）。"""
    try:
        algorithm, n, r, p, salt_b64, hash_b64 = encoded.split("$")
        if algorithm != _ALGORITHM:
            return False
        derived = hashlib.scrypt(
            normalize(password).encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(base64.b64decode(hash_b64)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, base64.b64decode(hash_b64))


def dummy_verify() -> None:
    """存在しないユーザーでも同等の計算量を消費し、応答時間で存在を推測させない。"""
    hashlib.scrypt(os.urandom(16), salt=os.urandom(_SALT_BYTES), n=_N, r=_R, p=_P, dklen=_DKLEN)


def needs_rehash(encoded: str) -> bool:
    """保存済みハッシュが現行のコスト係数より弱い場合に True。"""
    try:
        algorithm, n, r, p, _salt, _hash = encoded.split("$")
    except ValueError:
        return True
    return algorithm != _ALGORITHM or (int(n), int(r), int(p)) != (_N, _R, _P)
