"""ローカル IdP 相当の認証・JWT・RBAC（設計書 9.3 / 13）。

ユーザーは Metadata DB の `app_user` で管理する。パスワードは KDF 派生値のみを
保持し、平文は保存も出力もしない（`passwords` モジュール）。

初回起動時に利用者が 1 件も存在しない場合に限り、`LOCAL_USERS` 環境変数から
初期管理者を作成する（ブートストラップ）。以後の管理は管理画面／API で行う。

本番相当環境では組織 OIDC IdP に置き換え、公開鍵検証へ差し替える。
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass

import jwt

from . import config, db, passwords, problem

_LOG = logging.getLogger(__name__)

ROLES = ("viewer", "operator", "admin")
#: 上位 role は下位 role の権限を含む
_ROLE_RANK = {name: index for index, name in enumerate(ROLES)}

ALGORITHM = "HS256"

#: 認可判定のためのユーザー情報キャッシュ。無効化・権限変更・パスワード変更を
#: 短時間で反映しつつ、リクエストごとの DB 参照を避ける。
_CACHE_TTL_SECONDS = 30.0
_user_cache: dict[str, tuple[float, "AccountUser"]] = {}
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class AccountUser:
    user_id: str
    username: str
    role: str
    email: str
    is_active: bool
    password_changed_at: float

    def has_role(self, required: str) -> bool:
        return _ROLE_RANK[self.role] >= _ROLE_RANK[required]


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    user_id: str | None = None
    email: str | None = None

    def has_role(self, required: str) -> bool:
        return _ROLE_RANK[self.role] >= _ROLE_RANK[required]


def _row_to_user(row: dict) -> AccountUser:
    return AccountUser(
        user_id=str(row["user_id"]),
        username=row["username"],
        role=row["role"],
        email=row["email"],
        is_active=bool(row["is_active"]),
        password_changed_at=row["password_changed_at"].timestamp(),
    )


def invalidate_cache(user_id: str | None = None) -> None:
    with _cache_lock:
        if user_id is None:
            _user_cache.clear()
        else:
            _user_cache.pop(user_id, None)


def load_user(user_id: str, *, use_cache: bool = True) -> AccountUser | None:
    if use_cache:
        with _cache_lock:
            entry = _user_cache.get(user_id)
            if entry is not None and entry[0] > time.monotonic():
                return entry[1]
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn,
            "SELECT user_id, username, role, email, is_active, password_changed_at "
            "FROM app_user WHERE user_id = %s",
            (user_id,),
        )
    if row is None:
        return None
    user = _row_to_user(row)
    with _cache_lock:
        _user_cache[user_id] = (time.monotonic() + _CACHE_TTL_SECONDS, user)
    return user


# --- ブートストラップ ---------------------------------------------------------


def _parse_local_users() -> list[tuple[str, str, str]]:
    """LOCAL_USERS 環境変数（user:password:role のカンマ区切り）を解析する。"""
    raw = os.environ.get("LOCAL_USERS", "")
    entries: list[tuple[str, str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3 or parts[2] not in ROLES:
            raise RuntimeError("LOCAL_USERS entry must be 'user:password:viewer|operator|admin'")
        entries.append((parts[0], parts[1], parts[2]))
    return entries


def bootstrap_users() -> int:
    """利用者が 0 件のときだけ LOCAL_USERS から初期ユーザーを作成する。"""
    created = 0
    with db.metadata_conn() as conn:
        existing = db.fetch_one(conn, "SELECT count(*) AS count FROM app_user")
        if existing and existing["count"] > 0:
            return 0

        domain = config.env("BOOTSTRAP_EMAIL_DOMAIN", "local.example")
        for username, password, role in _parse_local_users():
            db.execute(
                conn,
                "INSERT INTO app_user "
                "  (username, password_hash, role, email, memo, created_by, updated_by) "
                "VALUES (%s, %s, %s, %s, %s, 'bootstrap', 'bootstrap') "
                "ON CONFLICT (username) DO NOTHING",
                (
                    username,
                    passwords.hash_password(password),
                    role,
                    f"{username}@{domain}",
                    "LOCAL_USERS から自動作成された初期ユーザー。"
                    "運用開始前にメールアドレスとパスワードを更新してください。",
                ),
            )
            created += 1
    if created:
        _LOG.warning(
            "bootstrapped initial users from LOCAL_USERS; "
            "update their e-mail addresses and passwords before operational use",
            extra={"fields": {"count": created}},
        )
    return created


# --- 認証 --------------------------------------------------------------------


def authenticate(username: str, password: str) -> AccountUser:
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn,
            "SELECT user_id, username, role, email, is_active, password_changed_at, "
            "password_hash FROM app_user WHERE username = %s",
            (username,),
        )
    if row is None:
        # 存在しない利用者でも同等の計算量を消費し、応答時間差で存在を推測させない
        passwords.dummy_verify()
        raise problem.unauthorized("ユーザー名またはパスワードが正しくありません。")
    if not passwords.verify_password(password, row["password_hash"]):
        raise problem.unauthorized("ユーザー名またはパスワードが正しくありません。")
    if not row["is_active"]:
        raise problem.unauthorized(
            "このアカウントは無効化されています。管理者へ連絡してください。"
        )

    with db.metadata_conn() as conn:
        db.execute(
            conn,
            "UPDATE app_user SET last_login_at = now() WHERE user_id = %s",
            (row["user_id"],),
        )
        # コスト係数を引き上げた場合はログイン時に再ハッシュする
        if passwords.needs_rehash(row["password_hash"]):
            db.execute(
                conn,
                "UPDATE app_user SET password_hash = %s WHERE user_id = %s",
                (passwords.hash_password(password), row["user_id"]),
            )
    return _row_to_user(row)


def issue_token(user: AccountUser) -> tuple[str, int]:
    ttl = config.env_int("JWT_TTL_SECONDS", 28800)
    issued_at = time.time()
    now = int(issued_at)
    payload = {
        "iss": config.env("JWT_ISSUER", "nihcc-rag-local"),
        "aud": config.env("JWT_AUDIENCE", "nihcc-rag-api"),
        "sub": user.username,
        "uid": user.user_id,
        "role": user.role,
        "iat": now,
        # 標準の iat は秒単位へ切り捨てられ、パスワード変更時刻との前後判定が
        # 同一秒内で狂う。失効判定にはミリ秒精度の発行時刻を使う。
        "iat_ms": round(issued_at, 3),
        "nbf": now,
        "exp": now + ttl,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, config.jwt_secret(), algorithm=ALGORITHM), ttl


def verify_token(token: str) -> Principal:
    """署名検証に加え、利用者の有効性・権限・パスワード変更時刻を確認する。

    パスワード変更・無効化・権限変更を、発行済みトークンへも反映させる。
    """
    try:
        payload = jwt.decode(
            token,
            config.jwt_secret(),
            algorithms=[ALGORITHM],
            audience=config.env("JWT_AUDIENCE", "nihcc-rag-api"),
            issuer=config.env("JWT_ISSUER", "nihcc-rag-local"),
        )
    except jwt.ExpiredSignatureError:
        raise problem.unauthorized("アクセストークンの有効期限が切れています。") from None
    except jwt.InvalidTokenError:
        raise problem.unauthorized("アクセストークンが不正です。") from None

    user_id = payload.get("uid")
    if not user_id:
        raise problem.unauthorized("アクセストークンに利用者 ID が含まれていません。")

    user = load_user(str(user_id))
    if user is None:
        raise problem.unauthorized("利用者が存在しません。")
    if not user.is_active:
        raise problem.unauthorized("このアカウントは無効化されています。")
    # パスワード変更前に発行されたトークンは失効させる。
    # 秒単位の iat では同一秒内の前後を判定できないため iat_ms を使う
    # （iat_ms を持たないトークンは iat + 1 秒で保守的に判定する）。
    issued_at = float(payload.get("iat_ms") or (float(payload.get("iat", 0)) + 1))
    if issued_at < user.password_changed_at:
        raise problem.unauthorized("パスワードが変更されたため、再ログインが必要です。")
    if user.role not in ROLES:
        raise problem.unauthorized("利用者の権限が不正です。")

    return Principal(
        subject=user.username, role=user.role, user_id=user.user_id, email=user.email
    )
