"""ユーザー管理とパスワード運用（機能追加）。

- 管理者によるユーザー一覧・詳細・追加・編集・削除
- 全利用者によるパスワード変更（既存パスワード照合 + 新パスワード 2 回入力）
- パスワード忘れによる再設定（メール送付・10 分有効・1 回限りの URL）

方針:
- パスワードは KDF 派生値のみ保存し、平文は保存も応答もログにも出さない
- 再設定要求はメールアドレスの存在有無で応答を変えない（利用者列挙の防止）
- URL に載せる UUID は DB へ平文保存せず SHA-256 のみ保持する
- 最後の有効な管理者を失う操作は拒否する（締め出しの防止）
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from typing import Any

from nihcc_rag import auth, config, db, mail, passwords, problem

_LOG = logging.getLogger("backend-api.users")

#: 仮パスワード変更 URL の有効時間（分）
RESET_TOKEN_TTL_MINUTES = 10
#: 同一利用者が短時間に要求できる回数（総当たり・メール爆撃の抑止）
RESET_REQUEST_LIMIT = 3
RESET_REQUEST_WINDOW_MINUTES = 10

_USER_COLUMNS = (
    "user_id, username, role, email, memo, is_active, password_changed_at, "
    "last_login_at, created_by, updated_by, created_at, updated_at"
)


def _token_hash(token: str) -> str:
    return hashlib.sha256(f"password-reset:{token}".encode()).hexdigest()


def _row(record: dict[str, Any]) -> dict[str, Any]:
    """DB 行を API 応答向けへ変換する。password_hash は決して含めない。"""
    return {key: value for key, value in record.items() if key != "password_hash"}


# --- 一覧・詳細 ---------------------------------------------------------------


def list_users(*, include_inactive: bool = True) -> list[dict[str, Any]]:
    condition = "" if include_inactive else "WHERE is_active"
    with db.metadata_conn() as conn:
        rows = db.fetch_all(
            conn,
            f"SELECT {_USER_COLUMNS} FROM app_user {condition} ORDER BY username",
        )
    return [_row(row) for row in rows]


def get_user(user_id: uuid.UUID) -> dict[str, Any]:
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn, f"SELECT {_USER_COLUMNS} FROM app_user WHERE user_id = %s", (str(user_id),)
        )
    if row is None:
        raise problem.not_found("user")
    return _row(row)


def _active_admin_count(conn, *, excluding: str | None = None) -> int:
    row = db.fetch_one(
        conn,
        "SELECT count(*) AS count FROM app_user "
        "WHERE role = 'admin' AND is_active AND (%s::uuid IS NULL OR user_id <> %s::uuid)",
        (excluding, excluding),
    )
    return int(row["count"]) if row else 0


def _conflict(exc: Exception) -> problem.ProblemError:
    message = str(exc).lower()
    if "app_user_email_lower_key" in message or "email" in message:
        return problem.ProblemError(
            409,
            "EMAIL_ALREADY_EXISTS",
            "Conflict",
            "このメールアドレスは既に登録されています。",
            errors=[{"field": "email", "reason": "already exists"}],
        )
    return problem.ProblemError(
        409,
        "USERNAME_ALREADY_EXISTS",
        "Conflict",
        "このアカウント名は既に使用されています。",
        errors=[{"field": "username", "reason": "already exists"}],
    )


# --- 追加・編集・削除 ---------------------------------------------------------


def create_user(payload, *, actor: str) -> dict[str, Any]:
    import psycopg

    password = passwords.validate_confirmation(payload.password, payload.password_confirmation)
    try:
        with db.metadata_conn() as conn:
            row = db.fetch_one(
                conn,
                "INSERT INTO app_user "
                "  (username, password_hash, role, email, memo, is_active, created_by, updated_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                f"RETURNING {_USER_COLUMNS}",
                (
                    payload.username,
                    passwords.hash_password(password),
                    payload.role,
                    str(payload.email),
                    payload.memo,
                    payload.is_active,
                    actor,
                    actor,
                ),
            )
    except psycopg.errors.UniqueViolation as exc:
        raise _conflict(exc) from exc
    _LOG.info("user created", extra={"fields": {"username": payload.username, "role": payload.role}})
    return _row(row)


def update_user(user_id: uuid.UUID, payload, *, actor: str, actor_user_id: str | None) -> dict[str, Any]:
    import psycopg

    updates: list[str] = []
    params: dict[str, Any] = {"user_id": str(user_id), "actor": actor}

    with db.metadata_conn() as conn:
        current = db.fetch_one(
            conn,
            "SELECT user_id, username, role, is_active FROM app_user WHERE user_id = %s",
            (str(user_id),),
        )
        if current is None:
            raise problem.not_found("user")

        # 有効な管理者が 0 人になる変更を拒否する
        losing_admin = current["role"] == "admin" and current["is_active"] and (
            (payload.role is not None and payload.role != "admin")
            or (payload.is_active is False)
        )
        if losing_admin and _active_admin_count(conn, excluding=str(user_id)) == 0:
            raise problem.unprocessable(
                "LAST_ADMIN_PROTECTED",
                "有効な管理者が 0 人になる変更はできません。先に別の管理者を用意してください。",
            )

    if payload.role is not None:
        updates.append("role = %(role)s")
        params["role"] = payload.role
    if payload.email is not None:
        updates.append("email = %(email)s")
        params["email"] = str(payload.email)
    if payload.memo is not None:
        updates.append("memo = %(memo)s")
        params["memo"] = payload.memo
    if payload.is_active is not None:
        updates.append("is_active = %(is_active)s")
        params["is_active"] = payload.is_active
    if payload.password is not None:
        password = passwords.validate_confirmation(payload.password, payload.password_confirmation)
        updates.append("password_hash = %(password_hash)s")
        updates.append("password_changed_at = now()")
        params["password_hash"] = passwords.hash_password(password)

    if not updates:
        raise problem.unprocessable("NO_FIELDS_TO_UPDATE", "更新する項目がありません。")

    updates.append("updated_by = %(actor)s")
    updates.append("updated_at = now()")

    try:
        with db.metadata_conn() as conn:
            row = db.fetch_one(
                conn,
                f"UPDATE app_user SET {', '.join(updates)} WHERE user_id = %(user_id)s "
                f"RETURNING {_USER_COLUMNS}",
                params,
            )
            if payload.password is not None:
                # 管理者によるパスワード変更は、未使用の再設定 token も無効化する
                db.execute(
                    conn,
                    "UPDATE password_reset_token SET invalidated_at = now() "
                    "WHERE user_id = %s AND used_at IS NULL AND invalidated_at IS NULL",
                    (str(user_id),),
                )
    except psycopg.errors.UniqueViolation as exc:
        raise _conflict(exc) from exc

    auth.invalidate_cache(str(user_id))
    _LOG.info(
        "user updated",
        extra={"fields": {"user_id": str(user_id), "fields": sorted(params.keys())}},
    )
    return _row(row)


def delete_user(user_id: uuid.UUID, *, actor_user_id: str | None) -> None:
    with db.metadata_conn() as conn:
        current = db.fetch_one(
            conn,
            "SELECT user_id, username, role, is_active FROM app_user WHERE user_id = %s",
            (str(user_id),),
        )
        if current is None:
            raise problem.not_found("user")
        if actor_user_id is not None and str(user_id) == actor_user_id:
            raise problem.unprocessable(
                "SELF_DELETE_FORBIDDEN", "自分自身のアカウントは削除できません。"
            )
        if current["role"] == "admin" and _active_admin_count(conn, excluding=str(user_id)) == 0:
            raise problem.unprocessable(
                "LAST_ADMIN_PROTECTED",
                "最後の管理者アカウントは削除できません。先に別の管理者を用意してください。",
            )
        db.execute(conn, "DELETE FROM app_user WHERE user_id = %s", (str(user_id),))
    auth.invalidate_cache(str(user_id))
    _LOG.info("user deleted", extra={"fields": {"user_id": str(user_id)}})


# --- パスワード変更（本人） ---------------------------------------------------


def change_own_password(user_id: str, payload) -> None:
    new_password = passwords.validate_confirmation(
        payload.new_password, payload.new_password_confirmation, field="new_password"
    )
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn, "SELECT password_hash FROM app_user WHERE user_id = %s", (user_id,)
        )
    if row is None:
        raise problem.not_found("user")
    if not passwords.verify_password(payload.current_password, row["password_hash"]):
        raise problem.unprocessable(
            "CURRENT_PASSWORD_MISMATCH",
            "現在のパスワードが正しくありません。",
            errors=[{"field": "current_password", "reason": "mismatch"}],
        )
    if passwords.verify_password(new_password, row["password_hash"]):
        raise problem.unprocessable(
            "PASSWORD_UNCHANGED",
            "新しいパスワードが現在のパスワードと同じです。",
            errors=[{"field": "new_password", "reason": "must differ from current password"}],
        )

    with db.metadata_conn() as conn:
        with conn.transaction():
            db.execute(
                conn,
                "UPDATE app_user SET password_hash = %s, password_changed_at = now(), "
                "updated_at = now(), updated_by = username WHERE user_id = %s",
                (passwords.hash_password(new_password), user_id),
            )
            db.execute(
                conn,
                "UPDATE password_reset_token SET invalidated_at = now() "
                "WHERE user_id = %s AND used_at IS NULL AND invalidated_at IS NULL",
                (user_id,),
            )
    auth.invalidate_cache(user_id)


# --- パスワード忘れによる再設定 -----------------------------------------------


def reset_url(token: str) -> str:
    base = config.env("APP_BASE_URL", "http://localhost:8080").rstrip("/")
    path = config.env("PASSWORD_RESET_PATH", "/reset").strip("/")
    return f"{base}/{path}?token={token}"


def request_password_reset(email: str, *, client_ip: str | None) -> None:
    """再設定 URL をメール送付する。

    メールアドレスの存在有無で応答も所要時間も変えない（利用者列挙の防止）。
    呼び出し側は常に 202 を返すこと。
    """
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn,
            "SELECT user_id, email, is_active FROM app_user WHERE lower(email) = lower(%s)",
            (email,),
        )
        if row is None or not row["is_active"]:
            _LOG.info("password reset requested for unknown or inactive address")
            return

        recent = db.fetch_one(
            conn,
            "SELECT count(*) AS count FROM password_reset_token "
            "WHERE user_id = %s AND created_at > now() - make_interval(mins => %s)",
            (row["user_id"], RESET_REQUEST_WINDOW_MINUTES),
        )
        if recent and recent["count"] >= RESET_REQUEST_LIMIT:
            _LOG.warning(
                "password reset rate limit reached",
                extra={"fields": {"user_id": str(row["user_id"])}},
            )
            return

        token = str(uuid.uuid4())
        db.execute(
            conn,
            "INSERT INTO password_reset_token "
            "  (token_hash, user_id, expires_at, requested_by_ip) "
            "VALUES (%s, %s, now() + make_interval(mins => %s), %s)",
            (_token_hash(token), row["user_id"], RESET_TOKEN_TTL_MINUTES, client_ip),
        )

    try:
        mail.send_password_reset(
            to=row["email"], reset_url=reset_url(token), valid_minutes=RESET_TOKEN_TTL_MINUTES
        )
    except mail.MailError:
        # 送信失敗時は token を無効化する（宛先へ届いていない URL を残さない）
        with db.metadata_conn() as conn:
            db.execute(
                conn,
                "UPDATE password_reset_token SET invalidated_at = now() WHERE token_hash = %s",
                (_token_hash(token),),
            )
        _LOG.error("password reset mail delivery failed", extra={"fields": {"code": "MAIL_FAILED"}})


def confirm_password_reset(payload) -> str:
    """token を検証して新パスワードを設定し、対象の username を返す。"""
    new_password = passwords.validate_confirmation(
        payload.new_password, payload.new_password_confirmation, field="new_password"
    )
    token_hash = _token_hash(str(payload.token))

    with db.metadata_conn() as conn:
        with conn.transaction():
            # 同時使用を避けるため行ロックを取り、有効性を確認する
            row = db.fetch_one(
                conn,
                "SELECT t.token_hash, t.user_id, t.expires_at, t.used_at, t.invalidated_at, "
                "       u.username, u.is_active "
                "FROM password_reset_token t JOIN app_user u USING (user_id) "
                "WHERE t.token_hash = %s FOR UPDATE OF t",
                (token_hash,),
            )
            if (
                row is None
                or row["used_at"] is not None
                or row["invalidated_at"] is not None
                or not row["is_active"]
            ):
                raise problem.unprocessable(
                    "RESET_TOKEN_INVALID",
                    "この URL は無効です。パスワード再設定をやり直してください。",
                )

            expired = db.fetch_one(
                conn,
                "SELECT (expires_at < now()) AS expired FROM password_reset_token "
                "WHERE token_hash = %s",
                (token_hash,),
            )
            if expired and expired["expired"]:
                raise problem.unprocessable(
                    "RESET_TOKEN_EXPIRED",
                    f"この URL は有効期限（{RESET_TOKEN_TTL_MINUTES} 分）を過ぎています。"
                    "パスワード再設定をやり直してください。",
                )

            db.execute(
                conn,
                "UPDATE app_user SET password_hash = %s, password_changed_at = now(), "
                "updated_at = now(), updated_by = 'password-reset' WHERE user_id = %s",
                (passwords.hash_password(new_password), row["user_id"]),
            )
            db.execute(
                conn,
                "UPDATE password_reset_token SET used_at = now() WHERE token_hash = %s",
                (token_hash,),
            )
            # 同一利用者の他の未使用 token も無効化する
            db.execute(
                conn,
                "UPDATE password_reset_token SET invalidated_at = now() "
                "WHERE user_id = %s AND used_at IS NULL AND invalidated_at IS NULL",
                (row["user_id"],),
            )

    auth.invalidate_cache(str(row["user_id"]))
    _LOG.info("password reset completed", extra={"fields": {"user_id": str(row["user_id"])}})
    return row["username"]


def purge_expired_tokens() -> int:
    """期限切れ token を削除する（常駐ジョブから定期実行）。"""
    with db.metadata_conn() as conn:
        return db.execute(
            conn,
            "DELETE FROM password_reset_token "
            "WHERE expires_at < now() - interval '1 day' OR used_at < now() - interval '1 day'",
        )


def generate_initial_password() -> str:
    """管理画面の新規追加でパスワード候補を提示するための生成器。"""
    return secrets.token_urlsafe(16)
