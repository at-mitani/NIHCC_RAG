"""ユーザー管理 API（機能追加）。

管理者のみ: 一覧・詳細・追加・編集・削除
全利用者   : 自身のパスワード変更
未認証     : パスワード忘れによる再設定の要求・確定
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Query, Request, Response
from nihcc_rag import auth as auth_module
from nihcc_rag.contract import (
    AcceptedResponse,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    UserCreateRequest,
    UserDetail,
    UserListResponse,
    UserSummary,
    UserUpdateRequest,
)

from ..deps import Admin, Viewer, audit
from ..services import users as user_service

_LOG = logging.getLogger("backend-api.users")

router = APIRouter(prefix="/api/v1/users", tags=["users"])
account_router = APIRouter(prefix="/api/v1/account", tags=["account"])
public_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else None


# --- 管理者によるユーザー管理 -------------------------------------------------


@router.get("", response_model=UserListResponse)
def list_users(
    principal: Admin,
    include_inactive: bool = Query(default=True, description="Include deactivated users"),
) -> UserListResponse:
    rows = user_service.list_users(include_inactive=include_inactive)
    return UserListResponse(users=[UserSummary(**row) for row in rows], total=len(rows))


@router.post("", response_model=UserDetail, status_code=201)
def create_user(payload: UserCreateRequest, principal: Admin, request: Request) -> UserDetail:
    created = user_service.create_user(payload, actor=principal.subject)
    audit(
        principal.subject,
        "USER_CREATE",
        target_type="app_user",
        target_id=str(created["user_id"]),
        trace_id=getattr(request.state, "trace_id", None),
        # パスワードは監査ログにも残さない
        detail={"username": created["username"], "role": created["role"]},
    )
    return UserDetail(**created)


@router.get("/{user_id}", response_model=UserDetail)
def get_user(user_id: uuid.UUID, principal: Admin) -> UserDetail:
    return UserDetail(**user_service.get_user(user_id))


@router.patch("/{user_id}", response_model=UserDetail)
def update_user(
    user_id: uuid.UUID, payload: UserUpdateRequest, principal: Admin, request: Request
) -> UserDetail:
    updated = user_service.update_user(
        user_id, payload, actor=principal.subject, actor_user_id=principal.user_id
    )
    audit(
        principal.subject,
        "USER_UPDATE",
        target_type="app_user",
        target_id=str(user_id),
        trace_id=getattr(request.state, "trace_id", None),
        detail={
            "username": updated["username"],
            "changed": [
                name
                for name in ("role", "email", "memo", "is_active")
                if getattr(payload, name) is not None
            ]
            + (["password"] if payload.password is not None else []),
        },
    )
    return UserDetail(**updated)


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: uuid.UUID, principal: Admin, request: Request) -> Response:
    target = user_service.get_user(user_id)
    user_service.delete_user(user_id, actor_user_id=principal.user_id)
    audit(
        principal.subject,
        "USER_DELETE",
        target_type="app_user",
        target_id=str(user_id),
        trace_id=getattr(request.state, "trace_id", None),
        detail={"username": target["username"], "role": target["role"]},
    )
    return Response(status_code=204)


# --- 本人によるパスワード変更 -------------------------------------------------


@account_router.get("/me", response_model=UserDetail)
def me(principal: Viewer) -> UserDetail:
    if principal.user_id is None:
        raise_not_found()
    return UserDetail(**user_service.get_user(uuid.UUID(principal.user_id)))


def raise_not_found() -> None:
    from nihcc_rag import problem

    raise problem.not_found("user")


@account_router.post("/password", response_model=AcceptedResponse)
def change_password(
    payload: PasswordChangeRequest, principal: Viewer, request: Request
) -> AcceptedResponse:
    if principal.user_id is None:
        raise_not_found()
    user_service.change_own_password(principal.user_id, payload)
    audit(
        principal.subject,
        "PASSWORD_CHANGE",
        target_type="app_user",
        target_id=principal.user_id,
        trace_id=getattr(request.state, "trace_id", None),
    )
    return AcceptedResponse(
        status="succeeded",
        detail="Your password was changed. For security, please sign in again.",
        message_id="PASSWORD_CHANGED",
    )


# --- パスワード忘れによる再設定（未認証） -------------------------------------


@public_router.post("/password-reset/request", response_model=AcceptedResponse, status_code=202)
def request_reset(payload: PasswordResetRequest, request: Request) -> AcceptedResponse:
    """常に 202 を返す。メールアドレスの登録有無を応答から判別させない。"""
    user_service.request_password_reset(str(payload.email), client_ip=_client_ip(request))
    audit(
        "anonymous",
        "PASSWORD_RESET_REQUEST",
        target_type="email",
        # メールアドレスそのものは監査ログへ残さない
        target_id=None,
        trace_id=getattr(request.state, "trace_id", None),
    )
    return AcceptedResponse(
        detail=(
            "If the email address you entered is registered, a URL for resetting your password "
            f"has been sent. The URL expires in {user_service.RESET_TOKEN_TTL_MINUTES} minutes."
        ),
        message_id="PASSWORD_RESET_REQUESTED",
        params={"ttl": user_service.RESET_TOKEN_TTL_MINUTES},
    )


@public_router.post("/password-reset/confirm", response_model=AcceptedResponse)
def confirm_reset(payload: PasswordResetConfirmRequest, request: Request) -> AcceptedResponse:
    username = user_service.confirm_password_reset(payload)
    audit(
        username,
        "PASSWORD_RESET_CONFIRM",
        target_type="app_user",
        trace_id=getattr(request.state, "trace_id", None),
    )
    return AcceptedResponse(
        status="succeeded",
        detail="Your password was reset. Please sign in with the new password.",
        message_id="PASSWORD_RESET_DONE",
    )


@public_router.get("/password-policy")
def password_policy() -> dict[str, object]:
    """画面側の入力検証と表示に使う規則。"""
    from nihcc_rag import passwords

    return {
        "min_length": passwords.MIN_LENGTH,
        "max_length": passwords.MAX_LENGTH,
        "confirmation_required": True,
        "reset_token_ttl_minutes": user_service.RESET_TOKEN_TTL_MINUTES,
        "roles": list(auth_module.ROLES),
    }
