"""パスワード規則・ハッシュ化とユーザー管理の契約テスト（機能追加）。"""

from __future__ import annotations

import uuid

import pytest
from nihcc_rag import passwords, problem
from nihcc_rag.contract import (
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    UserCreateRequest,
    UserUpdateRequest,
)
from pydantic import ValidationError

VALID = "correct-horse-8"


# --- ハッシュ化 ---------------------------------------------------------------


def test_hash_is_not_reversible_and_verifies() -> None:
    encoded = passwords.hash_password(VALID)
    assert VALID not in encoded
    assert encoded.startswith("scrypt$")
    assert passwords.verify_password(VALID, encoded)
    assert not passwords.verify_password(VALID + "x", encoded)


def test_hash_is_salted_so_same_password_differs() -> None:
    assert passwords.hash_password(VALID) != passwords.hash_password(VALID)


def test_verify_rejects_malformed_hash_without_raising() -> None:
    for broken in ("", "not-a-hash", "scrypt$1$2", "md5$1$1$1$aaaa$bbbb"):
        assert passwords.verify_password(VALID, broken) is False


def test_needs_rehash_detects_foreign_format() -> None:
    assert passwords.needs_rehash("md5$deadbeef") is True
    assert passwords.needs_rehash(passwords.hash_password(VALID)) is False


def test_unicode_normalisation_is_applied() -> None:
    """全角と半角が混在しても同一パスワードとして扱う（NFKC 正規化）。"""
    encoded = passwords.hash_password("ｐａｓｓword1")
    assert passwords.verify_password("password1", encoded)


# --- 入力規則（8 文字以上 128 文字以下・2 回入力） ----------------------------


@pytest.mark.parametrize("password", ["", "short", "1234567"])
def test_policy_rejects_too_short(password: str) -> None:
    with pytest.raises(problem.ProblemError) as excinfo:
        passwords.validate_policy(password)
    assert excinfo.value.code == "PASSWORD_POLICY_VIOLATION"
    assert excinfo.value.status == 422


def test_policy_rejects_too_long() -> None:
    with pytest.raises(problem.ProblemError):
        passwords.validate_policy("a" * 129)


def test_policy_accepts_boundaries() -> None:
    assert passwords.validate_policy("a" * 8)
    assert passwords.validate_policy("a" * 128)


def test_confirmation_mismatch_is_rejected() -> None:
    with pytest.raises(problem.ProblemError) as excinfo:
        passwords.validate_confirmation(VALID, VALID + "different")
    assert excinfo.value.code == "PASSWORD_CONFIRMATION_MISMATCH"


def test_confirmation_match_returns_normalised_password() -> None:
    assert passwords.validate_confirmation(VALID, VALID) == VALID


# --- API 契約 ----------------------------------------------------------------


def test_user_create_requires_matching_confirmation() -> None:
    with pytest.raises(ValidationError):
        UserCreateRequest(
            username="taro",
            password=VALID,
            password_confirmation="another-password",
            role="viewer",
            email="taro@example.com",
        )


def test_user_create_requires_valid_email() -> None:
    with pytest.raises(ValidationError):
        UserCreateRequest(
            username="taro",
            password=VALID,
            password_confirmation=VALID,
            role="viewer",
            email="not-an-email",
        )


def test_user_create_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        UserCreateRequest(
            username="taro",
            password=VALID,
            password_confirmation=VALID,
            role="superuser",
            email="taro@example.com",
        )


def test_user_create_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        UserCreateRequest(
            username="taro",
            password="short",
            password_confirmation="short",
            role="viewer",
            email="taro@example.com",
        )


def test_user_create_rejects_invalid_username_characters() -> None:
    with pytest.raises(ValidationError):
        UserCreateRequest(
            username="ta ro/../etc",
            password=VALID,
            password_confirmation=VALID,
            role="viewer",
            email="taro@example.com",
        )


def test_user_create_accepts_optional_memo() -> None:
    request = UserCreateRequest(
        username="taro",
        password=VALID,
        password_confirmation=VALID,
        role="operator",
        email="Taro@Example.com",
        memo=None,
    )
    assert request.memo is None
    assert request.is_active is True


def test_user_update_allows_partial_fields() -> None:
    request = UserUpdateRequest(role="admin")
    assert request.role == "admin"
    assert request.email is None
    assert request.password is None


def test_user_update_requires_password_pair_together() -> None:
    with pytest.raises(ValidationError):
        UserUpdateRequest(password=VALID)
    with pytest.raises(ValidationError):
        UserUpdateRequest(password_confirmation=VALID)


def test_password_change_requires_matching_new_password() -> None:
    with pytest.raises(ValidationError):
        PasswordChangeRequest(
            current_password="old-password",
            new_password=VALID,
            new_password_confirmation="mismatch-value",
        )
    ok = PasswordChangeRequest(
        current_password="old-password",
        new_password=VALID,
        new_password_confirmation=VALID,
    )
    assert ok.new_password == VALID


def test_password_reset_request_validates_email() -> None:
    with pytest.raises(ValidationError):
        PasswordResetRequest(email="nope")
    assert PasswordResetRequest(email="a@example.com").email == "a@example.com"


def test_password_reset_confirm_requires_uuid_token_and_match() -> None:
    token = uuid.uuid4()
    with pytest.raises(ValidationError):
        PasswordResetConfirmRequest(
            token="not-a-uuid", new_password=VALID, new_password_confirmation=VALID
        )
    with pytest.raises(ValidationError):
        PasswordResetConfirmRequest(
            token=token, new_password=VALID, new_password_confirmation="different-one"
        )
    ok = PasswordResetConfirmRequest(
        token=token, new_password=VALID, new_password_confirmation=VALID
    )
    assert ok.token == token
