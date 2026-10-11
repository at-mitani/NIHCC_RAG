# User management and password operations
English | [日本語](user-management.md)

Users are managed in `app_user` in the Metadata DB and operated through the admin UI and the API.
The `LOCAL_USERS` environment variable is used only to create the initial users ([7](#7-initial-users-bootstrap)).

---

## 1. Features

| Feature | Permission | UI | API |
|---|---|---|---|
| List users | admin | User management tab (「ユーザー管理」) | `GET /api/v1/users` |
| Add a new user | admin | Add new button (「新規追加」) | `POST /api/v1/users` |
| User details | admin | "Details" (「詳細」) in the list | `GET /api/v1/users/{user_id}` |
| Edit a user | admin | "Edit" (「編集」) in the details | `PATCH /api/v1/users/{user_id}` |
| Delete a user | admin | "Delete" (「削除」) in the details | `DELETE /api/v1/users/{user_id}` |
| Change password | All users | Change password tab (「パスワード変更」) | `POST /api/v1/account/password` |
| Forgotten password (request) | Unauthenticated | Link on the login screen | `POST /api/v1/auth/password-reset/request` |
| Forgotten password (confirm) | Unauthenticated | Temporary password-change URL | `POST /api/v1/auth/password-reset/confirm` |
| Own information | All users | - | `GET /api/v1/account/me` |
| Get input rules | Unauthenticated | Fetched automatically by the UI | `GET /api/v1/auth/password-policy` |

## 2. User information

| Field | Required | Description |
|---|---|---|
| Account name `username` | Required | 1–64 characters, `A-Za-z0-9._@-`. Unique. Cannot be changed after creation (it is an identifier) |
| Password `password` | Required | 8–128 characters. **Only the scrypt hash is stored** (the plaintext is never stored, returned, or logged) |
| Role `role` | Required | `viewer` / `operator` / `admin`. Higher roles include the permissions of lower ones |
| Email address `email` | Required | Unique, case-insensitive. Destination for password resets |
| Memo `memo` | Optional | Up to 2000 characters |

Additional fields: `is_active` (active/inactive), `last_login_at`, `password_changed_at`,
`created_by` / `updated_by` / `created_at` / `updated_at`.

## 3. Password storage

The standard library's `hashlib.scrypt` (a memory-hard KDF) is used, and passwords are stored in the following format.

```
scrypt$16384$8$1$<base64 salt>$<base64 derived key>
```

- The salt is generated randomly for each entry (the stored value differs every time, even for the same password)
- The cost parameters are included in the stored format, so existing hashes can still be verified after strength is raised later
  (if `needs_rehash` is true, the password is rehashed automatically at the next login)
- Comparison is constant-time. Even for a nonexistent user the same amount of computation is spent, so the existence of an
  account cannot be inferred from differences in response time

## 4. Changing the password (after login)

1. Open the Change password tab (「パスワード変更」) in the menu
2. Enter the current password, the new password, and the new password (confirmation)
3. The new password must be **8 to 128 characters** and **both entries must match**

If the current password is wrong, `CURRENT_PASSWORD_MISMATCH` (422) is returned;
if it is the same as the current password, `PASSWORD_UNCHANGED` (422) is returned.

On a successful change, **all access tokens issued before the change are revoked** and you must log in again.
Unused password reset URLs are invalidated at the same time.

## 5. Resetting a forgotten password

```
Login screen
  └─ 「パスワードをお忘れの方はこちら」 (Forgot your password?)
        └─ Enter the email address → the same message is always shown (202)
              └─ If registered, an email is sent: <APP_BASE_URL>/reset?token=<UUID>
                    └─ Open within 10 minutes → enter the new password twice → updated
```

- The URL is valid for **10 minutes** and **one use only**. Used, expired, and invalidated URLs all return 422
- The UUID in the URL is not stored in plaintext in the DB; **only its SHA-256 is kept**. Even with access to the DB,
  a valid URL cannot be reconstructed
- The response does not differ by whether the address is registered (prevents user enumeration). The destination email address is not recorded in the audit log either
- Requests for the same user are limited to 3 per 10 minutes
- If sending the email fails, the token is invalidated immediately so that no undelivered URL remains valid
- The email body contains only the reset URL and its expiry; no medical or patient information is included

## 6. Preventing lockout

| Operation | Behavior |
|---|---|
| Deleting yourself | Rejected (`SELF_DELETE_FORBIDDEN`) |
| Deleting the last active administrator | Rejected (`LAST_ADMIN_PROTECTED`) |
| Demoting or deactivating the last active administrator | Rejected (`LAST_ADMIN_PROTECTED`) |

## 7. Initial users (bootstrap)

Only when `app_user` has 0 rows, initial users are created from `LOCAL_USERS` at startup
(email address `<username>@${BOOTSTRAP_EMAIL_DOMAIN}`). After creation, the environment variable is no longer read.

> Initial users have auto-generated email addresses, so **before starting operation, update them to real
> email addresses in the admin UI and change their passwords**. Otherwise,
> forgotten-password reset emails will not be delivered. A warning is also written to the startup log.

## 8. Sending email

A `mailpit` container (SMTP 1025 / Web UI <http://localhost:8025>) is bundled for local testing.
Sent emails can be checked in the Web UI. To use your organization's SMTP relay, set
`SMTP_HOST` / `SMTP_PORT` / `SMTP_USE_STARTTLS` / `SMTP_USERNAME` / `SMTP_PASSWORD` in `.env`
and stop `mailpit` (`docker compose stop mailpit`).

| Setting | Default | Description |
|---|---|---|
| `APP_BASE_URL` | `http://localhost:8080` | Base for generating reset URLs. **Change it to match the public URL** |
| `MAIL_ENABLED` | `1` | 0 disables sending (requests are accepted but nothing is sent) |
| `MAIL_FROM` | `no-reply@local.example` | Sender |
| `SMTP_HOST` / `SMTP_PORT` | `mailpit` / `1025` | Destination SMTP |
| `SMTP_USE_STARTTLS` / `SMTP_USE_SSL` | `0` | Enable when using an organizational relay |
| `BOOTSTRAP_EMAIL_DOMAIN` | `local.example` | Email domain for initial users |

## 9. Auditing

The following are recorded in `audit_log` (passwords, hashes, and destination email addresses are not recorded).

`LOGIN` / `USER_CREATE` / `USER_UPDATE` / `USER_DELETE` /
`PASSWORD_CHANGE` / `PASSWORD_RESET_REQUEST` / `PASSWORD_RESET_CONFIRM`

## 10. Verification

```bash
# Unit tests (hashing, input rules, API contract)
python -m pytest tests/unit -q

# Integration tests (list, add, details, edit, delete, password change, reset: 43 checks)
python tests/integration/user_admin_test.py --password "<admin password>"

# UI check (Playwright; screenshots in tmp/ui-users/)
python tests/integration/ui_users_check.py --password "<admin password>"
```

## 11. Implementation decisions

- **Account names cannot be changed after creation.** Records such as the audit log and `created_by`
  reference the account name, so allowing changes would break the link to past records.
  If a rename is needed, create a new account and deactivate the old one.
- **Deletion is physical deletion** (as required). The audit log remains, but the `app_user` row is removed.
  To stop use while keeping the record, use "deactivate" (「無効化」, `is_active = false`).
- **Token revocation uses a millisecond-precision issue time (`iat_ms`).** The standard `iat` is
  truncated to seconds, so the order of a password change and a token issued in the same second
  cannot be determined, causing missed revocations (or wrongful revocation of a valid login).
- User information for authorization checks is cached for 30 seconds. Deactivation, role changes, and password changes
  each explicitly invalidate the cache, so they take effect immediately.
