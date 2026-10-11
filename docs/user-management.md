# ユーザー管理・パスワード運用
[English](user-management_EN.md) | 日本語

利用者は Metadata DB の `app_user` で管理し、管理画面と API から運用します。
`LOCAL_USERS` 環境変数は初期ユーザーの作成にだけ使います（[7](#7-初期ユーザーブートストラップ)）。

---

## 1. 機能一覧

| 機能 | 権限 | 画面 | API |
|---|---|---|---|
| ユーザー一覧 | admin | ユーザー管理タブ | `GET /api/v1/users` |
| ユーザー新規追加 | admin | 新規追加ボタン | `POST /api/v1/users` |
| ユーザー詳細 | admin | 一覧の「詳細」 | `GET /api/v1/users/{user_id}` |
| ユーザー編集 | admin | 詳細の「編集」 | `PATCH /api/v1/users/{user_id}` |
| ユーザー削除 | admin | 詳細の「削除」 | `DELETE /api/v1/users/{user_id}` |
| パスワード変更 | 全ユーザー | パスワード変更タブ | `POST /api/v1/account/password` |
| パスワード忘れ（要求） | 未認証 | ログイン画面のリンク | `POST /api/v1/auth/password-reset/request` |
| パスワード忘れ（確定） | 未認証 | 仮パスワード変更 URL | `POST /api/v1/auth/password-reset/confirm` |
| 自分の情報 | 全ユーザー | - | `GET /api/v1/account/me` |
| 入力規則の取得 | 未認証 | 画面が自動取得 | `GET /api/v1/auth/password-policy` |

## 2. ユーザー情報

| 項目 | 必須 | 説明 |
|---|---|---|
| アカウント名 `username` | 必須 | 1〜64 文字、`A-Za-z0-9._@-`。一意。作成後は変更不可（識別子のため） |
| パスワード `password` | 必須 | 8〜128 文字。**scrypt のハッシュ値のみ保存**（平文は保存も応答もログ出力もしない） |
| 権限 `role` | 必須 | `viewer` / `operator` / `admin`。上位は下位の権限を含む |
| メールアドレス `email` | 必須 | 大小文字を区別せず一意。パスワード再設定の宛先 |
| メモ `memo` | 任意 | 2000 文字以内 |

付随項目: `is_active`（有効/無効）、`last_login_at`、`password_changed_at`、
`created_by` / `updated_by` / `created_at` / `updated_at`。

## 3. パスワードの保存方式

標準ライブラリの `hashlib.scrypt`（メモリハード KDF）を使用し、次の形式で保存します。

```
scrypt$16384$8$1$<base64 salt>$<base64 derived key>
```

- ソルトは 1 件ごとにランダム生成（同じパスワードでも保存値は毎回異なる）
- コスト係数を保存形式へ含めるため、後から強度を上げても既存ハッシュを検証できる
  （`needs_rehash` が真なら次回ログイン時に自動で再ハッシュ）
- 照合は定数時間比較。存在しないユーザーでも同等の計算量を消費し、応答時間差で
  アカウントの存在を推測させない

## 4. パスワード変更（ログイン後）

1. メニューの「パスワード変更」タブを開く
2. 現在のパスワード、新しいパスワード、新しいパスワード（確認）を入力
3. 新しいパスワードは **8 文字以上 128 文字以下**、**2 回入力が一致**していること

現在のパスワードが誤っていれば `CURRENT_PASSWORD_MISMATCH`（422）、
現在と同じパスワードなら `PASSWORD_UNCHANGED`（422）を返します。

変更に成功すると **変更前に発行されたアクセストークンは全て失効**し、再ログインが必要です。
未使用のパスワード再設定 URL も同時に無効化されます。

## 5. パスワード忘れによる再設定

```
ログイン画面
  └─「パスワードをお忘れの方はこちら」
        └─ メールアドレスを入力 → 常に同じ案内を表示（202）
              └─ 登録があればメール送信: <APP_BASE_URL>/reset?token=<UUID>
                    └─ 10 分以内にアクセス → 新パスワードを 2 回入力 → 更新
```

- 有効期限は **10 分**、**1 回限り**。使用済み・期限切れ・無効化済みはいずれも 422
- URL に載せる UUID は DB へ平文で保存せず **SHA-256 のみ保持**します。DB を参照できても
  有効な URL を再構成できません
- 応答は登録の有無で変えません（利用者列挙の防止）。宛先メールアドレスは監査ログにも残しません
- 同一利用者からの要求は 10 分あたり 3 回までに制限します
- メール送信に失敗した場合は、届いていない URL を残さないよう token を即座に無効化します
- メール本文には再設定 URL と有効期限のみを記載し、医療情報・患者情報は含めません

## 6. 締め出しの防止

| 操作 | 挙動 |
|---|---|
| 自分自身の削除 | 拒否（`SELF_DELETE_FORBIDDEN`） |
| 最後の有効な管理者の削除 | 拒否（`LAST_ADMIN_PROTECTED`） |
| 最後の有効な管理者の降格・無効化 | 拒否（`LAST_ADMIN_PROTECTED`） |

## 7. 初期ユーザー（ブートストラップ）

`app_user` が 0 件のときに限り、起動時に `LOCAL_USERS` から初期ユーザーを作成します
（メールアドレスは `<username>@${BOOTSTRAP_EMAIL_DOMAIN}`）。作成後は環境変数を参照しません。

> 初期ユーザーは自動生成されたメールアドレスを持つため、**運用開始前に管理画面から
> 実在のメールアドレスへ更新し、パスワードを変更してください**。更新しないと
> パスワード忘れの再設定メールが届きません。起動ログにも警告を出力します。

## 8. メール送信

ローカル検証用に `mailpit` コンテナ（SMTP 1025 / Web UI <http://localhost:8025>）を同梱しています。
送信されたメールは Web UI で確認できます。組織の SMTP リレーを使う場合は `.env` の
`SMTP_HOST` / `SMTP_PORT` / `SMTP_USE_STARTTLS` / `SMTP_USERNAME` / `SMTP_PASSWORD` を設定し、
`mailpit` を停止してください（`docker compose stop mailpit`）。

| 設定 | 既定 | 説明 |
|---|---|---|
| `APP_BASE_URL` | `http://localhost:8080` | 再設定 URL の生成元。**公開 URL に合わせて変更する** |
| `MAIL_ENABLED` | `1` | 0 で送信を無効化（要求は受理するが送信しない） |
| `MAIL_FROM` | `no-reply@local.example` | 差出人 |
| `SMTP_HOST` / `SMTP_PORT` | `mailpit` / `1025` | 送信先 SMTP |
| `SMTP_USE_STARTTLS` / `SMTP_USE_SSL` | `0` | 組織リレー利用時に有効化する |
| `BOOTSTRAP_EMAIL_DOMAIN` | `local.example` | 初期ユーザーのメールドメイン |

## 9. 監査

`audit_log` へ次を記録します（パスワード・ハッシュ・宛先メールアドレスは記録しません）。

`LOGIN` / `USER_CREATE` / `USER_UPDATE` / `USER_DELETE` /
`PASSWORD_CHANGE` / `PASSWORD_RESET_REQUEST` / `PASSWORD_RESET_CONFIRM`

## 10. 動作確認

```bash
# 単体テスト（ハッシュ化・入力規則・API 契約）
python -m pytest tests/unit -q

# 結合テスト（一覧・追加・詳細・編集・削除・パスワード変更・再設定：43 項目）
python tests/integration/user_admin_test.py --password "<admin のパスワード>"

# 画面確認（Playwright。tmp/ui-users/ にスクリーンショット）
python tests/integration/ui_users_check.py --password "<admin のパスワード>"
```

## 11. 実装上の判断

- **アカウント名は作成後に変更不可**としました。監査ログ・`created_by` などの記録が
  アカウント名を参照しているため、変更を許すと過去の記録との対応が取れなくなります。
  名称変更が必要な場合は新規作成と旧アカウントの無効化で対応してください。
- **削除は物理削除**です（要件どおり）。監査ログは残りますが、`app_user` の行は消えます。
  記録を保持したまま利用停止にする場合は「無効化」（`is_active = false`）を使ってください。
- **トークン失効の判定にミリ秒精度の発行時刻（`iat_ms`）を使用**しています。標準の `iat` は
  秒単位へ切り捨てられるため、パスワード変更と同一秒に発行されたトークンの前後関係を
  判定できず、失効漏れ（または正常なログインの誤失効）が起きます。
- 認可判定のためのユーザー情報は 30 秒キャッシュします。無効化・権限変更・パスワード変更の
  各操作はキャッシュを明示的に破棄するため、即座に反映されます。
