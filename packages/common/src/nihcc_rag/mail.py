"""メール送信（パスワード再設定 URL の通知）。

医療情報は本文へ含めない。送信するのは「再設定 URL と有効期限」だけとし、
利用者名・患者情報・検索内容は載せない。SMTP 送信先はローカル/組織内の
リレーに限定し、外部サービスへは送らない前提の設定とする。
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid

from . import config

_LOG = logging.getLogger(__name__)


class MailError(RuntimeError):
    """SMTP 送信に失敗した。"""


def is_enabled() -> bool:
    return config.env_bool("MAIL_ENABLED", True)


def _connect() -> smtplib.SMTP:
    host = config.env("SMTP_HOST", "mailpit")
    port = config.env_int("SMTP_PORT", 1025)
    timeout = config.env_int("SMTP_TIMEOUT_SECONDS", 10)

    if config.env_bool("SMTP_USE_SSL", False):
        client: smtplib.SMTP = smtplib.SMTP_SSL(
            host, port, timeout=timeout, context=ssl.create_default_context()
        )
    else:
        client = smtplib.SMTP(host, port, timeout=timeout)
        if config.env_bool("SMTP_USE_STARTTLS", False):
            client.starttls(context=ssl.create_default_context())

    username = config.optional_secret("SMTP_USERNAME")
    password = config.optional_secret("SMTP_PASSWORD")
    if username and password:
        client.login(username, password)
    return client


def send(*, to: str, subject: str, body: str) -> str:
    """メールを送信し、Message-ID を返す。"""
    if not is_enabled():
        raise MailError("mail delivery is disabled (MAIL_ENABLED=0)")

    message = EmailMessage()
    message["From"] = formataddr(
        (config.env("MAIL_FROM_NAME", "NIHCC RAG"), config.env("MAIL_FROM", "no-reply@local.example"))
    )
    message["To"] = to
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    message_id = make_msgid(domain=config.env("MAIL_MESSAGE_ID_DOMAIN", "local.example"))
    message["Message-ID"] = message_id
    message["Auto-Submitted"] = "auto-generated"
    message.set_content(body)

    try:
        with _connect() as client:
            client.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise MailError(f"smtp delivery failed: {type(exc).__name__}") from exc
    # 宛先や URL はログへ出さない（設計書 13）
    _LOG.info("mail sent", extra={"fields": {"message_id": message_id, "subject": subject}})
    return message_id


def send_password_reset(*, to: str, reset_url: str, valid_minutes: int) -> str:
    subject = "【NIHCC 胸部X線画像検索】パスワード再設定のご案内"
    body = (
        "パスワード再設定の要求を受け付けました。\n\n"
        f"次の URL から {valid_minutes} 分以内に新しいパスワードを設定してください。\n"
        f"{reset_url}\n\n"
        "・この URL は 1 回のみ有効です。\n"
        f"・{valid_minutes} 分を過ぎた場合は、再度パスワード再設定を要求してください。\n"
        "・心当たりがない場合は、このメールを破棄してください。パスワードは変更されません。\n\n"
        "本メールは送信専用です。返信されても対応できません。\n"
    )
    return send(to=to, subject=subject, body=body)
