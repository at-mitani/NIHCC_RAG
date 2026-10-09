"""検索 API（設計書 9.5）。multipart/form-data で JSON part と画像 part を受け取る。"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Request, Response, UploadFile
from nihcc_rag import config, csvspec, db, imaging, problem, queueing, text_norm
from nihcc_rag.contract import SearchAccepted, SearchRequest, SearchResponse

from ..deps import Viewer, audit, publisher
from ..services import search as search_service

_LOG = logging.getLogger("backend-api.searches")
router = APIRouter(prefix="/api/v1/searches", tags=["searches"])

MAX_UPLOAD = config.env_int("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)


def _parse_request(raw: str) -> SearchRequest:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise problem.unprocessable("REQUEST_JSON_INVALID", "request part is not valid JSON") from exc
    return SearchRequest.model_validate(payload)


async def _read_image(upload: UploadFile | None) -> tuple[bytes | None, str | None]:
    if upload is None:
        return None, None
    data = await upload.read()
    if len(data) > MAX_UPLOAD:
        raise problem.ProblemError(
            413,
            "IMAGE_TOO_LARGE",
            "Payload too large",
            f"Uploaded image exceeds {MAX_UPLOAD} bytes.",
        )
    try:
        facts = imaging.inspect(data, max_bytes=MAX_UPLOAD)
    except imaging.ImageValidationError as exc:
        status = 415 if exc.code in ("IMAGE_UNSUPPORTED_FORMAT", "IMAGE_MIME_MISMATCH") else 422
        raise problem.ProblemError(status, exc.code, "Invalid image", str(exc)) from exc
    return data, facts.sha256


@router.post("", response_model=SearchResponse, responses={202: {"model": SearchAccepted}})
async def create_search(
    request: Request,
    principal: Viewer,
    response: Response,
    request_json: Annotated[str, Form(alias="request")] = "{}",
    image: Annotated[UploadFile | None, File()] = None,
    prefer: Annotated[str | None, Header()] = None,
):
    search_request = _parse_request(request_json)
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    content, sha256 = await _read_image(image)

    # 設計書 9.5 の入力制約（mode と image/text の組み合わせ）
    if search_request.mode in ("IMAGE_TO_IMAGE", "HYBRID") and content is None:
        raise problem.unprocessable(
            "SEARCH_INPUT_MISSING",
            f"{search_request.mode} mode requires an image part.",
            errors=[{"field": "image", "reason": "required"}],
        )
    if search_request.mode == "TEXT_TO_IMAGE" and content is not None:
        raise problem.unprocessable(
            "SEARCH_INPUT_UNEXPECTED",
            "TEXT_TO_IMAGE mode does not accept an image part.",
            errors=[{"field": "image", "reason": "not allowed"}],
        )

    search_id = uuid.uuid4()
    asked_async = (prefer or "").lower().replace(" ", "").find("respond-async") >= 0
    median = search_service.median_duration_ms()
    threshold = config.env_int("SEARCH_ASYNC_THRESHOLD_MS", 2000)
    go_async = asked_async and (median is None or median >= threshold or asked_async)

    if go_async:
        _enqueue_async(search_id, search_request, content, sha256, principal.subject, trace_id)
        response.status_code = 202
        response.headers["Location"] = f"/api/v1/searches/{search_id}"
        return SearchAccepted(
            search_id=search_id, status="PENDING", status_url=f"/api/v1/searches/{search_id}"
        )

    try:
        result = search_service.execute_search(
            search_request, image=content, search_id=search_id, actor=principal.subject
        )
    except problem.ProblemError as exc:
        search_service.persist_audit(
            search_id,
            search_request,
            None,
            actor=principal.subject,
            query_image_sha256=sha256,
            error={"code": exc.code, "status": exc.status},
        )
        raise
    search_service.persist_audit(
        search_id, search_request, result, actor=principal.subject, query_image_sha256=sha256
    )
    audit(
        principal.subject,
        "SEARCH",
        target_type="search",
        target_id=str(search_id),
        trace_id=trace_id,
        detail={
            "mode": search_request.mode,
            "duration_ms": result.duration_ms,
            "partial": result.partial,
        },
    )
    return result


def _enqueue_async(
    search_id: uuid.UUID,
    request: SearchRequest,
    content: bytes | None,
    sha256: str | None,
    actor: str,
    trace_id: str,
) -> None:
    retention = config.env_int("QUERY_IMAGE_RETENTION_MINUTES", 60)
    with db.metadata_conn() as conn:
        with conn.transaction():
            db.execute(
                conn,
                """
                INSERT INTO search_audit
                  (search_id, search_mode, query_image_sha256, query_text_original,
                   embedding_model_version, request_params, created_by)
                VALUES (%s, %s, %s, %s, 'pending', %s, %s)
                """,
                (
                    str(search_id),
                    request.mode,
                    sha256,
                    request.text,
                    json.dumps(request.model_dump(mode="json", by_alias=True), ensure_ascii=False),
                    actor,
                ),
            )
            db.execute(
                conn,
                "INSERT INTO search_result (search_id, status) VALUES (%s, 'PENDING')",
                (str(search_id),),
            )
            if content is not None:
                db.execute(
                    conn,
                    "INSERT INTO search_query_image (search_id, content, mime_type, sha256, expires_at) "
                    "VALUES (%s, %s, %s, %s, now() + make_interval(mins => %s))",
                    (str(search_id), content, "image/png", sha256, retention),
                )
    publisher().publish(
        queueing.ROUTING_SEARCH,
        {
            "search_id": str(search_id),
            "trace_id": trace_id,
            "actor": actor,
            "request": request.model_dump(mode="json", by_alias=True),
            "has_image": content is not None,
        },
    )


@router.get("/{search_id}", response_model=SearchResponse)
def get_search(search_id: uuid.UUID, principal: Viewer) -> SearchResponse:
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn,
            "SELECT sr.status, sr.response, sr.error, sa.search_mode "
            "FROM search_result sr JOIN search_audit sa USING (search_id) WHERE sr.search_id = %s",
            (str(search_id),),
        )
    if row is None:
        raise problem.not_found("search_id")
    if row["status"] == "SUCCEEDED" and row["response"]:
        return SearchResponse.model_validate(row["response"])
    if row["status"] == "FAILED":
        detail = (row["error"] or {}).get("detail", "search failed")
        raise problem.ProblemError(
            500, (row["error"] or {}).get("code", "SEARCH_FAILED"), "Search failed", detail
        )
    return SearchResponse(search_id=search_id, status=row["status"], mode=row["search_mode"])


@router.get("/{search_id}/chest")
def query_chest(search_id: uuid.UUID, principal: Viewer, request: Request) -> Response:
    """検索元画像に胸部野の輪郭を重ねたものを返す（結果一覧の「検索元」表示用）。

    候補は同じ見せ方（`/images/{image_id}/chest`）で並ぶ。何と何を比べたかを
    目で確かめられるようにする。

    検索を実行した本人と admin のみ取得できる。保持期間を過ぎたものは 404 とし、
    存在しない search_id と区別しない。
    """
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn,
            "SELECT c.content, a.created_by, (c.expires_at < now()) AS expired "
            "FROM search_query_chest c JOIN search_audit a USING (search_id) "
            "WHERE c.search_id = %s",
            (str(search_id),),
        )
    if row is None or row["expired"]:
        raise problem.not_found("query chest")
    if row["created_by"] != principal.subject and not principal.has_role("admin"):
        # 他人の検索入力は参照させない。存在有無も区別させない。
        raise problem.not_found("query chest")

    return Response(
        bytes(row["content"]),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.get("/{search_id}/lung-outline")
def query_lung_outline(search_id: uuid.UUID, principal: Viewer, request: Request) -> Response:
    """検索元画像に肺野外周を描いたものを返す（胸郭ランキングの「検索元」表示用）。

    候補は同じ見せ方（`/images/{image_id}/lung-outline`）で並ぶ。比べる相手である
    検索元が無いと、何に似ているのかが判らない。

    検索を実行した本人と admin のみ取得できる。保持期間を過ぎたものは 404 とし、
    存在しない search_id と区別しない（胸部野の画像と同じ扱い）。
    """
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn,
            "SELECT o.content, a.created_by, (o.expires_at < now()) AS expired "
            "FROM search_query_outline o JOIN search_audit a USING (search_id) "
            "WHERE o.search_id = %s",
            (str(search_id),),
        )
    if row is None or row["expired"]:
        raise problem.not_found("query lung outline")
    if row["created_by"] != principal.subject and not principal.has_role("admin"):
        raise problem.not_found("query lung outline")

    return Response(
        bytes(row["content"]),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.get("/support/phrases", tags=["searches"])
def phrases(principal: Viewer) -> dict[str, object]:
    """TEXT_TO_IMAGE / HYBRID で使える定型日本語所見（設計書 7.3.2）と所見ラベルの対訳。

    所見ラベルの対訳を**画面に持たせず、ここから配る。** 表示名と入力の受け口は
    同じ表から作らないと、画面には出ているのに絞り込みでは一致しない語が生まれる。
    """
    return {
        "translation_model_version": text_norm.TRANSLATION_MODEL_VERSION,
        "note": "日本語入力は承認済みローカル辞書のみで英語化する。外部翻訳 API へは送信しない。",
        "phrases": text_norm.known_phrases(),
        "findings": [
            {
                "en": label,
                "ja": csvspec.finding_label_ja(label),
                #: 絞り込みで受け付ける言い換え。画面の入力補助に使う。
                "aliases": list(csvspec.FINDING_ALIASES_JA.get(label, ())),
            }
            for label in csvspec.ALLOWED_FINDINGS
        ],
    }
