"""HTTP API службы: приём файла, готовность, ошибки с устойчивыми кодами."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.application.dto import (
    ErrorBody,
    ErrorResponse,
    ModelInfo,
    ReadinessResponse,
    TranscriptionResponse,
)
from src.application.errors import FileTooLargeError, TranscriptionError
from src.application.ports import TranscriptionService

log = logging.getLogger("speech_gigaam.http")

_STATUS = {
    "bad_audio": 422,
    "audio_too_long": 413,
    "file_too_large": 413,
    "busy": 503,
    "not_ready": 503,
    "timeout": 504,
    "engine_failed": 502,
    "worker_failed": 500,
    "internal": 500,
}
_RETRY = {"busy", "not_ready"}


@dataclass(frozen=True, slots=True)
class HttpLimits:
    """Пределы запроса и подсказка, когда повторять после «занято»."""

    max_upload_bytes: int
    retry_after_s: int


async def _read_body(request: Request, limit: int) -> bytes:
    """Прочитать тело запроса, прерывая чтение на превышении предела.

    Ошибка ``FileTooLargeError`` — если тело или ``Content-Length`` больше ``limit``.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise FileTooLargeError(f"Файл больше {limit // (1024 * 1024)} МБ")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise FileTooLargeError(f"Файл больше {limit // (1024 * 1024)} МБ")
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    service: TranscriptionService,
    model: ModelInfo,
    triton_ready: Callable[[], bool],
    limits: HttpLimits,
) -> FastAPI:
    """Собрать приложение FastAPI.

    Пул запускается при старте приложения и останавливается при выходе. ``triton_ready``
    — блокирующая проверка сервера моделей для ``/ready``; она вызывается в потоке.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title="speech-gigaam", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(TranscriptionError)
    async def transcription_error(_: Request, exc: TranscriptionError) -> JSONResponse:
        status = _STATUS.get(exc.code, 500)
        headers = {"Retry-After": str(limits.retry_after_s)} if exc.code in _RETRY else None
        body = ErrorResponse(error=ErrorBody(code=exc.code, message=str(exc)))
        return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Сбой обработки запроса")
        body = ErrorResponse(error=ErrorBody(code="internal", message=type(exc).__name__))
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.post(
        "/v1/transcriptions",
        response_model=TranscriptionResponse,
        responses={code: {"model": ErrorResponse} for code in (413, 422, 500, 502, 503, 504)},
        summary="Распознать файл",
        description=(
            "Тело запроса — сам файл записи (`Content-Type: application/octet-stream`), "
            "любой формат, который читает FFmpeg. Ответ — реплики с таймкодами."
        ),
    )
    async def transcribe(request: Request) -> TranscriptionResponse:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        data = await _read_body(request, limits.max_upload_bytes)
        try:
            outcome = await service.transcribe(data)
        except TranscriptionError as exc:
            log.warning("Запрос %s: %d байт, ошибка %s: %s", request_id, len(data), exc.code, exc)
            raise
        transcript = outcome.transcript
        log.info(
            "Запрос %s: %d байт, звук %.1f с, фрагментов %d, реплик %d, "
            "ожидание %d мс, обработка %d мс",
            request_id,
            len(data),
            transcript.duration_s,
            transcript.speech_segments,
            len(transcript.replies),
            outcome.wait_ms,
            transcript.processing_ms,
        )
        return TranscriptionResponse.build(transcript, model, outcome.wait_ms)

    @app.get("/health", summary="Процесс жив")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
        summary="Готовность: обработчики загрузили модели, Triton отвечает",
    )
    async def ready() -> JSONResponse:
        status = service.status()
        triton = await asyncio.to_thread(triton_ready) if status.ready else False
        ok = status.ready and triton
        body = ReadinessResponse(
            status="ready" if ok else "not_ready",
            workers=status.workers,
            alive=status.alive,
            busy=status.busy,
            waiting=status.waiting,
            capacity=status.capacity,
            triton=triton,
        )
        return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())

    @app.get("/v1/model", response_model=ModelInfo, summary="Какая модель отвечает")
    async def model_info() -> ModelInfo:
        return model

    return app
