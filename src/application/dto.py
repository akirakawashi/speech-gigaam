"""Ответ API распознавания.

Поля движка, модели, языка, длительности и реплик (``seq``, ``start_ms``, ``end_ms``,
``text``) совпадают по смыслу с контрактом приёма распознавания бэкенда.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.domain.transcript import Transcript


class WordOut(BaseModel):
    """Слово реплики со временем от начала записи, мс."""

    model_config = ConfigDict(frozen=True)
    text: str
    start_ms: int
    end_ms: int


class ReplyOut(BaseModel):
    """Реплика: фрагмент речи с текстом; ``channels_db`` — громкость каналов, дБ."""

    model_config = ConfigDict(frozen=True)
    seq: int
    start_ms: int
    end_ms: int
    text: str
    words: list[WordOut]
    channels_db: list[float] | None


class PauseOut(BaseModel):
    """Промежуток вне речи по оценке детектора, мс."""

    model_config = ConfigDict(frozen=True)
    start_ms: int
    end_ms: int


class ModelInfo(BaseModel):
    """Какая модель отвечает: веса, движок TensorRT и его версия."""

    model_config = ConfigDict(frozen=True)
    model_name: str
    model_digest: str
    engine_digest: str
    tensorrt: str


class TranscriptionResponse(BaseModel):
    """Результат распознавания одного файла."""

    model_config = ConfigDict(frozen=True)
    engine: Literal["gigaam"] = "gigaam"
    model_name: str
    model_digest: str
    engine_digest: str
    language: Literal["ru"] = "ru"
    duration_s: float
    source_channels: int
    pcm_sha256: str
    speech_segments: int
    skipped_short: int
    replies: list[ReplyOut]
    pauses: list[PauseOut]
    processing_ms: int
    wait_ms: int

    @classmethod
    def build(
        cls, transcript: Transcript, model: ModelInfo, wait_ms: int
    ) -> "TranscriptionResponse":
        """Собрать ответ из результата обработчика, сведений о модели и ожидания в очереди."""
        return cls(
            model_name=model.model_name,
            model_digest=model.model_digest,
            engine_digest=model.engine_digest,
            duration_s=round(transcript.duration_s, 3),
            source_channels=transcript.source_channels,
            pcm_sha256=transcript.pcm_sha256,
            speech_segments=transcript.speech_segments,
            skipped_short=transcript.skipped_short,
            replies=[
                ReplyOut(
                    seq=r.seq,
                    start_ms=r.start_ms,
                    end_ms=r.end_ms,
                    text=r.text,
                    words=[
                        WordOut(text=w.text, start_ms=w.start_ms, end_ms=w.end_ms) for w in r.words
                    ],
                    channels_db=list(r.channels_db) if r.channels_db is not None else None,
                )
                for r in transcript.replies
            ],
            pauses=[PauseOut(start_ms=p.start_ms, end_ms=p.end_ms) for p in transcript.pauses],
            processing_ms=transcript.processing_ms,
            wait_ms=wait_ms,
        )


class ErrorBody(BaseModel):
    """Описание ошибки: устойчивый код и сообщение для человека."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""

    error: ErrorBody


class ReadinessResponse(BaseModel):
    """Готовность службы: обработчики, очередь и сервер моделей."""

    status: Literal["ready", "not_ready"]
    workers: int
    alive: int
    busy: int
    waiting: int
    capacity: int
    triton: bool
