"""Настройки службы из переменных окружения ``GIGAAM_*``."""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Все параметры службы. Пояснения к каждому — в docs/configuration.md."""

    model_config = SettingsConfigDict(env_prefix="GIGAAM_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    log_level: str = "INFO"

    model_dir: Path = Path("/model")
    verify_engine: bool = True

    triton_url: str = "triton:8001"
    triton_model: str = "gigaam_encoder"
    triton_timeout_s: float = Field(60.0, gt=0)

    workers: int = Field(4, ge=1, le=64)
    threads_per_worker: int = Field(2, ge=1, le=32)
    queue_limit: int | None = Field(None, ge=0)
    request_timeout_s: float = Field(600.0, gt=0)
    worker_start_timeout_s: float = Field(180.0, gt=0)
    retry_after_s: int = Field(2, ge=1)

    max_upload_mb: int = Field(200, ge=1)
    max_audio_seconds: float = Field(7200.0, gt=0)

    # Параметры поиска речи — как на стенде temptest; от них зависит совпадение текста.
    vad_threshold: float = Field(0.5, gt=0, lt=1)
    vad_min_silence: float = Field(0.45, gt=0)
    vad_min_speech: float = Field(0.25, gt=0)
    vad_max_speech: float = Field(15.0, gt=0, le=24)

    @model_validator(mode="after")
    def default_queue(self) -> "Settings":
        """Если очередь не задана, держать её равной числу обработчиков."""
        if self.queue_limit is None:
            self.queue_limit = self.workers
        return self

    @property
    def max_upload_bytes(self) -> int:
        """Предел тела запроса в байтах."""
        return self.max_upload_mb * 1024 * 1024
