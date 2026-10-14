"""Сборка HTTP-процесса: проверка модели, пул обработчиков, проверка Triton, приложение."""

import os
from collections.abc import Callable

from fastapi import FastAPI

from src.application.dto import ModelInfo
from src.di.engine import EngineConfig, build_engine
from src.infrastructure.model_files import MODEL_NAME, ModelFiles
from src.infrastructure.pool import ProcessPool
from src.infrastructure.triton import connect, is_ready
from src.presentation.http.app import HttpLimits, create_app
from src.setting.settings import Settings

_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def triton_probe(url: str, model: str) -> Callable[[], bool]:
    """Блокирующая проверка готовности Triton и модели с коротким ожиданием."""
    client = connect(url)

    def ready() -> bool:
        return is_ready(client, model, timeout=2.0)

    return ready


def build_app(settings: Settings) -> FastAPI:
    """Собрать приложение.

    Проверяет папку модели до запуска обработчиков: при ошибке служба не стартует
    (``ModelFilesError``). Побочный эффект: задаёт число потоков математических
    библиотек для процессов-обработчиков через переменные окружения.
    """
    model = ModelFiles.load(settings.model_dir, settings.verify_engine)
    for name in _THREAD_VARIABLES:
        os.environ[name] = str(settings.threads_per_worker)
    os.environ["GIGAAM_LOG_LEVEL"] = settings.log_level
    config = EngineConfig(
        model=model,
        triton_url=settings.triton_url,
        triton_model=settings.triton_model,
        triton_timeout_s=settings.triton_timeout_s,
        threads=settings.threads_per_worker,
        vad_threshold=settings.vad_threshold,
        vad_min_silence=settings.vad_min_silence,
        vad_min_speech=settings.vad_min_speech,
        vad_max_speech=settings.vad_max_speech,
        max_audio_seconds=settings.max_audio_seconds,
    )
    pool = ProcessPool(
        factory=build_engine,
        config=config,
        workers=settings.workers,
        queue_limit=settings.queue_limit or 0,
        request_timeout=settings.request_timeout_s,
        start_timeout=settings.worker_start_timeout_s,
    )
    info = ModelInfo(
        model_name=MODEL_NAME,
        model_digest=model.checkpoint_sha256,
        engine_digest=model.engine_sha256,
        tensorrt=model.tensorrt,
    )
    return create_app(
        pool,
        info,
        triton_probe(settings.triton_url, settings.triton_model),
        HttpLimits(
            max_upload_bytes=settings.max_upload_bytes, retry_after_s=settings.retry_after_s
        ),
    )
