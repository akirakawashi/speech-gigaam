"""Поддельный движок для тестов пула: поведение задаётся содержимым «файла»."""

import os
import time
from typing import Any

from src.application.errors import AudioDecodeError
from src.domain.transcript import Transcript


class FakeEngine:
    """``sleep:<с>`` — ждать; ``crash`` — уронить процесс; ``bad`` — плохой звук; ``bug`` — сбой."""

    def __call__(self, data: bytes) -> Transcript:
        command = data.decode()
        if command.startswith("sleep:"):
            time.sleep(float(command.removeprefix("sleep:")))
        elif command == "crash":
            os._exit(3)
        elif command == "bad":
            raise AudioDecodeError("плохой звук")
        elif command == "bug":
            raise ValueError("ошибка в коде")
        return Transcript(
            duration_s=1.0,
            source_channels=1,
            pcm_sha256=f"{os.getpid()}:{command}",
            speech_segments=0,
            skipped_short=0,
            replies=(),
            pauses=(),
            processing_ms=0,
        )


def fake_factory(config: dict[str, Any]) -> FakeEngine:
    """Фабрика для пула; ``{"fail": True}`` — обработчик не загружается."""
    if config.get("fail"):
        raise RuntimeError("модели не загрузились")
    return FakeEngine()
