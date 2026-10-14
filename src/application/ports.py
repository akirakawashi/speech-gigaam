"""Что нужно сценарию распознавания от звука, детектора речи и модели."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.domain.transcript import Transcript

Samples = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """Фрагмент речи от детектора.

    ``start`` — номер первого отсчёта 16 кГц от начала записи; ``lead`` и ``trail`` —
    поля тишины в отсчётах, добавленные перед речью и оставленные после неё.
    """

    start: int
    samples: Samples
    lead: int
    trail: int

    @property
    def end(self) -> int:
        """Номер отсчёта сразу после фрагмента."""
        return self.start + int(self.samples.size)


@dataclass(frozen=True, slots=True)
class Recognition:
    """Ответ модели на один фрагмент: текст, токены и время токенов в секундах."""

    text: str
    tokens: tuple[str, ...]
    stamps: tuple[float, ...]


class DecodedAudio(Protocol):
    """Декодированная запись: моно 16 кГц для распознавания и громкость каналов."""

    @property
    def mono(self) -> Samples:
        """Отсчёты моно 16 кГц в диапазоне [-1, 1)."""
        ...

    @property
    def source_channels(self) -> int:
        """Число каналов исходного файла."""
        ...

    @property
    def pcm_sha256(self) -> str:
        """SHA256 моно PCM 16 бит — для сверки декодирования со стендом."""
        ...

    def channel_levels(self, start: int, end: int) -> tuple[float, ...] | None:
        """Громкость каждого канала на отрезке отсчётов, дБ; ``None`` для моно."""
        ...


class AudioDecoder(Protocol):
    """Декодер файла в звук."""

    def decode(self, data: bytes) -> DecodedAudio:
        """Декодировать файл; ``AudioDecodeError``, если это не звук или он пуст."""
        ...


class SpeechDetector(Protocol):
    """Поиск речи в записи."""

    def segments(self, samples: Samples) -> list[SpeechSegment]:
        """Вернуть фрагменты речи в порядке времени."""
        ...


class Recognizer(Protocol):
    """Распознавание одного фрагмента речи."""

    def recognize(self, samples: Samples) -> Recognition | None:
        """Распознать фрагмент; ``None``, если он короче минимального входа модели.

        ``EngineError`` — при сбое модели или сервера моделей.
        """
        ...


class TranscriptionService(Protocol):
    """Служба распознавания с пулом обработчиков; её вызывает HTTP-слой."""

    async def start(self) -> None:
        """Запустить обработчики и дождаться загрузки моделей."""
        ...

    async def stop(self) -> None:
        """Остановить обработчики."""
        ...

    async def transcribe(self, data: bytes) -> "Outcome":
        """Распознать файл; ошибки — подклассы ``TranscriptionError``."""
        ...

    def status(self) -> "ServiceStatus":
        """Текущее состояние обработчиков и очереди."""
        ...


@dataclass(frozen=True, slots=True)
class Outcome:
    """Результат запроса: распознавание и сколько запрос ждал свободного обработчика, мс."""

    transcript: Transcript
    wait_ms: int


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Состояние пула: сколько обработчиков готово, занято и сколько запросов ждёт."""

    ready: bool
    workers: int
    alive: int
    busy: int
    waiting: int
    capacity: int
