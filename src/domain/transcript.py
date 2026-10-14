"""Итог распознавания одной записи: реплики, слова и паузы со временем в миллисекундах."""

from dataclasses import dataclass

SAMPLE_RATE = 16000


def to_ms(seconds: float) -> int:
    """Перевести секунды в целые миллисекунды с округлением до ближайшего.

    Принимает время от начала записи в секундах, возвращает миллисекунды.
    Ошибок и побочных эффектов нет.
    """
    return round(seconds * 1000)


@dataclass(frozen=True, slots=True)
class Word:
    """Слово реплики; время отсчитывается от начала записи."""

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class Reply:
    """Фрагмент речи с распознанным текстом.

    Границы — фрагмент детектора речи вместе с короткими полями тишины по краям,
    как на стенде temptest. ``channels_db`` — громкость каждого канала исходной
    записи на этом отрезке в дБ относительно полной шкалы; ``None`` для моно.
    """

    seq: int
    start_ms: int
    end_ms: int
    text: str
    words: tuple[Word, ...]
    channels_db: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class Pause:
    """Промежуток, который детектор не отнёс к речи; это оценка детектора."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class Transcript:
    """Результат одной записи.

    ``speech_segments`` — все фрагменты детектора, включая те, где модель не нашла
    слов; ``skipped_short`` — фрагменты короче минимального входа энкодера.
    ``pcm_sha256`` — контрольная сумма моно PCM 16 бит 16 кГц, как ``sha256_pcm``
    стенда: по ней проверяется, что звук декодирован так же. ``processing_ms`` —
    время обработки в обработчике, без ожидания очереди.
    """

    duration_s: float
    source_channels: int
    pcm_sha256: str
    speech_segments: int
    skipped_short: int
    replies: tuple[Reply, ...]
    pauses: tuple[Pause, ...]
    processing_ms: int
