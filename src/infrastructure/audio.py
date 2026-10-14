"""Декодирование файла через PyAV: моно 16 кГц для распознавания и громкость каналов.

Моно декодируется так же, как на стенде temptest (``faster_whisper.audio.decode_audio``
1.2.1 поверх PyAV 15.1.0): кадры группируются по 500 000 отсчётов и переводятся в
16 бит моно 16 кГц. Поэтому SHA256 моно PCM совпадает со ``sha256_pcm`` стенда.
"""

import gc
import hashlib
import io
import itertools
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass

import av
import numpy as np
from numpy.typing import NDArray

from src.application.errors import AudioDecodeError
from src.application.ports import Samples
from src.domain.transcript import SAMPLE_RATE

_GROUP_SAMPLES = 500_000
# Громкость каналов копится блоками по 10 мс: память не растёт с числом отсчётов.
_LEVEL_BLOCK = SAMPLE_RATE // 100
_MAX_LEVEL_CHANNELS = 8
_SILENCE_DB = -120.0

log = logging.getLogger("speech_gigaam.audio")
_LEVEL_ERRORS = (av.error.FFmpegError, ValueError, TypeError)


@dataclass(frozen=True, slots=True)
class PyAvAudio:
    """Декодированная запись.

    ``energy`` — средний квадрат отсчётов каждого канала по блокам 10 мс, форма
    ``[блоки, каналы]``; ``None`` для моно.
    """

    mono: Samples
    source_channels: int
    pcm_sha256: str
    energy: NDArray[np.float64] | None

    def channel_levels(self, start: int, end: int) -> tuple[float, ...] | None:
        """Громкость каждого канала на отрезке отсчётов 16 кГц в дБ полной шкалы.

        Принимает границы отрезка в отсчётах. Возвращает по числу на канал,
        округлённому до 0,1 дБ, или ``None`` для моно. Тишина — −120 дБ.
        """
        if self.energy is None or self.energy.shape[0] == 0:
            return None
        first = max(0, min(start // _LEVEL_BLOCK, self.energy.shape[0] - 1))
        last = max(first + 1, min(math.ceil(end / _LEVEL_BLOCK), self.energy.shape[0]))
        mean = self.energy[first:last].mean(axis=0)
        return tuple(
            round(10 * math.log10(value), 1) if value > 1e-12 else _SILENCE_DB for value in mean
        )


class _LevelMeter:
    """Копит средний квадрат отсчётов каждого канала по блокам 10 мс."""

    def __init__(self, channels: int) -> None:
        self._channels = channels
        self._carry = np.zeros((0, channels), dtype=np.float64)
        self._blocks: list[NDArray[np.float64]] = []

    def add(self, interleaved: NDArray[np.int16]) -> None:
        """Добавить чередующиеся отсчёты 16 бит всех каналов."""
        values = interleaved.reshape(-1, self._channels).astype(np.float64) / 32768.0
        values = np.concatenate([self._carry, values * values])
        whole = values.shape[0] // _LEVEL_BLOCK * _LEVEL_BLOCK
        if whole:
            self._blocks.append(
                values[:whole].reshape(-1, _LEVEL_BLOCK, self._channels).mean(axis=1)
            )
        self._carry = values[whole:]

    def result(self) -> NDArray[np.float64]:
        """Вернуть энергию по блокам; неполный последний блок усредняется как есть."""
        blocks = list(self._blocks)
        if self._carry.shape[0]:
            blocks.append(self._carry.mean(axis=0, keepdims=True))
        if not blocks:
            return np.zeros((0, self._channels), dtype=np.float64)
        return np.concatenate(blocks)


def _ignore_invalid_frames(frames: Iterator[av.AudioFrame]) -> Iterator[av.AudioFrame]:
    """Пропустить повреждённые кадры, как faster-whisper."""
    while True:
        try:
            yield next(frames)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames: Iterator[av.AudioFrame], num_samples: int) -> Iterator[av.AudioFrame]:
    """Собрать кадры в куски около ``num_samples`` отсчётов, как faster-whisper."""
    fifo = av.audio.fifo.AudioFifo()
    for frame in frames:
        frame.pts = None
        fifo.write(frame)
        if fifo.samples >= num_samples and (chunk := fifo.read()) is not None:
            yield chunk
    if fifo.samples > 0 and (rest := fifo.read()) is not None:
        yield rest


class PyAvDecoder:
    """Декодер любого формата, который читает FFmpeg из состава PyAV."""

    def decode(self, data: bytes) -> PyAvAudio:
        """Декодировать файл из памяти.

        Возвращает моно 16 кГц, число каналов источника, SHA256 моно PCM и энергию
        каналов для громкости реплик. Ошибка ``AudioDecodeError`` — если это не звук,
        в нём нет звуковой дорожки или звук пустой. Побочных эффектов нет.
        """
        if not data:
            raise AudioDecodeError("Пустой файл")
        mono = io.BytesIO()
        meter: _LevelMeter | None = None
        try:
            with av.open(io.BytesIO(data), mode="r", metadata_errors="ignore") as container:
                if not container.streams.audio:
                    raise AudioDecodeError("В файле нет звуковой дорожки")
                stream = container.streams.audio[0]
                channels = int(stream.codec_context.channels or 0)
                mono_resampler = av.audio.resampler.AudioResampler(
                    format="s16", layout="mono", rate=SAMPLE_RATE
                )
                level_resampler = None
                if 2 <= channels <= _MAX_LEVEL_CHANNELS:
                    try:
                        # Раскладка — объект потока, а не её имя: у части файлов имя нестандартное.
                        level_resampler = av.audio.resampler.AudioResampler(
                            format="s16", layout=stream.codec_context.layout, rate=SAMPLE_RATE
                        )
                        meter = _LevelMeter(channels)
                    except _LEVEL_ERRORS as exc:
                        log.warning("Громкость каналов не посчитана: %s", exc)
                frames = _group_frames(
                    _ignore_invalid_frames(container.decode(audio=0)), _GROUP_SAMPLES
                )
                for frame in itertools.chain(frames, [None]):
                    for out in mono_resampler.resample(frame):
                        mono.write(out.to_ndarray())
                    if level_resampler is not None and meter is not None:
                        try:
                            for out in level_resampler.resample(frame):
                                meter.add(np.asarray(out.to_ndarray(), dtype=np.int16))
                        except _LEVEL_ERRORS as exc:
                            # Громкость — дополнение: без неё запись всё равно распознаётся.
                            log.warning("Громкость каналов не посчитана: %s", exc)
                            level_resampler = meter = None
                del mono_resampler, level_resampler
        except AudioDecodeError:
            raise
        except (av.error.FFmpegError, ValueError, OSError) as exc:
            raise AudioDecodeError(f"Файл не читается как звук: {exc}") from exc
        finally:
            # Ресемплер PyAV освобождает память только после сборки мусора (faster-whisper #390).
            gc.collect()
        if mono.tell() == 0:
            raise AudioDecodeError("В файле нет звука")
        with mono.getbuffer() as pcm:
            # Без копии байтов: на двухчасовой записи это сотни мегабайт.
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            digest = hashlib.sha256(pcm).hexdigest()
        return PyAvAudio(
            mono=samples,
            source_channels=channels,
            pcm_sha256=digest,
            energy=meter.result() if meter is not None else None,
        )
