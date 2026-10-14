"""Поиск речи Silero VAD на процессоре.

Детектор перенесён со стенда temptest (``app/vad.py``) без изменения логики: модель
каждые 32 мс даёт вероятность речи, фрагмент начинается выше порога и кончается
после паузы ``min_silence``; речь без пауз дольше ``max_speech`` режется по самому
тихому месту второй половины. Запись подаётся порциями по 100 мс, как на стенде.
"""

import importlib.util
import os
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.application.ports import Samples, SpeechSegment
from src.domain.transcript import SAMPLE_RATE

WINDOW = 512
CONTEXT = 64
_FEED = 1600


def silero_model_path() -> str:
    """Путь к ``silero_vad.onnx`` из пакета silero-vad, без импорта самого пакета.

    Ошибка ``FileNotFoundError`` — если пакета или файла модели нет.
    """
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Пакет silero-vad не установлен")
    path = os.path.join(list(spec.submodule_search_locations)[0], "data", "silero_vad.onnx")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


class _Segment:
    """Фрагмент речи внутри детектора: начало, звук, вероятности и поля тишины."""

    __slots__ = ("start", "samples", "lead", "trail")

    def __init__(self, start: int, samples: NDArray[np.float32], lead: int, trail: int) -> None:
        self.start = start
        self.samples = samples
        self.lead = lead
        self.trail = trail


class StreamingVad:
    """Потоковый детектор; состояние модели живёт в пределах одной записи."""

    def __init__(
        self,
        session: Any,
        threshold: float,
        min_silence: float,
        min_speech: float,
        max_speech: float,
        pad: float = 0.10,
    ) -> None:
        self.sess = session
        self.threshold = threshold
        self.neg_threshold = max(0.05, threshold - 0.15)
        self.min_silence = int(min_silence * SAMPLE_RATE)
        self.min_speech = int(min_speech * SAMPLE_RATE)
        self.max_speech = int(max_speech * SAMPLE_RATE)
        self.pad = int(pad * SAMPLE_RATE)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Сбросить состояние модели и текущий фрагмент."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._rest = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._preroll: deque[NDArray[np.float32]] = deque(maxlen=8)
        self._frames: list[NDArray[np.float32]] = []
        self._probs: list[float] = []
        self._seg_start = 0
        self._lead = 0
        self._silence_at: int | None = None
        self.speaking = False

    def _prob(self, frame: NDArray[np.float32]) -> float:
        x = np.concatenate([self._context, frame])[None, :]
        out, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = frame[-CONTEXT:]
        return float(out[0][0])

    def accept(self, x: NDArray[np.float32]) -> list[_Segment]:
        """Принять порцию звука; вернуть завершившиеся фрагменты речи."""
        out: list[_Segment] = []
        buf = np.concatenate([self._rest, x.astype(np.float32, copy=False)])
        n = buf.size // WINDOW
        for i in range(n):
            frame = buf[i * WINDOW : (i + 1) * WINDOW]
            seg = self._step(frame, self._prob(frame))
            if seg is not None:
                out.append(seg)
            self._pos += WINDOW
        self._rest = buf[n * WINDOW :].copy()
        return out

    def _step(self, frame: NDArray[np.float32], p: float) -> _Segment | None:
        if not self.speaking:
            if p >= self.threshold:
                self.speaking = True
                pre = list(self._preroll)[-max(1, self.pad // WINDOW + 1) :]
                self._frames = pre + [frame]
                self._probs = [self.neg_threshold] * len(pre) + [p]
                self._seg_start = self._pos - len(pre) * WINDOW
                self._lead = len(pre) * WINDOW
                self._silence_at = None
                self._preroll.clear()
            else:
                self._preroll.append(frame)
            return None

        self._frames.append(frame)
        self._probs.append(p)
        k = len(self._frames) - 1
        if p >= self.threshold:
            self._silence_at = None
        elif p < self.neg_threshold and self._silence_at is None:
            self._silence_at = k

        if self._silence_at is not None and (k - self._silence_at + 1) * WINDOW >= self.min_silence:
            return self._close(self._silence_at)
        if len(self._frames) * WINDOW >= self.max_speech:
            return self._close_long()
        return None

    def _close(self, silence_from: int) -> _Segment | None:
        keep = min(len(self._frames), silence_from + self.pad // WINDOW + 1)
        seg = self._make(keep, trail=max(0, keep - silence_from) * WINDOW)
        tail = self._frames[keep:]
        self._preroll.clear()
        self._preroll.extend(tail[-(self._preroll.maxlen or 0) :])
        self._frames, self._probs = [], []
        self.speaking = False
        self._silence_at = None
        return seg

    def _close_long(self) -> _Segment | None:
        probs = np.asarray(self._probs)
        half = len(probs) // 2
        cut = half + int(np.argmin(probs[half:]))
        if probs[cut] >= self.threshold or len(probs) - cut < 3:
            cut = len(probs)
        seg = self._make(cut)
        rest_frames, rest_probs = self._frames[cut:], self._probs[cut:]
        self._seg_start += cut * WINDOW
        self._lead = 0
        self._frames, self._probs = rest_frames, rest_probs
        self._silence_at = None
        if not self._frames:
            self._seg_start = self._pos + WINDOW
        return seg

    def _make(self, n_frames: int, trail: int = 0) -> _Segment | None:
        if n_frames <= 0:
            return None
        samples = np.concatenate(self._frames[:n_frames])
        if samples.size - self._lead - trail < self.min_speech:
            return None
        return _Segment(self._seg_start, samples, self._lead, trail)

    def flush(self) -> list[_Segment]:
        """Конец записи: вернуть незавершённый фрагмент, если он есть."""
        out: list[_Segment] = []
        if self.speaking and self._frames:
            end = self._silence_at if self._silence_at is not None else len(self._frames)
            keep = min(len(self._frames), end + self.pad // WINDOW + 1)
            seg = self._make(keep, trail=max(0, keep - end) * WINDOW)
            if seg is not None:
                out.append(seg)
        pos = self._pos + self._rest.size
        self.reset()
        self._pos = pos
        return out


class SileroSpeechDetector:
    """Поиск речи во всей записи: новый детектор на каждую запись, одна сессия ONNX."""

    def __init__(
        self,
        session: Any,
        threshold: float,
        min_silence: float,
        min_speech: float,
        max_speech: float,
    ) -> None:
        self._session = session
        self._settings = (threshold, min_silence, min_speech, max_speech)

    def segments(self, samples: Samples) -> list[SpeechSegment]:
        """Найти фрагменты речи в записи.

        Принимает отсчёты моно 16 кГц, кратные 1/32768, как их отдаёт декодер. Стенд
        пропускал запись через PCM 16 бит (вход микрофона); для таких отсчётов это
        тождество, поэтому лишние копии записи здесь не делаются. Звук подаётся
        порциями по 100 мс, как на стенде. Возвращает фрагменты в порядке времени.
        Побочных эффектов нет.
        """
        vad = StreamingVad(self._session, *self._settings)
        found: list[_Segment] = []
        for offset in range(0, samples.size, _FEED):
            found.extend(vad.accept(samples[offset : offset + _FEED]))
        found.extend(vad.flush())
        return [SpeechSegment(s.start, s.samples, s.lead, s.trail) for s in found]
