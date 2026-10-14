"""Распознавание фрагмента: признаки → энкодер в Triton → жадный декодер → токены."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.application.errors import EngineError
from src.application.ports import Recognition, Samples
from src.domain.transcript import SAMPLE_RATE
from src.infrastructure.decoder import GreedyDecoder, Tokenizer
from src.infrastructure.features import LogMelFeatures


class Encoder(Protocol):
    """Энкодер: признаки → выход и число кадров."""

    def forward(
        self, features: NDArray[np.float32], length: int
    ) -> tuple[NDArray[np.float32], int]:
        """Посчитать энкодер."""
        ...


class GigaamRecognizer:
    """GigaAM v3 e2e RNN-T: энкодер на сервере моделей, остальное на процессоре."""

    def __init__(
        self,
        features: LogMelFeatures,
        encoder: Encoder,
        decoder: GreedyDecoder,
        tokenizer: Tokenizer,
        min_frames: int,
        max_frames: int,
    ) -> None:
        self._features = features
        self._encoder = encoder
        self._decoder = decoder
        self._tokenizer = tokenizer
        self._min_frames = min_frames
        self._max_frames = max_frames

    def recognize(self, samples: Samples) -> Recognition | None:
        """Распознать фрагмент речи.

        Возвращает текст, токены и время токенов в секундах от начала фрагмента;
        ``None`` — если фрагмент короче минимального входа движка. ``EngineError`` —
        фрагмент длиннее профиля движка или сбой сервера моделей.
        """
        values, length = self._features(samples)
        if length < self._min_frames:
            return None
        if length > self._max_frames:
            raise EngineError(
                f"Фрагмент речи {length / 100:.1f} с длиннее профиля движка "
                f"{self._max_frames / 100:.0f} с"
            )
        encoded, frames = self._encoder.forward(values, length)
        ids, emitted = self._decoder.decode(encoded, frames)
        shift = samples.size / SAMPLE_RATE / frames
        return Recognition(
            text=self._tokenizer.decode(ids).strip(),
            tokens=tuple(self._tokenizer.piece(i) for i in ids),
            stamps=tuple(float(frame * shift) for frame in emitted),
        )
