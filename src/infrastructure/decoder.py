"""Жадный декодер RNN-T на ONNX Runtime и словарь SentencePiece.

Декодер перенесён со стенда temptest (``benchmark/gigaam_trt.py``): тот же разбор,
что в официальном пакете GigaAM, — до ``max_symbols`` символов на кадр энкодера.
"""

from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece
from numpy.typing import NDArray


def cpu_session(path: Path) -> Any:
    """Открыть модель ONNX на процессоре в один поток, как на стенде.

    Ошибки ONNX Runtime пробрасываются: без модели служба не стартует.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class GreedyDecoder:
    """Жадный разбор выхода энкодера в номера токенов."""

    def __init__(
        self, decoder: Any, joint: Any, blank: int, hidden: int, layers: int, max_symbols: int
    ) -> None:
        self._decoder = decoder
        self._joint = joint
        self._blank = blank
        self._max_symbols = max_symbols
        zeros = np.zeros((layers, 1, hidden), dtype=np.float32)
        # Вложение пустого символа в весах нулевое: тот же старт, что у PyTorch с нулевым входом.
        self._start = self._predict(blank, zeros, zeros)

    def _predict(self, label: int, h: NDArray[np.float32], c: NDArray[np.float32]) -> Any:
        return self._decoder.run(None, {"x": np.array([[label]], dtype=np.int64), "hi": h, "ci": c})

    def decode(self, encoded: NDArray[np.float32], frames: int) -> tuple[list[int], list[int]]:
        """Разобрать ``frames`` кадров энкодера ``[1, 768, T]``.

        Возвращает номера токенов и номер кадра, на котором появился каждый токен.
        """
        ids: list[int] = []
        emitted: list[int] = []
        dec, h, c = self._start
        for t in range(frames):
            frame = np.ascontiguousarray(encoded[:, :, t : t + 1])
            for _ in range(self._max_symbols):
                logits = self._joint.run(None, {"enc": frame, "dec": dec.swapaxes(1, 2)})[0]
                token = int(logits[0, 0, 0].argmax())
                if token == self._blank:
                    break
                ids.append(token)
                emitted.append(t)
                dec, h, c = self._predict(token, h, c)
        return ids, emitted


class Tokenizer:
    """Словарь SentencePiece модели GigaAM."""

    def __init__(self, path: Path) -> None:
        self._model = sentencepiece.SentencePieceProcessor()
        self._model.load(str(path))

    def __len__(self) -> int:
        return int(self._model.get_piece_size())

    def decode(self, ids: list[int]) -> str:
        """Текст из номеров токенов."""
        return str(self._model.decode(ids))

    def piece(self, token_id: int) -> str:
        """Токен как строка, с маркером начала слова ``▁``."""
        return str(self._model.id_to_piece(token_id))
