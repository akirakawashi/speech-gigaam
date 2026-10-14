"""Сборка движка распознавания внутри процесса-обработчика.

``build_engine`` вызывается в каждом процессе пула один раз: загружает модели,
подключается к Triton, сверяет модель на сервере и прогоняет пробный фрагмент.
Функция и её настройки лежат на верхнем уровне модуля — так их можно передать в
процесс, запущенный методом ``spawn``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.application.errors import EngineError
from src.application.transcribe import TranscribeRecording
from src.domain.transcript import SAMPLE_RATE
from src.infrastructure.audio import PyAvDecoder
from src.infrastructure.decoder import GreedyDecoder, Tokenizer, cpu_session
from src.infrastructure.features import LogMelFeatures
from src.infrastructure.model_files import ModelFiles, ModelFilesError
from src.infrastructure.recognizer import GigaamRecognizer
from src.infrastructure.triton import TritonEncoder, connect, readiness_problem, validate_model
from src.infrastructure.vad import SileroSpeechDetector, silero_model_path


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Всё, что нужно обработчику; передаётся в процесс целиком."""

    model: ModelFiles
    triton_url: str
    triton_model: str
    triton_timeout_s: float
    threads: int
    vad_threshold: float
    vad_min_silence: float
    vad_min_speech: float
    vad_max_speech: float
    max_audio_seconds: float


def build_engine(config: EngineConfig) -> TranscribeRecording:
    """Собрать сценарий распознавания с загруженными моделями.

    Ошибки: ``ModelFilesError`` — словарь не подходит к модели; ``EngineError`` —
    Triton или модель на нём не готовы либо настроены не так. Побочные эффекты:
    открывает соединение с Triton и ограничивает потоки PyTorch числом ``threads``.
    """
    import torch

    torch.set_num_threads(config.threads)
    model = config.model
    detector = SileroSpeechDetector(
        cpu_session(Path(silero_model_path())),
        threshold=config.vad_threshold,
        min_silence=config.vad_min_silence,
        min_speech=config.vad_min_speech,
        max_speech=config.vad_max_speech,
    )
    features = LogMelFeatures(model.part("preprocessor.pt"), model.preprocessor)
    tokenizer = Tokenizer(model.part("tokenizer.model"))
    if len(tokenizer) != model.blank_id:
        raise ModelFilesError(
            "Словарь не соответствует модели: размер не равен номеру пустого символа"
        )
    decoder = GreedyDecoder(
        cpu_session(model.part("decoder.onnx")),
        cpu_session(model.part("joint.onnx")),
        blank=model.blank_id,
        hidden=model.pred_hidden,
        layers=model.pred_rnn_layers,
        max_symbols=model.max_symbols,
    )
    client = connect(config.triton_url)
    check_timeout = min(config.triton_timeout_s, 30.0)
    problem = readiness_problem(client, config.triton_model, check_timeout)
    if problem is not None:
        raise EngineError(f"Triton {config.triton_url} не готов: {problem}")
    validate_model(client, config.triton_model, check_timeout)
    recognizer = GigaamRecognizer(
        features,
        TritonEncoder(client, config.triton_model, config.triton_timeout_s),
        decoder,
        tokenizer,
        min_frames=model.min_frames,
        max_frames=model.max_frames,
    )
    # Прогрев: первый вызов энкодера и сессий ONNX дольше остальных.
    noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)
    recognizer.recognize(noise)
    detector.segments(np.zeros(SAMPLE_RATE // 10, dtype=np.float32))
    return TranscribeRecording(PyAvDecoder(), detector, recognizer, config.max_audio_seconds)
