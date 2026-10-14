"""Лог-мел признаки GigaAM на процессоре с окном и мел-фильтрами из весов модели.

Модуль повторяет ``gigaam.preprocess.FeatureExtractor`` (MIT, GigaChat Team) той же
структурой, чтобы состояние ``*_preprocessor.pt`` загружалось без переименований.
Окно и мел-фильтры берутся из весов обученной модели: созданные заново, они дают
другой текст примерно в четверти фрагментов — это найдено на стенде temptest.
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from numpy.typing import NDArray

from src.application.ports import Samples


class _SpecScaler(torch.nn.Module):
    """Натуральный логарифм спектра с ограничением снизу и сверху, как в GigaAM."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class _FeatureExtractor(torch.nn.Module):
    """Та же структура слоёв, что у ``gigaam.preprocess.FeatureExtractor``."""

    def __init__(
        self,
        sample_rate: int,
        features: int,
        win_length: int,
        hop_length: int,
        n_fft: int,
        center: bool,
    ) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.featurizer = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=features,
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
                center=center,
            ),
            _SpecScaler(),
        )

    def out_len(self, length: int) -> int:
        """Число кадров признаков для ``length`` отсчётов."""
        if self.center:
            return length // self.hop_length + 1
        return (length - self.win_length) // self.hop_length + 1


class LogMelFeatures:
    """Признаки фрагмента для энкодера: FP32 ``[1, 64, кадры]`` и число кадров."""

    def __init__(self, state_path: Path, preprocessor: dict[str, object]) -> None:
        """Построить извлекатель по параметрам модели и загрузить окно и фильтры.

        ``preprocessor`` — раздел ``preprocessor`` из ``v3_e2e_rnnt.yaml``. Ошибки:
        ``ValueError`` — если параметры не те, что у GigaAM v3; ``RuntimeError`` от
        PyTorch — если состояние не подходит к слоям.
        """
        if preprocessor.get("_target_") != "gigaam.preprocess.FeatureExtractor":
            raise ValueError(f"Неизвестный извлекатель признаков: {preprocessor.get('_target_')}")
        self._module = _FeatureExtractor(
            sample_rate=int(str(preprocessor["sample_rate"])),
            features=int(str(preprocessor["features"])),
            win_length=int(str(preprocessor["win_length"])),
            hop_length=int(str(preprocessor["hop_length"])),
            n_fft=int(str(preprocessor["n_fft"])),
            center=bool(preprocessor.get("center", True)),
        ).eval()
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self._module.load_state_dict(state, strict=True)

    def __call__(self, samples: Samples) -> tuple[NDArray[np.float32], int]:
        """Посчитать признаки фрагмента; побочных эффектов нет."""
        with torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
            values = self._module.featurizer(wav)
        return np.ascontiguousarray(values.numpy(), dtype=np.float32), self._module.out_len(
            int(wav.shape[-1])
        )
