import os
from pathlib import Path

import numpy as np
import pytest

from src.setting.settings import Settings


def test_queue_defaults_to_number_of_workers(monkeypatch):
    monkeypatch.setenv("GIGAAM_WORKERS", "6")
    settings = Settings()
    assert settings.queue_limit == 6 and settings.max_upload_bytes == 200 * 1024 * 1024
    monkeypatch.setenv("GIGAAM_QUEUE_LIMIT", "0")
    assert Settings().queue_limit == 0


MODEL_DIR = os.environ.get("GIGAAM_TEST_MODEL_DIR")


@pytest.mark.skipif(not MODEL_DIR, reason="нужна папка модели: GIGAAM_TEST_MODEL_DIR")
def test_features_from_model_weights():
    from src.infrastructure.features import LogMelFeatures
    from src.infrastructure.model_files import ModelFiles

    model = ModelFiles.load(Path(str(MODEL_DIR)), verify_engine=False)
    features = LogMelFeatures(model.part("preprocessor.pt"), model.preprocessor)
    values, length = features(np.zeros(16000, dtype=np.float32))
    assert values.shape == (1, 64, 99) and length == 99 and values.dtype == np.float32
