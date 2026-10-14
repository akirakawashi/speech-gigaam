import hashlib
import json
import os
from pathlib import Path

import pytest

from src.infrastructure.model_files import MODEL_NAME, ModelFiles, ModelFilesError

YAML = """model_class: rnnt
preprocessor:
  _target_: gigaam.preprocess.FeatureExtractor
  sample_rate: 16000
  features: 64
  win_length: 320
  hop_length: 160
  n_fft: 320
  center: false
"""


def make_model_dir(root: Path) -> Path:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    files = {}
    contents = {
        f"{MODEL_NAME}.yaml": YAML.encode(),
        f"{MODEL_NAME}_decoder.onnx": b"decoder",
        f"{MODEL_NAME}_joint.onnx": b"joint",
        f"{MODEL_NAME}_preprocessor.pt": b"pre",
        f"{MODEL_NAME}_tokenizer.model": b"tok",
    }
    for name, data in contents.items():
        (artifacts / name).write_bytes(data)
        files[name] = {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (root / "артефакты.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "blank_id": 1024,
                "max_symbols": 10,
                "pred_hidden": 320,
                "pred_rnn_layers": 1,
                "source": {"checkpoint_sha256": "c" * 64},
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    engine = root / "repository" / "gigaam_encoder" / "1" / "model.plan"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"plan")
    (root / "движок.json").write_text(
        json.dumps(
            {
                "engine_sha256": hashlib.sha256(b"plan").hexdigest(),
                "fingerprint": {
                    "trtexec_version": "10.11.0",
                    "command": [
                        "trtexec",
                        "--minShapes=audio_signal:1x64x16,length:1",
                        "--maxShapes=audio_signal:1x64x3200,length:1",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_valid_folder_is_loaded(tmp_path):
    model = ModelFiles.load(make_model_dir(tmp_path), verify_engine=True)
    assert (model.blank_id, model.max_symbols, model.min_frames, model.max_frames) == (
        1024,
        10,
        16,
        3200,
    )
    assert model.preprocessor["n_fft"] == 320
    assert model.part("decoder.onnx").name == f"{MODEL_NAME}_decoder.onnx"
    assert model.tensorrt == "10.11.0" and model.checkpoint_sha256 == "c" * 64


def test_changed_part_is_rejected(tmp_path):
    root = make_model_dir(tmp_path)
    (root / "artifacts" / f"{MODEL_NAME}_joint.onnx").write_bytes(b"JOINT")
    with pytest.raises(ModelFilesError, match="SHA256"):
        ModelFiles.load(root, verify_engine=False)


def test_rebuilt_engine_is_rejected_only_when_checked(tmp_path):
    root = make_model_dir(tmp_path)
    (root / "repository" / "gigaam_encoder" / "1" / "model.plan").write_bytes(b"other")
    with pytest.raises(ModelFilesError, match="движка"):
        ModelFiles.load(root, verify_engine=True)
    assert ModelFiles.load(root, verify_engine=False).engine_sha256


def test_missing_part_is_rejected(tmp_path):
    root = make_model_dir(tmp_path)
    os.remove(root / "artifacts" / f"{MODEL_NAME}_tokenizer.model")
    with pytest.raises(ModelFilesError):
        ModelFiles.load(root, verify_engine=False)


def test_incomplete_description_is_a_clear_error(tmp_path):
    root = make_model_dir(tmp_path)
    data = json.loads((root / "артефакты.json").read_text(encoding="utf-8"))
    del data["blank_id"]
    (root / "артефакты.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelFilesError, match="неполное"):
        ModelFiles.load(root, verify_engine=False)
