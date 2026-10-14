"""Папка модели в формате стенда temptest и её проверка при старте.

Состав папки (``benchmark-cache/gigaam-trt`` стенда)::

    артефакты.json                    описание экспорта: модель, словарь, файлы и SHA256
    движок.json                       описание движка TensorRT: SHA256, версия, профиль
    artifacts/v3_e2e_rnnt.yaml        параметры модели
    artifacts/v3_e2e_rnnt_decoder.onnx, _joint.onnx, _preprocessor.pt, _tokenizer.model
    repository/gigaam_encoder/1/model.plan, config.pbtxt   — это читает Triton
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MODEL_NAME = "v3_e2e_rnnt"
TRITON_MODEL = "gigaam_encoder"
_USED_PARTS = ("decoder.onnx", "joint.onnx", "preprocessor.pt", "tokenizer.model")
_SHAPE = re.compile(r"--(min|max)Shapes=audio_signal:1x64x(\d+)")


class ModelFilesError(RuntimeError):
    """Папка модели неполная или не совпадает со своим описанием."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelFilesError(f"Не читается {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelFilesError(f"{path}: ожидался объект JSON")
    return data


@dataclass(frozen=True, slots=True)
class ModelFiles:
    """Проверенная папка модели: пути, параметры декодера и контрольные суммы."""

    root: Path
    blank_id: int
    max_symbols: int
    pred_hidden: int
    pred_rnn_layers: int
    preprocessor: dict[str, object]
    checkpoint_sha256: str
    engine_sha256: str
    tensorrt: str
    min_frames: int
    max_frames: int

    def part(self, name: str) -> Path:
        """Путь к части модели в ``artifacts``, например ``decoder.onnx``."""
        return self.root / "artifacts" / f"{MODEL_NAME}_{name}"

    @property
    def engine_path(self) -> Path:
        """Путь к движку TensorRT, который читает Triton."""
        return self.root / "repository" / TRITON_MODEL / "1" / "model.plan"

    @classmethod
    def load(cls, root: Path, verify_engine: bool) -> "ModelFiles":
        """Прочитать и проверить папку модели.

        Сверяет размеры и SHA256 частей, нужных службе, со ``артефакты.json`` и, если
        ``verify_engine``, SHA256 движка со ``движок.json``. Файл энкодера ONNX не
        нужен службе и не проверяется. Ошибка ``ModelFilesError`` — при любом
        несовпадении. Побочных эффектов нет; движок читается целиком, это 1–2 с.
        """
        try:
            return cls._load(root, verify_engine)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ModelFilesError(
                f"Описание модели в {root} неполное или неверное: {exc!r}"
            ) from exc

    @classmethod
    def _load(cls, root: Path, verify_engine: bool) -> "ModelFiles":
        artifacts = _json(root / "артефакты.json")
        engine = _json(root / "движок.json")
        if artifacts.get("model") != MODEL_NAME:
            raise ModelFilesError(
                f"Ожидалась модель {MODEL_NAME}, в описании {artifacts.get('model')}"
            )
        files = artifacts.get("files", {})
        for name in (*_USED_PARTS, "yaml"):
            filename = f"{MODEL_NAME}.yaml" if name == "yaml" else f"{MODEL_NAME}_{name}"
            path = root / "artifacts" / filename
            expected = files.get(filename)
            if not isinstance(expected, dict):
                raise ModelFilesError(f"В артефакты.json нет описания {filename}")
            if not path.is_file() or path.stat().st_size != expected.get("size_bytes"):
                raise ModelFilesError(f"Файл {path} отсутствует или другого размера")
            if _sha256(path) != expected.get("sha256"):
                raise ModelFilesError(f"SHA256 файла {path} не совпадает с артефакты.json")
        config = yaml.safe_load(
            (root / "artifacts" / f"{MODEL_NAME}.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(config, dict) or config.get("model_class") != "rnnt":
            raise ModelFilesError("Ожидалась модель RNN-T")
        preprocessor = config.get("preprocessor")
        if not isinstance(preprocessor, dict):
            raise ModelFilesError("В параметрах модели нет раздела preprocessor")
        shapes = {
            kind: int(frames)
            for kind, frames in _SHAPE.findall(
                " ".join(str(part) for part in engine.get("fingerprint", {}).get("command", []))
            )
        }
        if set(shapes) != {"min", "max"}:
            raise ModelFilesError("В движок.json нет профиля входа энкодера (min/max)")
        model = cls(
            root=root,
            blank_id=int(artifacts["blank_id"]),
            max_symbols=int(artifacts["max_symbols"]),
            pred_hidden=int(artifacts["pred_hidden"]),
            pred_rnn_layers=int(artifacts["pred_rnn_layers"]),
            preprocessor=preprocessor,
            checkpoint_sha256=str(artifacts.get("source", {}).get("checkpoint_sha256", "")),
            engine_sha256=str(engine.get("engine_sha256", "")),
            tensorrt=str(engine.get("fingerprint", {}).get("trtexec_version", "")),
            min_frames=shapes["min"],
            max_frames=shapes["max"],
        )
        if not model.checkpoint_sha256 or not model.engine_sha256:
            raise ModelFilesError("В описании нет SHA256 весов или движка")
        if verify_engine:
            if not model.engine_path.is_file():
                raise ModelFilesError(f"Нет движка {model.engine_path}")
            if _sha256(model.engine_path) != model.engine_sha256:
                raise ModelFilesError(
                    "SHA256 движка не совпадает с движок.json: движок пересобран?"
                )
        return model
