"""Энкодер GigaAM в Triton по gRPC и проверка модели на сервере."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.application.errors import EngineError

FEATURES = 64
HIDDEN = 768
VERSION = "1"


def connect(url: str) -> Any:
    """Создать gRPC-клиент Triton. Соединение открывается при первом вызове.

    Параметры канала — умолчания клиента Triton и запрет HTTP-прокси: Triton —
    соседний контейнер, а корпоративный прокси, который Docker может подставить в
    окружение (``http_proxy``), увёл бы туда и gRPC.
    """
    import tritonclient.grpc as grpc

    keepalive = grpc.KeepAliveOptions()
    largest = 2**31 - 1  # как MAX_GRPC_MESSAGE_SIZE клиента Triton
    options = [
        ("grpc.max_send_message_length", largest),
        ("grpc.max_receive_message_length", largest),
        ("grpc.keepalive_time_ms", keepalive.keepalive_time_ms),
        ("grpc.keepalive_timeout_ms", keepalive.keepalive_timeout_ms),
        ("grpc.keepalive_permit_without_calls", keepalive.keepalive_permit_without_calls),
        ("grpc.http2.max_pings_without_data", keepalive.http2_max_pings_without_data),
        ("grpc.enable_http_proxy", 0),
    ]
    return grpc.InferenceServerClient(url, channel_args=options)


def readiness_problem(client: Any, model: str, timeout: float) -> str | None:
    """Проверить, что сервер и модель готовы.

    Возвращает ``None``, если готовы, иначе причину словами — вместе с текстом ошибки
    gRPC, если сервер не ответил. Исключений не бросает.
    """
    try:
        if not client.is_server_ready(client_timeout=timeout):
            return "сервер отвечает, но не готов"
        if not client.is_model_ready(model, VERSION, client_timeout=timeout):
            return f"сервер готов, модель {model} не готова"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def is_ready(client: Any, model: str, timeout: float) -> bool:
    """Готовы ли сервер и модель; для проверки ``/ready``."""
    return readiness_problem(client, model, timeout) is None


def validate_model(client: Any, model: str, timeout: float) -> dict[str, Any]:
    """Сверить настройки модели в Triton с тем, что ждёт служба.

    Нужны: TensorRT на видеокарте, без батчей и накопления запросов, входы
    ``audio_signal`` FP32 ``[1, 64, -1]`` и ``length`` INT64 ``[1]``. Число копий
    модели не проверяется — это настройка нагрузки. Возвращает конфигурацию;
    ``EngineError`` — если она другая или сервер не ответил.
    """
    try:
        response = client.get_model_config(model, VERSION, as_json=True, client_timeout=timeout)
    except Exception as exc:
        raise EngineError(f"Triton не отдал настройки модели {model}: {exc}") from exc
    config: dict[str, Any] = response.get("config", response)
    groups = config.get("instance_group", [])
    if (
        config.get("platform") != "tensorrt_plan"
        or int(config.get("max_batch_size", 0)) != 0
        or "dynamic_batching" in config
        or not groups
        or any(group.get("kind") != "KIND_GPU" for group in groups)
    ):
        raise EngineError("Модель в Triton должна быть TensorRT на видеокарте, без батчей")
    expected = {"audio_signal": ("TYPE_FP32", [1, FEATURES, -1]), "length": ("TYPE_INT64", [1])}
    actual = {
        item["name"]: (item["data_type"], [int(d) for d in item["dims"]])
        for item in config.get("input", [])
    }
    if actual != expected:
        raise EngineError(f"Входы модели в Triton отличаются от ожидаемых: {actual}")
    return config


class TritonEncoder:
    """Вызов энкодера: признаки ``[1, 64, T]`` → выход ``[1, 768, T/4]`` и его длина."""

    def __init__(self, client: Any, model: str, timeout: float) -> None:
        import tritonclient.grpc as grpc

        self._client = client
        self._grpc = grpc
        self._model = model
        self._timeout = timeout

    def forward(
        self, features: NDArray[np.float32], length: int
    ) -> tuple[NDArray[np.float32], int]:
        """Посчитать энкодер на Triton.

        Ошибки: ``ValueError`` — неверная форма признаков; ``EngineError`` — сервер
        не ответил или вернул повреждённые данные.
        """
        if (
            features.ndim != 3
            or features.shape[:2] != (1, FEATURES)
            or features.dtype != np.float32
        ):
            raise ValueError("Энкодеру нужны признаки FP32 [1, 64, T]")
        grpc = self._grpc
        signal = grpc.InferInput("audio_signal", list(features.shape), "FP32")
        signal.set_data_from_numpy(features)
        frames = grpc.InferInput("length", [1], "INT64")
        frames.set_data_from_numpy(np.array([length], dtype=np.int64))
        try:
            response = self._client.infer(
                self._model,
                [signal, frames],
                model_version=VERSION,
                outputs=[
                    grpc.InferRequestedOutput("encoded"),
                    grpc.InferRequestedOutput("encoded_len"),
                ],
                client_timeout=self._timeout,
            )
        except Exception as exc:
            raise EngineError(f"Triton не посчитал энкодер: {exc}") from exc
        encoded = response.as_numpy("encoded")
        encoded_len = response.as_numpy("encoded_len")
        if (
            encoded is None
            or encoded_len is None
            or encoded.ndim != 3
            or encoded.shape[:2] != (1, HIDDEN)
            or encoded.dtype != np.float32
            or not np.isfinite(encoded).all()
        ):
            raise EngineError("Triton вернул повреждённый выход энкодера")
        count = int(np.asarray(encoded_len).reshape(-1)[0])
        if not 0 < count <= encoded.shape[2]:
            raise EngineError("Triton вернул неверную длину выхода энкодера")
        return encoded, count
