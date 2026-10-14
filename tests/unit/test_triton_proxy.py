"""Клиент Triton не должен уходить в HTTP-прокси из окружения: Triton — соседний контейнер."""

from concurrent import futures

import grpc
import pytest
from tritonclient.grpc import service_pb2, service_pb2_grpc

from src.infrastructure.triton import connect, readiness_problem


class ReadyServicer(service_pb2_grpc.GRPCInferenceServiceServicer):  # type: ignore[misc]
    def ServerReady(self, request, context):
        return service_pb2.ServerReadyResponse(ready=True)

    def ModelReady(self, request, context):
        return service_pb2.ModelReadyResponse(ready=True)


@pytest.fixture
def fake_triton():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(ReadyServicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield f"127.0.0.1:{port}"
    server.stop(0)


@pytest.fixture
def dead_proxy(monkeypatch):
    # Прокси, который никто не слушает, — как корпоративный прокси, недоступный из сети Docker.
    for name in ("no_proxy", "NO_PROXY", "no_grpc_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("http_proxy", "https_proxy", "grpc_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")


def test_default_client_goes_through_proxy_and_fails(fake_triton, dead_proxy):
    import tritonclient.grpc as triton

    assert readiness_problem(triton.InferenceServerClient(fake_triton), "m", 3) is not None


def test_our_client_ignores_proxy(fake_triton, dead_proxy):
    assert readiness_problem(connect(fake_triton), "m", 3) is None
