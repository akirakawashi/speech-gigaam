from fastapi.testclient import TestClient

from src.application.dto import ModelInfo
from src.application.errors import AudioDecodeError, RequestTimeoutError, ServiceBusyError
from src.application.ports import Outcome, ServiceStatus
from src.domain.transcript import Pause, Reply, Transcript, Word
from src.presentation.http.app import HttpLimits, create_app

MODEL = ModelInfo(
    model_name="v3_e2e_rnnt", model_digest="w" * 64, engine_digest="e" * 64, tensorrt="10.11.0"
)


class FakeService:
    def __init__(self):
        self.started = self.stopped = False
        self.error = None
        self.received = None
        self.ready = True

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def transcribe(self, data):
        self.received = data
        if self.error:
            raise self.error
        transcript = Transcript(
            duration_s=3.5,
            source_channels=2,
            pcm_sha256="p" * 64,
            speech_segments=2,
            skipped_short=0,
            replies=(Reply(0, 100, 2100, "Алло", (Word("алло", 300, 700),), (-20.0, -45.5)),),
            pauses=(Pause(0, 300),),
            processing_ms=42,
        )
        return Outcome(transcript=transcript, wait_ms=7)

    def status(self):
        return ServiceStatus(
            ready=self.ready, workers=2, alive=2 if self.ready else 0, busy=1, waiting=0, capacity=4
        )


def client(service, triton=True, limit=1024):
    app = create_app(
        service, MODEL, lambda: triton, HttpLimits(max_upload_bytes=limit, retry_after_s=3)
    )
    return TestClient(app)


def test_file_is_transcribed():
    service = FakeService()
    with client(service) as http:
        response = http.post(
            "/v1/transcriptions",
            content=b"audio",
            headers={"Content-Type": "application/octet-stream"},
        )
    assert service.started and service.stopped and service.received == b"audio"
    assert response.status_code == 200
    body = response.json()
    assert (
        body["engine"] == "gigaam"
        and body["model_name"] == "v3_e2e_rnnt"
        and body["language"] == "ru"
    )
    assert body["model_digest"] == "w" * 64 and body["engine_digest"] == "e" * 64
    assert body["replies"] == [
        {
            "seq": 0,
            "start_ms": 100,
            "end_ms": 2100,
            "text": "Алло",
            "words": [{"text": "алло", "start_ms": 300, "end_ms": 700}],
            "channels_db": [-20.0, -45.5],
        }
    ]
    assert body["pauses"] == [{"start_ms": 0, "end_ms": 300}]
    assert body["duration_s"] == 3.5 and body["wait_ms"] == 7 and body["processing_ms"] == 42


def test_errors_have_codes_statuses_and_retry_hint():
    service = FakeService()
    with client(service) as http:
        service.error = AudioDecodeError("не звук")
        bad = http.post("/v1/transcriptions", content=b"x")
        service.error = ServiceBusyError("занято")
        busy = http.post("/v1/transcriptions", content=b"x")
        service.error = RequestTimeoutError("долго")
        slow = http.post("/v1/transcriptions", content=b"x")
    assert bad.status_code == 422 and bad.json() == {
        "error": {"code": "bad_audio", "message": "не звук"}
    }
    assert busy.status_code == 503 and busy.headers["Retry-After"] == "3"
    assert busy.json()["error"]["code"] == "busy"
    assert slow.status_code == 504 and "Retry-After" not in slow.headers


def test_too_large_file_is_rejected_before_transcription():
    service = FakeService()
    with client(service, limit=10) as http:
        response = http.post("/v1/transcriptions", content=b"x" * 11)
    assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"
    assert service.received is None


def test_readiness_needs_workers_and_triton():
    service = FakeService()
    with client(service) as http:
        assert http.get("/health").json() == {"status": "ok"}
        ready = http.get("/ready")
        assert ready.status_code == 200 and ready.json()["status"] == "ready"
    with client(service, triton=False) as http:
        assert http.get("/ready").status_code == 503
    service.ready = False
    with client(service) as http:
        not_ready = http.get("/ready")
        assert not_ready.status_code == 503 and not_ready.json()["alive"] == 0


def test_model_endpoint():
    with client(FakeService()) as http:
        assert http.get("/v1/model").json()["tensorrt"] == "10.11.0"


def test_unexpected_error_is_json_internal():
    service = FakeService()
    service.error = KeyError("сломалось")
    app = create_app(
        service, MODEL, lambda: True, HttpLimits(max_upload_bytes=1024, retry_after_s=3)
    )
    with TestClient(app, raise_server_exceptions=False) as http:
        response = http.post("/v1/transcriptions", content=b"x")
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal", "message": "KeyError"}}
