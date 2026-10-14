from pathlib import Path

import numpy as np

from src.infrastructure.decoder import GreedyDecoder, cpu_session
from src.infrastructure.vad import WINDOW, SileroSpeechDetector, StreamingVad, silero_model_path

SR = 16000


class ScriptedVadSession:
    """Сессия ONNX, которая отдаёт заданные вероятности речи по одной на кадр 32 мс."""

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)

    def run(self, _outputs, inputs):
        assert inputs["input"].shape == (1, 64 + WINDOW)
        p = self.probabilities.pop(0) if self.probabilities else 0.0
        return np.array([[p]], dtype=np.float32), inputs["state"]


def run_vad(probabilities, **settings):
    vad = StreamingVad(ScriptedVadSession(probabilities), **settings)
    samples = np.zeros(len(probabilities) * WINDOW, dtype=np.float32)
    found = []
    for offset in range(0, samples.size, 1600):
        found.extend(vad.accept(samples[offset : offset + 1600]))
    return found + vad.flush()


def test_speech_between_silences_becomes_one_segment_with_padding():
    found = run_vad(
        [0.0] * 10 + [0.9] * 30 + [0.0] * 30,
        threshold=0.5,
        min_silence=0.45,
        min_speech=0.25,
        max_speech=15.0,
    )
    assert len(found) == 1
    segment = found[0]
    # Перед речью — поле тишины 0,1 с (4 кадра), после — паузa до порога плюс поле.
    assert segment.lead == 4 * WINDOW and segment.start == 6 * WINDOW
    assert segment.trail == 4 * WINDOW


def test_too_short_speech_is_dropped():
    found = run_vad(
        [0.0] * 10 + [0.9] * 3 + [0.0] * 30,
        threshold=0.5,
        min_silence=0.45,
        min_speech=0.25,
        max_speech=15.0,
    )
    assert found == []


def test_long_speech_is_cut_at_quietest_place():
    probs = [0.9] * 40 + [0.6] + [0.9] * 60
    found = run_vad(probs, threshold=0.5, min_silence=0.45, min_speech=0.25, max_speech=2.0)
    assert len(found) >= 2
    assert all(s.samples.size <= int(2.0 * SR) + WINDOW for s in found)


def test_real_silero_finds_nothing_in_silence():
    detector = SileroSpeechDetector(cpu_session(Path(silero_model_path())), 0.5, 0.45, 0.25, 15.0)
    assert detector.segments(np.zeros(SR * 2, dtype=np.float32)) == []


class FakeDecoderSession:
    def __init__(self):
        self.labels = []

    def run(self, _outputs, inputs):
        self.labels.append(int(inputs["x"][0, 0]))
        hidden = inputs["hi"]
        return [np.zeros((1, 320, 1), dtype=np.float32), hidden, inputs["ci"]]


class FakeJointSession:
    """Выдаёт токены по сценарию: для каждого кадра — список, затем пустой символ."""

    def __init__(self, script, blank):
        self.script = [list(tokens) for tokens in script]
        self.blank = blank
        self.frame = -1
        self.last_enc = None

    def run(self, _outputs, inputs):
        enc = inputs["enc"]
        if self.last_enc is None or not np.array_equal(enc, self.last_enc):
            self.frame += 1
            self.last_enc = enc.copy()
        tokens = self.script[self.frame]
        token = tokens.pop(0) if tokens else self.blank
        logits = np.zeros((1, 1, 1, self.blank + 1), dtype=np.float32)
        logits[0, 0, 0, token] = 1.0
        return [logits]


def test_greedy_decoder_emits_tokens_per_frame_and_caps_symbols():
    blank = 5
    encoded = np.arange(3, dtype=np.float32).reshape(1, 1, 3).repeat(768, axis=1)
    joint = FakeJointSession([[1, 2], [], [3, 3, 3, 3]], blank)
    decoder_session = FakeDecoderSession()
    decoder = GreedyDecoder(
        decoder_session, joint, blank=blank, hidden=320, layers=1, max_symbols=3
    )
    ids, emitted = decoder.decode(encoded, 3)
    assert ids == [1, 2, 3, 3, 3]
    assert emitted == [0, 0, 2, 2, 2]
    # Первый вызов предсказателя — пустой символ при сборке, затем каждый выданный токен.
    assert decoder_session.labels == [blank, 1, 2, 3, 3, 3]


class FakeTritonClient:
    def __init__(self, server=True, model=True, error=None):
        self.server, self.model, self.error = server, model, error

    def is_server_ready(self, client_timeout):
        if self.error:
            raise self.error
        return self.server

    def is_model_ready(self, name, version, client_timeout):
        return self.model


def test_triton_readiness_problem_names_the_cause():
    from src.infrastructure.triton import is_ready, readiness_problem

    assert readiness_problem(FakeTritonClient(), "m", 1) is None and is_ready(
        FakeTritonClient(), "m", 1
    )
    assert (
        readiness_problem(FakeTritonClient(server=False), "m", 1) == "сервер отвечает, но не готов"
    )
    assert (
        readiness_problem(FakeTritonClient(model=False), "m", 1)
        == "сервер готов, модель m не готова"
    )
    problem = readiness_problem(
        FakeTritonClient(error=ConnectionError("failed to connect")), "m", 1
    )
    assert problem == "ConnectionError: failed to connect"
