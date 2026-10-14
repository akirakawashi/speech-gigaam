from dataclasses import dataclass, field

import numpy as np
import pytest

from src.application.errors import AudioTooLongError
from src.application.ports import Recognition, SpeechSegment
from src.application.transcribe import TranscribeRecording
from src.domain.transcript import Pause

SR = 16000


@dataclass
class FakeAudio:
    mono: np.ndarray
    source_channels: int = 2
    pcm_sha256: str = "sha"
    levels: list = field(default_factory=list)

    def channel_levels(self, start: int, end: int):
        self.levels.append((start, end))
        return (-10.0, -30.0)


class FakeDecoder:
    def __init__(self, audio):
        self.audio = audio

    def decode(self, data: bytes):
        return self.audio


class FakeDetector:
    def __init__(self, segments):
        self._segments = segments

    def segments(self, samples):
        return self._segments


class FakeRecognizer:
    def __init__(self, answers):
        self.answers = list(answers)

    def recognize(self, samples):
        return self.answers.pop(0)


def segment(start_s: float, seconds: float, lead: float = 0.1, trail: float = 0.1) -> SpeechSegment:
    return SpeechSegment(
        int(start_s * SR),
        np.zeros(int(seconds * SR), dtype=np.float32),
        int(lead * SR),
        int(trail * SR),
    )


def test_replies_are_numbered_and_empty_fragments_skipped():
    audio = FakeAudio(np.zeros(10 * SR, dtype=np.float32))
    segments = [segment(1, 2), segment(4, 1), segment(6, 0.3), segment(7, 2)]
    answers = [
        Recognition("алло", ("▁алло",), (0.2,)),
        Recognition(" . ", ("▁.",), (0.1,)),
        None,
        Recognition("да", ("▁да",), (0.3,)),
    ]
    engine = TranscribeRecording(
        FakeDecoder(audio), FakeDetector(segments), FakeRecognizer(answers), 3600
    )
    result = engine(b"file")
    assert [(r.seq, r.text, r.start_ms, r.end_ms) for r in result.replies] == [
        (0, "Алло", 1000, 3000),
        (1, "Да", 7000, 9000),
    ]
    assert result.speech_segments == 4 and result.skipped_short == 1
    assert result.replies[0].channels_db == (-10.0, -30.0)
    # Громкость считается только для реплик с текстом.
    assert audio.levels == [(16000, 48000), (112000, 144000)]
    assert result.pauses[0] == Pause(0, 1100)
    assert result.duration_s == 10.0 and result.pcm_sha256 == "sha" and result.source_channels == 2


def test_too_long_record_is_rejected_before_recognition():
    audio = FakeAudio(np.zeros(3 * SR, dtype=np.float32))
    engine = TranscribeRecording(FakeDecoder(audio), FakeDetector([]), FakeRecognizer([]), 2)
    with pytest.raises(AudioTooLongError):
        engine(b"file")


def test_record_without_speech_has_no_replies_and_one_pause():
    audio = FakeAudio(np.zeros(2 * SR, dtype=np.float32))
    result = TranscribeRecording(FakeDecoder(audio), FakeDetector([]), FakeRecognizer([]), 60)(b"x")
    assert result.replies == () and result.pauses == (Pause(0, 2000),)
