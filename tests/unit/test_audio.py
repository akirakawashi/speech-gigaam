import hashlib
import io
import wave

import numpy as np
import pytest

from src.application.errors import AudioDecodeError
from src.infrastructure.audio import PyAvDecoder


def wav_bytes(channels: list[np.ndarray], rate: int) -> bytes:
    pcm = (
        np.stack(channels, axis=1).astype("<i2") if len(channels) > 1 else channels[0].astype("<i2")
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(len(channels))
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def tone(seconds: float, rate: int, amplitude: float) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 440 * t) * amplitude * 32767).astype(np.int16)


def test_stereo_record_gives_mono_16k_and_channel_levels():
    data = wav_bytes([tone(2.0, 8000, 0.5), tone(2.0, 8000, 0.05)], 8000)
    audio = PyAvDecoder().decode(data)
    assert audio.source_channels == 2
    assert abs(audio.mono.size - 32000) < 64 and audio.mono.dtype == np.float32
    levels = audio.channel_levels(0, audio.mono.size)
    assert levels is not None and len(levels) == 2
    # Синус амплитудой 0,5 — около −9 дБ, амплитудой 0,05 — около −29 дБ.
    assert levels[0] == pytest.approx(-9.0, abs=0.5)
    assert levels[1] == pytest.approx(-29.0, abs=0.5)
    pcm = (audio.mono * 32768).astype("<i2").tobytes()
    assert audio.pcm_sha256 == hashlib.sha256(pcm).hexdigest()


def test_levels_of_silence_and_part_of_record():
    left = np.concatenate([np.zeros(8000, dtype=np.int16), tone(1.0, 8000, 0.5)])
    data = wav_bytes([left, np.zeros_like(left)], 8000)
    audio = PyAvDecoder().decode(data)
    first = audio.channel_levels(0, 16000)
    second = audio.channel_levels(16000, 32000)
    # Ресемплер слегка размывает начало тона в тишину перед ним — это доли процента.
    assert first is not None and first[0] < -60 and first[1] == -120.0
    assert second is not None and second[0] > -12 and second[1] == -120.0


def test_mono_record_has_no_channel_levels():
    audio = PyAvDecoder().decode(wav_bytes([tone(1.0, 16000, 0.3)], 16000))
    assert audio.source_channels == 1 and audio.channel_levels(0, 100) is None
    assert audio.mono.size == 16000


@pytest.mark.parametrize("data", [b"", b"not an audio file at all" * 10])
def test_garbage_is_rejected(data):
    with pytest.raises(AudioDecodeError):
        PyAvDecoder().decode(data)


def test_channel_level_failure_does_not_break_decoding(monkeypatch):
    import av

    real = av.audio.resampler.AudioResampler

    class FailingForChannels:
        def __init__(self, format, layout, rate):
            self._inner = real(format=format, layout=layout, rate=rate)
            self._mono = layout == "mono"

        def resample(self, frame):
            if not self._mono:
                raise ValueError("раскладка не поддерживается")
            return self._inner.resample(frame)

    monkeypatch.setattr(av.audio.resampler, "AudioResampler", FailingForChannels)
    audio = PyAvDecoder().decode(wav_bytes([tone(1.0, 8000, 0.5), tone(1.0, 8000, 0.1)], 8000))
    assert audio.mono.size > 15000 and audio.source_channels == 2
    assert audio.channel_levels(0, 1000) is None


def test_decoded_samples_survive_pcm_round_trip_unchanged():
    # Поиск речи на стенде пропускал запись через PCM 16 бит; для отсчётов декодера
    # это тождество — на нём держится отказ от лишних копий в SileroSpeechDetector.
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    samples = pcm.astype(np.float32) / 32768.0
    round_trip = (np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2").astype(
        np.float32
    ) / 32768.0
    assert np.array_equal(samples.view(np.uint32), round_trip.view(np.uint32))
