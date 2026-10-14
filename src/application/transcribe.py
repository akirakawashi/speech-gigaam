"""Сценарий одной записи: файл → звук → фрагменты речи → реплики и паузы."""

import time

from src.application.errors import AudioTooLongError
from src.application.ports import AudioDecoder, Recognizer, SpeechDetector
from src.domain.replies import make_reply, pauses_between
from src.domain.transcript import SAMPLE_RATE, Reply, Transcript
from src.domain.words import has_words


class TranscribeRecording:
    """Распознать запись целиком в процессе-обработчике.

    Работает синхронно: фрагменты идут друг за другом в порядке времени, как в
    урезанном режиме стенда. Параллельность даёт пул процессов, а не этот сценарий.
    """

    def __init__(
        self,
        decoder: AudioDecoder,
        detector: SpeechDetector,
        recognizer: Recognizer,
        max_audio_seconds: float,
    ) -> None:
        self._decoder = decoder
        self._detector = detector
        self._recognizer = recognizer
        self._max_audio_seconds = max_audio_seconds

    def __call__(self, data: bytes) -> Transcript:
        """Распознать файл.

        Принимает байты файла в любом формате, который читает FFmpeg. Возвращает
        реплики с текстом, словами и громкостью каналов, паузы и сведения о звуке.
        Ошибки: ``AudioDecodeError`` — не звук или пусто; ``AudioTooLongError`` —
        запись длиннее предела; ``EngineError`` — сбой модели. Побочных эффектов нет.
        """
        started = time.perf_counter()
        audio = self._decoder.decode(data)
        samples = audio.mono
        duration = samples.size / SAMPLE_RATE
        if duration > self._max_audio_seconds:
            raise AudioTooLongError(
                f"Запись длится {duration:.0f} с, предел {self._max_audio_seconds:.0f} с"
            )
        segments = self._detector.segments(samples)
        replies: list[Reply] = []
        skipped = 0
        for segment in segments:
            recognition = self._recognizer.recognize(segment.samples)
            if recognition is None:
                skipped += 1
                continue
            if not has_words(recognition.text):
                continue
            reply = make_reply(
                seq=len(replies),
                start=segment.start / SAMPLE_RATE,
                end=segment.end / SAMPLE_RATE,
                duration=segment.samples.size / SAMPLE_RATE,
                text=recognition.text,
                tokens=recognition.tokens,
                stamps=recognition.stamps,
                channels_db=audio.channel_levels(segment.start, segment.end),
            )
            if reply is not None:
                replies.append(reply)
        voice = [
            ((s.start + s.lead) / SAMPLE_RATE, (s.end - s.trail) / SAMPLE_RATE) for s in segments
        ]
        return Transcript(
            duration_s=duration,
            source_channels=audio.source_channels,
            pcm_sha256=audio.pcm_sha256,
            speech_segments=len(segments),
            skipped_short=skipped,
            replies=tuple(replies),
            pauses=pauses_between(voice, duration),
            processing_ms=round((time.perf_counter() - started) * 1000),
        )
