"""Реплики и паузы записи из фрагментов детектора речи и распознанных токенов."""

from collections.abc import Sequence

from src.domain.transcript import Pause, Reply, Word, to_ms
from src.domain.words import has_words, tokens_to_words, words_text


def make_reply(
    seq: int,
    start: float,
    end: float,
    duration: float,
    text: str,
    tokens: Sequence[str],
    stamps: Sequence[float],
    channels_db: tuple[float, ...] | None,
) -> Reply | None:
    """Собрать реплику из распознанного фрагмента речи.

    Принимает номер реплики, границы фрагмента в секундах от начала записи, его
    длительность (число отсчётов / 16000 — так её считает стенд), текст модели,
    её токены со временем от начала фрагмента и громкость каналов.
    Возвращает реплику или ``None``, если в тексте нет ни букв, ни цифр.
    Текст реплики собирается из слов, а без слов берётся текст модели — как на стенде.
    """
    if not has_words(text):
        return None
    words = tokens_to_words(tokens, stamps, duration)
    body = words_text(words) if words else text.strip()
    return Reply(
        seq=seq,
        start_ms=to_ms(start),
        end_ms=to_ms(end),
        text=body,
        words=tuple(Word(w.text, to_ms(start + w.start), to_ms(start + w.end)) for w in words),
        channels_db=channels_db,
    )


def pauses_between(voice: Sequence[tuple[float, float]], duration: float) -> tuple[Pause, ...]:
    """Найти промежутки вне речи, включая начало и конец записи.

    Принимает границы речи каждого фрагмента детектора в секундах (без полей
    тишины) в порядке времени и длительность записи. Возвращает паузы в
    миллисекундах. Ошибок и побочных эффектов нет.
    """
    found: list[tuple[float, float]] = []
    previous = 0.0
    for voice_start, voice_end in voice:
        start = min(duration, max(0.0, voice_start))
        end = min(duration, max(start, voice_end))
        if start > previous:
            found.append((previous, start))
        previous = max(previous, end)
    if previous < duration:
        found.append((previous, duration))
    return tuple(Pause(to_ms(a), to_ms(b)) for a, b in found)
