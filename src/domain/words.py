"""Сборка слов и текста реплики из токенов модели.

Правила перенесены со стенда temptest (``app/session.py``) без изменений: от них
зависит совпадение текста службы со стендом слово в слово.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

_WORD_START = (" ", "▁")
_EDGE_DASH = re.compile(r"\s+[—–-]+$")


@dataclass(frozen=True, slots=True)
class TokenWord:
    """Слово из токенов; время в секундах от начала фрагмента речи."""

    text: str
    start: float
    end: float
    raw: tuple[str, ...]


@dataclass(slots=True)
class _Draft:
    """Слово в процессе сборки; поля меняются, пока идут его токены."""

    text: str
    start: float
    end: float
    raw: list[str]


def tokens_to_words(
    tokens: Sequence[str], stamps: Sequence[float], duration: float
) -> list[TokenWord]:
    """Собрать слова из токенов модели: токен с пробелом в начале открывает слово.

    Принимает токены, время каждого токена в секундах от начала фрагмента и
    длительность фрагмента. Возвращает слова; конец слова — начало следующего,
    но не дольше 0,6 с после последнего токена и не позже конца фрагмента.
    Ошибок и побочных эффектов нет. Лишние токены или отметки без пары пропускаются.
    """
    drafts: list[_Draft] = []
    for token, stamp in zip(tokens, stamps, strict=False):
        if not token:
            continue
        if token[0] in _WORD_START or not drafts:
            drafts.append(_Draft(token.lstrip(" ▁"), float(stamp), float(stamp), [token]))
        else:
            drafts[-1].text += token
            drafts[-1].raw.append(token)
        drafts[-1].end = float(stamp)
    for index, draft in enumerate(drafts):
        following = drafts[index + 1].start if index + 1 < len(drafts) else duration
        # Начало следующего слова берётся до его округления — как на стенде.
        draft.end = round(min(max(draft.end + 0.08, min(following, draft.end + 0.6)), duration), 3)
        draft.start = round(draft.start, 3)
    return [TokenWord(d.text, d.start, d.end, tuple(d.raw)) for d in drafts]


def words_text(words: Sequence[TokenWord]) -> str:
    """Склеить текст реплики из слов.

    Убирает тире диалога в начале и в конце реплики и делает первую букву
    заглавной. Возвращает текст; ошибок и побочных эффектов нет.
    """
    text = "".join("".join(word.raw) for word in words).replace("▁", " ").strip()
    text = text.lstrip("—–- ")
    text = _EDGE_DASH.sub("", text)
    return text[:1].upper() + text[1:] if text else text


def has_words(text: str) -> bool:
    """Проверить, есть ли в распознанном тексте хоть одна буква или цифра.

    Фрагмент без букв и цифр репликой не считается — как на стенде.
    """
    return any(char.isalnum() for char in text)
