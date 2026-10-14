import pytest

from src.domain.replies import make_reply, pauses_between
from src.domain.transcript import Pause
from src.domain.words import has_words, tokens_to_words, words_text


def test_tokens_to_words_and_text():
    tokens = [" Доб", "рый", " день", ".", " Как", " дела", "?"]
    stamps = [0.1, 0.3, 0.6, 0.8, 1.4, 1.7, 2.0]
    words = tokens_to_words(tokens, stamps, 2.5)
    assert [w.text for w in words] == ["Добрый", "день.", "Как", "дела?"]
    assert words[0].start == pytest.approx(0.1) and words[0].end <= words[1].start
    assert words[-1].end <= 2.5
    assert words_text(words) == "Добрый день. Как дела?"


def test_words_text_strips_dialogue_dash_and_capitalises():
    assert words_text(
        tokens_to_words([" —", " какие", " работы", "?"], [0.1, 0.2, 0.5, 0.7], 1.0)
    ) == ("Какие работы?")
    assert words_text(tokens_to_words([" работы", ".", " —"], [0.1, 0.4, 0.6], 1.0)) == "Работы."
    assert words_text(
        tokens_to_words([" кто", "-", "то", " пришёл"], [0.1, 0.2, 0.3, 0.6], 1.0)
    ) == ("Кто-то пришёл")


def test_word_end_uses_unrounded_start_of_next_word():
    words = tokens_to_words(["▁а", "▁б"], [0.1, 0.12345], 1.0)
    # Конец первого слова — начало второго до округления: min(max(0.18, 0.12345), 1.0).
    assert words[0].end == 0.18
    assert words[1].start == 0.123


def test_word_end_is_capped_by_duration():
    words = tokens_to_words(["▁да"], [0.95], 1.0)
    assert words[0].end == 1.0


def test_has_words():
    assert has_words("да")
    assert has_words("15")
    assert not has_words(" — . ?")


def test_make_reply_uses_words_and_absolute_time():
    reply = make_reply(
        3, 10.0, 12.0, 2.0, "добрый день", ["▁добрый", "▁день"], [0.2, 0.9], (-20.0, -40.0)
    )
    assert reply is not None
    assert reply.seq == 3 and reply.start_ms == 10_000 and reply.end_ms == 12_000
    assert reply.text == "Добрый день"
    assert [(w.text, w.start_ms) for w in reply.words] == [("добрый", 10_200), ("день", 10_900)]
    assert reply.channels_db == (-20.0, -40.0)


def test_make_reply_skips_text_without_letters():
    assert make_reply(0, 0.0, 1.0, 1.0, " . ", ["▁."], [0.1], None) is None


def test_make_reply_falls_back_to_model_text_without_tokens():
    reply = make_reply(0, 0.0, 1.0, 1.0, " да ", [], [], None)
    assert reply is not None and reply.text == "да" and reply.words == ()


def test_pauses_between_speech_and_edges():
    pauses = pauses_between([(1.0, 2.0), (1.5, 3.0), (4.0, 4.5)], 5.0)
    assert pauses == (Pause(0, 1000), Pause(3000, 4000), Pause(4500, 5000))


def test_pauses_are_clamped_to_record():
    assert pauses_between([(-0.5, 0.2), (0.9, 9.0)], 1.0) == (Pause(200, 900),)
    assert pauses_between([], 2.0) == (Pause(0, 2000),)
