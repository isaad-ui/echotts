"""Tests for the text front-end."""
from echotts.text import (build_symbols, clean_text, ids_to_text, number_to_words,
                          text_to_ids)


def test_symbol_table_order(cfg) -> None:
    symbols = build_symbols(cfg["text"])
    assert symbols[0] == "_" and symbols[1] == "~"
    assert len(symbols) == len(set(symbols))


def test_numbers() -> None:
    assert number_to_words(12) == "twelve"
    assert number_to_words(105) == "one hundred five"
    assert number_to_words(2_000_001) == "two million one"


def test_clean_text(cfg) -> None:
    t = cfg["text"]
    assert clean_text("Dr. Smith paid $12.", t) == "doctor smith paid twelve dollars."
    assert clean_text("In 1984, the 3rd one.", t) == "in nineteen eighty four, the third one."
    assert clean_text("Hello   World; ok", t) == "hello world, ok"
    assert clean_text("café #1", t) == "caf one"


def test_roundtrip(cfg) -> None:
    t = cfg["text"]
    for sentence in ["Hello, world!", "Dr. Who is 12 years old?", "It's 3.5 percent."]:
        ids = text_to_ids(sentence, t)
        assert ids[-1] == 1                       # ends with end-of-sentence
        assert 0 not in ids                       # pad id never appears in real text
        assert ids_to_text(ids, t) == clean_text(sentence, t)
