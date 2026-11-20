"""Text front-end: clean raw English and turn it into character IDs.

Analogy: before reading aloud, a newsreader rewrites "Dr. Smith paid $12"
as "doctor smith paid twelve dollars". The model only ever sees the
rewritten version, one character at a time.
"""
import re
from typing import Any

# ------------------------------------------------------------ symbol table ---


def build_symbols(text_cfg: dict[str, Any]) -> list[str]:
    """Pad first (id 0), end-of-sentence second (id 1), then punctuation and letters."""
    return [text_cfg["pad_symbol"], text_cfg["eos_symbol"]] + list(text_cfg["punctuation"]) \
        + list(text_cfg["characters"])


# ------------------------------------------------------- number to words ---

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")]
_ORDINAL_FIX = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def number_to_words(n: int) -> str:
    """12 -> 'twelve', 1234 -> 'one thousand two hundred thirty four'."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])
    for value, name in _SCALES:
        if n >= value:
            high, rest = divmod(n, value)
            words = number_to_words(high) + " " + name
            return words if rest == 0 else words + " " + number_to_words(rest)
    raise ValueError(n)  # unreachable


def year_to_words(n: int) -> str:
    """Read years the way people say them: 1984 -> 'nineteen eighty four'."""
    if 2000 <= n <= 2009 or n % 1000 == 0:
        return number_to_words(n)
    high, low = divmod(n, 100)
    if low == 0:
        return number_to_words(high) + " hundred"
    low_words = ("oh " + _ONES[low]) if low < 10 else number_to_words(low)
    return number_to_words(high) + " " + low_words


def ordinal_to_words(n: int) -> str:
    """3 -> 'third', 21 -> 'twenty first', 40 -> 'fortieth'."""
    words = number_to_words(n).split(" ")
    last = words[-1]
    if last in _ORDINAL_FIX:
        words[-1] = _ORDINAL_FIX[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _expand_money(m: re.Match) -> str:
    dollars = int(m.group(1).replace(",", ""))
    cents = int(m.group(2)) if m.group(2) else 0
    parts = []
    if dollars:
        parts.append(number_to_words(dollars) + (" dollar" if dollars == 1 else " dollars"))
    if cents:
        parts.append(number_to_words(cents) + (" cent" if cents == 1 else " cents"))
    return " and ".join(parts) or "zero dollars"


def _expand_plain_number(m: re.Match) -> str:
    n = int(m.group(0))
    if 1100 <= n <= 2099 and len(m.group(0)) == 4:   # four digits in this range: probably a year
        return year_to_words(n)
    return number_to_words(n)


def expand_numbers(text: str) -> str:
    """Rewrite every number in the text as words."""
    text = re.sub(r"\$([\d,]+)(?:\.(\d{2}))?", _expand_money, text)          # $12.50
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)                             # 1,000 -> 1000
    text = re.sub(r"(\d+)\.(\d+)",                                             # 3.14
                  lambda m: number_to_words(int(m.group(1))) + " point "
                  + " ".join(_ONES[int(d)] for d in m.group(2)), text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: ordinal_to_words(int(m.group(1))), text)
    text = re.sub(r"\d+", _expand_plain_number, text)
    return text


# ------------------------------------------------------- abbreviations ---

_ABBREVIATIONS = {
    "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor", "st": "saint", "co": "company",
    "jr": "junior", "sr": "senior", "maj": "major", "gen": "general", "drs": "doctors",
    "rev": "reverend", "lt": "lieutenant", "hon": "honorable", "sgt": "sergeant",
    "capt": "captain", "esq": "esquire", "ltd": "limited", "col": "colonel", "ft": "fort",
    "vs": "versus", "etc": "et cetera", "no": "number",
}
# Only expand when followed by a dot, so the word "no" stays "no".
_ABBREV_RE = re.compile(r"\b(" + "|".join(_ABBREVIATIONS) + r")\.", re.IGNORECASE)


def expand_abbreviations(text: str) -> str:
    """'Dr. Smith' -> 'doctor Smith'."""
    return _ABBREV_RE.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()], text)


# ------------------------------------------------------------- cleaner ---


def clean_text(text: str, text_cfg: dict[str, Any]) -> str:
    """Full cleaning pipeline: abbreviations, numbers, lowercase, allowed symbols only."""
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = text.lower()
    text = text.replace("&", " and ").replace("%", " percent")
    text = re.sub(r"[;:]", ",", text)                 # treat these as a short pause
    text = re.sub(r"[\"“”()\[\]]", " ", text)          # drop quotes and brackets
    text = text.replace("’", "'").replace("—", " - ").replace("–", " - ")
    allowed = set(text_cfg["characters"]) | set(text_cfg["punctuation"])
    text = "".join(ch if ch in allowed else " " for ch in text)  # unknown -> space
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------ text <-> ids ---


def text_to_ids(text: str, text_cfg: dict[str, Any]) -> list[int]:
    """Clean text and map it to IDs, with the end-of-sentence ID appended."""
    symbols = build_symbols(text_cfg)
    lookup = {s: i for i, s in enumerate(symbols)}
    cleaned = clean_text(text, text_cfg)
    return [lookup[ch] for ch in cleaned] + [lookup[text_cfg["eos_symbol"]]]


def ids_to_text(ids: list[int], text_cfg: dict[str, Any]) -> str:
    """Map IDs back to characters, dropping padding and end-of-sentence."""
    symbols = build_symbols(text_cfg)
    skip = {text_cfg["pad_symbol"], text_cfg["eos_symbol"]}
    return "".join(symbols[i] for i in ids if symbols[i] not in skip)
