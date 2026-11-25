"""Interpretable linguistic features for AI-text detection.

Every feature is computed with the standard library and NumPy only, so the
extractor is fast, deterministic and easy to audit. Features fall into five
groups:

* **Structure** - sentence/word counts and lengths, burstiness.
* **Lexical richness** - type-token ratio, hapax ratio, Yule's K.
* **Function words & style** - stopword, pronoun, contraction and
  discourse-marker rates.
* **Punctuation & characters** - per-word rates of punctuation marks,
  uppercase/digit ratios and character entropy.
* **Readability & repetition** - Flesch reading ease and repeated-bigram rate.

Large language models tend to write with uniform sentence lengths, heavy use of
discourse markers ("moreover", "overall") and few contractions or first-person
pronouns. These features make those tendencies measurable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*")
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because
    been before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself just me more most my myself
    no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours
    yourself yourselves""".split()
)

FIRST_PERSON = frozenset("i me my mine myself we us our ours ourselves".split())
SECOND_PERSON = frozenset("you your yours yourself yourselves".split())

# Connectives that are heavily over-represented in instruction-tuned LLM output.
DISCOURSE_MARKERS = frozenset(
    """additionally furthermore moreover however overall consequently therefore
    thus hence notably importantly ultimately essentially specifically
    conversely nevertheless nonetheless""".split()
)
DISCOURSE_PHRASES = (
    "in conclusion",
    "in summary",
    "it is important to note",
    "it's important to note",
    "on the other hand",
    "in addition",
    "as a result",
    "in other words",
    "it is worth noting",
    "plays a crucial role",
)

HEDGES = frozenset("may might could perhaps possibly generally typically often".split())

FEATURE_NAMES: tuple[str, ...] = (
    # structure
    "n_words",
    "n_sentences",
    "mean_sentence_len",
    "std_sentence_len",
    "burstiness",
    "mean_word_len",
    "std_word_len",
    "long_word_ratio",
    # lexical richness
    "type_token_ratio",
    "hapax_ratio",
    "yules_k",
    # function words & style
    "stopword_ratio",
    "first_person_rate",
    "second_person_rate",
    "contraction_rate",
    "discourse_marker_rate",
    "hedge_rate",
    # punctuation & characters
    "comma_rate",
    "semicolon_rate",
    "colon_rate",
    "exclamation_rate",
    "question_rate",
    "dash_rate",
    "paren_rate",
    "quote_rate",
    "uppercase_ratio",
    "digit_ratio",
    "char_entropy",
    # readability & repetition
    "flesch_reading_ease",
    "repeated_bigram_rate",
    "paragraph_rate",
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using a punctuation + capitalisation heuristic."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; contractions such as "don't" stay one token."""
    return [w.lower() for w in _WORD.findall(text)]


def count_syllables(word: str) -> int:
    """Approximate syllable count (vowel-group heuristic, minimum one)."""
    word = word.lower()
    groups = _VOWEL_GROUPS.findall(word)
    count = len(groups)
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ee")):
        count -= 1
    return max(count, 1)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _yules_k(counts: Counter[str], n_tokens: int) -> float:
    """Yule's K lexical-repetition statistic (higher = more repetitive)."""
    if n_tokens < 2:
        return 0.0
    freq_of_freq = Counter(counts.values())
    m2 = sum(freq * (i**2) for i, freq in freq_of_freq.items())
    return 1e4 * (m2 - n_tokens) / (n_tokens**2)


def _char_entropy(text: str) -> float:
    """Shannon entropy (bits) of the lower-cased character distribution."""
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_features(text: str) -> dict[str, float]:
    """Compute every feature in :data:`FEATURE_NAMES` for a single document."""
    sentences = split_sentences(text)
    tokens = tokenize(text)
    n_words = len(tokens)
    n_sent = max(len(sentences), 1) if n_words else 0

    sent_lens = np.array([len(tokenize(s)) for s in sentences] or [0], dtype=float)
    word_lens = np.array([len(t) for t in tokens] or [0], dtype=float)
    mean_sl = float(sent_lens.mean())
    std_sl = float(sent_lens.std())

    counts = Counter(tokens)
    hapax = sum(1 for c in counts.values() if c == 1)
    lowered = text.lower()
    n_chars = len(text)
    letters = sum(ch.isalpha() for ch in text)

    discourse = sum(counts[m] for m in DISCOURSE_MARKERS) + sum(lowered.count(p) for p in DISCOURSE_PHRASES)

    bigrams = list(zip(tokens, tokens[1:], strict=False))
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)

    syllables = sum(count_syllables(t) for t in tokens if t.isalpha())
    if n_words:
        flesch = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / n_words)
    else:
        flesch = 0.0

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    feats = {
        "n_words": float(n_words),
        "n_sentences": float(n_sent),
        "mean_sentence_len": mean_sl,
        "std_sentence_len": std_sl,
        "burstiness": _safe_div(std_sl, mean_sl),
        "mean_word_len": float(word_lens.mean()),
        "std_word_len": float(word_lens.std()),
        "long_word_ratio": _safe_div(sum(len(t) >= 7 for t in tokens), n_words),
        "type_token_ratio": _safe_div(len(counts), n_words),
        "hapax_ratio": _safe_div(hapax, n_words),
        "yules_k": _yules_k(counts, n_words),
        "stopword_ratio": _safe_div(sum(counts[w] for w in STOPWORDS), n_words),
        "first_person_rate": _safe_div(sum(counts[w] for w in FIRST_PERSON), n_words),
        "second_person_rate": _safe_div(sum(counts[w] for w in SECOND_PERSON), n_words),
        "contraction_rate": _safe_div(sum(1 for t in tokens if "'" in t), n_words),
        "discourse_marker_rate": _safe_div(discourse, n_words),
        "hedge_rate": _safe_div(sum(counts[w] for w in HEDGES), n_words),
        "comma_rate": _safe_div(text.count(","), n_words),
        "semicolon_rate": _safe_div(text.count(";"), n_words),
        "colon_rate": _safe_div(text.count(":"), n_words),
        "exclamation_rate": _safe_div(text.count("!"), n_words),
        "question_rate": _safe_div(text.count("?"), n_words),
        "dash_rate": _safe_div(text.count("-") + text.count("—") + text.count("–"), n_words),
        "paren_rate": _safe_div(text.count("("), n_words),
        "quote_rate": _safe_div(text.count('"') + text.count("“"), n_words),
        "uppercase_ratio": _safe_div(sum(ch.isupper() for ch in text), letters),
        "digit_ratio": _safe_div(sum(ch.isdigit() for ch in text), n_chars),
        "char_entropy": _char_entropy(text),
        "flesch_reading_ease": float(flesch),
        "repeated_bigram_rate": _safe_div(repeated_bigrams, max(len(bigrams), 1)),
        "paragraph_rate": _safe_div(len(paragraphs), n_sent),
    }
    return feats


def feature_matrix(texts: Iterable[str]) -> np.ndarray:
    """Vectorise many documents into an ``(n_docs, n_features)`` float matrix."""
    rows = [[extract_features(t)[name] for name in FEATURE_NAMES] for t in texts]
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)))
    matrix = np.asarray(rows, dtype=float)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
