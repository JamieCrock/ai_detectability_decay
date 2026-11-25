"""Evasion transformations applied to AI-generated text.

Each transformation is a pure function ``f(text, rng, strength) -> text``
that nudges a document towards human-like style without changing its meaning
much. They mimic what people actually do to slip past detectors: swapping in
plainer words, adding contractions and fillers, varying sentence length,
cutting stock connectives and introducing small typos.

Transformations are composed by :class:`TransformationSchedule` and applied
*successively* - generation ``t`` is produced from generation ``t-1`` - which
is what the decay experiment measures.

Everything is seeded, so the same seed always produces the same text.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .features import split_sentences

Transform = Callable[[str, random.Random, float], str]

# Formal or "LLM-flavoured" vocabulary mapped to plainer alternatives.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "utilize": ("use",),
    "utilise": ("use",),
    "utilization": ("use",),
    "additionally": ("also", "plus"),
    "furthermore": ("also", "on top of that"),
    "moreover": ("also", "and"),
    "however": ("but", "still"),
    "therefore": ("so",),
    "consequently": ("so", "as a result"),
    "significant": ("big", "major", "real"),
    "significantly": ("a lot", "hugely"),
    "crucial": ("key", "vital"),
    "essential": ("key", "needed"),
    "numerous": ("lots of", "many"),
    "individuals": ("people",),
    "individual": ("person",),
    "demonstrate": ("show",),
    "demonstrates": ("shows",),
    "facilitate": ("help", "make easier"),
    "obtain": ("get",),
    "purchase": ("buy",),
    "assist": ("help",),
    "commence": ("start",),
    "approximately": ("about", "around"),
    "sufficient": ("enough",),
    "regarding": ("about", "on"),
    "various": ("different", "all sorts of"),
    "ensure": ("make sure",),
    "enhance": ("improve", "boost"),
    "optimal": ("best",),
    "comprehensive": ("full", "thorough"),
    "potential": ("possible",),
    "typically": ("usually", "normally"),
    "particularly": ("especially",),
    "ultimately": ("in the end",),
    "overall": ("all in all", "generally"),
    "important": ("big", "key"),
    "difficult": ("hard", "tough"),
    "beneficial": ("helpful", "good"),
    "require": ("need",),
    "requires": ("needs",),
    "indicate": ("show", "suggest"),
    "indicates": ("shows", "suggests"),
    "individually": ("one by one",),
    "frequently": ("often",),
    "rapidly": ("quickly", "fast"),
    "primarily": ("mainly", "mostly"),
    "subsequently": ("then", "later"),
    "nevertheless": ("still", "even so"),
    "delve": ("dig",),
    "leverage": ("use",),
    "robust": ("solid", "strong"),
    "vital": ("key",),
}

CONTRACTIONS: tuple[tuple[str, str], ...] = (
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("is not", "isn't"),
    ("are not", "aren't"),
    ("was not", "wasn't"),
    ("cannot", "can't"),
    ("can not", "can't"),
    ("will not", "won't"),
    ("would not", "wouldn't"),
    ("should not", "shouldn't"),
    ("it is", "it's"),
    ("that is", "that's"),
    ("there is", "there's"),
    ("they are", "they're"),
    ("we are", "we're"),
    ("you are", "you're"),
    ("i am", "I'm"),
    ("i have", "I've"),
    ("they have", "they've"),
    ("you will", "you'll"),
    ("it will", "it'll"),
)

LEADING_MARKERS = re.compile(
    r"^(Additionally|Furthermore|Moreover|However|Overall|Ultimately|Consequently|"
    r"Therefore|Thus|Notably|Importantly|In conclusion|In summary|In addition|"
    r"As a result|On the other hand|It is important to note that|It's important to note that|"
    r"It is worth noting that),?\s+",
    flags=re.IGNORECASE,
)

FILLERS = (
    "Honestly, ",
    "I think ",
    "To be fair, ",
    "Basically, ",
    "Tbh ",
    "Thing is, ",
    "In my experience, ",
    "Mind you, ",
    "Funnily enough, ",
)

_WORD_RE = re.compile(r"[A-Za-z]+")


def _match_case(replacement: str, original: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _lower_first(sentence: str) -> str:
    if sentence[:2] == "I " or sentence[:2] == "I'":
        return sentence
    return sentence[:1].lower() + sentence[1:]


def _upper_first(sentence: str) -> str:
    return sentence[:1].upper() + sentence[1:]


def synonym_swap(text: str, rng: random.Random, strength: float) -> str:
    """Replace formal vocabulary with plainer synonyms."""

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        options = SYNONYMS.get(word.lower())
        if options and rng.random() < strength:
            return _match_case(rng.choice(options), word)
        return word

    return _WORD_RE.sub(repl, text)


def contract(text: str, rng: random.Random, strength: float) -> str:
    """Turn expanded forms ("do not") into contractions ("don't")."""
    for full, short in CONTRACTIONS:
        pattern = re.compile(rf"\b{re.escape(full)}\b", flags=re.IGNORECASE)

        def repl(match: re.Match[str], short: str = short) -> str:
            if rng.random() >= strength:
                return match.group(0)
            return _match_case(short, match.group(0))

        text = pattern.sub(repl, text)
    return text


def drop_discourse_markers(text: str, rng: random.Random, strength: float) -> str:
    """Remove stock sentence-initial connectives ("Moreover, ...")."""
    out = []
    for sentence in split_sentences(text):
        if LEADING_MARKERS.match(sentence) and rng.random() < strength:
            sentence = _upper_first(LEADING_MARKERS.sub("", sentence, count=1))
        out.append(sentence)
    return " ".join(out)


def vary_sentence_length(text: str, rng: random.Random, strength: float) -> str:
    """Increase burstiness by merging short sentences and splitting long ones."""
    sentences = split_sentences(text)
    out: list[str] = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        n_words = len(current.split())
        roll = rng.random()
        if roll < strength / 2 and i + 1 < len(sentences) and n_words < 25:
            joiner = rng.choice((", and ", " - ", ", so "))
            merged = current.rstrip(".!?") + joiner + _lower_first(sentences[i + 1])
            out.append(merged)
            i += 2
            continue
        if roll < strength and n_words > 18:
            for sep in (", and ", "; ", ", but "):
                if sep in current:
                    head, tail = current.split(sep, 1)
                    out.append(head.rstrip(",;") + ".")
                    current = "But " + tail if sep == ", but " else _upper_first(tail)
                    break
        out.append(current)
        i += 1
    return " ".join(out)


def insert_fillers(text: str, rng: random.Random, strength: float) -> str:
    """Prepend conversational fillers to a fraction of sentences."""
    out = []
    for sentence in split_sentences(text):
        if rng.random() < strength * 0.5:
            sentence = rng.choice(FILLERS) + _lower_first(sentence)
        out.append(sentence)
    return " ".join(out)


def punctuation_shift(text: str, rng: random.Random, strength: float) -> str:
    """Swap some commas for dashes and some semicolons for full stops."""

    def comma(match: re.Match[str]) -> str:
        return " - " if rng.random() < strength * 0.3 else match.group(0)

    def semicolon(match: re.Match[str]) -> str:
        if rng.random() < strength:
            return ". " + match.group(1).upper()
        return match.group(0)

    text = re.sub(r", ", comma, text)
    text = re.sub(r"; ([a-z])", semicolon, text)
    return text


def typo_noise(text: str, rng: random.Random, strength: float) -> str:
    """Introduce occasional adjacent-character swaps in longer words."""

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        if len(word) > 4 and rng.random() < strength * 0.04:
            i = rng.randrange(1, len(word) - 2)
            return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        return word

    return _WORD_RE.sub(repl, text)


def sentence_swap(text: str, rng: random.Random, strength: float) -> str:
    """Swap a few adjacent sentence pairs (keeps the opening sentence fixed)."""
    sentences = split_sentences(text)
    for i in range(1, len(sentences) - 1):
        if rng.random() < strength * 0.2:
            sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]
    return " ".join(sentences)


TRANSFORMS: dict[str, Transform] = {
    "synonym_swap": synonym_swap,
    "contract": contract,
    "drop_discourse_markers": drop_discourse_markers,
    "vary_sentence_length": vary_sentence_length,
    "insert_fillers": insert_fillers,
    "punctuation_shift": punctuation_shift,
    "typo_noise": typo_noise,
    "sentence_swap": sentence_swap,
}


@dataclass
class TransformationSchedule:
    """Applies ``per_step`` randomly chosen transforms at every generation.

    Parameters
    ----------
    names:
        Which transforms may be sampled (defaults to all of them).
    per_step:
        How many distinct transforms are applied in each generation.
    strength:
        Probability-like intensity in ``[0, 1]`` passed to each transform.
    seed:
        Base seed; generation ``t`` of document ``i`` uses ``(seed, t, i)``.
    """

    names: Sequence[str] = field(default_factory=lambda: tuple(TRANSFORMS))
    per_step: int = 2
    strength: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        unknown = set(self.names) - set(TRANSFORMS)
        if unknown:
            raise ValueError(f"Unknown transforms: {sorted(unknown)}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        if not 1 <= self.per_step <= len(self.names):
            raise ValueError("per_step must be between 1 and the number of transforms")

    def apply(self, text: str, generation: int, doc_index: int = 0) -> tuple[str, list[str]]:
        """Apply one generation of transforms and return ``(text, applied_names)``."""
        rng = random.Random(f"{self.seed}-{generation}-{doc_index}")
        chosen = rng.sample(list(self.names), self.per_step)
        for name in chosen:
            text = TRANSFORMS[name](text, rng, self.strength)
        return text, chosen

    def apply_all(self, texts: Sequence[str], generation: int) -> list[str]:
        return [self.apply(t, generation, i)[0] for i, t in enumerate(texts)]
