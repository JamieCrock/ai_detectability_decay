"""Small synthetic corpus for tests and the zero-download demo.

The generator writes short paragraphs in two deliberately different registers:
a formal, connective-heavy "AI-style" register and a casual, bursty
"human-style" register. It exists so the test suite and the web demo run
offline in seconds. It is **not** a substitute for real data - train on HC3
(``make data``) for any meaningful result.
"""

from __future__ import annotations

import random

import pandas as pd

TOPICS = (
    ("remote work", "video calls", "productivity"),
    ("electric cars", "charging points", "running costs"),
    ("online learning", "recorded lectures", "flexibility"),
    ("social media", "short videos", "attention spans"),
    ("home cooking", "meal prep", "saving money"),
    ("public transport", "bus routes", "commuting"),
    ("smartphones", "notifications", "sleep"),
    ("renting a flat", "letting agents", "deposits"),
    ("learning to code", "online tutorials", "career changes"),
    ("gym memberships", "group classes", "motivation"),
    ("streaming services", "subscriptions", "monthly budgets"),
    ("climate change", "recycling schemes", "local councils"),
)

AI_OPENERS = (
    "{s} has become a significant aspect of modern life.",
    "{s} plays a crucial role in how many individuals approach daily life.",
    "In recent years, {s} has attracted considerable attention.",
    "{s} is an increasingly important topic for numerous individuals.",
)
AI_BODY = (
    "Additionally, it is important to note that {o} can significantly enhance {b}.",
    "Furthermore, numerous individuals utilize {o} in order to improve {b}.",
    "Moreover, {o} provides a comprehensive framework that facilitates {b}.",
    "However, it is essential to consider the potential drawbacks of {o}.",
    "Consequently, organizations must ensure that {o} is implemented effectively.",
    "It is worth noting that {o} does not guarantee improvements in {b}.",
    "Research indicates that {o} is typically associated with better {b}.",
)
AI_CLOSERS = (
    "In conclusion, {s} offers various benefits, although careful consideration is required.",
    "Overall, {s} represents a valuable opportunity to enhance {b}.",
    "Ultimately, the impact of {s} depends on how individuals choose to utilize it.",
)

HUMAN_LINES = (
    "Honestly I've been doing {s} for about two years now and it's fine.",
    "Not gonna lie, {o} drive me up the wall!",
    "My mate swears by {o} but I'm not convinced.",
    "Is it worth it? Depends who you ask.",
    "We tried it last winter. Total disaster, to be fair.",
    "I don't really get the hype around {o}, if I'm honest.",
    "The {b} thing is overblown imo - it's mostly about habits.",
    "Can't complain though.",
    "My sister reckons {s} saved her a fortune, but she's like that.",
    "Took me ages to sort out {o}, and I still mess it up sometimes!",
    "Anyway, that's my two cents.",
    "Would I do it again? Probably, yeah.",
    "Weirdly the best bit was {b}, which I didn't expect at all.",
)


def generate_corpus(n_per_class: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return a shuffled DataFrame with ``text`` and ``label`` (1 = AI-style)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_per_class):
        s, o, b = rng.choice(TOPICS)
        sentences = [rng.choice(AI_OPENERS)]
        sentences += rng.sample(AI_BODY, rng.randint(3, 5))
        sentences.append(rng.choice(AI_CLOSERS))
        text = " ".join(x.format(s=s, o=o, b=b) for x in sentences)
        rows.append({"text": text[0].upper() + text[1:], "label": 1, "topic": s})

        s, o, b = rng.choice(TOPICS)
        lines = rng.sample(HUMAN_LINES, rng.randint(5, 8))
        text = " ".join(x.format(s=s, o=o, b=b) for x in lines)
        rows.append({"text": text, "label": 0, "topic": s})

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
