import random

import pytest

from detectability.features import extract_features
from detectability.transforms import (
    TRANSFORMS,
    TransformationSchedule,
    contract,
    drop_discourse_markers,
    insert_fillers,
    synonym_swap,
    vary_sentence_length,
)

AI_TEXT = (
    "Additionally, it is important to note that numerous individuals utilize online tools. "
    "Furthermore, they do not always demonstrate significant improvements in productivity. "
    "However, these tools can facilitate collaboration across various teams, and they are typically "
    "inexpensive to adopt for small organisations that are growing quickly. "
    "In conclusion, it is essential to ensure that adoption is carefully planned."
)


def rng(seed=0):
    return random.Random(seed)


def test_synonym_swap_full_strength_replaces_formal_words():
    out = synonym_swap(AI_TEXT, rng(), 1.0).lower()
    for word in ("utilize", "numerous", "individuals", "facilitate"):
        assert word not in out


def test_synonym_swap_preserves_capitalisation():
    out = synonym_swap("Additionally, we go.", rng(), 1.0)
    assert out[0].isupper()


def test_contract_creates_contractions():
    out = contract("They do not know. It is late.", rng(), 1.0)
    assert out == "They don't know. It's late."


def test_drop_discourse_markers_removes_leading_connectives():
    out = drop_discourse_markers(AI_TEXT, rng(), 1.0)
    assert not out.startswith("Additionally")
    assert "In conclusion" not in out
    assert "Furthermore" not in out


def test_vary_sentence_length_changes_structure():
    out = vary_sentence_length(AI_TEXT, rng(3), 1.0)
    assert out != AI_TEXT


def test_insert_fillers_makes_text_more_conversational():
    out = insert_fillers(AI_TEXT, rng(1), 1.0)
    assert len(out) > len(AI_TEXT)


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_every_transform_is_deterministic_and_zero_strength_is_safe(name):
    fn = TRANSFORMS[name]
    assert fn(AI_TEXT, rng(7), 0.6) == fn(AI_TEXT, rng(7), 0.6)
    out = fn(AI_TEXT, rng(7), 0.0)
    assert len(out.split()) == pytest.approx(len(AI_TEXT.split()), abs=2)


def test_schedule_is_reproducible_and_reports_applied_transforms():
    schedule = TransformationSchedule(per_step=3, strength=0.8, seed=11)
    a, applied_a = schedule.apply(AI_TEXT, generation=1)
    b, applied_b = schedule.apply(AI_TEXT, generation=1)
    assert a == b and applied_a == applied_b
    assert len(applied_a) == 3 and len(set(applied_a)) == 3


def test_successive_generations_move_style_towards_human():
    schedule = TransformationSchedule(per_step=3, strength=0.8, seed=5)
    text = AI_TEXT
    for gen in range(1, 7):
        text, _ = schedule.apply(text, gen)
    before, after = extract_features(AI_TEXT), extract_features(text)
    assert after["discourse_marker_rate"] < before["discourse_marker_rate"]


@pytest.mark.parametrize(
    "kwargs",
    [{"names": ["nope"]}, {"strength": 1.5}, {"per_step": 0}, {"names": ["contract"], "per_step": 2}],
)
def test_schedule_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        TransformationSchedule(**kwargs)
