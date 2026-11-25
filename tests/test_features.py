import math

import numpy as np
import pytest

from detectability.features import (
    FEATURE_NAMES,
    count_syllables,
    extract_features,
    feature_matrix,
    split_sentences,
    tokenize,
)


def test_every_feature_is_present_and_finite():
    feats = extract_features("This is a short test. It has two sentences!")
    assert set(feats) == set(FEATURE_NAMES)
    assert all(math.isfinite(v) for v in feats.values())


def test_empty_text_returns_zeros_not_errors():
    feats = extract_features("")
    assert feats["n_words"] == 0
    assert feats["type_token_ratio"] == 0
    assert feats["flesch_reading_ease"] == 0


def test_sentence_split_and_tokenize():
    text = "Dr Smith arrived. She didn't stay long! Why? 3.5 hours later she left."
    assert len(split_sentences(text)) == 4
    tokens = tokenize(text)
    assert "didn't" in tokens
    assert "3.5" in tokens


@pytest.mark.parametrize("word,expected", [("cat", 1), ("table", 2), ("banana", 3), ("rhythm", 1), ("make", 1)])
def test_syllable_heuristic(word, expected):
    assert count_syllables(word) == expected


def test_burstiness_is_higher_for_varied_sentence_lengths():
    uniform = "The cat sat on the mat. The dog lay on the rug. The bird sat in the tree."
    varied = "Stop. The dog, after a very long and tiring day out in the fields, lay on the rug. Why?"
    assert extract_features(varied)["burstiness"] > extract_features(uniform)["burstiness"]


def test_style_features_capture_ai_vs_human_markers():
    ai = "Additionally, it is important to note that individuals utilize tools. Moreover, this is crucial."
    human = "I don't think it's that deep tbh. We tried it and honestly it was fine!"
    f_ai, f_human = extract_features(ai), extract_features(human)
    assert f_ai["discourse_marker_rate"] > f_human["discourse_marker_rate"]
    assert f_human["contraction_rate"] > f_ai["contraction_rate"]
    assert f_human["first_person_rate"] > f_ai["first_person_rate"]
    assert f_human["exclamation_rate"] > 0


def test_type_token_ratio_and_repetition():
    repetitive = "the cat the cat the cat the cat"
    feats = extract_features(repetitive)
    assert feats["type_token_ratio"] == pytest.approx(2 / 8)
    assert feats["repeated_bigram_rate"] > 0.5


def test_feature_matrix_shape_and_order():
    X = feature_matrix(["One two three.", "Four five six seven. Eight!"])
    assert X.shape == (2, len(FEATURE_NAMES))
    assert X[1, FEATURE_NAMES.index("n_sentences")] == 2
    assert np.isfinite(X).all()


def test_feature_matrix_empty_input():
    assert feature_matrix([]).shape == (0, len(FEATURE_NAMES))
