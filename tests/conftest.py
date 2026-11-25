import pytest

from detectability import data
from detectability.evaluate import compare_detectors
from detectability.synthetic import generate_corpus


@pytest.fixture(scope="session")
def corpus():
    return generate_corpus(n_per_class=120, seed=1)


@pytest.fixture(scope="session")
def splits(corpus):
    return data.split(corpus, test_size=0.25, seed=1)


@pytest.fixture(scope="session")
def trained(splits):
    """Every detector fitted once per test session."""
    table, bundles = compare_detectors(
        splits.train,
        splits.test,
        ["random_forest", "logistic", "tfidf_logistic"],
        trained_on="synthetic test corpus",
        seed=1,
    )
    return table, bundles
