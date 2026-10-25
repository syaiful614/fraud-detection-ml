import pytest

from fraudlab.data import make_synthetic, prepare


@pytest.fixture(scope="session")
def synthetic_df():
    return prepare(make_synthetic(n=6000, fraud_rate=0.02, seed=1))
