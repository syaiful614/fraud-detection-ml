import numpy as np
import pandas as pd
import pytest

from fraudlab.data import DataValidationError, make_synthetic, prepare, validate
from fraudlab.features import FEATURE_NAMES, FeatureBuilder
from fraudlab.splits import expanding_window_splits, temporal_holdout


def test_prepare_sorts_dedupes_and_validates():
    raw = make_synthetic(n=500, seed=3)
    raw = pd.concat([raw, raw.iloc[:10]]).sample(frac=1.0, random_state=0)
    out = prepare(raw)
    assert out["Time"].is_monotonic_increasing
    assert len(out) == 500


def test_validate_rejects_bad_data():
    raw = make_synthetic(n=300, seed=3)
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(raw.drop(columns=["V3"]))
    bad = raw.copy()
    bad.loc[0, "Amount"] = np.nan
    with pytest.raises(DataValidationError, match="missing values"):
        validate(bad)
    bad = raw.copy()
    bad["Class"] = 0
    with pytest.raises(DataValidationError, match="No positive"):
        validate(bad)


def test_feature_builder_outputs_expected_columns(synthetic_df):
    X = synthetic_df.drop(columns="Class")
    out = FeatureBuilder().fit(X).transform(X)
    assert list(out.columns) == FEATURE_NAMES
    assert not out.isna().any().any()
    assert "Time" not in out.columns  # raw counter must never reach the model


def test_time_of_day_is_cyclical():
    base = {c: [0.0, 0.0, 0.0] for c in make_synthetic(10).columns if c.startswith("V")}
    X = pd.DataFrame({"Time": [0.0, 21_600.0, 86_400.0], "Amount": [0.0, 10.0, 10.0], **base})
    out = FeatureBuilder().fit(X).transform(X)
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0)
    assert out["hour_sin"].iloc[1] == pytest.approx(1.0)  # 06:00
    assert out["hour_cos"].iloc[2] == pytest.approx(1.0)  # exactly one day later
    assert out["log_amount"].iloc[0] == 0.0


def test_feature_builder_requires_raw_columns(synthetic_df):
    with pytest.raises(ValueError, match="Missing raw columns"):
        FeatureBuilder().fit(synthetic_df.drop(columns=["Amount", "Class"]))


def test_temporal_holdout_has_no_overlap(synthetic_df):
    dev, test = temporal_holdout(synthetic_df, 0.2)
    assert len(dev) + len(test) == len(synthetic_df)
    assert dev["Time"].max() <= test["Time"].min()


def test_temporal_holdout_rejects_unsorted(synthetic_df):
    with pytest.raises(ValueError, match="sorted"):
        temporal_holdout(synthetic_df.sample(frac=1.0, random_state=0), 0.2)


def test_expanding_window_never_trains_on_the_future():
    gap = 50
    splits = expanding_window_splits(5000, n_splits=4, gap=gap)
    assert len(splits) == 4
    sizes = [len(tr) for tr, _ in splits]
    assert sizes == sorted(sizes)
    for tr, va in splits:
        assert tr.max() + gap < va.min()
