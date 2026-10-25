from pathlib import Path

import pytest

from fraudlab.config import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["default.yaml", "fast.yaml"])
def test_shipped_configs_are_valid(name):
    cfg = Config.load(ROOT / "configs" / name)
    assert cfg.split.test_fraction == 0.20
    assert "lgbm" in cfg.candidates


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown config key"):
        Config.from_dict({"sed": 1})


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown key"):
        Config.from_dict({"split": {"test_frac": 0.2}})


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown candidate"):
        Config.from_dict({"candidates": ["xgboost_magic"]})
