"""Typed, strictly validated experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

VALID_CANDIDATES = (
    "dummy",
    "logreg",
    "isolation_forest",
    "random_forest",
    "lgbm",
    "lgbm_weighted",
    "lgbm_smote",
)


@dataclass
class DataConfig:
    raw_path: str = "data/raw/creditcard.csv"
    drop_duplicates: bool = True


@dataclass
class SplitConfig:
    test_fraction: float = 0.20
    cv_splits: int = 4
    cv_gap_rows: int = 1000


@dataclass
class CostConfig:
    review_cost: float = 5.0


@dataclass
class TuningConfig:
    n_trials: int = 30


@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 500
    alert_budgets: list[float] = field(default_factory=lambda: [0.001, 0.0025, 0.005])
    cv_alert_rate: float = 0.0025


@dataclass
class Config:
    seed: int = 42
    n_jobs: int = -1
    models_dir: str = "models"
    reports_dir: str = "reports"
    candidates: list[str] = field(default_factory=lambda: list(VALID_CANDIDATES))
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        bad = [c for c in self.candidates if c not in VALID_CANDIDATES]
        if bad:
            raise ValueError(f"Unknown candidate model(s) {bad}; valid: {VALID_CANDIDATES}")
        if not 0.05 <= self.split.test_fraction <= 0.5:
            raise ValueError("split.test_fraction must be between 0.05 and 0.5")
        if self.split.cv_splits < 2:
            raise ValueError("split.cv_splits must be >= 2")
        if self.cost.review_cost < 0:
            raise ValueError("cost.review_cost must be >= 0")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Config:
        raw = dict(raw or {})
        sections = {
            "data": DataConfig,
            "split": SplitConfig,
            "cost": CostConfig,
            "tuning": TuningConfig,
            "evaluation": EvaluationConfig,
        }
        top_level = {f.name for f in fields(cls)}
        unknown = set(raw) - top_level
        if unknown:
            raise ValueError(f"Unknown config key(s): {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in sections:
                section_cls = sections[key]
                allowed = {f.name for f in fields(section_cls)}
                extra = set(value or {}) - allowed
                if extra:
                    raise ValueError(f"Unknown key(s) in '{key}': {sorted(extra)}")
                kwargs[key] = section_cls(**(value or {}))
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
