"""Persist and load the deployable model bundle (pipeline + threshold + metadata)."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

import fraudlab

MODEL_FILE = "fraud_model.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class ModelBundle:
    model: Any
    metadata: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "fraudlab": fraudlab.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def save_bundle(model: Any, metadata: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "environment": environment_info(),
    }
    joblib.dump(model, out_dir / MODEL_FILE, compress=3)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def load_bundle(model_dir: str | Path) -> ModelBundle:
    model_dir = Path(model_dir)
    model_path, meta_path = model_dir / MODEL_FILE, model_dir / METADATA_FILE
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No model bundle in {model_dir}. Run `fraudlab train` first.")
    return ModelBundle(
        model=joblib.load(model_path),
        metadata=json.loads(meta_path.read_text(encoding="utf-8")),
    )
