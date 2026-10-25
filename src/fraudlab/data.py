"""Data access: download, integrity check, validation and a synthetic generator for tests."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *V_COLUMNS, "Amount"]
TARGET = "Class"
RAW_COLUMNS = [*RAW_FEATURES, TARGET]

# Official source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (Worldline & ULB).
# The mirror below is a convenience copy that needs no Kaggle account.
MIRROR_URL = (
    "https://raw.githubusercontent.com/nsethi31/"
    "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
)
EXPECTED_SHA256 = "33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59"


class DataValidationError(ValueError):
    """Raised when the input data violates the expected schema."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(dest: str | Path = "data/raw/creditcard.csv", source: str = "mirror") -> Path:
    """Download the dataset. `source`: 'mirror' (no account needed) or 'kaggle' (kaggle CLI)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source == "mirror":
        log.info("Downloading %s ...", MIRROR_URL)
        with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent) as tmp:
            tmp_path = Path(tmp.name)
        try:
            urllib.request.urlretrieve(MIRROR_URL, tmp_path)  # noqa: S310
            shutil.move(str(tmp_path), dest)
        finally:
            tmp_path.unlink(missing_ok=True)
    elif source == "kaggle":
        subprocess.run(  # noqa: S603
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "mlg-ulb/creditcardfraud",
                "-p",
                str(dest.parent),
                "--unzip",
            ],
            check=True,
        )
    else:
        raise ValueError("source must be 'mirror' or 'kaggle'")
    digest = sha256_file(dest)
    if digest != EXPECTED_SHA256:
        log.warning(
            "SHA-256 differs from the file this project was developed with (%s...). "
            "Results may differ slightly.",
            EXPECTED_SHA256[:12],
        )
    return dest


def validate(df: pd.DataFrame) -> None:
    """Fail fast on schema problems instead of training on garbage."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if df[RAW_COLUMNS].isna().any().any():
        raise DataValidationError("Dataset contains missing values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise DataValidationError("Target must be binary (0/1)")
    if df[TARGET].sum() == 0:
        raise DataValidationError("No positive (fraud) examples found")
    if (df["Amount"] < 0).any() or (df["Time"] < 0).any():
        raise DataValidationError("Amount and Time must be non-negative")


def load_raw(path: str | Path, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load, validate, de-duplicate and sort chronologically (stable)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `fraudlab download` first.")
    df = pd.read_csv(path)
    return prepare(df, drop_duplicates=drop_duplicates)


def prepare(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    validate(df)
    df = df[RAW_COLUMNS].copy()
    df[TARGET] = df[TARGET].astype("int8")
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        log.info("Dropped %d exact duplicate rows", before - len(df))
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def make_synthetic(n: int = 6000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic dataset with the same schema. For tests/CI only, not for reported results."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype("int8")
    # Guarantee positives early and late so every temporal fold has some.
    y[:: max(1, n // 60)] = 1
    X = rng.normal(size=(n, len(V_COLUMNS)))
    shift = np.zeros(len(V_COLUMNS))
    shift[[2, 3, 9, 13, 16]] = [-2.0, 1.5, -1.8, -2.2, -1.5]
    X += y[:, None] * shift
    amount = rng.lognormal(mean=3.4, sigma=1.1, size=n)
    amount[y == 1] *= 1.6
    df = pd.DataFrame(X, columns=V_COLUMNS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)))
    df["Amount"] = amount
    df[TARGET] = y
    return df
