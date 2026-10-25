"""Command line interface: `fraudlab download | train | score | serve`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fraudlab.artifacts import load_bundle
from fraudlab.config import Config
from fraudlab.data import RAW_FEATURES, download


def _cmd_download(args: argparse.Namespace) -> None:
    path = download(args.dest, args.source)
    print(f"Dataset ready at {path}")


def _cmd_train(args: argparse.Namespace) -> None:
    from fraudlab.pipeline import run

    metrics = run(Config.load(args.config))
    t = metrics["test"]
    print(f"\nTest average precision: {t['average_precision']:.3f} | ROC-AUC: {t['roc_auc']:.3f}")
    print("Full report: reports/results.md")


def _cmd_score(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.model_dir)
    df = pd.read_csv(args.input)
    proba = bundle.model.predict_proba(df[RAW_FEATURES])[:, 1]
    out = df.copy()
    out["fraud_probability"] = proba
    out["is_fraud_alert"] = proba >= bundle.threshold
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out)} rows -> {args.output} ({int(out['is_fraud_alert'].sum())} alerts)")


def _cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ["MODEL_DIR"] = str(args.model_dir)
    uvicorn.run("fraudlab.serve.app:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(prog="fraudlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the dataset")
    p.add_argument("--dest", default="data/raw/creditcard.csv")
    p.add_argument("--source", choices=["mirror", "kaggle"], default="mirror")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("train", help="run the full pipeline and write models/ and reports/")
    p.add_argument("--config", default="configs/default.yaml", type=Path)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("score", help="score a CSV with raw columns using the trained model")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="scored.csv")
    p.add_argument("--model-dir", default="models", type=Path)
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("serve", help="start the FastAPI scoring service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=8000, type=int)
    p.add_argument("--model-dir", default="models", type=Path)
    p.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)
