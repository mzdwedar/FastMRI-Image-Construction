"""Packaging CLI: `python -m fastmri_knee.packaging register --checkpoint PATH`."""

import argparse
import json
import sys
from pathlib import Path

from fastmri_knee.packaging.registry import MODEL_NAME, RegistryError, register, register_random


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fastmri_knee.packaging")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help=f"register a checkpoint as a new version of {MODEL_NAME}")
    source = reg.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--random-init", action="store_true", help="register untrained weights, tagged random_init=true"
    )
    reg.add_argument("--summary", type=Path, help="run_summary.json written by training")
    reg.add_argument("--run-id", help="MLflow run the checkpoint came from (adds its val metrics)")
    reg.add_argument("--tracking-uri", help="default: MLFLOW_TRACKING_URI, else ./mlruns")

    args = parser.parse_args(argv)
    if args.random_init and (args.summary or args.run_id):
        parser.error("--random-init cannot be combined with --summary or --run-id")
    try:
        if args.random_init:
            version = register_random(tracking_uri=args.tracking_uri)
            print(f"registered {MODEL_NAME} version {version} (random_init=true)")
            return 0
        summary = json.loads(args.summary.read_text()) if args.summary else None
        version = register(args.checkpoint, tracking_uri=args.tracking_uri, summary=summary, run_id=args.run_id)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"registered {MODEL_NAME} version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
