"""Training skeleton for our own object detector (P5 scaffold).

This is intentionally a SKELETON. v1 ships with the pretrained YOLO weights and
:class:`~src.extractors.learned.LearnedObjectExtractor` falls back to them when
no checkpoint exists. When we are ready to train our own detector, fill in the
TODOs below and produce a checkpoint at ``--out`` (default
``checkpoints/detector.pt``); the learned extractor will then load it
automatically.

We fine-tune an existing detector backbone rather than training from scratch
(per the task constraint: do not train large models from scratch here).

Usage (once implemented):
    python scripts/train_detector.py --data data/detector.yaml --epochs 50 \
        --base yolov8n.pt --out checkpoints/detector.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train-detector")


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune our object detector (skeleton)")
    parser.add_argument("--data", type=Path, default=Path("data/detector.yaml"),
                        help="Ultralytics-style dataset YAML (images + labels).")
    parser.add_argument("--base", type=str, default="yolov8n.pt",
                        help="Backbone/checkpoint to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/detector.pt"))
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Fine-tune the detector and save a checkpoint.

    TODO(P5): implement training. A concrete first version can simply call
    ultralytics fine-tuning, e.g.::

        from ultralytics import YOLO
        model = YOLO(args.base)
        model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.out))

    Later, replace this with our own detector architecture + training loop and
    save a checkpoint in a format that ``LearnedObjectExtractor`` can load.
    """
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "scripts/train_detector.py is a skeleton. Implement the TODO above to "
        f"fine-tune {args.base} on {args.data} and write {args.out}."
    )


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    logger.info("Training config: %s", vars(args))
    train(args)


if __name__ == "__main__":
    main()
