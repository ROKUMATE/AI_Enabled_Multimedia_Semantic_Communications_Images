"""Experiment runner: folder of images x configs + baselines.

Runs every image through the semantic pipeline and the JPEG-matched and
text-only baselines, writes a results table (CSV + JSON), and saves side-by-side
original/reconstructed comparison images under ``results/comparisons/``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import iter_images, load_config_file, save_json, setup_logging
from src.baselines import jpeg_baseline, text_only_payload
from src.metrics import Metrics
from src.pipeline import PipelineSettings, SemanticPipeline


logger = logging.getLogger("semantic-experiment")
_FONT = ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    """Parse experiment CLI arguments (keeps legacy flags accepted)."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_file(pre_args.config)

    parser = argparse.ArgumentParser(description="Semantic image communication experiment runner")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--image-dir", type=Path, default=Path(str(config_defaults["image_dir"])))
    parser.add_argument("--results-dir", type=Path, default=Path(str(config_defaults["results_dir"])))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--deep-features",
        action=argparse.BooleanOptionalAction,
        default=bool((config_defaults.get("metrics") or {}).get("deep_features", False)),
    )
    # Legacy noise-sweep flags: accepted but ignored (v1 has no lossy channel).
    parser.add_argument("--noise-start", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--noise-stop", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--noise-step", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _panel(image: np.ndarray, caption: str, height: int) -> Image.Image:
    """Resize an image to a fixed height and add a caption bar above it."""
    pil = Image.fromarray(image)
    scale = height / pil.height
    pil = pil.resize((max(1, int(pil.width * scale)), height), Image.BILINEAR)
    bar = 18
    panel = Image.new("RGB", (pil.width, height + bar), (20, 20, 20))
    panel.paste(pil, (0, bar))
    draw = ImageDraw.Draw(panel)
    draw.text((2, 4), caption, fill=(255, 255, 255), font=_FONT)
    return panel


def save_side_by_side(path: Path, panels: list[tuple[str, np.ndarray]], height: int = 256) -> None:
    """Save a horizontal strip of captioned panels."""
    rendered = [_panel(image, caption, height) for caption, image in panels]
    total_width = sum(panel.width for panel in rendered) + 4 * (len(rendered) - 1)
    strip = Image.new("RGB", (total_width, rendered[0].height), (20, 20, 20))
    x = 0
    for panel in rendered:
        strip.paste(panel, (x, 0))
        x += panel.width + 4
    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist experiment rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace) -> None:
    """Run the full experiment over the image folder."""
    config = load_config_file(args.config)
    config["metrics"] = {**(config.get("metrics") or {}), "deep_features": args.deep_features}
    settings = PipelineSettings.from_config(config)

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    comparisons_dir = args.results_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SemanticPipeline(settings)
    metrics = Metrics(
        downstream_extractor=pipeline.extractor,
        ocr_backend=pipeline.mode_classifier.ocr_backend,
        deep_features=args.deep_features,
        use_lpips=bool((config.get("metrics") or {}).get("lpips", False)),
    )

    images = iter_images(args.image_dir)
    if args.max_images is not None:
        images = images[: args.max_images]
    logger.info("Running experiment over %d image(s).", len(images))

    rows: list[dict[str, Any]] = []
    for image_path in images:
        try:
            raw_bytes = image_path.stat().st_size
            out = pipeline.run(image_path)
            original = out.original_image
            semantic_recon = out.reconstruction.image
            payload_bytes = out.size_report["total_bytes"]

            semantic_metrics = metrics.compute(
                original, semantic_recon, payload_bytes, raw_bytes, out.objects
            )

            text_payload = text_only_payload(out.received_payload)
            text_recon = pipeline.reconstructor.reconstruct(
                text_payload, pipeline.appearance_encoder
            )
            text_metrics = metrics.compute(
                original, text_recon.image, text_payload.size_report()["total_bytes"],
                raw_bytes, out.objects,
            )

            jpeg_image, jpeg_bytes = jpeg_baseline(original, payload_bytes)
            jpeg_metrics = metrics.compute(
                original, jpeg_image, jpeg_bytes, raw_bytes, out.objects
            )

            for method, result in (
                ("semantic", semantic_metrics),
                ("text_only", text_metrics),
                ("jpeg_matched", jpeg_metrics),
            ):
                rows.append({"image_id": out.image_id, "method": method, **result.to_dict()})

            save_side_by_side(
                comparisons_dir / f"{out.image_id}.png",
                [
                    ("original", original),
                    ("semantic", semantic_recon),
                    ("text-only", text_recon.image),
                    ("jpeg-matched", jpeg_image),
                ],
            )
            logger.info(
                "%s: semantic PSNR=%.2f recall=%.2f | jpeg PSNR=%.2f | text-only PSNR=%.2f",
                out.image_id,
                semantic_metrics.psnr,
                semantic_metrics.downstream_class_recall,
                jpeg_metrics.psnr,
                text_metrics.psnr,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Failed on %s: %s", image_path.name, exc)

    write_csv(args.results_dir / "experiment_results.csv", rows)
    save_json(
        args.results_dir / "experiment_results.json",
        {"rows": rows, "summary": _summarize(rows), "image_count": len(images)},
    )
    logger.info("Experiment complete: %d row(s) over %d image(s).", len(rows), len(images))
    _log_summary(_summarize(rows))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Average the key numeric metrics per method."""
    summary: dict[str, dict[str, float]] = {}
    methods = sorted({row["method"] for row in rows})
    keys = ["compression_ratio", "psnr", "downstream_class_recall", "payload_bytes"]
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: float(np.mean([row[key] for row in group if row.get(key) is not None]))
            for key in keys
            if any(row.get(key) is not None for row in group)
        }
    return summary


def _log_summary(summary: dict[str, dict[str, float]]) -> None:
    """Print a compact final results summary."""
    logger.info("==== RESULTS SUMMARY (means) ====")
    for method, values in summary.items():
        logger.info(
            "%-13s | compression=%.1fx PSNR=%.2f recall=%.2f payload=%.0fB",
            method,
            values.get("compression_ratio", 0.0),
            values.get("psnr", 0.0),
            values.get("downstream_class_recall", 0.0),
            values.get("payload_bytes", 0.0),
        )


def main() -> None:
    """Application entrypoint."""
    args = parse_args()
    log_path = setup_logging(args.results_dir)
    logger.info("Logging to %s", log_path)
    if args.noise_start is not None or args.noise_stop is not None or args.noise_step is not None:
        logger.warning("Noise-sweep flags are ignored in v1 (the channel is pass-through).")
    run_experiment(args)


if __name__ == "__main__":
    main()
