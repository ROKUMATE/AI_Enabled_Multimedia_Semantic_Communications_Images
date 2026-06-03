"""Main entrypoint for the v1 semantic image communication pipeline.

Runs each image in the input folder through the transmitter -> IdentityChannel
-> receiver, then writes the scene graph, payload sizes, the reconstructed
image, and a text description to ``results/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.pipeline import PipelineSettings, SemanticPipeline


DEFAULT_CONFIG: dict[str, Any] = {
    "image_dir": "data/images",
    "results_dir": "results",
    "model_path": "yolov8n.pt",
    "conf_threshold": 0.25,
    "max_objects": 20,
    "near_distance_threshold": 120.0,
    "seed": 42,
    # v1 image-pipeline defaults (see README / PLAN.md)
    "extractor": "yolo",
    "relation_builder": "rule_based",
    "reconstructor": "compositional",
    "channel": "identity",
    "background_color": [127, 127, 127],
    "preserve_classes": ["person"],
    "importance": {"budget": 3},
    "appearance": {"format": "JPEG", "preserve_quality": 95, "regenerate_quality": 35},
    "ocr": {"enabled": True, "backend": "auto"},
    "diffusion": {"enabled": False, "model_id": "stabilityai/sd-turbo"},
    "metrics": {"deep_features": False, "lpips": False},
    "streams": {"structure_priority": 0, "appearance_priority": 1},
    # legacy keys retained for backward compatibility (unused by the v1 image path)
    "noise_level": 0.2,
    "enable_privacy": True,
}


def setup_logging(results_dir: Path) -> Path:
    """Configure logging handlers and return the log file path."""
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    return log_path


def load_config_file(config_path: Path) -> dict[str, Any]:
    """Load YAML config and merge it over the defaults (all keys kept)."""
    merged = dict(DEFAULT_CONFIG)
    if not config_path.exists():
        return merged
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a dictionary: {config_path}")
    merged.update(loaded)
    return merged


def parse_args() -> tuple[dict[str, Any], Path, Path]:
    """Parse CLI args, returning (config dict, image_dir, results_dir)."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config_file(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Semantic image communication (v1): image -> payload -> image + text"
    )
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--image-dir", type=Path, default=Path(str(config["image_dir"])))
    parser.add_argument("--results-dir", type=Path, default=Path(str(config["results_dir"])))
    parser.add_argument("--model-path", type=str, default=str(config["model_path"]))
    parser.add_argument("--conf-threshold", type=float, default=float(config["conf_threshold"]))
    parser.add_argument("--max-objects", type=int, default=int(config["max_objects"]))
    parser.add_argument(
        "--near-distance-threshold",
        type=float,
        default=float(config["near_distance_threshold"]),
    )
    parser.add_argument("--seed", type=int, default=int(config["seed"]))
    parser.add_argument("--extractor", type=str, default=str(config["extractor"]))
    parser.add_argument("--reconstructor", type=str, default=str(config["reconstructor"]))
    # Legacy flags retained so existing invocations keep working (unused in v1).
    parser.add_argument("--noise-level", type=float, default=float(config["noise_level"]))
    parser.add_argument(
        "--enable-privacy",
        action=argparse.BooleanOptionalAction,
        default=bool(config["enable_privacy"]),
    )

    args = parser.parse_args()
    config.update(
        {
            "model_path": args.model_path,
            "conf_threshold": args.conf_threshold,
            "max_objects": args.max_objects,
            "near_distance_threshold": args.near_distance_threshold,
            "seed": args.seed,
            "extractor": args.extractor,
            "reconstructor": args.reconstructor,
            "noise_level": args.noise_level,
            "enable_privacy": args.enable_privacy,
        }
    )
    return config, args.image_dir, args.results_dir


def iter_images(image_dir: Path) -> list[Path]:
    """Return the sorted list of image files in a directory."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(image_dir.glob(pattern)))
    return paths


def save_json(path: Path, data: Any) -> None:
    """Save data as formatted JSON on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def run_pipeline(config: dict[str, Any], image_dir: Path, results_dir: Path) -> None:
    """Execute the v1 pipeline over a folder of images."""
    logger = logging.getLogger("semantic-pipeline")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    for sub in ("text", "semantic", "reconstructed"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    settings = PipelineSettings.from_config(config)
    pipeline = SemanticPipeline(settings)

    images = iter_images(image_dir)
    logger.info("Found %d image(s) in %s", len(images), image_dir)
    if not images:
        logger.warning("No images found. Add files under %s to run the pipeline.", image_dir)

    summaries: list[dict[str, Any]] = []
    for image_path in images:
        logger.info("Processing image: %s", image_path.name)
        try:
            out = pipeline.run(image_path)

            Image.fromarray(out.reconstruction.image).save(
                results_dir / "reconstructed" / f"{out.image_id}.png"
            )
            (results_dir / "text" / f"{out.image_id}.txt").write_text(
                out.reconstruction.text + "\n", encoding="utf-8"
            )
            semantic_output = {
                "image_id": out.image_id,
                "scene_graph": out.scene_graph(),
                "size_report": out.size_report,
                "objects": [obj.to_dict() for obj in out.objects],
                "relations": [rel.to_dict() for rel in out.relations],
                "text": out.reconstruction.text,
            }
            save_json(results_dir / "semantic" / f"{out.image_id}.json", semantic_output)

            summaries.append(
                {
                    "image_id": out.image_id,
                    "num_objects": len(out.objects),
                    "num_crops": out.size_report["num_crops"],
                    "total_bytes": out.size_report["total_bytes"],
                }
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to process %s: %s", image_path.name, exc)

    save_json(results_dir / "dataset.json", summaries)
    logger.info("Completed processing %d image(s).", len(summaries))


def main() -> None:
    """Application entrypoint."""
    config, image_dir, results_dir = parse_args()
    log_path = setup_logging(results_dir)
    logging.getLogger("semantic-pipeline").info("Logging to %s", log_path)
    run_pipeline(config, image_dir, results_dir)


if __name__ == "__main__":
    main()
