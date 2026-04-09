"""Main pipeline for semantic image communication with OAR representation."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.channel import NoisyChannel
from src.decoder import OARDecoder
from src.encoder import OAREncoder
from src.evaluate import Evaluator
from src.extract import ObjectExtractor
from src.oar_builder import OARBuilder
from src.reconstruct import SemanticReconstructor


@dataclass
class PipelineConfig:
    """Runtime configuration for semantic communication pipeline."""

    image_dir: Path
    results_dir: Path
    model_path: str
    noise_level: float
    max_objects: int
    near_distance_threshold: float
    conf_threshold: float
    seed: int | None
    enable_privacy: bool


DEFAULT_CONFIG: dict[str, Any] = {
    "image_dir": "data/images",
    "results_dir": "results",
    "model_path": "yolov8n.pt",
    "noise_level": 0.2,
    "max_objects": 20,
    "near_distance_threshold": 120.0,
    "conf_threshold": 0.25,
    "seed": 42,
    "enable_privacy": True,
}


def setup_logging(results_dir: Path) -> Path:
    """Configure logging handlers and return log file path."""
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def parse_args() -> PipelineConfig:
    """Parse CLI arguments and map them into pipeline configuration."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_file(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Semantic Image Communication using OAR and AI-based Reconstruction"
    )
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument(
        "--image-dir", type=Path, default=Path(str(config_defaults["image_dir"]))
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(str(config_defaults["results_dir"]))
    )
    parser.add_argument("--model-path", type=str, default=str(config_defaults["model_path"]))
    parser.add_argument("--noise-level", type=float, default=float(config_defaults["noise_level"]))
    parser.add_argument("--max-objects", type=int, default=int(config_defaults["max_objects"]))
    parser.add_argument(
        "--near-distance-threshold",
        type=float,
        default=float(config_defaults["near_distance_threshold"]),
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=float(config_defaults["conf_threshold"])
    )
    parser.add_argument("--seed", type=int, default=int(config_defaults["seed"]))
    parser.add_argument(
        "--enable-privacy",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults["enable_privacy"]),
    )

    args = parser.parse_args()
    return PipelineConfig(
        image_dir=args.image_dir,
        results_dir=args.results_dir,
        model_path=args.model_path,
        noise_level=args.noise_level,
        max_objects=args.max_objects,
        near_distance_threshold=args.near_distance_threshold,
        conf_threshold=args.conf_threshold,
        seed=args.seed,
        enable_privacy=args.enable_privacy,
    )


def load_config_file(config_path: Path) -> dict[str, Any]:
    """Load YAML config and merge values with defaults."""
    merged = dict(DEFAULT_CONFIG)
    if not config_path.exists():
        return merged

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a dictionary: {config_path}")

    for key, value in loaded.items():
        if key in merged:
            merged[key] = value
    return merged


def iter_images(image_dir: Path) -> list[Path]:
    """Return image file list from input directory."""
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


def run_pipeline(config: PipelineConfig) -> None:
    """Execute full end-to-end semantic communication pipeline."""
    logger = logging.getLogger("semantic-pipeline")

    if not config.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {config.image_dir}")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "text").mkdir(exist_ok=True)
    (config.results_dir / "semantic").mkdir(exist_ok=True)

    extractor = ObjectExtractor(
        model_path=config.model_path,
        conf_threshold=config.conf_threshold,
        max_objects=config.max_objects,
    )
    builder = OARBuilder(near_distance_threshold=config.near_distance_threshold)
    encoder = OAREncoder()
    channel = NoisyChannel(noise_level=config.noise_level, seed=config.seed)
    decoder = OARDecoder()
    reconstructor = SemanticReconstructor()
    evaluator = Evaluator()

    dataset_entries: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []

    images = iter_images(config.image_dir)
    logger.info("Found %d image(s) in %s", len(images), config.image_dir)
    if not images:
        logger.warning("No images found. Add files under %s to run semantic conversion.", config.image_dir)

    for image_path in images:
        image_id = image_path.stem
        logger.info("Processing image: %s", image_path.name)
        try:
            objects = extractor.extract(image_path)
            original_oar = builder.build(objects)

            encoded_packet = encoder.encode(original_oar)
            transmitted_packet = (
                channel.transmit(encoded_packet) if config.enable_privacy else encoded_packet
            )
            decoded_oar = decoder.decode(transmitted_packet)

            source_text = reconstructor.reconstruct_text(original_oar)
            reconstructed_text = reconstructor.reconstruct_text(decoded_oar)

            metrics = evaluator.evaluate(
                original_oar=original_oar,
                decoded_oar=decoded_oar,
                original_text=source_text,
                reconstructed_text=reconstructed_text,
                original_image_path=image_path,
                semantic_size_bytes=transmitted_packet.semantic_size_bytes,
                noise_level=config.noise_level,
            )
            quality_label = evaluator.label_quality(metrics)

            semantic_output = {
                "image_id": image_id,
                "original_oar": original_oar.to_dict(),
                "decoded_oar": decoded_oar.to_dict(),
                "encoded_packet": encoded_packet.to_dict(),
                "transmitted_packet": transmitted_packet.to_dict(),
                "semantic_text": reconstructed_text,
                "metrics": metrics.to_dict(),
                "quality_label": quality_label,
                "original_image_size_kb": metrics.original_image_size_kb,
                "semantic_size_bytes": metrics.semantic_size_bytes,
                "compression_ratio": metrics.compression_ratio,
            }
            save_json(config.results_dir / "semantic" / f"{image_id}.json", semantic_output)

            text_path = config.results_dir / "text" / f"{image_id}.txt"
            text_path.write_text(reconstructed_text + "\n", encoding="utf-8")

            dataset_entries.append(
                {
                    "image_id": image_id,
                    "objects": [item.to_dict() for item in decoded_oar.objects],
                    "relations": [item.to_dict() for item in decoded_oar.relations],
                    "semantic_text": reconstructed_text,
                    "bit_estimate": transmitted_packet.bit_estimate,
                    "semantic_size_bytes": transmitted_packet.semantic_size_bytes,
                    "compression_ratio": metrics.compression_ratio,
                    "semantic_score": metrics.semantic_score,
                    "quality_label": quality_label,
                }
            )

            metric_row = {"image_id": image_id, **metrics.to_dict()}
            evaluation_rows.append(metric_row)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to process %s: %s", image_path.name, exc)

    save_json(config.results_dir / "dataset.json", dataset_entries)
    save_json(config.results_dir / "evaluation_metrics.json", evaluation_rows)
    logger.info("Completed processing %d image(s).", len(dataset_entries))


def main() -> None:
    """Application entrypoint."""
    config = parse_args()
    log_path = setup_logging(config.results_dir)
    logging.getLogger("semantic-pipeline").info("Logging to %s", log_path)
    run_pipeline(config)


if __name__ == "__main__":
    main()
