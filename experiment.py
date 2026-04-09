"""Experiment runner for semantic robustness and compression analysis."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import PipelineConfig, iter_images, load_config_file, save_json
from src.channel import NoisyChannel
from src.decoder import OARDecoder
from src.encoder import OAREncoder
from src.evaluate import Evaluator
from src.extract import ObjectExtractor
from src.oar_builder import OARBuilder
from src.reconstruct import SemanticReconstructor


def setup_experiment_logging(results_dir: Path) -> Path:
    """Configure experiment-specific logging."""
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "experiment.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    return log_path


def parse_args() -> tuple[PipelineConfig, float, float, float, int | None]:
    """Parse experiment CLI arguments."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_config_file(pre_args.config)

    parser = argparse.ArgumentParser(description="Semantic communication experiment runner")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--image-dir", type=Path, default=Path(str(config_defaults["image_dir"])))
    parser.add_argument("--results-dir", type=Path, default=Path(str(config_defaults["results_dir"])))
    parser.add_argument("--model-path", type=str, default=str(config_defaults["model_path"]))
    parser.add_argument("--noise-level", type=float, default=float(config_defaults["noise_level"]))
    parser.add_argument("--max-objects", type=int, default=int(config_defaults["max_objects"]))
    parser.add_argument(
        "--near-distance-threshold",
        type=float,
        default=float(config_defaults["near_distance_threshold"]),
    )
    parser.add_argument("--conf-threshold", type=float, default=float(config_defaults["conf_threshold"]))
    parser.add_argument("--seed", type=int, default=int(config_defaults["seed"]))
    parser.add_argument(
        "--enable-privacy",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults["enable_privacy"]),
    )
    parser.add_argument("--noise-start", type=float, default=0.0)
    parser.add_argument("--noise-stop", type=float, default=0.5)
    parser.add_argument("--noise-step", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=None)

    args = parser.parse_args()
    config = PipelineConfig(
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
    return config, args.noise_start, args.noise_stop, args.noise_step, args.max_images


def build_noise_schedule(start: float, stop: float, step: float) -> list[float]:
    """Create a monotonic noise schedule."""
    if step <= 0.0:
        raise ValueError("noise-step must be positive")

    levels: list[float] = []
    current = start
    while current <= stop + 1e-9:
        levels.append(round(current, 4))
        current += step
    return levels


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist experiment rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(results_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    """Generate compression/accuracy and noise/semantic plots."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        return

    noise_levels = [row["noise_level"] for row in summary_rows]
    compression = [row["mean_compression_ratio"] for row in summary_rows]
    semantic_score = [row["mean_semantic_score"] for row in summary_rows]

    indices = range(len(noise_levels))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar([index - width / 2 for index in indices], compression, width=width, label="Compression ratio")
    plt.bar([index + width / 2 for index in indices], semantic_score, width=width, label="Semantic score")
    plt.xticks(list(indices), [f"{level:.1f}" for level in noise_levels])
    plt.xlabel("Noise level")
    plt.ylabel("Mean value")
    plt.title("Compression ratio vs semantic accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "compression_vs_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(noise_levels, semantic_score, marker="o", linewidth=2.0)
    plt.xlabel("Noise level")
    plt.ylabel("Mean semantic score")
    plt.title("Noise robustness of semantic communication")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_vs_semantic_score.png", dpi=200)
    plt.close()


def run_experiment(
    config: PipelineConfig,
    noise_start: float,
    noise_stop: float,
    noise_step: float,
    max_images: int | None,
) -> None:
    """Run the semantic communication experiment across images and noise levels."""
    logger = logging.getLogger("semantic-experiment")

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
    decoder = OARDecoder()
    reconstructor = SemanticReconstructor()
    evaluator = Evaluator()

    images = iter_images(config.image_dir)
    if max_images is not None:
        images = images[:max_images]
    logger.info("Found %d image(s) for experiment.", len(images))

    samples: list[dict[str, Any]] = []
    for image_path in images:
        try:
            objects = extractor.extract(image_path)
            original_oar = builder.build(objects)
            encoded_packet = encoder.encode(original_oar)
            original_text = reconstructor.reconstruct_text(original_oar)
            samples.append(
                {
                    "image_id": image_path.stem,
                    "image_path": image_path,
                    "original_oar": original_oar,
                    "encoded_packet": encoded_packet,
                    "original_text": original_text,
                }
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Skipping %s because extraction failed: %s", image_path.name, exc)

    if not samples:
        logger.warning("No successful image samples were produced; experiment will not continue.")
        return

    noise_levels = build_noise_schedule(noise_start, noise_stop, noise_step)
    rows: list[dict[str, Any]] = []

    for noise_level in noise_levels:
        logger.info("Running experiment at noise level %.2f", noise_level)
        channel = NoisyChannel(noise_level=noise_level, seed=config.seed)

        for sample in samples:
            transmitted_packet = sample["encoded_packet"]
            if config.enable_privacy:
                transmitted_packet = channel.transmit(sample["encoded_packet"])

            decoded_oar = decoder.decode(transmitted_packet)
            reconstructed_text = reconstructor.reconstruct_text(decoded_oar)
            metrics = evaluator.evaluate(
                original_oar=sample["original_oar"],
                decoded_oar=decoded_oar,
                original_text=sample["original_text"],
                reconstructed_text=reconstructed_text,
                original_image_path=sample["image_path"],
                semantic_size_bytes=transmitted_packet.semantic_size_bytes,
                noise_level=noise_level,
            )
            quality_label = evaluator.label_quality(metrics)

            row = {
                "image_id": sample["image_id"],
                "noise_level": noise_level,
                "semantic_text": reconstructed_text,
                "quality_label": quality_label,
                **metrics.to_dict(),
            }
            rows.append(row)

    summary_rows = evaluator.evaluate_noise_robustness(rows)
    output_payload = {
        "noise_levels": noise_levels,
        "summary_by_noise": summary_rows,
        "rows": rows,
        "sample_count": len(samples),
    }

    save_json(config.results_dir / "experiment_results.json", output_payload)
    write_csv(config.results_dir / "experiment_results.csv", rows)
    save_plots(config.results_dir, summary_rows)
    logger.info("Experiment complete with %d samples and %d rows.", len(samples), len(rows))


def main() -> None:
    """Application entrypoint."""
    config, noise_start, noise_stop, noise_step, max_images = parse_args()
    log_path = setup_experiment_logging(config.results_dir)
    logging.getLogger("semantic-experiment").info("Logging to %s", log_path)
    run_experiment(config, noise_start, noise_stop, noise_step, max_images)


if __name__ == "__main__":
    main()
