# Semantic Image Communication using OAR and AI-based Reconstruction

This project provides a modular, research-grade baseline for **6G semantic communication** experiments with images.

It simulates a semantic communication pipeline where:
1. Images are analyzed with object detection.
2. Detections are converted to an OAR representation (Object-Attribute-Relation).
3. OAR is encoded into a compact payload.
4. A noisy wireless channel drops part of semantic content.
5. OAR is decoded and reconstructed into text.
6. Multi-layer metrics are computed and saved.

## Project Structure

```text
data/
	images/
src/
	__init__.py
	types.py
	extract.py
	oar_builder.py
	encoder.py
	channel.py
	decoder.py
	reconstruct.py
	evaluate.py
models/
results/
notebooks/
main.py
requirements.txt
README.md
```

## Module Overview

- `src/extract.py`: YOLOv8-based object extraction (`name`, `bbox`, `confidence`).
- `src/oar_builder.py`: Rule-based OAR construction and spatial relation inference.
- `src/encoder.py`: OAR compression with `zlib + base64`, plus bit-size estimation.
- `src/channel.py`: Configurable noise channel (drops objects/relations probabilistically).
- `src/decoder.py`: OAR packet decoding.
- `src/reconstruct.py`: Semantic-to-text reconstruction (and image-generation placeholder).
- `src/evaluate.py`: Traditional placeholders (PSNR/SSIM), semantic retention metrics, text similarity.
- `main.py`: End-to-end orchestration, logging, and output persistence.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add input images to `data/images/`.

## Run

Basic run:

```bash
python main.py
```

Run with config file defaults:

```bash
python main.py --config config.yaml
```

Run with custom semantic channel parameters:

```bash
python main.py \
	--noise-level 0.30 \
	--max-objects 12 \
	--near-distance-threshold 140 \
	--model-path yolov8n.pt
```

## Configuration Parameters

- `--noise-level`: Probability of dropping semantic units in channel (`0.0` to `1.0`).
- `--max-objects`: Maximum objects retained from detector.
- `--near-distance-threshold`: Pixel threshold for `near` relation creation.
- `--conf-threshold`: Detection confidence threshold for YOLOv8.
- `--seed`: Random seed for reproducible channel corruption.

You can set these defaults in `config.yaml`, and still override them from CLI flags.

## Outputs

The pipeline writes all outputs under `results/`:

- `results/semantic/<image_id>.json`: Full per-image semantic trace.
- `results/text/<image_id>.txt`: Reconstructed semantic text.
- `results/dataset.json`: Structured dataset with fields:
	- `image_id`
	- `objects`
	- `relations`
	- `semantic_text`
	- `bit_estimate`
	- `quality_label`
- `results/evaluation_metrics.json`: Per-image metric table.
- `results/logs/pipeline.log`: Debug and run logs.

## Sample Semantic Text

Example reconstructed text:

```text
Detected objects: person (obj_0), bicycle (obj_1). Relations: obj_0 near obj_1; obj_0 interacting_with obj_1.
```

## Notes for Research Extension

- Replace rule-based relation extraction with a learned relation model.
- Replace placeholder traditional metrics with true image-domain PSNR/SSIM when paired reconstructions are available.
- Add a semantic source coder beyond generic compression.
- Extend reconstruction to text-to-image generation for visual regeneration studies.

