# Commands Used To Check The Project

This document records the commands used to verify the v1 semantic image
communication pipeline builds and runs end-to-end on CPU.

## 1. Compile the Python sources

```bash
.venv/bin/python -m compileall main.py experiment.py scripts src
```

## 2. Run the unit tests

```bash
.venv/bin/python -m unittest discover -s tests
```

Covers payload (de)serialization, importance scoring, mode classification, and
metrics.

## 3. Run the main pipeline

```bash
.venv/bin/python main.py --config config.yaml
```

Writes reconstructed images, text descriptions, and per-image scene graphs under
`results/` (`reconstructed/`, `text/`, `semantic/`).

## 4. Run the experiment runner (configs + baselines + side-by-sides)

```bash
.venv/bin/python experiment.py --config config.yaml --max-images 2
```

Writes `results/experiment_results.{csv,json}` and side-by-side comparison images
under `results/comparisons/` (original | semantic | text-only | jpeg-matched).

## 5. Run everything

```bash
make check
```

Runs compile + tests + main pipeline + experiment in order.

## Notes

- The core path runs on CPU with no GPU. Diffusion, OCR, LPIPS, and the
  deep-feature metric are optional and degrade gracefully when their
  dependencies are not installed (see `requirements.txt`).
- All outputs are written under `results/`.
