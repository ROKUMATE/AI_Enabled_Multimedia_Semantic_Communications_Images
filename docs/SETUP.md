# Setup, Running, Training & GPU Guide

Everything you need to set up the project, run it, read the results, and later
swap in trained/GPU components. The **core pipeline runs on CPU with no GPU and
no heavy dependencies**; the optional features (OCR, LPIPS, diffusion, the
analysis notebook) and all training/GPU work are described here and live behind
`requirements-extra.txt`.

> Design reference: [PLAN.md](../PLAN.md). Module overview: [README.md](../README.md).

---

## 1. Which machine do I need?

| Task | Machine | Extra deps |
|------|---------|-----------|
| Run the full image→payload→image+text pipeline, all baselines, PSNR + downstream-detector metrics, side-by-side images | **CPU** (any) | none (just `requirements.txt`) |
| OCR text-region detection + OCR-legibility metric | CPU | `easyocr` *or* `pytesseract` |
| LPIPS perceptual metric | CPU (GPU faster) | `lpips` |
| Deep-feature (VGG) distance metric | CPU | none (downloads torchvision weights on first use) |
| Diffusion background reconstruction | **GPU** (CUDA; Apple MPS partial) | `diffusers`, `transformers`, `accelerate`, CUDA `torch` |
| Train our own detector / relations / embeddings | **GPU** strongly recommended | see [§6](#6-what-to-train) |
| Run the analysis notebook | CPU | `pandas`, `seaborn`, `jupyter` |

**Python version:** the core pipeline works on the repo's interpreter, but several
optional packages (easyocr, lpips, diffusers, transformers, pandas, seaborn) do
**not** yet ship wheels for very new Python releases. On a training/GPU machine
use **Python 3.10–3.12**.

---

## 2. Installation

### 2.1 Core (CPU, required)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 Optional / training / GPU extras

Install everything, or uncomment just the group you need in the file:

```bash
pip install -r requirements-extra.txt
```

Group-specific notes:

- **OCR** — install ONE backend.
  - `pip install easyocr` (pure-pip), **or**
  - `pip install pytesseract` *and* the system binary:
    `brew install tesseract` (macOS) / `sudo apt-get install tesseract-ocr` (Debian/Ubuntu).
- **Diffusion (GPU)** — install a CUDA build of torch **first**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install diffusers transformers accelerate
  ```
- **Analysis notebook** — `pip install pandas seaborn jupyter`.

---

## 3. How to run

### 3.1 Single run over a folder

```bash
python main.py                       # uses config.yaml + data/images/
python main.py --image-dir path/to/imgs --results-dir out/
python main.py --extractor yolo --reconstructor compositional --seed 7
```

Writes per image: `results/reconstructed/<id>.png`, `results/text/<id>.txt`,
`results/semantic/<id>.json` (scene graph + payload size report), and a
`results/dataset.json` summary.

### 3.2 Experiment runner (configs + baselines + comparisons)

```bash
python experiment.py                 # all images
python experiment.py --max-images 5  # subset
python experiment.py --deep-features # also compute the VGG deep-feature metric
```

Writes `results/experiment_results.{csv,json}` (one row per image × method) and
`results/comparisons/<id>.png` (original | semantic | text-only | jpeg-matched).

### 3.3 Tests & full check

```bash
python -m unittest discover -s tests
make check                           # compile + tests + main + experiment
```

### 3.4 Analysis notebook

```bash
pip install -r requirements-extra.txt    # pandas, seaborn, jupyter
jupyter notebook notebooks/experiment_results.ipynb
```

All knobs are in `config.yaml` (documented in the README table) and most have a
CLI override on `main.py`.

---

## 4. What to check (reading the results)

- **`results/comparisons/<id>.png`** — eyeball the reconstruction: important
  objects should be sharp (their crops), background is the solid canvas (or
  diffusion scene), text-only panel has labeled boxes but no crop detail.
- **`results/experiment_results.csv`** columns:
  - `compression_ratio` — raw image bytes ÷ payload bytes (higher = smaller payload).
  - `psnr` — pixel fidelity in dB (JPEG wins here; expected).
  - `downstream_class_recall` — fraction of original objects re-detected on the
    reconstruction. **The key semantic metric.** Expect
    `semantic ≳ jpeg_matched ≫ text_only` at matched payload size — that is the
    proof that sending crops helps.
  - `downstream_center_error` — mean normalized position error of matched objects.
  - `deep_feature_distance`, `lpips`, `ocr_legibility` — `null`/blank unless the
    corresponding optional dep/flag is enabled.
  - `num_preserve_text` — count of preserve-mode text objects in the image.
- **`results/semantic/<id>.json`** → `size_report`: `structure_bytes`,
  `appearance_bytes`, `crops_raw_bytes`, `total_bytes` — the per-stream byte
  budget.
- **`results/logs/`** — per-stage logging; warnings tell you when an optional
  feature fell back (no OCR backend, no GPU for diffusion, no learned checkpoint).

---

## 5. What to replace / swap (config toggles & stubs)

Every model-bearing step is behind a base class and selected from config, so you
replace a component by implementing the concrete class and flipping a config key.

| Component | Base class | Default | Swap to | Where to implement |
|-----------|-----------|---------|---------|--------------------|
| Detector | `ObjectExtractor` | `YoloExtractor` | `extractor: learned` | [src/extractors/learned.py](../src/extractors/learned.py) |
| Relations | `RelationBuilder` | `RuleBasedRelationBuilder` | `relation_builder: learned` | [src/relations/learned.py](../src/relations/learned.py) |
| Appearance | `AppearanceEncoder` | `CropCompressor` | (embeddings) | [src/appearance/embedding.py](../src/appearance/embedding.py) |
| Reconstructor | `Reconstructor` | `CompositionalReconstructor` | `reconstructor: diffusion` | [src/reconstructors/diffusion.py](../src/reconstructors/diffusion.py) |
| Channel | `Channel` | `IdentityChannel` | (AWGN/Rayleigh) | [src/channels/](../src/channels/) |

All current stubs **fall back gracefully** (learned extractor → YOLO, learned
relations → rule-based, diffusion → compositional, embeddings → raise), so you
can flip the switch before the implementation is finished and the pipeline still
runs.

---

## 6. What to train

> Constraint: do **not** train large models from scratch. Fine-tune pretrained /
> frozen backbones and save checkpoints the pipeline loads automatically.

### 6.1 Our own object detector

Skeleton: [scripts/train_detector.py](../scripts/train_detector.py).

1. **Prepare a dataset** in Ultralytics format — a `data/detector.yaml`:
   ```yaml
   path: data/detector            # dataset root
   train: images/train
   val: images/val
   names: {0: person, 1: car, ...}
   ```
   with YOLO-format label `.txt` files alongside the images.
2. **Implement the TODO** in `scripts/train_detector.py` (a first version can just
   call ultralytics fine-tuning — the exact snippet is in the file's docstring).
3. **Run** (GPU machine):
   ```bash
   python scripts/train_detector.py --data data/detector.yaml \
       --base yolov8n.pt --epochs 50 --imgsz 640 --batch 16 \
       --out checkpoints/detector.pt
   ```
4. **Use it** — set in `config.yaml`:
   ```yaml
   extractor: learned
   checkpoint_path: checkpoints/detector.pt
   ```
   `LearnedObjectExtractor` loads the checkpoint if present, else warns and falls
   back to YOLO. Compare side by side by running once with `extractor: yolo` and
   once with `extractor: learned`.

### 6.2 Learned relation model

Stub: [src/relations/learned.py](../src/relations/learned.py) (`LearnedRelationBuilder`).

- **Goal:** predict relations (a scene graph) from object features instead of the
  geometry heuristics in `RuleBasedRelationBuilder`.
- **Keep the interface:** `build(objects) -> list[Relation]`.
- **Data:** object pairs + relation labels (e.g. Visual Genome–style triples, or
  bootstrap from the rule-based output and refine).
- **Wire up:** replace the `_fallback` delegation with your model; select via
  `relation_builder: learned`.

### 6.3 Appearance embeddings (CLIP)

Stub: [src/appearance/embedding.py](../src/appearance/embedding.py) (`EmbeddingEncoder`).

- **Goal:** send a compact learned embedding per object instead of a raw crop;
  the receiver conditions a generator on it.
- **Encode:** run the crop through a frozen CLIP image encoder, serialize the
  vector (e.g. float16 bytes) in `encode()`; `decode()` reverses it.
- **Keep the interface:** `encode(obj, image_rgb) -> bytes` / `decode(bytes) -> np.ndarray`
  (or adapt the reconstructor to consume embeddings).
- It is **not** selected by default; `CropCompressor` stays the default until this
  is implemented and validated.

---

## 7. GPU diffusion reconstruction

Implementation: [src/reconstructors/diffusion.py](../src/reconstructors/diffusion.py).

1. Install the diffusion extras + CUDA torch (see [§2.2](#22-optional--training--gpu-extras)).
2. Enable in `config.yaml`:
   ```yaml
   reconstructor: diffusion
   diffusion:
     enabled: true
     model_id: stabilityai/sd-turbo     # any diffusers text2img pipeline
   ```
3. Behavior: the background/scene is generated from the scene text; the received
   **crops are always composited on top** and **text/`preserve` regions are never
   generated** (diffusion produces unreadable text). With no GPU or missing deps
   it logs a warning and falls back to the compositional background — so the same
   config is safe to commit and run anywhere.
4. First run downloads the model weights from Hugging Face; ensure network access
   and disk space. Lower `num_inference_steps` / use a turbo model for speed.

---

## 8. Reproducibility

- `seed` in `config.yaml` (or `--seed`) threads through detection and any sampling.
- The two-stream payload round-trips through binary (de)serialization on every
  run (via `IdentityChannel`), so the on-wire format is exercised end-to-end.
- When the wireless `Channel` lands, seed it too for repeatable degradation.

---

## 9. Quick troubleshooting

| Symptom (log line) | Meaning / fix |
|--------------------|---------------|
| `No OCR backend available …` | Install `easyocr` or `pytesseract`; until then text regions rely on `preserve_classes` only. |
| `DiffusionReconstructor: no GPU available …` | Expected on CPU; install CUDA torch + diffusers on a GPU box to enable. |
| `Learned detector checkpoint not found …` | Train one (`scripts/train_detector.py`) or keep `extractor: yolo`. |
| `Deep-feature distance unavailable` | torchvision weights couldn't download; metric is reported as `null`. |
| `EmbeddingEncoder … is a stub` | Expected; `CropCompressor` is the active appearance encoder. |
