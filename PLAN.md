# PLAN — Semantic Image Communication v1 (Prove the Pipeline)

This document is the architecture + delivery plan for the v1 milestone: an
**image-in / image-out** semantic communication pipeline. The transmitter
detects objects, ranks them, sends a *compressed crop* for the important ones
and *text only* for the rest, plus a compact scene graph. The receiver
regenerates a similar image from the scene text and composites the received
crops back at their locations. There is **no lossy channel yet** — the only
channel is a pass-through `IdentityChannel` — but every interface is shaped so
a real wireless channel, learned models, and embeddings slot in later without
changing callers.

## Existing pipeline (before this milestone)

```
image -> YoloExtractor (src/extract.py)
      -> rule-based spatial relations (src/oar_builder.py)
      -> compact pipe-separated tokens (src/semantic_codec.py, src/encoder.py)
      -> probabilistic token-dropping channel (src/channel.py NoisyChannel)
      -> graph repair (src/decoder.py)
      -> template text reconstruction (src/reconstruct.py)
      -> custom semantic score (src/evaluate.py)
```

Only model: pretrained `yolov8n.pt`. Nothing else trained.

## Target pipeline (after this milestone)

```
TRANSMITTER
  image
   -> ObjectExtractor            (YoloExtractor | LearnedObjectExtractor)
   -> RelationBuilder            (RuleBasedRelationBuilder | LearnedRelationBuilder)
   -> ImportanceScorer           (area * confidence * centrality -> top-k)
   -> ObjectModeClassifier       (regenerate | preserve; OCR text -> preserve)
   -> AppearanceEncoder          (CropCompressor: JPEG/WebP per-mode quality)
   -> SemanticPayload            (structure stream + appearance stream, priorities, sizes)
CHANNEL
   -> Channel                    (IdentityChannel only for now; pass-through)
RECEIVER
   -> Reconstructor              (CompositionalReconstructor CPU | DiffusionReconstructor GPU)
        - background/scene from text
        - composite received crops at their boxes
        - preserve-text: paste hi-q crop or re-render crisp OCR text
   -> text description           (reused template generator)
METRICS / BASELINES / RUNNER
   -> Metrics                    (payload size, compression ratio, PSNR, LPIPS*,
                                  deep-feature cosine*, downstream detector match,
                                  OCR legibility for preserve-text)
   -> Baselines                  (JPEG @ matched payload size; text-only no-crops)
   -> ExperimentRunner           (folder x configs + baselines -> table + side-by-sides)
```
`*` = optional, behind a flag / graceful fallback when the dep is absent.

## Module plan (base class + concrete impls; everything swappable)

| # | Base class | Concrete impls | File(s) |
|---|------------|----------------|---------|
| 1 | `ObjectExtractor` | `YoloExtractor` (default), `LearnedObjectExtractor` (P5 scaffold) | `src/extractors/{base,yolo,learned}.py` |
| 2 | `RelationBuilder` | `RuleBasedRelationBuilder` (default), `LearnedRelationBuilder` (P5 stub) | `src/relations/{base,rule_based,learned}.py` |
| 3 | `ImportanceScorer` | `HeuristicImportanceScorer` (area·conf·centrality) | `src/importance.py` |
| 4 | `ObjectModeClassifier` | OCR/text-region + forced-class config | `src/mode_classifier.py` |
| 5 | `AppearanceEncoder` | `CropCompressor` (default), `EmbeddingEncoder` (P5 CLIP stub) | `src/appearance/{base,crop,embedding}.py` |
| 6 | `SemanticPayload` | two-stream serializable payload | `src/payload.py` |
| 7 | `Channel` | `IdentityChannel` only (pass-through) | `src/channels/{base,identity}.py` |
| 8 | `Reconstructor` | `CompositionalReconstructor` (CPU default), `DiffusionReconstructor` (flag) | `src/reconstructors/{base,compositional,diffusion,text}.py` |
| 9 | `Metrics` | PSNR / LPIPS* / deep-feature* / downstream / OCR-legibility | `src/metrics.py` |
| 10 | Baselines | `jpeg_baseline`, `text_only_baseline` | `src/baselines.py` |
| 11 | `ExperimentRunner` | folder × configs + baselines, side-by-sides | `experiment.py` (rewritten) |

Shared OCR backend abstraction: `src/ocr.py` (auto: easyocr -> pytesseract -> none).
Pipeline orchestration: `src/pipeline.py` (transmit + receive).

### Final file layout

```
src/
  types.py            # legacy types + new: ObjectMode, SceneObject, Stream
  extractors/         # ObjectExtractor base + Yolo + Learned
  relations/          # RelationBuilder base + RuleBased + Learned
  importance.py       # ImportanceScorer base + Heuristic
  mode_classifier.py  # ObjectModeClassifier (+ forced classes, OCR)
  ocr.py              # OCR backend abstraction (optional deps)
  appearance/         # AppearanceEncoder base + CropCompressor + EmbeddingEncoder stub
  payload.py          # SemanticPayload (two streams) + binary (de)serialization
  channels/           # Channel base + IdentityChannel
  reconstructors/     # Reconstructor base + Compositional + Diffusion + text
  metrics.py          # Metrics
  baselines.py        # JPEG-matched + text-only
  pipeline.py         # SemanticPipeline orchestration
  # legacy (retained, still live): oar_builder, semantic_codec, encoder,
  #   channel(NoisyChannel), decoder, reconstruct, evaluate, extract
main.py               # runs the NEW v1 pipeline (keeps all old CLI flags)
experiment.py         # ExperimentRunner (configs + baselines + side-by-sides)
scripts/train_detector.py  # P5 training skeleton
tests/                # unittest: payload, importance, mode, metrics
```

## Payload design (future-proofing for a real channel)

`SemanticPayload` carries **two independently-degradable streams**, each tagged
with an integer `priority` (0 = highest protection):

1. **Structure stream** (priority 0): compact scene graph — object ids,
   classes, boxes, relations, per-object mode, OCR text, original image size.
   Serialized as JSON.
2. **Appearance stream** (priority 1): `object_id -> compressed crop bytes`
   for selected/important + preserve objects.

`to_bytes()` packs a **length-prefixed binary container** (magic + version +
structure block + crop entries + priorities + image size). No base64, so the
reported byte sizes are the true on-wire sizes. `size_report()` returns
`structure_bytes`, `appearance_bytes`, `total_bytes`. A future `Channel` will
corrupt each stream independently (unequal error protection by priority);
`IdentityChannel` round-trips through `to_bytes()/from_bytes()` so serialization
is exercised end-to-end today.

## Defaults chosen (ambiguous decisions recorded here)

- **Importance score** = `norm_area * confidence * centrality`, where
  `centrality = 1 - dist(center, image_center)/max_dist`. Top-k selection by a
  count budget (`importance.budget`, default **3**). Preserve-mode objects are
  *always* selected for a crop regardless of budget (their appearance must
  survive).
- **Mode**: default `regenerate`. Forced to `preserve` if (a) class is in
  `preserve_classes` (default `[person]`), or (b) OCR detects legible text in
  the crop (text stored in the scene graph).
- **Crop quality tiers**: `preserve` -> JPEG quality **95**, `regenerate` ->
  JPEG quality **35**. Format default **JPEG** (`appearance.format`, WEBP
  optional).
- **Reconstruction canvas** (CPU compositional): solid background
  (`background_color`, default mid-gray `[127,127,127]`); text-only objects are
  drawn as labeled rectangles; selected objects get their decompressed crop
  pasted at the box; preserve-text objects get the hi-q crop pasted and, when
  re-render is on, crisp OCR text drawn over it. Diffusion is **never** used on
  text regions.
- **Channel**: `IdentityChannel` (pass-through) is the only channel and the
  default. `noise_level` config is retained but unused by the v1 image path.
- **Metrics that need heavy/absent deps are optional and default off or
  degrade gracefully**: LPIPS (needs `lpips`), deep-feature cosine (needs
  torchvision weights download), OCR legibility (needs an OCR backend). PSNR,
  payload size / compression ratio, and downstream detector match always run.
- **Reproducibility**: single `seed` threads through extractor + any sampling.
- **Python 3.14 / CPU-first**: the repo runs on Python 3.14 where `easyocr`,
  `pytesseract`, `lpips`, `diffusers`, `transformers` are not installable/loaded;
  all are *optional*. The core CPU path needs only `numpy, pillow, opencv,
  torch, torchvision, ultralytics, matplotlib, scipy, PyYAML`.
- **Tests** use the stdlib `unittest` runner (no `pytest` dependency).

## Config additions (documented in README; old keys all still work)

```yaml
extractor: yolo                 # yolo | learned
relation_builder: rule_based    # rule_based | learned
reconstructor: compositional    # compositional | diffusion
channel: identity               # identity (only option in v1)
background_color: [127, 127, 127]
preserve_classes: [person]
importance:
  budget: 3                     # top-k objects sent as crops
appearance:
  format: JPEG                  # JPEG | WEBP
  preserve_quality: 95
  regenerate_quality: 35
ocr:
  enabled: true
  backend: auto                 # auto | easyocr | pytesseract | none
diffusion:
  enabled: false
  model_id: stabilityai/sd-turbo
metrics:
  deep_features: false          # torchvision VGG cosine (downloads weights)
  lpips: false                  # needs `lpips` package
streams:
  structure_priority: 0
  appearance_priority: 1
```

## Delivery order (main stays runnable throughout)

- **P0** — explore + this PLAN.md. *(done)*
- **P1** — base classes + `YoloExtractor`, `RuleBasedRelationBuilder`,
  `IdentityChannel`, text reconstruction wired through `SemanticPipeline`;
  behavior intact.
- **P2** — `HeuristicImportanceScorer`, `ObjectModeClassifier` (+OCR),
  `CropCompressor`, two-stream `SemanticPayload` with reported sizes.
- **P3** — `CompositionalReconstructor` (full image-in/image-out on CPU); then
  `DiffusionReconstructor` (flag-gated, falls back to compositional).
- **P4** — `Metrics`, JPEG + text-only baselines, `ExperimentRunner` with
  side-by-side original/reconstructed outputs.
- **P5** — scaffolds only: `LearnedObjectExtractor` + `scripts/train_detector.py`
  skeleton; `LearnedRelationBuilder` stub; `EmbeddingEncoder` (CLIP) stub;
  confirm `Channel` base + payload streams are ready for a real channel.

## TODO backlog (future work; scaffolded now)

- [ ] **Wireless channel**: `AWGNChannel`, `RayleighChannel` subclasses of
  `Channel` with per-stream unequal error protection driven by `priority` and a
  signal-strength / SNR parameter. The `Channel.transmit(payload)` signature and
  the per-stream `priority` tags already support this — no caller changes needed.
- [ ] **Learned relations**: train `LearnedRelationBuilder` (scene-graph
  generation) replacing the rule-based geometry heuristics.
- [ ] **Appearance embeddings**: finish `EmbeddingEncoder` (CLIP image
  embeddings) so the appearance stream carries embeddings instead of raw crops;
  receiver uses a conditioned generator.
- [ ] **Own detector**: implement training in `scripts/train_detector.py` and
  load the checkpoint in `LearnedObjectExtractor` (falls back to YOLO today).
- [ ] **Diffusion background**: enable `DiffusionReconstructor` with a frozen
  `diffusers` model on GPU for scene/background synthesis (text regions stay
  composited, never generated).
- [ ] **Metrics**: add SSIM (needs `scikit-image`) and turn on LPIPS /
  deep-feature cosine once those deps are installed in the target env.
