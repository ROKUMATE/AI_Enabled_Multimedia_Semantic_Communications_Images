# Commands Used To Check The Project

This document records the commands I ran to verify that the project builds and runs.

## 1. Compile the Python sources

```bash
/Users/rokum/Rokum/sem/wct/sementic-analysis-dataset-image-to-text/.venv/bin/python -m compileall main.py experiment.py src
```

Result: passed. All Python modules compiled successfully.

## 2. Smoke test the semantic codec path

```python
from src.encoder import OAREncoder
from src.decoder import OARDecoder
from src.channel import NoisyChannel
from src.types import OARRepresentation, DetectedObject, Relation

oar = OARRepresentation(
    objects=[
        DetectedObject(object_id='obj_0', name='person', bbox=(0, 0, 10, 10), confidence=0.9),
        DetectedObject(object_id='obj_1', name='dog', bbox=(20, 20, 30, 30), confidence=0.8),
    ],
    relations=[Relation(subject_id='obj_0', predicate='near', object_id='obj_1')],
)
encoder = OAREncoder()
channel = NoisyChannel(noise_level=0.2, seed=42)
decoder = OARDecoder()
packet = encoder.encode(oar)
noisy_packet = channel.transmit(packet)
decoded = decoder.decode(noisy_packet)
print(packet.to_dict())
print(noisy_packet.to_dict())
print(decoded.to_dict())
```

Result: passed. The payload encoded as compact semantic tokens, channel transmission succeeded, and decoding returned a valid partial OAR graph.

## 3. Run the main pipeline without privacy noise

```bash
/Users/rokum/Rokum/sem/wct/sementic-analysis-dataset-image-to-text/.venv/bin/python main.py --config config.yaml --no-enable-privacy
```

Result: passed. The pipeline found 2 images in `data/images/` and completed processing both images successfully.

## 4. Run the experiment sweep on one image

```bash
/Users/rokum/Rokum/sem/wct/sementic-analysis-dataset-image-to-text/.venv/bin/python experiment.py --config config.yaml --max-images 1 --noise-start 0.0 --noise-stop 0.5 --noise-step 0.5
```

Result: passed. The experiment runner completed a 2-point noise sweep, wrote experiment outputs, and generated plots.

## 5. Run all checks with one command

```bash
make check
```

Result: this target runs the three verification steps above in order: compile the Python sources, run the main pipeline without channel noise, and run the lightweight experiment sweep.

## Notes

- All commands were run from the project virtual environment.
- The main pipeline writes outputs under `results/`.
- The experiment runner writes JSON, CSV, logs, and plots under `results/`.
