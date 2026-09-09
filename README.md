# SAM3 Forensic Object Detection

Open-vocabulary object detection and segmentation for digital & physical
forensic evidence triage, built on Meta AI's **SAM 3** (Segment Anything
Model 3) and the `ultralytics` SAM3 predictor.

Given a photo of a scene (e.g. a seized device pile, a search-and-seizure
photograph, or an evidence table), this project uses SAM 3's text-prompted
detection to automatically locate and segment items commonly relevant to
forensic and evidence-cataloguing workflows — digital media (laptops, hard
drives, SIM/SD cards, routers, DVRs) as well as other flagged physical items —
using **plain-language class names** instead of a fixed, pre-trained label
set.

> ⚠️ **Scope note:** This tool is a *visual triage aid*. It draws bounding
> boxes/masks around objects that visually match a text prompt. It does not
> identify, authenticate, fingerprint, or forensically validate evidence, and
> its output must always be verified by a qualified examiner before any
> investigative or legal use.

---

## How it works

1. **Model**: [`facebook/sam3`](https://huggingface.co/facebook/sam3), loaded
   via `huggingface_hub.snapshot_download` and run through
   `ultralytics.models.sam.SAM3SemanticPredictor`.
2. **Prompting**: SAM 3 supports open-vocabulary "concept prompts" — you pass
   a list of free-text class names (e.g. `"laptop"`, `"USB drive"`,
   `"SIM card"`) instead of retraining a detector.
3. **Inference**: the predictor loads an image, runs `predictor(text=classes)`,
   and returns per-class masks/boxes with confidence scores.
4. **Visualization**: results are rendered with `result.plot()` and saved to
   `outputs/sample_detections/`.

See [`docs/methodology.md`](docs/methodology.md) for a fuller write-up of the
approach, prompt design, and known limitations.

---

## Repository structure

```
SAM3-Forensic-Object-Detection/
│
├── README.md                     # This file
├── SAM3_Object_Detection.ipynb   # Main notebook (Colab-ready)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT license (code only — see note below)
│
├── images/
│   ├── input/                    # Put your source images here
│   └── results/                  # Annotated output images land here
│
├── outputs/
│   └── sample_detections/        # Example detection runs / logs
│
└── docs/
    └── methodology.md            # Detailed methodology & limitations
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/SAM3-Forensic-Object-Detection.git
cd SAM3-Forensic-Object-Detection
pip install -r requirements.txt
```

### 2. Get access to SAM 3 weights

SAM 3 is gated on Hugging Face. You need a Hugging Face account with access
approved for `facebook/sam3`.

```python
from huggingface_hub import notebook_login
notebook_login()  # or `huggingface-cli login` from the terminal
```

### 3. Download the checkpoint

```python
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="facebook/sam3")
```

Locate the `.pt` checkpoint inside the downloaded snapshot and copy it
somewhere convenient, e.g. `./sam3.pt`.

### 4. Run detection

```python
from ultralytics.models.sam import SAM3SemanticPredictor

overrides = {
    "conf": 0.25,
    "task": "segment",
    "mode": "predict",
    "model": "./sam3.pt",
    "save": True,
}
predictor = SAM3SemanticPredictor(overrides=overrides)

predictor.set_image("images/input/your_image.jpg")

classes = [
    "laptop", "cell phone", "external hard drive", "USB drive",
    "WiFi router", "SD card", "memory card", "SIM card",
    "camera", "DVR", "hard drive", "SSD",
]

results = predictor(text=classes)
```

### 5. Visualize / save results

```python
from PIL import Image

for result in results:
    plotted = result.plot()[:, :, ::-1]
    Image.fromarray(plotted).save("images/results/detection.png")
```

The full, runnable, Colab-friendly version of this workflow is in
[`SAM3_Object_Detection.ipynb`](SAM3_Object_Detection.ipynb).

---

## Customizing the class list

Because SAM 3 is open-vocabulary, you can adapt `classes` to whatever your
use case requires — no retraining. Effective prompts are usually short,
concrete noun phrases (`"external hard drive"`, `"SIM card"`) rather than
abstract categories (`"evidence"`, `"contraband"`).

## Notes, ethics & limitations

- **Human-in-the-loop only.** This is a candidate-generation / triage tool.
  All detections require review by a trained examiner before being relied on
  for any investigative, legal, or safety-critical decision.
- **No weapon-handling or exploitation guidance.** This repository only
  performs visual object localization on user-supplied images; it does not
  provide instructions for building, modifying, or operating any device or
  weapon.
- **False positives/negatives.** Open-vocabulary segmentation is sensitive to
  lighting, occlusion, image quality, and prompt wording. Always validate
  against the source imagery.
- **Model license.** SAM 3 weights are distributed by Meta AI under their own
  license/terms on Hugging Face — review and accept those terms before use.
  This repo's MIT license covers the code in this repo only, not the model
  weights.
- **Data privacy/chain-of-custody.** If used on real case material, follow
  your organization's evidence-handling, retention, and privacy policies;
  this tool does not implement chain-of-custody logging.

## License

Code in this repository is released under the [MIT License](LICENSE).
