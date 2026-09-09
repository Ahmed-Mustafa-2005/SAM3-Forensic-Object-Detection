# Methodology

## 1. Objective

Provide a lightweight, no-training-required pipeline for locating and
segmenting objects of interest in still images, aimed at forensic
evidence-cataloguing and triage workflows — most notably digital storage and
communication devices (laptops, external drives, SD/SIM cards, routers,
DVRs, phones) that commonly need to be identified and inventoried from scene
photographs.

## 2. Model choice: SAM 3

[SAM 3](https://huggingface.co/facebook/sam3) (Segment Anything Model 3) is
Meta AI's open-vocabulary segmentation model. Unlike a conventional object
detector trained on a fixed label set (e.g. COCO's 80 classes), SAM 3 accepts
**free-text concept prompts** at inference time and returns segmentation
masks and bounding boxes for any matching instances in the image.

This is well suited to evidence triage because:

- The set of "interesting" objects varies case to case and can't be fully
  enumerated in advance.
- New object categories (e.g. a new model of storage device) can be added by
  simply adding a string to the prompt list — no annotation or retraining.
- Segmentation masks (not just boxes) support downstream cropping,
  redaction, or measurement tasks.

The project uses the `ultralytics` library's `SAM3SemanticPredictor` wrapper,
which exposes SAM 3 through the same ergonomic API as other Ultralytics
models (`predict`, `.plot()`, result objects with `.boxes`, `.masks`, etc.).

## 3. Pipeline

```
 ┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
 │  Input image     │ --> │  SAM3 predictor    │ --> │  Per-class masks /   │
 │  (scene photo)   │     │  + text prompts    │     │  boxes + confidence  │
 └─────────────────┘     └───────────────────┘     └──────────────────────┘
                                                              │
                                                              v
                                                     ┌──────────────────┐
                                                     │  Annotated image  │
                                                     │  (result.plot())  │
                                                     └──────────────────┘
```

1. **Acquire weights** — `snapshot_download(repo_id="facebook/sam3")` via
   Hugging Face Hub (gated download, requires accepted license + auth token).
2. **Initialize predictor** — `SAM3SemanticPredictor` with `conf=0.25`,
   `task="segment"`, `mode="predict"`.
3. **Set image** — `predictor.set_image(path)` loads and preprocesses a
   single image.
4. **Prompt with text classes** — `predictor(text=classes)` runs one forward
   pass per prompt set and returns Ultralytics `Results` objects containing
   masks, boxes, class labels, and confidence scores.
5. **Visualize / persist** — `result.plot()` draws the annotated overlay;
   this is written to `images/results/` (or `outputs/sample_detections/` for
   example runs).

## 4. Prompt design

Class prompts are plain noun phrases. Good prompts in practice are:

- **Concrete and specific**: `"external hard drive"` outperforms
  `"storage device"`.
- **Singular, not category names**: individual object types work better
  than umbrella terms like `"electronics"`.
- **Short**: 1–3 words per class tends to segment more reliably than long
  descriptive phrases.

The confidence threshold (`conf`) trades off recall vs. precision; `0.25` is
a reasonable starting point for triage (favoring recall, i.e. don't miss
candidate items) with the expectation that a human reviews all detections.

## 5. Evaluation approach

Because this is an open-vocabulary, zero-shot pipeline, formal
precision/recall benchmarking requires a labeled evidence-photo dataset
specific to the deployment context. Recommended evaluation approach for
adopters:

1. Curate a small held-out set of representative scene photos with manual
   ground-truth boxes for the classes you care about.
2. Run the pipeline with your target class list and confidence threshold.
3. Compute per-class precision/recall and inspect false positives/negatives
   qualitatively (lighting, occlusion, unusual angles are the common
   failure modes).
4. Tune `conf` and prompt wording iteratively.

## 6. Limitations

- **Not a forensic authentication tool.** The pipeline localizes objects
  visually; it makes no claims about object authenticity, ownership, or
  evidentiary status.
- **Sensitive to prompt phrasing and image quality.** Zero-shot performance
  varies with lighting, resolution, occlusion, and unusual object
  orientations.
- **No chain-of-custody or audit logging** is implemented; this must be
  layered on separately if used in a real casework pipeline.
- **Gated model weights.** SAM 3 requires accepting Meta's license on
  Hugging Face; weights are not redistributed in this repository.
- **Human review is required** before any detection here is used to inform
  an investigative or legal decision.

## 7. References

- Meta AI, "Segment Anything Model 3" model card —
  [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
- Ultralytics documentation —
  [docs.ultralytics.com](https://docs.ultralytics.com)
