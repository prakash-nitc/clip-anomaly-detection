---
title: Zero-Shot Anomaly Detection CLIP
emoji: "\U0001F50D"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Zero-Shot Industrial Anomaly Detection with CLIP

Upload a product image, pick its category - the model flags defects
**without any task-specific training** using a vision-language model (CLIP).

---

## How it works

1. CLIP encodes the image into a joint vision-language embedding.
2. Ensembles of "normal" and "abnormal" text prompts are encoded and mean-pooled.
3. The image is scored by softmax probability of matching the "abnormal" ensemble.

No threshold tuning, no training data.
Based on the **WinCLIP** idea (Jeong et al., CVPR 2023).

---

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Tech stack

Python - PyTorch - OpenCLIP - Gradio

## Key reference

Jeong et al., *"WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation"*, CVPR 2023.