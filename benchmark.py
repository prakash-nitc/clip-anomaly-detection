"""
MVTec AD Benchmark — CLIP Zero-Shot vs. One-Class SVM
======================================================
Reports image-level AUROC: CLIP Zero-Shot vs. trained One-Class SVM.


Run on a GPU (Kaggle free T4 works):
    python benchmark.py --data /path/to/mvtec-ad

Dataset: https://kaggle.com/datasets/ipythonx/mvtec-ad
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────
CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]
DISPLAY = {
    "carpet": "carpet", "grid": "grid", "leather": "leather", "tile": "tile",
    "wood": "wood", "bottle": "bottle", "cable": "cable", "capsule": "capsule",
    "hazelnut": "hazelnut", "metal_nut": "metal nut", "pill": "pill",
    "screw": "screw", "toothbrush": "toothbrush", "transistor": "transistor",
    "zipper": "zipper",
}
NORMAL_STATES   = ["good", "perfect", "flawless", "pristine", "normal", "unblemished"]
ABNORMAL_STATES = ["damaged", "defective", "broken", "flawed", "abnormal", "imperfect"]
TEMPLATES = [
    "a photo of a {state} {object}",
    "a {state} {object}",
    "a photo of a {state} {object} for quality inspection",
    "a close-up photo of a {state} {object}",
]



def build_prompts(category, states):
    obj = DISPLAY.get(category, category)
    return [t.format(state=s, object=obj) for t in TEMPLATES for s in states]


# ── Dataset ───────────────────────────────────────────────────────
class MVTecSplit(Dataset):
    """Images from one split (train/test) of a single category."""

    def __init__(self, root, category, split, transform):
        self.samples, self.transform = [], transform
        split_dir = Path(root) / category / split
        for subdir in sorted(split_dir.iterdir()):
            if not subdir.is_dir():
                continue
            label = 0 if subdir.name == "good" else 1
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self.samples.append((str(f), label))
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,              help="MVTec AD root")
    parser.add_argument("--model",      default="ViT-L-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--batch-size", type=int, default=32)
                        help="Skip the slow pixel-level AUROC stage")
    args = parser.parse_args()

    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, args.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    print(f"Model  : {args.model} ({args.pretrained})\n")

    @torch.no_grad()
    def encode_text(prompts):
        tokens = tokenizer(prompts).to(device)
        feats  = F.normalize(model.encode_text(tokens), dim=-1)
        return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)

    @torch.no_grad()
    def text_pair(normal_prompts, abnormal_prompts):
        return torch.cat([encode_text(normal_prompts),
                          encode_text(abnormal_prompts)], dim=0)

    @torch.no_grad()
    def extract_features(category, split):
        ds = MVTecSplit(args.data, category, split, preprocess)
        ld = DataLoader(ds, batch_size=args.batch_size, num_workers=2,
                        pin_memory=True, shuffle=False)
        feats, labels = [], []
        for imgs, lbls in ld:
            f = F.normalize(model.encode_image(imgs.to(device)), dim=-1)
            feats.append(f.cpu().numpy())
            labels.append(lbls.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    @torch.no_grad()
if __name__ == "__main__":
    main()
