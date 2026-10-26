"""Handcrafted baseline. Patch evidence is texture selection, not lesion localization."""

from pathlib import Path
import csv
import hashlib

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

MODES = ("global", "patch", "combined")


def load_manifest(path):
    path = Path(path).resolve()
    rows, seen = [], {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "image_path", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires patient_id,image_path,label (0=benign, 1=malignant)")
        for row in reader:
            patient = row["patient_id"].strip()
            if not patient or row["label"] not in ("0", "1"):
                raise ValueError("Every row needs a patient ID and label 0 or 1")
            image_path = (path.parent / row["image_path"]).resolve()
            if not image_path.is_file():
                raise ValueError(f"Missing image: {image_path}")
            # Prevent the same exported image being assigned to multiple patients or splits.
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate image content: {image_path}; first seen at {seen[digest]}")
            seen[digest] = str(image_path)
            density = row.get("density", "").strip()
            if density not in ("", "1", "2", "3", "4"):
                raise ValueError("Density must be empty or a category from 1 to 4")
            rows.append(dict(patient_id=patient, image_path=str(image_path),
                             label=int(row["label"]), density=density, sha256=digest, source_split=row.get("source_split", "").strip()))
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def patient_split(rows, seed=42):
    """Fixed 60/20/20 patient-group split; never select a seed using test performance."""
    groups = np.array([r["patient_id"] for r in rows])
    if len(set(groups)) < 10:
        raise ValueError("Need at least 10 distinct patients for this baseline split")
    indices = np.arange(len(rows))
    declared = {r.get("source_split", "") for r in rows}
    if declared == {"train", "test"}:
        trainval = np.array([i for i, r in enumerate(rows) if r["source_split"] == "train"])
        test = np.array([i for i, r in enumerate(rows) if r["source_split"] == "test"])
        if set(groups[trainval]) & set(groups[test]):
            raise ValueError("Official partitions overlap patients; audit or exclude training views first")
    elif declared == {""}:
        trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                               random_state=seed).split(indices, groups=groups))
    else:
        raise ValueError("source_split must be train/test for all rows, or blank for all rows")
    train_sub, val_sub = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                               random_state=seed).split(trainval, groups=groups[trainval]))
    parts = dict(train=trainval[train_sub], validation=trainval[val_sub], test=test)
    for name, ids in parts.items():
        if {rows[i]["label"] for i in ids} != {0, 1}:
            raise ValueError(f"{name} lacks a class; use a larger, representative cohort")
    return parts


def prepare_image(source, blur=0., noise=0., seed=42):
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original)
        if image.mode in ("P", "RGBA", "LA", "CMYK"):
            image = image.convert("RGB")
        # Normalize grayscale intensities before converting, preserving 16-bit exports.
        pixels = np.asarray(image, dtype=np.float32)
        if pixels.ndim == 3:
            pixels = pixels[..., :3].mean(axis=2)
        lo, hi = np.percentile(pixels, [1, 99.5])
        pixels = np.clip((pixels - lo) / max(float(hi - lo), 1e-6), 0, 1)
        image = Image.fromarray((pixels * 255).astype(np.uint8)).resize((256, 256))
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    pixels = np.asarray(image, dtype=np.float32) / 255
    if noise:
        pixels = np.clip(pixels + np.random.default_rng(seed).normal(0, noise, pixels.shape), 0, 1)
    return pixels


def texture(array):
    histogram = np.histogram(array, bins=16, range=(0, 1))[0].astype(float)
    histogram /= histogram.sum()
    gx, gy = np.diff(array, axis=1), np.diff(array, axis=0)
    return np.r_[histogram, array.mean(), array.std(), np.quantile(array, [.1, .5, .9]),
                 np.abs(gx).mean(), np.abs(gy).mean()]


def features(pixels, mode="combined"):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    # Select four texture-rich 64x64 grid patches, excluding near-empty background.
    patches = [(x, y, pixels[y:y+64, x:x+64])
               for y in range(0, 256, 64) for x in range(0, 256, 64)]
    patches.sort(key=lambda p: float(p[2].std()) if (p[2] > .05).mean() > .1 else -1,
                 reverse=True)
    selected = patches[:4]
    local = np.stack([texture(p[2]) for p in selected])
    global_features = texture(pixels)
    patch_features = np.r_[local.mean(axis=0), local.max(axis=0)]
    vector = {"global": global_features, "patch": patch_features,
              "combined": np.r_[global_features, patch_features]}[mode]
    return vector, [(x, y, x+64, y+64) for x, y, _ in selected]


def metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    return dict(n=len(y), auroc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                brier=float(brier_score_loss(y, p)),
                sensitivity=float(tp / (tp + fn)) if tp + fn else None,
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def coverage_curve(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    result = []
    for threshold in (.5, .6, .7, .8, .9):
        keep = np.maximum(p, 1-p) >= threshold
        result.append(dict(threshold=threshold, retained=int(keep.sum()),
                           coverage=float(keep.mean()),
                           error_rate=float(((p[keep] >= .5) != y[keep]).mean()) if keep.any() else None))
    return result
