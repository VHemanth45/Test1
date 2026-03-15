#!/usr/bin/env python3
"""
RenAIssance OCR Pipeline — Steps 1.3 through 2.3
=================================================
Step 1.3: Split labelled dataset into 80/20 train/val (stratified by source)
Step 2.1: Choose YOLOv8n model
Step 2.2: Fine-tune YOLOv8n on the dataset
Step 2.3: Evaluate YOLO performance
"""

import os
import re
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict
from PIL import Image
import PIL.Image

# Disable PIL decompression bomb limit — our scanned page images are very
# high-resolution but perfectly valid.  YOLO resizes to 640px anyway.
Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = None
# Also neuter the bomb check itself — ultralytics worker processes and
# internal patches can reset MAX_IMAGE_PIXELS, so we replace the check
# function with a no-op to guarantee large scans are never rejected.
PIL.Image._decompression_bomb_check = lambda size: None

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
LABELLED_DIR = BASE_DIR / "Data" / "labelled_dataset"
YOLO_DATASET_DIR = BASE_DIR / "Data" / "yolo_dataset"
RUNS_DIR = BASE_DIR / "runs"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
RANDOM_SEED = 42

# Training hyperparameters
YOLO_MODEL = "yolov8n.pt"  # Nano variant — fast, lightweight, sufficient for 1 class
EPOCHS = 100
PATIENCE = 15  # Early stopping patience
BATCH_SIZE = 4            # Low VRAM — RTX 3050 Ti ~4 GB
IMG_SIZE = 480            # Smaller than 640 to save memory
WORKERS = 2               # Fewer dataloader workers to limit RAM/VRAM spikes


# ──────────────────────────────────────────────────────────────────────────────
# Step 1.3 — Split into Train/Val with Stratified Sampling
# ──────────────────────────────────────────────────────────────────────────────
def extract_source(filename: str) -> str:
    """Extract source identifier (e.g. 'source1') from filename like
    '05ecf430-source4_page014.png'."""
    match = re.search(r"(source\d+)", filename)
    return match.group(1) if match else "unknown"


def is_multibox(label_path: Path) -> bool:
    """Check if a label file has more than 1 bounding box (two-column page)."""
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return len(lines) > 1


def stratified_split(labelled_dir: Path, train_ratio: float, seed: int):
    """
    Split images into train/val ensuring:
    - Both splits have pages from ALL 6 sources
    - Both splits have single-column AND multi-column pages
    """
    random.seed(seed)

    images_dir = labelled_dir / "images"
    labels_dir = labelled_dir / "labels"

    # Group files by (source, is_multibox) for stratification
    groups = defaultdict(list)
    for img_file in sorted(images_dir.glob("*.png")):
        stem = img_file.stem
        label_file = labels_dir / f"{stem}.txt"
        if not label_file.exists():
            print(f"  WARNING: No label for {img_file.name}, skipping")
            continue
        source = extract_source(img_file.name)
        multi = is_multibox(label_file)
        groups[(source, multi)].append(stem)

    train_stems = []
    val_stems = []

    print("\n  Stratified split breakdown:")
    print(f"  {'Group':<30} {'Total':>5} {'Train':>5} {'Val':>5}")
    print("  " + "-" * 50)

    for key in sorted(groups.keys()):
        items = groups[key]
        random.shuffle(items)
        n_train = max(1, round(len(items) * train_ratio))
        # Ensure at least 1 in val if group has ≥ 2 items
        if len(items) >= 2:
            n_train = min(n_train, len(items) - 1)

        train_items = items[:n_train]
        val_items = items[n_train:]

        train_stems.extend(train_items)
        val_stems.extend(val_items)

        source, multi = key
        label = f"{source} ({'multi' if multi else 'single'})"
        print(f"  {label:<30} {len(items):>5} {len(train_items):>5} {len(val_items):>5}")

    print("  " + "-" * 50)
    print(f"  {'TOTAL':<30} {len(train_stems) + len(val_stems):>5} {len(train_stems):>5} {len(val_stems):>5}")

    return train_stems, val_stems


def create_yolo_dataset(labelled_dir: Path, output_dir: Path, train_stems, val_stems):
    """Create the YOLO dataset directory structure by copying images and labels."""
    # Clean existing dataset
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split, stems in [("train", train_stems), ("val", val_stems)]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            src_img = labelled_dir / "images" / f"{stem}.png"
            src_lbl = labelled_dir / "labels" / f"{stem}.txt"

            shutil.copy2(src_img, img_dir / f"{stem}.png")
            shutil.copy2(src_lbl, lbl_dir / f"{stem}.txt")

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["maintext"],
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\n  Dataset created at: {output_dir}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Train: {len(train_stems)} images")
    print(f"  Val:   {len(val_stems)} images")

    return yaml_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 2.1 + 2.2 — Fine-tune YOLOv8n
# ──────────────────────────────────────────────────────────────────────────────
def train_yolo(data_yaml: Path):
    """Fine-tune YOLOv8n on the prepared dataset."""
    from ultralytics import YOLO

    # Re-apply after ultralytics import so its internal PIL calls respect it
    Image.MAX_IMAGE_PIXELS = None

    print(f"\n  Loading pretrained model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    print(f"  Starting training for up to {EPOCHS} epochs (patience={PATIENCE})...")
    print(f"  Batch size: {BATCH_SIZE}, Image size: {IMG_SIZE}")
    print(f"  Results will be saved to: {RUNS_DIR}/\n")

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=0,  # GPU
        workers=WORKERS,
        project=str(RUNS_DIR),
        name="yolo_maintext",
        exist_ok=True,
        amp=True,         # Mixed precision — halves VRAM for activations
        cache=False,      # Don't cache images in RAM
        # Minimal augmentation for document images
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.1,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,       # Disabled — saves VRAM (no 4-image composite)
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        verbose=False,
    )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 2.3 — Evaluate YOLO performance
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_yolo(data_yaml: Path):
    """Evaluate the best checkpoint on the validation set."""
    from ultralytics import YOLO

    # Re-apply after ultralytics import so its internal PIL calls respect it
    Image.MAX_IMAGE_PIXELS = None

    best_model_path = RUNS_DIR / "yolo_maintext" / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"  ERROR: Best model not found at {best_model_path}")
        return None

    print(f"\n  Loading best model: {best_model_path}")
    model = YOLO(str(best_model_path))

    print("  Running validation...")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name="yolo_maintext_eval",
        exist_ok=True,
        verbose=False,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  YOLO EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    print("=" * 60)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Visual inspection helper — run predictions on a few val images
# ──────────────────────────────────────────────────────────────────────────────
def visual_inspect(data_yaml: Path, n_samples: int = 8):
    """Run predictions on sample validation images and save visualizations."""
    from ultralytics import YOLO

    # Re-apply after ultralytics import so its internal PIL calls respect it
    Image.MAX_IMAGE_PIXELS = None

    best_model_path = RUNS_DIR / "yolo_maintext" / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"  ERROR: Best model not found at {best_model_path}")
        return

    model = YOLO(str(best_model_path))

    val_images_dir = YOLO_DATASET_DIR / "images" / "val"
    val_images = sorted(val_images_dir.glob("*.png"))[:n_samples]

    if not val_images:
        print("  No validation images found for inspection.")
        return

    print(f"\n  Running predictions on {len(val_images)} validation images...")
    results = model.predict(
        source=[str(p) for p in val_images],
        imgsz=IMG_SIZE,
        device=0,
        save=True,
        project=str(RUNS_DIR),
        name="yolo_maintext_inspect",
        exist_ok=True,
        conf=0.25,
    )

    print(f"  Visual predictions saved to: {RUNS_DIR}/yolo_maintext_inspect/")

    # Print per-image box counts
    for img_path, result in zip(val_images, results):
        n_boxes = len(result.boxes)
        source = extract_source(img_path.name)
        print(f"    {img_path.name} ({source}): {n_boxes} box(es) detected")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RenAIssance OCR Pipeline — Steps 1.3 to 2.3")
    print("=" * 60)

    # Step 1.3: Split
    print("\n[Step 1.3] Splitting labelled dataset into train/val...")
    train_stems, val_stems = stratified_split(LABELLED_DIR, TRAIN_RATIO, RANDOM_SEED)

    print("\n[Step 1.3] Creating YOLO dataset structure...")
    data_yaml = create_yolo_dataset(LABELLED_DIR, YOLO_DATASET_DIR, train_stems, val_stems)

    # Step 2.1 + 2.2: Fine-tune
    print("\n[Step 2.1] Using YOLOv8n (nano) — pretrained on COCO")
    print("[Step 2.2] Fine-tuning on main_text detection...")
    train_yolo(data_yaml)

    # Step 2.3: Evaluate
    print("\n[Step 2.3] Evaluating YOLO performance...")
    evaluate_yolo(data_yaml)

    print("\n" + "=" * 60)
    print("  Pipeline complete! Check results in: runs/yolo_maintext/")
    print("=" * 60)


if __name__ == "__main__":
    main()
