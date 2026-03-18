"""Step 2: Split dataset, fine-tune YOLOv8n, and evaluate."""

import re, shutil, random, yaml
from pathlib import Path
from collections import defaultdict
from PIL import Image
import PIL.Image

# Allow high-res scanned pages
Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.Image._decompression_bomb_check = lambda size: None

BASE_DIR = Path(__file__).resolve().parent
LABELLED_DIR = BASE_DIR / "Data" / "labelled_dataset"
YOLO_DIR = BASE_DIR / "Data" / "yolo_dataset"
RUNS_DIR = BASE_DIR / "runs"

# Training config
YOLO_MODEL = "yolov8n.pt"
EPOCHS, PATIENCE = 100, 15
BATCH, IMGSZ, WORKERS = 4, 480, 2


def split_dataset(train_ratio=0.8, seed=42):
    """Stratified 80/20 train/val split by source and box count."""
    random.seed(seed)
    imgs_dir = LABELLED_DIR / "images"
    lbls_dir = LABELLED_DIR / "labels"

    groups = defaultdict(list)
    for img in sorted(imgs_dir.glob("*.png")):
        lbl = lbls_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        src = re.search(r"(source\d+)", img.name)
        key = (src.group(1) if src else "unknown", lbl.read_text().strip().count("\n") > 0)
        groups[key].append(img.stem)

    train, val = [], []
    for items in groups.values():
        random.shuffle(items)
        n = max(1, round(len(items) * train_ratio))
        if len(items) >= 2:
            n = min(n, len(items) - 1)
        train.extend(items[:n])
        val.extend(items[n:])

    print(f"  Split: {len(train)} train, {len(val)} val")
    return train, val


def create_yolo_dataset(train_stems, val_stems):
    """Copy images/labels into YOLO directory structure and write data.yaml."""
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    for split, stems in [("train", train_stems), ("val", val_stems)]:
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for s in stems:
            shutil.copy2(LABELLED_DIR / "images" / f"{s}.png", img_dir)
            shutil.copy2(LABELLED_DIR / "labels" / f"{s}.txt", lbl_dir)

    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(yaml.dump({
        "path": str(YOLO_DIR.resolve()), "train": "images/train",
        "val": "images/val", "nc": 1, "names": ["maintext"],
    }))
    print(f"  YOLO dataset: {YOLO_DIR}")
    return yaml_path


def train_yolo(data_yaml):
    """Fine-tune YOLOv8n on the dataset."""
    from ultralytics import YOLO
    Image.MAX_IMAGE_PIXELS = None

    model = YOLO(YOLO_MODEL)
    model.train(
        data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE,
        batch=BATCH, imgsz=IMGSZ, device=0, workers=WORKERS,
        project=str(RUNS_DIR), name="yolo_maintext", exist_ok=True,
        amp=True, cache=False, verbose=False,
        # Minimal augmentation for documents
        hsv_h=0, hsv_s=0, hsv_v=0.1, degrees=0, translate=0,
        scale=0, flipud=0, fliplr=0, mosaic=0, mixup=0,
        copy_paste=0, erasing=0,
    )
    print("  Training complete.")


def evaluate_yolo(data_yaml):
    """Evaluate best checkpoint on validation set."""
    from ultralytics import YOLO
    Image.MAX_IMAGE_PIXELS = None

    best = RUNS_DIR / "yolo_maintext" / "weights" / "best.pt"
    if not best.exists():
        print(f"  ERROR: {best} not found"); return

    metrics = YOLO(str(best)).val(
        data=str(data_yaml), imgsz=IMGSZ, batch=BATCH,
        device=0, workers=WORKERS, project=str(RUNS_DIR),
        name="yolo_maintext_eval", exist_ok=True, verbose=False,
    )
    print(f"  mAP@50: {metrics.box.map50:.4f}  |  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}  |  Recall: {metrics.box.mr:.4f}")


def main():
    print("[Step 1.3] Splitting dataset...")
    train_stems, val_stems = split_dataset()

    print("[Step 1.3] Creating YOLO dataset...")
    data_yaml = create_yolo_dataset(train_stems, val_stems)

    print("[Step 2.1-2.2] Fine-tuning YOLOv8n...")
    train_yolo(data_yaml)

    print("[Step 2.3] Evaluating...")
    evaluate_yolo(data_yaml)

    print(f"\nDone! Results in {RUNS_DIR / 'yolo_maintext'}")


if __name__ == "__main__":
    main()
