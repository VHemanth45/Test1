#!/home/hemanth/Documents/RenAIssance/Test1/.venv/bin/python
"""
RenAIssance OCR Pipeline — Steps 2.4 and 2.5
=============================================
Step 2.4: Run fine-tuned YOLO on ALL page images, save detection manifest
Step 2.5: Crop detected main-text regions from each page

Cropping logic:
  - 1 bounding box  → single crop:  sourceX_pageYYY_full.png
  - 2 bounding boxes → sorted by x-centre as left/right:
        sourceX_pageYYY_left.png, sourceX_pageYYY_right.png
  - 0 bounding boxes → page skipped (logged as warning)
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

try:
    from PIL import Image
    import PIL.Image
except ImportError:
    print("ERROR: Pillow not installed. Run with the project venv:")
    print("  .venv/bin/python step2_4_to_2_5_yolo_crop.py")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Disable PIL decompression bomb limit — our scanned pages are high-res
# ──────────────────────────────────────────────────────────────────────────────
Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.Image._decompression_bomb_check = lambda size: None

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PAGE_IMAGES_DIR = BASE_DIR / "Data" / "page_images"
RUNS_DIR = BASE_DIR / "runs"
BEST_MODEL_PATH = RUNS_DIR / "yolo_maintext" / "weights" / "best.pt"

# Output directories
MANIFEST_PATH = BASE_DIR / "Data" / "yolo_detections_manifest.json"
CROPS_DIR = BASE_DIR / "Data" / "cropped_main_text"

# Inference settings (match training settings)
IMG_SIZE = 480
CONFIDENCE_THRESHOLD = 0.25

# Device — try GPU first, fall back to CPU
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# Sources to process
SOURCES = ["source1", "source2", "source3", "source4", "source5", "source6"]

# Padding (pixels) to add around each detected box to avoid clipping text
PAD_PX = 5


# ──────────────────────────────────────────────────────────────────────────────
# Step 2.4 — Run YOLO on Full Dataset
# ──────────────────────────────────────────────────────────────────────────────
def run_yolo_on_all_pages() -> dict:
    """
    Run the fine-tuned YOLO model on every page image across all sources.
    Returns a manifest dict mapping source/page to detected bounding boxes.
    """
    from ultralytics import YOLO
    import torch

    # Re-apply after ultralytics import (it can reset this)
    Image.MAX_IMAGE_PIXELS = None

    if not BEST_MODEL_PATH.exists():
        print(f"  ERROR: Best model not found at {BEST_MODEL_PATH}")
        print("  Please run steps 1.3–2.3 first to train the YOLO model.")
        sys.exit(1)

    print(f"  Loading model: {BEST_MODEL_PATH}")
    model = YOLO(str(BEST_MODEL_PATH))

    manifest = {
        "model": str(BEST_MODEL_PATH),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "img_size": IMG_SIZE,
        "sources": {},
    }

    total_pages = 0
    total_detections = 0
    pages_with_no_detection = []

    for source in SOURCES:
        source_dir = PAGE_IMAGES_DIR / source
        if not source_dir.exists():
            print(f"  WARNING: Source directory not found: {source_dir}")
            continue

        page_files = sorted(source_dir.glob("*.png"))
        if not page_files:
            print(f"  WARNING: No PNG images found in {source_dir}")
            continue

        print(f"\n  Processing {source}: {len(page_files)} pages...")
        source_results = {}

        # Process in small batches — 4GB VRAM safe
        BATCH_SIZE = 4
        for batch_start in range(0, len(page_files), BATCH_SIZE):
            batch_files = page_files[batch_start : batch_start + BATCH_SIZE]
            batch_paths = [str(p) for p in batch_files]

            try:
                results = model.predict(
                    source=batch_paths,
                    imgsz=IMG_SIZE,
                    device=DEVICE,
                    conf=CONFIDENCE_THRESHOLD,
                    save=False,
                    verbose=False,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # OOM — fall back to processing one image at a time on CPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"\n    WARNING: OOM on batch, processing one-by-one on CPU...")
                results = []
                for bp in batch_paths:
                    r = model.predict(
                        source=bp,
                        imgsz=IMG_SIZE,
                        device="cpu",
                        conf=CONFIDENCE_THRESHOLD,
                        save=False,
                        verbose=False,
                    )
                    results.extend(r)

            for img_path, result in zip(batch_files, results):
                page_name = img_path.stem  # e.g. "source1_page001"
                boxes = result.boxes

                # Get image dimensions for absolute coordinates
                img_h, img_w = result.orig_shape  # (height, width)

                detections = []
                for i in range(len(boxes)):
                    # xyxy = absolute pixel coordinates [x1, y1, x2, y2]
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
                    conf = float(boxes.conf[i].cpu())
                    cls = int(boxes.cls[i].cpu())

                    # Also store normalized coordinates (YOLO format)
                    cx, cy, w, h = boxes.xywhn[i].cpu().tolist()

                    detections.append({
                        "class": cls,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [round(v, 1) for v in [x1, y1, x2, y2]],
                        "bbox_xywhn": [round(v, 6) for v in [cx, cy, w, h]],
                        "image_width": img_w,
                        "image_height": img_h,
                    })

                source_results[page_name] = {
                    "image_path": str(img_path),
                    "num_detections": len(detections),
                    "detections": detections,
                }

                total_pages += 1
                total_detections += len(detections)

                if len(detections) == 0:
                    pages_with_no_detection.append(f"{source}/{page_name}")

            # Free GPU memory between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Progress indicator
            done = min(batch_start + BATCH_SIZE, len(page_files))
            print(f"    {done}/{len(page_files)} pages processed", end="\r")

        print(f"    {len(page_files)}/{len(page_files)} pages processed — "
              f"{sum(r['num_detections'] for r in source_results.values())} detections")

        manifest["sources"][source] = source_results

    # Summary stats
    manifest["summary"] = {
        "total_pages_processed": total_pages,
        "total_detections": total_detections,
        "pages_with_no_detection": len(pages_with_no_detection),
        "no_detection_pages": pages_with_no_detection,
    }

    return manifest


def save_manifest(manifest: dict):
    """Save the detection manifest to JSON."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved to: {MANIFEST_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2.5 — Crop Main Text Regions
# ──────────────────────────────────────────────────────────────────────────────
def crop_main_text_regions(manifest: dict):
    """
    Crop detected main-text regions from each page image.

    Naming convention:
      - 1 detection:  sourceX_pageYYY_full.png
      - 2 detections: sourceX_pageYYY_left.png, sourceX_pageYYY_right.png
                      (sorted by x-centre, leftmost first)
      - 3+ detections: sourceX_pageYYY_col1.png, sourceX_pageYYY_col2.png, ...
                       (sorted left to right by x-centre)
    """
    # Clean and recreate output directory
    if CROPS_DIR.exists():
        import shutil
        shutil.rmtree(CROPS_DIR)

    total_crops = 0
    skipped_pages = 0
    crop_log = {}  # Track crop info for each source

    for source in SOURCES:
        if source not in manifest["sources"]:
            continue

        source_crops_dir = CROPS_DIR / source
        source_crops_dir.mkdir(parents=True, exist_ok=True)
        source_data = manifest["sources"][source]
        source_crop_count = 0

        for page_name, page_data in sorted(source_data.items()):
            detections = page_data["detections"]
            img_path = Path(page_data["image_path"])

            if len(detections) == 0:
                skipped_pages += 1
                continue

            # Open the original full-resolution image
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Sort detections by x-centre (left to right)
            detections_sorted = sorted(
                detections,
                key=lambda d: d["bbox_xyxy"][0]  # sort by x1 (left edge)
            )

            num_boxes = len(detections_sorted)

            for idx, det in enumerate(detections_sorted):
                x1, y1, x2, y2 = det["bbox_xyxy"]

                # Apply padding (clamp to image bounds)
                x1 = max(0, x1 - PAD_PX)
                y1 = max(0, y1 - PAD_PX)
                x2 = min(img_w, x2 + PAD_PX)
                y2 = min(img_h, y2 + PAD_PX)

                # Determine crop suffix
                if num_boxes == 1:
                    suffix = "full"
                elif num_boxes == 2:
                    suffix = "left" if idx == 0 else "right"
                else:
                    suffix = f"col{idx + 1}"

                crop_filename = f"{page_name}_{suffix}.png"
                crop_path = source_crops_dir / crop_filename

                # Crop and save
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                crop.save(crop_path, "PNG")

                total_crops += 1
                source_crop_count += 1
                del crop

            img.close()
            del img
            gc.collect()

        print(f"  {source}: {source_crop_count} crops saved to {source_crops_dir}")
        crop_log[source] = source_crop_count

    # Save a crop manifest for downstream use
    crop_manifest = {
        "crops_dir": str(CROPS_DIR),
        "total_crops": total_crops,
        "skipped_pages_no_detection": skipped_pages,
        "pad_px": PAD_PX,
        "crops_per_source": crop_log,
        "naming_convention": {
            "1_box": "<page_name>_full.png",
            "2_boxes": "<page_name>_left.png, <page_name>_right.png (sorted by x-pos)",
            "3+_boxes": "<page_name>_col1.png, <page_name>_col2.png, ... (sorted by x-pos)",
        },
        "crops": {},
    }

    # Build detailed crop list for traceability
    for source in SOURCES:
        if source not in manifest["sources"]:
            continue

        source_data = manifest["sources"][source]
        source_crops = {}

        for page_name, page_data in sorted(source_data.items()):
            detections = page_data["detections"]
            if len(detections) == 0:
                continue

            detections_sorted = sorted(
                detections,
                key=lambda d: d["bbox_xyxy"][0]
            )
            num_boxes = len(detections_sorted)

            page_crops = []
            for idx, det in enumerate(detections_sorted):
                if num_boxes == 1:
                    suffix = "full"
                elif num_boxes == 2:
                    suffix = "left" if idx == 0 else "right"
                else:
                    suffix = f"col{idx + 1}"

                crop_filename = f"{page_name}_{suffix}.png"
                page_crops.append({
                    "crop_file": crop_filename,
                    "crop_path": str(CROPS_DIR / source / crop_filename),
                    "source_image": page_data["image_path"],
                    "bbox_xyxy": det["bbox_xyxy"],
                    "confidence": det["confidence"],
                    "column_type": suffix,
                })

            source_crops[page_name] = page_crops

        crop_manifest["crops"][source] = source_crops

    crop_manifest_path = BASE_DIR / "Data" / "crop_manifest.json"
    with open(crop_manifest_path, "w") as f:
        json.dump(crop_manifest, f, indent=2)

    print(f"\n  Crop manifest saved to: {crop_manifest_path}")

    return crop_manifest


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RenAIssance OCR Pipeline — Steps 2.4 & 2.5")
    print("=" * 60)

    # ── Step 2.4: Run YOLO on all pages ──────────────────────────────────────
    print("\n[Step 2.4] Running YOLO on all page images...")
    start = time.time()
    manifest = run_yolo_on_all_pages()
    elapsed = time.time() - start

    save_manifest(manifest)

    summary = manifest["summary"]
    print(f"\n  Summary:")
    print(f"    Pages processed:       {summary['total_pages_processed']}")
    print(f"    Total detections:      {summary['total_detections']}")
    print(f"    Pages w/ no detection: {summary['pages_with_no_detection']}")
    print(f"    Inference time:        {elapsed:.1f}s")

    if summary["pages_with_no_detection"] > 0:
        print(f"\n  WARNING: {summary['pages_with_no_detection']} pages had no detections:")
        for p in summary["no_detection_pages"][:20]:
            print(f"    - {p}")
        if len(summary["no_detection_pages"]) > 20:
            print(f"    ... and {len(summary['no_detection_pages']) - 20} more")

    # ── Step 2.5: Crop main text regions ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 2.5] Cropping main text regions...")
    print("=" * 60)

    crop_manifest = crop_main_text_regions(manifest)

    print(f"\n  Cropping complete!")
    print(f"    Total crops created: {crop_manifest['total_crops']}")
    print(f"    Skipped (no detect): {crop_manifest['skipped_pages_no_detection']}")
    print(f"    Crops per source:")
    for src, count in crop_manifest["crops_per_source"].items():
        print(f"      {src}: {count}")
    print(f"\n    Crops saved to: {CROPS_DIR}")

    print("\n" + "=" * 60)
    print("  Steps 2.4 & 2.5 complete!")
    print("  Next: Phase 3 — Line Segmentation")
    print("=" * 60)


if __name__ == "__main__":
    main()
