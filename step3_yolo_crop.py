"""Step 3: Run YOLO on all pages and crop detected main-text regions."""

import json, shutil
from pathlib import Path
from PIL import Image
import PIL.Image

Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.Image._decompression_bomb_check = lambda size: None

BASE_DIR = Path(__file__).resolve().parent
PAGE_IMAGES_DIR = BASE_DIR / "Data" / "page_images"
BEST_MODEL = BASE_DIR / "runs" / "yolo_maintext" / "weights" / "best.pt"
CROPS_DIR = BASE_DIR / "Data" / "cropped_main_text"
MANIFEST_PATH = BASE_DIR / "Data" / "yolo_detections_manifest.json"

IMGSZ, CONF, PAD = 480, 0.25, 5
SOURCES = [f"source{i}" for i in range(1, 7)]


def run_yolo_inference():
    """Run YOLO on all page images and return detections dict."""
    from ultralytics import YOLO
    import torch
    Image.MAX_IMAGE_PIXELS = None

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(BEST_MODEL))
    detections = {}  # {source: {page_stem: [(x1,y1,x2,y2,conf), ...]}}

    for src in SOURCES:
        src_dir = PAGE_IMAGES_DIR / src
        if not src_dir.exists():
            continue
        pages = sorted(src_dir.glob("*.png"))
        if not pages:
            continue

        src_dets = {}
        for p in pages:
            results = model.predict(str(p), imgsz=IMGSZ, device=device,
                                    conf=CONF, save=False, verbose=False)
            boxes = results[0].boxes
            src_dets[p.stem] = {
                "path": str(p),
                "boxes": [[*b.xyxy[0].cpu().tolist(), float(b.conf[0])]
                          for b in [boxes[i:i+1] for i in range(len(boxes))]],
            }
        detections[src] = src_dets
        total = sum(len(d["boxes"]) for d in src_dets.values())
        print(f"  {src}: {len(pages)} pages, {total} detections")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(detections, indent=2))
    print(f"  Manifest: {MANIFEST_PATH}")
    return detections


def crop_regions(detections):
    """Crop detected regions from page images."""
    if CROPS_DIR.exists():
        shutil.rmtree(CROPS_DIR)

    total = 0
    for src, pages in detections.items():
        out_dir = CROPS_DIR / src
        out_dir.mkdir(parents=True, exist_ok=True)

        for stem, data in sorted(pages.items()):
            boxes = data["boxes"]
            if not boxes:
                continue

            img = Image.open(data["path"])
            w, h = img.size
            # Sort by x1 (left to right)
            boxes_sorted = sorted(boxes, key=lambda b: b[0])
            n = len(boxes_sorted)

            for i, box in enumerate(boxes_sorted):
                x1, y1, x2, y2 = box[:4]
                x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
                x2, y2 = min(w, x2 + PAD), min(h, y2 + PAD)

                if n == 1:
                    suffix = "full"
                elif n == 2:
                    suffix = "left" if i == 0 else "right"
                else:
                    suffix = f"col{i+1}"

                img.crop((int(x1), int(y1), int(x2), int(y2))).save(
                    out_dir / f"{stem}_{suffix}.png")
                total += 1
            img.close()

        print(f"  {src}: cropped to {out_dir}")

    print(f"  Total crops: {total}")


def main():
    print("[Step 2.4] Running YOLO inference...")
    detections = run_yolo_inference()

    print("[Step 2.5] Cropping main-text regions...")
    crop_regions(detections)

    print(f"\nDone! Crops saved in {CROPS_DIR}")


if __name__ == "__main__":
    main()
