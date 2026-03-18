"""Step 6: Source-level train/val split + augmentation of training images."""

import csv, re, shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Data" / "ocr_dataset"
AUG_PER_IMAGE = 5
SEED = 42
N_VAL_SOURCES = 2


def slugify(text):
    return re.sub(r"[^a-z0-9_\-]", "", re.sub(r"\s+", "_", text.strip().lower())) or "unknown"


def read_aligned_rows(csv_path):
    """Read aligned rows from line_alignment.csv (skip unaligned/empty)."""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            img = (row.get("image_path") or "").strip()
            gt = (row.get("ground_truth_text") or "").strip()
            status = (row.get("status") or "aligned").strip().lower()
            if img and gt and status == "aligned" and Path(img).exists():
                rows.append(row)
    return rows


# ── Augmentation functions ───────────────────────────────────────────────────

def apply_rotation(img, rng):
    h, w = img.shape[:2]
    angle = float(rng.uniform(-2.0, 2.0))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def apply_blur(img, rng):
    k = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_brightness(img, rng):
    return np.clip(img.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)


def apply_noise(img, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)


def apply_elastic(img, rng):
    h, w = img.shape[:2]
    alpha, sigma = rng.uniform(1, 3), rng.uniform(6, 10)
    dx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def augment_image(img, rng):
    out = img.copy()
    if rng.random() < 0.70: out = apply_rotation(out, rng)
    if rng.random() < 0.50: out = apply_brightness(out, rng)
    if rng.random() < 0.45: out = apply_blur(out, rng)
    if rng.random() < 0.50: out = apply_noise(out, rng)
    if rng.random() < 0.30: out = apply_elastic(out, rng)
    return out


# ── Main pipeline ────────────────────────────────────────────────────────────

def make_row(split, row, dst, is_aug=False, aug_id=0, orig=""):
    return {"split": split, "source": row["source"], "source_id": row["source_id"],
            "page": row["page"], "line": row["line"], "image_path": str(Path(dst).resolve()),
            "ground_truth_text": row["ground_truth_text"],
            "is_augmented": "1" if is_aug else "0", "aug_id": str(aug_id), "orig_image_path": orig}


def main():
    # Find alignment CSV
    csv_path = None
    for p in [ROOT / "line_alignment.csv", ROOT / "Data" / "line_alignment.csv"]:
        if p.exists():
            csv_path = p; break
    if csv_path is None:
        raise FileNotFoundError("line_alignment.csv not found")

    rows = read_aligned_rows(csv_path)
    if not rows:
        raise ValueError("No valid aligned rows found")

    # Pick val sources
    rng = np.random.default_rng(SEED)
    all_ids = sorted(set(r["source_id"] for r in rows))
    val_ids = set(rng.choice(all_ids, size=N_VAL_SOURCES, replace=False))
    print(f"Train sources: {sorted(set(all_ids) - val_ids)}")
    print(f"Val sources:   {sorted(val_ids)}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    train_rows, val_rows = [], []

    for row in rows:
        src = Path(row["image_path"])
        slug = slugify(row["source"])

        if row["source_id"] in val_ids:
            dst = OUTPUT_DIR / "val" / "images" / slug / f"{src.stem}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            val_rows.append(make_row("val", row, dst, orig=str(src)))
        else:
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            for i in range(1, AUG_PER_IMAGE + 1):
                aug = augment_image(img, np.random.default_rng(int(rng.integers(0, 2**31))))
                dst = OUTPUT_DIR / "train" / "images" / slug / f"{src.stem}__aug_{i:02d}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), aug)
                train_rows.append(make_row("train", row, dst, is_aug=True, aug_id=i, orig=str(src)))

    # Write manifests
    fields = ["split", "source", "source_id", "page", "line", "image_path",
              "ground_truth_text", "is_augmented", "aug_id", "orig_image_path"]
    for name, data in [("train_manifest.csv", train_rows), ("val_manifest.csv", val_rows),
                       ("full_manifest.csv", train_rows + val_rows)]:
        p = OUTPUT_DIR / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    print(f"\nDone. Train: {len(train_rows)}, Val: {len(val_rows)} → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
