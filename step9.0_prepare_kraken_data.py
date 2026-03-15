#!/usr/bin/env python3
"""
prepare_kraken_data.py
Convert train/val manifest CSVs into Kraken ground-truth pair format.

Each image is copied into kraken_gt/{train,val}/ with a corresponding .gt.txt
file containing the ground truth text.
"""

import shutil
from pathlib import Path

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "Data" / "ocr_dataset" / "train_manifest.csv"
VAL_CSV = ROOT / "Data" / "ocr_dataset" / "val_manifest.csv"
OUT_DIR = ROOT / "kraken_gt"
MISSING_LOG = ROOT / "missing_images.log"


def build_filename(row: pd.Series) -> str:
    """Return a unique filename stem from manifest columns.

    For augmented images (is_augmented == 1) the aug_id is appended so that
    multiple augmentations of the same line don't collide.
    """
    base = f"{row['source_id']}_p{row['page']}_l{int(row['line']):03d}"
    if int(row.get("is_augmented", 0)) == 1 and int(row.get("aug_id", 0)) > 0:
        base += f"_aug{int(row['aug_id']):02d}"
    return base


def process_split(csv_path: Path, split_name: str) -> tuple[int, list[str]]:
    """Read a manifest CSV, copy images & write .gt.txt files.

    Returns (count_written, list_of_missing_paths).
    """
    df = pd.read_csv(csv_path, dtype=str)

    # If a 'status' column exists, keep only aligned rows; otherwise keep all
    if "status" in df.columns:
        df = df[df["status"] == "aligned"]

    out = OUT_DIR / split_name
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    missing: list[str] = []

    for _, row in df.iterrows():
        src = Path(row["image_path"])
        if not src.exists():
            missing.append(str(src))
            continue

        stem = build_filename(row)
        dst_img = out / f"{stem}.png"
        dst_txt = out / f"{stem}.gt.txt"

        shutil.copy2(src, dst_img)

        gt = str(row["ground_truth_text"]).replace("\ufeff", "")  # strip BOM
        dst_txt.write_text(gt, encoding="utf-8")

        written += 1

    return written, missing


def main() -> None:
    print("Preparing Kraken ground-truth pairs …")

    all_missing: list[str] = []

    train_n, train_miss = process_split(TRAIN_CSV, "train")
    all_missing.extend(train_miss)

    val_n, val_miss = process_split(VAL_CSV, "val")
    all_missing.extend(val_miss)

    # log missing images
    if all_missing:
        MISSING_LOG.write_text("\n".join(all_missing) + "\n", encoding="utf-8")
        print(f"⚠  {len(all_missing)} missing images logged to {MISSING_LOG}")
    else:
        print("✓  No missing images")

    print(f"\nSummary")
    print(f"  Train pairs written : {train_n}")
    print(f"  Val   pairs written : {val_n}")


if __name__ == "__main__":
    main()
