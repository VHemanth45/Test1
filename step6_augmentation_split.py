"""
Phase 5 + 6 — Augmentation and Source-level Split
==================================================

Implements:
  - Phase 6.1: split aligned pairs by source (4 train sources, 2 val sources)
  - Phase 5.1: augment training images only (5-8 variants per image)

Input:
  line_alignment.csv (or Data/line_alignment.csv)

Output (default under Data/ocr_dataset):
  - train/images/<source_slug>/*.png
  - val/images/<source_slug>/*.png
  - train_manifest.csv
  - val_manifest.csv
  - full_manifest.csv
  - split_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_CANDIDATES = [
    ROOT / "line_alignment.csv",
    ROOT / "Data" / "line_alignment.csv",
]
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "ocr_dataset"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text or "unknown_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split aligned OCR lines by source and augment train set.",
    )
    parser.add_argument(
        "--alignment-csv",
        type=Path,
        default=None,
        help="Path to alignment CSV (auto-detected if omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--val-sources",
        type=str,
        default="",
        help="Comma-separated source_id list for validation (e.g. source2,source6).",
    )
    parser.add_argument(
        "--n-val-sources",
        type=int,
        default=2,
        help="Number of validation sources when --val-sources is not provided.",
    )
    parser.add_argument(
        "--aug-per-image",
        type=int,
        default=5,
        help="Number of augmented variants per training image (recommended 5-8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split/augmentation.",
    )
    parser.add_argument(
        "--copy-train-original",
        action="store_true",
        help="Also copy original (non-augmented) train images into train split.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing output directory before writing new files.",
    )
    return parser.parse_args()


def resolve_alignment_csv(path_arg: Path | None) -> Path:
    if path_arg is not None:
        if not path_arg.exists():
            raise FileNotFoundError(f"Alignment CSV not found: {path_arg}")
        return path_arg

    for candidate in DEFAULT_CSV_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find alignment CSV. Tried: "
        + ", ".join(str(p) for p in DEFAULT_CSV_CANDIDATES)
    )


def read_alignment_rows(alignment_csv: Path) -> list[dict]:
    required = {"source", "source_id", "page", "line", "image_path", "ground_truth_text"}
    rows: list[dict] = []

    with alignment_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Alignment CSV has no header row.")

        reader.fieldnames = [(name or "").strip() for name in reader.fieldnames]

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Alignment CSV missing columns: {sorted(missing)}")

        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            gt = (row.get("ground_truth_text") or "").strip()
            status = (row.get("status") or "aligned").strip().lower()

            if not image_path or not gt:
                continue
            if status not in {"aligned", ""}:
                continue

            p = Path(image_path)
            if not p.exists():
                log.warning(f"Missing image (skipped): {p}")
                continue

            rows.append(
                {
                    "source": row["source"].strip(),
                    "source_id": row["source_id"].strip(),
                    "page": str(row["page"]).strip(),
                    "line": str(row["line"]).strip(),
                    "image_path": str(p),
                    "ground_truth_text": gt,
                }
            )

    if not rows:
        raise ValueError("No valid aligned rows found in CSV.")
    return rows


def pick_val_sources(all_source_ids: list[str], args: argparse.Namespace) -> set[str]:
    source_ids = sorted(set(all_source_ids))

    if args.val_sources.strip():
        requested = {s.strip() for s in args.val_sources.split(",") if s.strip()}
        unknown = requested - set(source_ids)
        if unknown:
            raise ValueError(f"Unknown val source_id(s): {sorted(unknown)}")
        if len(requested) == 0:
            raise ValueError("--val-sources was provided but empty after parsing.")
        return requested

    if args.n_val_sources <= 0:
        raise ValueError("--n-val-sources must be >= 1")
    if args.n_val_sources >= len(source_ids):
        raise ValueError("--n-val-sources must be smaller than total number of sources")

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(source_ids, size=args.n_val_sources, replace=False)
    return set(str(s) for s in selected)


def apply_rotation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-2.0, 2.0))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def apply_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(img, (k, k), sigmaX=0.0)


def apply_brightness(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.uniform(0.9, 1.1))
    beta = float(rng.uniform(-12, 12))
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(2.0, 8.0))
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_elastic(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    alpha = float(rng.uniform(1.0, 3.0))
    sigma = float(rng.uniform(6.0, 10.0))

    dx = rng.normal(0, 1, (h, w)).astype(np.float32)
    dy = rng.normal(0, 1, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def augment_image(img_gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img_gray.copy()

    if rng.random() < 0.70:
        out = apply_rotation(out, rng)
    if rng.random() < 0.50:
        out = apply_brightness(out, rng)
    if rng.random() < 0.45:
        out = apply_blur(out, rng)
    if rng.random() < 0.50:
        out = apply_noise(out, rng)
    if rng.random() < 0.30:
        out = apply_elastic(out, rng)

    return out


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_augmented(img: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {dst}")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "source",
        "source_id",
        "page",
        "line",
        "image_path",
        "ground_truth_text",
        "is_augmented",
        "aug_id",
        "orig_image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    alignment_csv = resolve_alignment_csv(args.alignment_csv)
    output_dir = args.output_dir

    if args.aug_per_image < 1:
        raise ValueError("--aug-per-image must be >= 1")

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    rows = read_alignment_rows(alignment_csv)

    source_ids = [r["source_id"] for r in rows]
    val_source_ids = pick_val_sources(source_ids, args)
    train_source_ids = set(source_ids) - val_source_ids

    if not train_source_ids:
        raise ValueError("Train source set is empty. Adjust val source selection.")

    log.info("=" * 68)
    log.info("PHASE 6 — Source-level split")
    log.info("=" * 68)
    log.info(f"Alignment CSV: {alignment_csv}")
    log.info(f"Total aligned rows: {len(rows)}")
    log.info(f"Train sources ({len(train_source_ids)}): {sorted(train_source_ids)}")
    log.info(f"Val sources   ({len(val_source_ids)}): {sorted(val_source_ids)}")

    train_img_dir = output_dir / "train" / "images"
    val_img_dir = output_dir / "val" / "images"

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for row in rows:
        split = "val" if row["source_id"] in val_source_ids else "train"
        src = Path(row["image_path"])
        src_slug = slugify(row["source"])
        base_name = f"{src.stem}.png"

        if split == "val":
            dst = val_img_dir / src_slug / base_name
            copy_image(src, dst)
            val_rows.append(
                {
                    "split": "val",
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "page": row["page"],
                    "line": row["line"],
                    "image_path": str(dst.resolve()),
                    "ground_truth_text": row["ground_truth_text"],
                    "is_augmented": "0",
                    "aug_id": "0",
                    "orig_image_path": str(src),
                }
            )
            continue

        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            log.warning(f"Unreadable image (skipped): {src}")
            continue

        if args.copy_train_original:
            dst0 = train_img_dir / src_slug / f"{src.stem}__orig.png"
            copy_image(src, dst0)
            train_rows.append(
                {
                    "split": "train",
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "page": row["page"],
                    "line": row["line"],
                    "image_path": str(dst0.resolve()),
                    "ground_truth_text": row["ground_truth_text"],
                    "is_augmented": "0",
                    "aug_id": "0",
                    "orig_image_path": str(src),
                }
            )

        for aug_idx in range(1, args.aug_per_image + 1):
            local_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            local_rng = np.random.default_rng(local_seed)
            aug = augment_image(img, local_rng)

            dst = train_img_dir / src_slug / f"{src.stem}__aug_{aug_idx:02d}.png"
            save_augmented(aug, dst)

            train_rows.append(
                {
                    "split": "train",
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "page": row["page"],
                    "line": row["line"],
                    "image_path": str(dst.resolve()),
                    "ground_truth_text": row["ground_truth_text"],
                    "is_augmented": "1",
                    "aug_id": str(aug_idx),
                    "orig_image_path": str(src),
                }
            )

    full_rows = train_rows + val_rows

    write_manifest(output_dir / "train_manifest.csv", train_rows)
    write_manifest(output_dir / "val_manifest.csv", val_rows)
    write_manifest(output_dir / "full_manifest.csv", full_rows)

    stats = {
        "alignment_csv": str(alignment_csv),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "aug_per_image": args.aug_per_image,
        "copy_train_original": bool(args.copy_train_original),
        "n_val_sources": len(val_source_ids),
        "train_source_ids": sorted(train_source_ids),
        "val_source_ids": sorted(val_source_ids),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "full_rows": len(full_rows),
        "unique_sources": sorted(set(source_ids)),
    }

    stats_path = output_dir / "split_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 68)
    log.info("PHASE 5 + 6 COMPLETE")
    log.info("=" * 68)
    log.info(f"Output dir:     {output_dir}")
    log.info(f"Train rows:     {len(train_rows)}")
    log.info(f"Val rows:       {len(val_rows)}")
    log.info(f"Full rows:      {len(full_rows)}")
    log.info(f"Stats JSON:     {stats_path}")
    log.info(f"Train manifest: {output_dir / 'train_manifest.csv'}")
    log.info(f"Val manifest:   {output_dir / 'val_manifest.csv'}")


if __name__ == "__main__":
    main()
