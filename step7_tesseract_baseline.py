"""
Phase 7 — Tesseract Baseline
=============================

Implements FinalPlan.md Phase 7:
  - Run out-of-the-box Tesseract on validation line images
  - Use Spanish language (`spa`) and PSM 6
  - Compute CER, WER, chrF

Expected input from Phase 5/6:
  Data/ocr_dataset/val_manifest.csv

Outputs:
  Data/tesseract_baseline/predictions.csv
  Data/tesseract_baseline/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
DEFAULT_VAL_MANIFEST = ROOT / "Data" / "ocr_dataset" / "val_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "tesseract_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tesseract baseline on validation set.")
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lang", type=str, default="spa", help="Tesseract language code.")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode.")
    parser.add_argument("--oem", type=int, default=3, help="Tesseract OCR Engine Mode.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows for quick test.")
    return parser.parse_args()


def check_tesseract_available(lang: str) -> None:
    binary = subprocess.run(["bash", "-lc", "command -v tesseract"], capture_output=True, text=True)
    if binary.returncode != 0:
        raise RuntimeError(
            "Tesseract is not installed or not in PATH. "
            "Install `tesseract-ocr` and language pack `tesseract-ocr-spa` (for lang=spa)."
        )

    langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    if langs.returncode != 0:
        raise RuntimeError("Failed to query Tesseract languages via `tesseract --list-langs`.")

    installed = {line.strip() for line in langs.stdout.splitlines() if line.strip() and not line.startswith("List of available")}
    if lang not in installed:
        raise RuntimeError(
            f"Tesseract language '{lang}' not found. Installed languages: {sorted(installed)}"
        )


def load_val_manifest(path: Path, limit: int = 0) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {path}")

    required = {
        "split",
        "source",
        "source_id",
        "page",
        "line",
        "image_path",
        "ground_truth_text",
    }
    rows: list[dict] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Validation manifest has no header.")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Validation manifest missing columns: {sorted(missing)}")

        for row in reader:
            if row.get("split", "").strip() != "val":
                continue
            image_path = Path((row.get("image_path") or "").strip())
            gt = (row.get("ground_truth_text") or "").strip()
            if not image_path.exists() or not gt:
                continue

            rows.append(
                {
                    "source": row["source"].strip(),
                    "source_id": row["source_id"].strip(),
                    "page": row["page"].strip(),
                    "line": row["line"].strip(),
                    "image_path": str(image_path),
                    "ground_truth_text": gt,
                }
            )

            if limit > 0 and len(rows) >= limit:
                break

    if not rows:
        raise ValueError("No usable validation rows found in manifest.")
    return rows


def run_tesseract(image_path: str, lang: str, psm: int, oem: int) -> str:
    cmd = [
        "tesseract",
        image_path,
        "stdout",
        "-l",
        lang,
        "--psm",
        str(psm),
        "--oem",
        str(oem),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or "unknown tesseract error"
        raise RuntimeError(f"Tesseract failed for {image_path}: {err}")

    text = proc.stdout.replace("\x0c", " ")
    text = " ".join(text.split())
    return text


def levenshtein_distance(seq_a: list[str], seq_b: list[str]) -> int:
    n = len(seq_a)
    m = len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    return prev[m]


def cer_stats(ref: str, hyp: str) -> tuple[int, int]:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    edits = levenshtein_distance(ref_chars, hyp_chars)
    return edits, max(1, len(ref_chars))


def wer_stats(ref: str, hyp: str) -> tuple[int, int]:
    ref_words = ref.split()
    hyp_words = hyp.split()
    edits = levenshtein_distance(ref_words, hyp_words)
    return edits, max(1, len(ref_words))


def char_ngrams(text: str, n: int) -> Counter:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf_corpus(refs: list[str], hyps: list[str], n_max: int = 6, beta: float = 2.0) -> float:
    beta2 = beta * beta
    f_scores: list[float] = []

    for n in range(1, n_max + 1):
        match_total = 0
        ref_total = 0
        hyp_total = 0

        for ref, hyp in zip(refs, hyps):
            ref_ng = char_ngrams(ref, n)
            hyp_ng = char_ngrams(hyp, n)
            if not ref_ng and not hyp_ng:
                continue

            ref_total += sum(ref_ng.values())
            hyp_total += sum(hyp_ng.values())
            overlap = ref_ng & hyp_ng
            match_total += sum(overlap.values())

        precision = match_total / hyp_total if hyp_total > 0 else 0.0
        recall = match_total / ref_total if ref_total > 0 else 0.0

        if precision == 0.0 and recall == 0.0:
            f_n = 0.0
        else:
            f_n = (1 + beta2) * precision * recall / (beta2 * precision + recall)

        f_scores.append(f_n)

    if not f_scores:
        return 0.0
    return float(sum(f_scores) / len(f_scores))


def aggregate_metrics(rows: list[dict]) -> dict:
    refs = [r["ground_truth_text"] for r in rows]
    hyps = [r["prediction"] for r in rows]

    cer_edits = 0
    cer_chars = 0
    wer_edits = 0
    wer_words = 0

    for row in rows:
        c_edits, c_chars = cer_stats(row["ground_truth_text"], row["prediction"])
        w_edits, w_words = wer_stats(row["ground_truth_text"], row["prediction"])
        cer_edits += c_edits
        cer_chars += c_chars
        wer_edits += w_edits
        wer_words += w_words

    cer = cer_edits / cer_chars if cer_chars > 0 else 0.0
    wer = wer_edits / wer_words if wer_words > 0 else 0.0
    chrf = chrf_corpus(refs, hyps)

    return {
        "num_samples": len(rows),
        "CER": cer,
        "WER": wer,
        "chrF": chrf,
        "cer_edits": cer_edits,
        "cer_chars": cer_chars,
        "wer_edits": wer_edits,
        "wer_words": wer_words,
    }


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "source_id",
        "page",
        "line",
        "image_path",
        "ground_truth_text",
        "prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    check_tesseract_available(args.lang)

    val_rows = load_val_manifest(args.val_manifest, limit=args.limit)
    log.info("=" * 68)
    log.info("PHASE 7 — Tesseract Baseline")
    log.info("=" * 68)
    log.info(f"Validation manifest: {args.val_manifest}")
    log.info(f"Samples: {len(val_rows)}")
    log.info(f"Lang: {args.lang} | PSM: {args.psm} | OEM: {args.oem}")

    prediction_rows: list[dict] = []
    for idx, row in enumerate(val_rows, 1):
        pred = run_tesseract(row["image_path"], args.lang, args.psm, args.oem)
        pred_row = dict(row)
        pred_row["prediction"] = pred
        prediction_rows.append(pred_row)

        if idx % 50 == 0 or idx == len(val_rows):
            log.info(f"Processed {idx}/{len(val_rows)}")

    overall = aggregate_metrics(prediction_rows)

    by_source = defaultdict(list)
    for row in prediction_rows:
        by_source[row["source_id"]].append(row)

    per_source = {sid: aggregate_metrics(rows) for sid, rows in sorted(by_source.items())}

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "predictions.csv"
    metrics_json = output_dir / "metrics.json"

    write_predictions(predictions_csv, prediction_rows)

    metrics_payload = {
        "phase": "Phase 7 - Tesseract baseline",
        "tesseract": {
            "lang": args.lang,
            "psm": args.psm,
            "oem": args.oem,
        },
        "overall": overall,
        "per_source": per_source,
        "inputs": {
            "val_manifest": str(args.val_manifest),
            "num_samples": len(val_rows),
        },
        "outputs": {
            "predictions_csv": str(predictions_csv),
            "metrics_json": str(metrics_json),
        },
    }

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 68)
    log.info("BASELINE COMPLETE")
    log.info("=" * 68)
    log.info(f"CER : {overall['CER']:.4f}")
    log.info(f"WER : {overall['WER']:.4f}")
    log.info(f"chrF: {overall['chrF']:.4f}")
    log.info(f"Predictions: {predictions_csv}")
    log.info(f"Metrics:     {metrics_json}")


if __name__ == "__main__":
    main()
