"""
Phase 9.5 — Evaluate TrOCR raw vs LLM-corrected predictions.

Compares metrics against ground truth:
  - CER
  - WER
  - chrF

Usage:
  python step9_5_evaluate_post_correction.py \
      --raw-csv runs/trocr_finetune/predictions.csv \
      --corrected-csv runs/trocr_finetune/predictions_corrected.csv \
      --output-json runs/trocr_finetune/post_correction_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OCR metrics before/after LLM post-correction.")
    p.add_argument("--raw-csv", type=Path, required=True)
    p.add_argument("--corrected-csv", type=Path, required=True)
    p.add_argument("--ref-column", default="ground_truth_text")
    p.add_argument("--raw-column", default="prediction")
    p.add_argument("--corrected-column", default="llm_corrected_text")
    p.add_argument("--source-column", default="source")
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def levenshtein_distance(seq_a: list, seq_b: list) -> int:
    n, m = len(seq_a), len(seq_b)
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
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def cer_value(ref: str, hyp: str) -> tuple[int, int]:
    edits = levenshtein_distance(list(ref), list(hyp))
    return edits, max(1, len(ref))


def wer_value(ref: str, hyp: str) -> tuple[int, int]:
    edits = levenshtein_distance(ref.split(), hyp.split())
    return edits, max(1, len(ref.split()))


def char_ngrams(text: str, n: int) -> Counter:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf_corpus(refs: list[str], hyps: list[str], n_max: int = 6, beta: float = 2.0) -> float:
    beta2 = beta * beta
    f_scores: list[float] = []
    for n in range(1, n_max + 1):
        match_total = ref_total = hyp_total = 0
        for ref, hyp in zip(refs, hyps):
            ref_ng = char_ngrams(ref, n)
            hyp_ng = char_ngrams(hyp, n)
            if not ref_ng and not hyp_ng:
                continue
            ref_total += sum(ref_ng.values())
            hyp_total += sum(hyp_ng.values())
            match_total += sum((ref_ng & hyp_ng).values())
        p = match_total / hyp_total if hyp_total else 0.0
        r = match_total / ref_total if ref_total else 0.0
        f_n = (1 + beta2) * p * r / (beta2 * p + r) if (p + r) > 0 else 0.0
        f_scores.append(f_n)
    return float(sum(f_scores) / len(f_scores)) if f_scores else 0.0


def aggregate_metrics(refs: list[str], hyps: list[str]) -> dict:
    cer_edits = cer_chars = wer_edits = wer_words = 0
    for ref, hyp in zip(refs, hyps):
        ce, cc = cer_value(ref, hyp)
        we, ww = wer_value(ref, hyp)
        cer_edits += ce
        cer_chars += cc
        wer_edits += we
        wer_words += ww

    return {
        "num_samples": len(refs),
        "CER": cer_edits / cer_chars if cer_chars else 0.0,
        "WER": wer_edits / wer_words if wer_words else 0.0,
        "chrF": chrf_corpus(refs, hyps),
    }


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def key_of(row: dict) -> tuple[str, str, str, str, str]:
    return (
        (row.get("source") or "").strip(),
        (row.get("source_id") or "").strip(),
        (row.get("page") or "").strip(),
        (row.get("line") or "").strip(),
        (row.get("image_path") or "").strip(),
    )


def main() -> None:
    args = parse_args()

    raw_rows = read_csv(args.raw_csv)
    corrected_rows = read_csv(args.corrected_csv)

    corrected_by_key = {key_of(r): r for r in corrected_rows}

    refs: list[str] = []
    raw_hyps: list[str] = []
    corrected_hyps: list[str] = []

    by_source = defaultdict(lambda: {"refs": [], "raw": [], "corr": []})
    missing = 0

    for row in raw_rows:
        k = key_of(row)
        corr_row = corrected_by_key.get(k)
        if corr_row is None:
            missing += 1
            continue

        ref = (row.get(args.ref_column) or "").strip()
        raw_hyp = (row.get(args.raw_column) or "").strip()
        corr_hyp = (corr_row.get(args.corrected_column) or "").strip()
        source = (row.get(args.source_column) or "").strip()

        refs.append(ref)
        raw_hyps.append(raw_hyp)
        corrected_hyps.append(corr_hyp)

        by_source[source]["refs"].append(ref)
        by_source[source]["raw"].append(raw_hyp)
        by_source[source]["corr"].append(corr_hyp)

    raw_metrics = aggregate_metrics(refs, raw_hyps)
    corrected_metrics = aggregate_metrics(refs, corrected_hyps)

    per_source = {}
    for src, vals in by_source.items():
        m_raw = aggregate_metrics(vals["refs"], vals["raw"])
        m_corr = aggregate_metrics(vals["refs"], vals["corr"])
        per_source[src] = {
            "raw": m_raw,
            "corrected": m_corr,
            "delta": {
                "CER": m_corr["CER"] - m_raw["CER"],
                "WER": m_corr["WER"] - m_raw["WER"],
                "chrF": m_corr["chrF"] - m_raw["chrF"],
            },
        }

    out = {
        "matched_rows": len(refs),
        "missing_in_corrected": missing,
        "overall": {
            "raw": raw_metrics,
            "corrected": corrected_metrics,
            "delta": {
                "CER": corrected_metrics["CER"] - raw_metrics["CER"],
                "WER": corrected_metrics["WER"] - raw_metrics["WER"],
                "chrF": corrected_metrics["chrF"] - raw_metrics["chrF"],
            },
        },
        "per_source": per_source,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote metrics: {args.output_json}")
    print(json.dumps(out["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
