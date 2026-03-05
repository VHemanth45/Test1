"""
Phase 8 — TrOCR Fine-tuning
============================

Implements FinalPlan.md Phase 8:
  - Step 8.1: Start from pretrained TrOCR (microsoft/trocr-base-printed)
  - Step 8.2: Train on aligned + augmented line crop pairs (50 epochs)
  - Step 8.3: Monitor for overfitting / underfitting via CER on val set
  - Step 8.4: Evaluate CER, WER, chrF on validation set

Input:
  Data/ocr_dataset/train_manifest.csv   (augmented training pairs)
  Data/ocr_dataset/val_manifest.csv     (pristine validation pairs)

Output (under runs/trocr_finetune/):
  - best/                  best model checkpoint (lowest val CER)
  - predictions.csv        val-set predictions from best checkpoint
  - metrics.json           final CER / WER / chrF + per-source breakdown
  - all_results.json       training history summary

Memory-optimised for 4 GB VRAM (RTX 3050 Ti):
  - Frozen encoder (only decoder trained) — critical for ≤4 GB VRAM
  - fp16 mixed precision
  - gradient checkpointing
  - 8-bit AdamW optimizer (bitsandbytes)
  - batch_size=1 + gradient_accumulation_steps=8 → effective batch 8

If you have ≥12 GB VRAM, you can use the larger model:
  --model microsoft/trocr-large-printed --no-freeze-encoder
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ────────────────────────────── paths ──────────────────────────────
ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_MANIFEST = ROOT / "Data" / "ocr_dataset" / "train_manifest.csv"
DEFAULT_VAL_MANIFEST   = ROOT / "Data" / "ocr_dataset" / "val_manifest.csv"
DEFAULT_OUTPUT_DIR     = ROOT / "runs" / "trocr_finetune"
TESSERACT_METRICS      = ROOT / "Data" / "tesseract_baseline" / "metrics.json"


# ────────────────── transformers 5.x compatibility ─────────────────
def fix_trocr_meta_tensors(model: VisionEncoderDecoderModel) -> None:
    """Fix meta-device tensors left behind by lazy-loading in transformers >= 5.x.

    TrOCRSinusoidalPositionalEmbedding stores ``weights`` as a plain attribute
    (NOT a registered buffer/parameter).  ``model.to(device)`` therefore never
    moves it.  When ``from_pretrained`` initialises on the meta device (which is
    the default since transformers 5.x) ``weights`` stays on meta permanently.
    This function recomputes the sinusoidal table and installs a forward
    pre-hook that auto-moves it to the right device on first use.
    """
    if not (hasattr(model, "decoder") and hasattr(model.decoder, "model")):
        return
    decoder_inner = model.decoder.model.decoder
    if not hasattr(decoder_inner, "embed_positions"):
        return
    ep = decoder_inner.embed_positions

    # Fix ``weights`` attribute (TrOCRSinusoidalPositionalEmbedding)
    if hasattr(ep, "weights") and ep.weights is not None:
        if ep.weights.device.type == "meta":
            from transformers.models.trocr.modeling_trocr import (
                TrOCRSinusoidalPositionalEmbedding,
            )
            num_pos = ep.weights.size(0)
            ep.weights = TrOCRSinusoidalPositionalEmbedding.get_embedding(
                num_pos, ep.embedding_dim, ep.padding_idx,
            )
            log.info("  Recomputed meta-device sinusoidal embed_positions.weights "
                     f"({num_pos} positions)")

            def _move_weights_hook(module, args):
                """Move sinusoidal weights to the same device as input_ids."""
                if module.weights is not None and len(args) > 0:
                    target_device = args[0].device
                    if module.weights.device != target_device:
                        module.weights = module.weights.to(target_device)

            ep.register_forward_pre_hook(_move_weights_hook)
            log.info("  Registered forward pre-hook to auto-move weights to device")

    # Fix ``_float_tensor`` buffer (TrOCRLearnedPositionalEmbedding)
    if hasattr(ep, "_float_tensor"):
        if ep._float_tensor.device.type == "meta":
            ep._float_tensor = torch.zeros(1, dtype=torch.float32)
            log.info("  Fixed meta-device _float_tensor in embed_positions")


# ──────────────────────────── argparse ─────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 8 — Fine-tune TrOCR on RenAIssance OCR dataset.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="HuggingFace model ID (default: microsoft/trocr-base-printed). "
             "Use trocr-large-printed if you have ≥12 GB VRAM.",
    )
    p.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    p.add_argument("--val-manifest",   type=Path, default=DEFAULT_VAL_MANIFEST)
    p.add_argument("--output-dir",     type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs",         type=int,  default=20)
    p.add_argument("--batch-size",     type=int,  default=1,
                   help="Per-device train batch size (keep low for 4 GB VRAM).")
    p.add_argument("--grad-accum",     type=int,  default=8,
                   help="Gradient accumulation steps (effective batch = batch-size × grad-accum).")
    p.add_argument("--lr",             type=float, default=4e-5,
                   help="Peak learning rate.")
    p.add_argument("--warmup-ratio",   type=float, default=0.05,
                   help="Fraction of total steps used for linear warmup.")
    p.add_argument("--max-target-length", type=int, default=128,
                   help="Max token length for ground truth labels.")
    p.add_argument("--num-beams",      type=int,  default=4,
                   help="Beam search width during generation / eval.")
    p.add_argument("--patience",       type=int,  default=7,
                   help="Early stopping patience (epochs w/o CER improvement).")
    p.add_argument("--eval-strategy",  type=str, default="epoch",
                   choices=["epoch", "steps"],
                   help="When to run evaluation.")
    p.add_argument("--eval-steps",     type=int, default=200,
                   help="Eval every N steps (only if --eval-strategy=steps).")
    p.add_argument("--save-total-limit", type=int, default=3,
                   help="Keep at most N checkpoints.")
    p.add_argument("--freeze-encoder", action="store_true", default=True,
                   help="Freeze the vision encoder — only train the decoder (default: True for ≤4 GB VRAM).")
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false",
                   help="Train the full model (encoder + decoder).")
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable fp16 mixed precision (uses fp32).")
    p.add_argument("--no-8bit-optim", action="store_true",
                   help="Use standard AdamW instead of 8-bit (needs more VRAM).")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="Path to checkpoint directory to resume training from.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, only run evaluation on val set.")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit rows for quick smoke test (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ──────────────────────────── dataset ──────────────────────────────
def load_manifest(path: Path, split_filter: str | None = None,
                  limit: int = 0) -> list[dict]:
    """Read a manifest CSV and return list of dicts with image_path + ground_truth_text."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_filter and row.get("split", "").strip() != split_filter:
                continue
            img = (row.get("image_path") or "").strip()
            gt  = (row.get("ground_truth_text") or "").strip()
            if not img or not gt:
                continue
            rows.append({
                "image_path":        img,
                "ground_truth_text": gt,
                "source":            row.get("source", "").strip(),
                "source_id":         row.get("source_id", "").strip(),
                "page":              row.get("page", "").strip(),
                "line":              row.get("line", "").strip(),
            })
            if 0 < limit <= len(rows):
                break

    if not rows:
        raise ValueError(f"No usable rows in {path} (filter={split_filter}).")
    return rows


class OCRDataset(Dataset):
    """PyTorch Dataset that feeds (pixel_values, labels) to TrOCR."""

    def __init__(self, records: list[dict], processor, max_target_length: int = 128):
        self.records = records
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze()

        labels = self.processor.tokenizer(
            rec["ground_truth_text"],
            padding="max_length",
            max_length=self.max_target_length,
        ).input_ids

        # Replace PAD token ids with -100 so they are ignored by the loss
        labels = [
            lbl if lbl != self.processor.tokenizer.pad_token_id else -100
            for lbl in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ──────────────────────────── metrics ──────────────────────────────
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


def chrf_corpus(refs: list[str], hyps: list[str],
                n_max: int = 6, beta: float = 2.0) -> float:
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
        cer_edits += ce;  cer_chars += cc
        wer_edits += we;  wer_words += ww

    return {
        "num_samples": len(refs),
        "CER":  cer_edits / cer_chars if cer_chars else 0.0,
        "WER":  wer_edits / wer_words if wer_words else 0.0,
        "chrF": chrf_corpus(refs, hyps),
        "cer_edits": cer_edits, "cer_chars": cer_chars,
        "wer_edits": wer_edits, "wer_words": wer_words,
    }


# ──────────────────────── training helpers ─────────────────────────
def build_compute_metrics(processor):
    """Return a compute_metrics function for Seq2SeqTrainer (CER only, fast)."""

    def compute_metrics(pred):
        label_ids = pred.label_ids
        pred_ids  = pred.predictions

        # Decode predictions
        pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # Replace -100 with pad so we can decode
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute corpus-level CER
        total_edits = total_chars = 0
        for ref, hyp in zip(label_str, pred_str):
            e, c = cer_value(ref, hyp)
            total_edits += e
            total_chars += c
        cer = total_edits / total_chars if total_chars else 0.0

        return {"cer": cer}

    return compute_metrics


# ────────────────────── full evaluation pass ───────────────────────
@torch.no_grad()
def full_evaluate(model, processor, val_records: list[dict],
                  num_beams: int = 4, max_length: int = 128,
                  batch_size: int = 4) -> tuple[list[dict], dict]:
    """
    Run beam-search generation on every val record.
    Returns (prediction_rows, overall_metrics).
    """
    device = next(model.parameters()).device
    model.eval()

    prediction_rows: list[dict] = []
    refs: list[str] = []
    hyps: list[str] = []

    for i in range(0, len(val_records), batch_size):
        batch_records = val_records[i : i + batch_size]
        images = [
            Image.open(r["image_path"]).convert("RGB") for r in batch_records
        ]
        pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)

        generated_ids = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_length=max_length,
        )
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for rec, pred_text in zip(batch_records, preds):
            gt = rec["ground_truth_text"]
            refs.append(gt)
            hyps.append(pred_text)
            prediction_rows.append({
                "source":            rec["source"],
                "source_id":         rec["source_id"],
                "page":              rec["page"],
                "line":              rec["line"],
                "image_path":        rec["image_path"],
                "ground_truth_text": gt,
                "prediction":        pred_text,
            })

        if (i // batch_size + 1) % 10 == 0:
            log.info(f"  Evaluated {min(i + batch_size, len(val_records))}/{len(val_records)}")

    overall = aggregate_metrics(refs, hyps)

    # Per-source breakdown
    by_source: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for row in prediction_rows:
        sid = row["source_id"]
        by_source[sid][0].append(row["ground_truth_text"])
        by_source[sid][1].append(row["prediction"])

    per_source = {
        sid: aggregate_metrics(r, h) for sid, (r, h) in sorted(by_source.items())
    }
    overall["per_source"] = per_source

    return prediction_rows, overall


# ─────────────────────────── main ──────────────────────────────────
def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    log.info("=" * 68)
    log.info("PHASE 8 — TrOCR Fine-tuning")
    log.info("=" * 68)
    log.info(f"Model         : {args.model}")
    log.info(f"Output dir    : {args.output_dir}")
    log.info(f"Epochs        : {args.epochs}")
    log.info(f"Batch (device): {args.batch_size}  ×  grad-accum {args.grad_accum}  "
             f"= effective {args.batch_size * args.grad_accum}")
    log.info(f"LR            : {args.lr}")
    log.info(f"fp16          : {not args.no_fp16}")
    log.info(f"8-bit optim   : {not args.no_8bit_optim}")
    log.info(f"Freeze encoder: {args.freeze_encoder}")
    log.info(f"Patience      : {args.patience}")
    log.info(f"Seed          : {args.seed}")

    # ── check GPU ──
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"GPU           : {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        log.warning("No GPU detected — training will be very slow.")
        if not args.no_fp16:
            log.warning("Disabling fp16 (no GPU).")
            args.no_fp16 = True

    # ── load manifests ──
    log.info("")
    log.info("Loading manifests …")
    train_records = load_manifest(
        args.train_manifest, split_filter="train", limit=args.limit,
    )
    val_records = load_manifest(
        args.val_manifest, split_filter="val", limit=args.limit,
    )
    log.info(f"  Train samples : {len(train_records)}")
    log.info(f"  Val   samples : {len(val_records)}")

    # ── processor & datasets ──
    log.info("Loading TrOCRProcessor …")
    processor = TrOCRProcessor.from_pretrained(args.model)

    train_dataset = OCRDataset(train_records, processor, args.max_target_length)
    eval_dataset  = OCRDataset(val_records,   processor, args.max_target_length)

    # Quick sanity check
    sample = train_dataset[0]
    log.info(f"  pixel_values shape : {sample['pixel_values'].shape}")
    log.info(f"  labels length      : {sample['labels'].shape}")

    # ── model ──
    log.info(f"Loading model: {args.model} …")
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    fix_trocr_meta_tensors(model)

    # Configure decoder / generation settings
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    # Generation params go on generation_config (required by transformers >= 5.x)
    model.generation_config.max_length             = args.max_target_length
    model.generation_config.early_stopping         = True
    model.generation_config.no_repeat_ngram_size   = 0   # OCR should allow repeated n-grams
    model.generation_config.length_penalty         = 1.0
    model.generation_config.num_beams              = args.num_beams

    # ── freeze encoder if requested (critical for ≤4 GB VRAM) ──
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        log.info("  Encoder frozen — only decoder will be trained")

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params   = total_params - trainable_params
    log.info(f"  Total params     : {total_params:,}")
    log.info(f"  Trainable params : {trainable_params:,}")
    log.info(f"  Frozen params    : {frozen_params:,}")

    # ── training arguments ──
    use_fp16 = (not args.no_fp16) and torch.cuda.is_available()
    optim_name = "adamw_bnb_8bit" if (not args.no_8bit_optim) else "adamw_torch"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_target_length,
        # ── epochs / batching ──
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 2),  # eval can use a bit more
        gradient_accumulation_steps=args.grad_accum,
        # ── optimiser ──
        optim=optim_name,
        learning_rate=args.lr,
        warmup_steps=int(args.warmup_ratio * args.epochs * len(train_records)
                       / (args.batch_size * args.grad_accum)),
        weight_decay=0.01,
        # ── mixed precision ──
        fp16=use_fp16,
        # ── memory optimisation ──
        gradient_checkpointing=True,
        # ── evaluation & saving ──
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        # ── logging ──
        logging_steps=10,
        report_to="none",
        # ── misc ──
        seed=args.seed,
        dataloader_pin_memory=False,   # save a bit of host memory
        dataloader_num_workers=2,
        remove_unused_columns=False,
        torch_compile=False,
    )

    # ── callbacks ──
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]

    # ── trainer ──
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=build_compute_metrics(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    # ── train / resume ──
    if not args.eval_only:
        log.info("")
        log.info("=" * 68)
        log.info("STARTING TRAINING")
        log.info("=" * 68)
        checkpoint = args.resume_from_checkpoint
        if checkpoint and not Path(checkpoint).exists():
            log.warning(f"Checkpoint {checkpoint} not found — training from scratch.")
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(output_dir / "best"))
        processor.save_pretrained(str(output_dir / "best"))

        # Save training results
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

        log.info(f"Best model saved to: {output_dir / 'best'}")
    else:
        # Load best checkpoint for eval-only mode
        best_dir = output_dir / "best"
        if best_dir.exists():
            log.info(f"Loading best model from {best_dir} for evaluation …")
            model = VisionEncoderDecoderModel.from_pretrained(
                str(best_dir),
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
            )
            fix_trocr_meta_tensors(model)
            if torch.cuda.is_available():
                model = model.to("cuda")
        else:
            log.warning("No best/ checkpoint found — evaluating current model weights.")

    # ── full evaluation on val set ──
    log.info("")
    log.info("=" * 68)
    log.info("FULL EVALUATION ON VALIDATION SET")
    log.info("=" * 68)

    if torch.cuda.is_available():
        model = model.to("cuda")

    pred_rows, overall_metrics = full_evaluate(
        model, processor, val_records,
        num_beams=args.num_beams,
        max_length=args.max_target_length,
        batch_size=max(args.batch_size, 2),
    )

    # ── save predictions CSV ──
    preds_csv = output_dir / "predictions.csv"
    fields = [
        "source", "source_id", "page", "line",
        "image_path", "ground_truth_text", "prediction",
    ]
    with preds_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pred_rows)
    log.info(f"Predictions saved: {preds_csv}")

    # ── load Tesseract baseline for comparison ──
    tesseract_cer = tesseract_wer = tesseract_chrf = None
    if TESSERACT_METRICS.exists():
        with TESSERACT_METRICS.open() as f:
            tess = json.load(f)
        if "overall" in tess:
            tesseract_cer  = tess["overall"].get("CER")
            tesseract_wer  = tess["overall"].get("WER")
            tesseract_chrf = tess["overall"].get("chrF")

    # ── build metrics payload ──
    per_source = overall_metrics.pop("per_source", {})

    metrics_payload = {
        "phase": "Phase 8 - TrOCR fine-tuned",
        "model": args.model,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "overall": overall_metrics,
        "per_source": per_source,
        "comparison": {
            "tesseract_baseline": {
                "CER":  tesseract_cer,
                "WER":  tesseract_wer,
                "chrF": tesseract_chrf,
            },
            "trocr_finetuned": {
                "CER":  overall_metrics["CER"],
                "WER":  overall_metrics["WER"],
                "chrF": overall_metrics["chrF"],
            },
            "improvement": {
                "CER_delta":  (tesseract_cer  - overall_metrics["CER"])  if tesseract_cer  is not None else None,
                "WER_delta":  (tesseract_wer  - overall_metrics["WER"])  if tesseract_wer  is not None else None,
                "chrF_delta": (overall_metrics["chrF"] - tesseract_chrf) if tesseract_chrf is not None else None,
            },
        },
        "inputs": {
            "train_manifest": str(args.train_manifest),
            "val_manifest":   str(args.val_manifest),
            "train_samples":  len(train_records),
            "val_samples":    len(val_records),
        },
        "outputs": {
            "best_model":      str(output_dir / "best"),
            "predictions_csv": str(preds_csv),
        },
    }

    metrics_json = output_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    # ── summary ──
    log.info("")
    log.info("=" * 68)
    log.info("PHASE 8 — RESULTS")
    log.info("=" * 68)
    log.info(f"{'Metric':<8} {'Tesseract':>12} {'TrOCR':>12} {'Δ (improvement)':>18}")
    log.info("-" * 52)

    def _fmt(v, delta=None):
        v_s = f"{v:.4f}" if v is not None else "   N/A"
        d_s = f"{delta:+.4f}" if delta is not None else "   N/A"
        return v_s, d_s

    cer_s, cer_d = _fmt(tesseract_cer, metrics_payload["comparison"]["improvement"]["CER_delta"])
    wer_s, wer_d = _fmt(tesseract_wer, metrics_payload["comparison"]["improvement"]["WER_delta"])
    chr_s, chr_d = _fmt(tesseract_chrf, metrics_payload["comparison"]["improvement"]["chrF_delta"])

    log.info(f"{'CER':<8} {cer_s:>12} {overall_metrics['CER']:>12.4f} {cer_d:>18}")
    log.info(f"{'WER':<8} {wer_s:>12} {overall_metrics['WER']:>12.4f} {wer_d:>18}")
    log.info(f"{'chrF':<8} {chr_s:>12} {overall_metrics['chrF']:>12.4f} {chr_d:>18}")
    log.info("")

    log.info("Per-source breakdown:")
    for sid, m in sorted(per_source.items()):
        log.info(f"  {sid:10s}  CER={m['CER']:.4f}  WER={m['WER']:.4f}  chrF={m['chrF']:.4f}  (n={m['num_samples']})")

    log.info("")
    log.info(f"Metrics: {metrics_json}")
    log.info(f"Model  : {output_dir / 'best'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
