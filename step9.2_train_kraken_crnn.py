#!/usr/bin/env python3
"""
train_kraken_crnn.py
Train a Kraken CRNN model on the compiled Arrow datasets, streaming stdout
in real time.
"""

import subprocess
import sys
import shutil
import os
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "kraken_models"
TRAIN_ARROW = ROOT / "train.arrow"
VAL_ARROW = ROOT / "val.arrow"
EVAL_MANIFEST = ROOT / "kraken_eval_files.txt"
BATCH_SIZE_CANDIDATES = [16, 12, 8, 6, 4, 2, 1]


def resolve_ketos_executable() -> str:
    """Return a runnable ketos executable path."""
    ketos_on_path = shutil.which("ketos")
    if ketos_on_path:
        return ketos_on_path

    venv_ketos = ROOT / ".venv" / "bin" / "ketos"
    if venv_ketos.exists():
        return str(venv_ketos)

    print("✗  Could not find 'ketos' executable.")
    print("   Install Kraken in the active environment: pip install kraken")
    sys.exit(1)


def main() -> None:
    # Verify Arrow files exist
    for f in (TRAIN_ARROW, VAL_ARROW):
        if not f.exists():
            print(f"✗  Required file not found: {f}")
            print("   Run compile_kraken_data.py first.")
            sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ketos_exe = resolve_ketos_executable()

    EVAL_MANIFEST.write_text(f"{VAL_ARROW}\n", encoding="utf-8")

    print(f"{'─' * 60}")
    print(f"Starting Kraken CRNN training")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"{'─' * 60}\n")

    def run_training(batch_size: int) -> Optional[str]:
        cmd = [
            ketos_exe,
            "-d", "cuda:0",
            "--precision", "16-mixed",
            "train",
            "--output", str(MODEL_DIR / "crnn"),
            "--format-type", "binary",
            "--evaluation-files", str(EVAL_MANIFEST),
            "--epochs", "50",
            "--batch-size", str(batch_size),
            "--lag", "10",
            "--augment",
            "--schedule", "cosine",
            str(TRAIN_ARROW),
        ]
        print(f"Trying batch size: {batch_size}")
        print(f"Command: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        captured_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            captured_lines.append(line)
            print(line, end="", flush=True)

        proc.wait()
        output_text = "".join(captured_lines)

        if proc.returncode == 0:
            return None

        if "OutOfMemoryError" in output_text or "CUDA out of memory" in output_text:
            print(f"\n⚠  CUDA OOM at batch size {batch_size}. Retrying with a smaller batch.")
            return "oom"

        print(f"\n✗  Training exited with code {proc.returncode}")
        sys.exit(1)

    for batch_size in BATCH_SIZE_CANDIDATES:
        status = run_training(batch_size)
        if status is None:
            break
    else:
        print("\n✗  Training failed for all batch-size fallbacks due to CUDA OOM.")
        sys.exit(1)

    # Locate best model
    best_model: Path | None = None
    for pattern in ("*best*", "*.mlmodel"):
        candidates = sorted(MODEL_DIR.glob(pattern))
        if candidates:
            best_model = candidates[-1]
            break

    if best_model:
        print(f"\n✓  Best model: {best_model}")
    else:
        print(f"\n✗  No model file found in {MODEL_DIR}")
        print("   Contents:", list(MODEL_DIR.iterdir()))
        sys.exit(1)


if __name__ == "__main__":
    main()
