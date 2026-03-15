#!/usr/bin/env python3
"""
compile_kraken_data.py
Compile Kraken ground-truth pairs into binary Arrow datasets using ketos compile.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GT_DIR = ROOT / "kraken_gt"


def run_compile(name: str, image_dir: Path, output: str) -> None:
    """Run 'ketos compile' and verify the output file."""
    out_path = ROOT / output

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        print(f"\n✗  No PNG files found for {name} in {image_dir}")
        sys.exit(1)

    cmd = [
        "ketos",
        "compile",
        "--format-type",
        "path",
        "--output",
        str(out_path),
        *[str(path) for path in image_paths],
    ]
    print(f"\n{'─' * 60}")
    print(f"Compiling {name} ({len(image_paths)} images)")
    print(f"Command: ketos compile --output {out_path} <{len(image_paths)} image paths>")
    print(f"{'─' * 60}")

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"\n✗  ketos compile failed for {name} (exit code {result.returncode})")
        sys.exit(1)

    if not out_path.exists():
        print(f"\n✗  Expected output {out_path} not found after compilation")
        sys.exit(1)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✓  {output} created — {size_mb:.2f} MB")


def main() -> None:
    train_dir = GT_DIR / "train"
    val_dir = GT_DIR / "val"

    run_compile("train", train_dir, "train.arrow")
    run_compile("val", val_dir, "val.arrow")

    print("\n✓  Both Arrow datasets compiled successfully.")


if __name__ == "__main__":
    main()
