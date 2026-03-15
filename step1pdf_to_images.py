#!/usr/bin/env python3
"""Step 1.1: Convert PDFs to high-resolution page images.

Outputs one subfolder per source PDF and names pages as:
    source1_page001.png, source1_page002.png, ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError as exc:
        raise SystemExit("Missing dependency: pymupdf. Install with `pip install pymupdf`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1.1 - Convert PDFs to PNG page images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Handwritten"),
        help="Directory containing input PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/page_images"),
        help="Directory where per-source image folders will be created.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization DPI (minimum recommended: 300).",
    )
    return parser.parse_args()


def list_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {input_dir}")

    return pdf_files


def render_pdf_to_pngs(pdf_path: Path, source_name: str, output_subdir: Path, dpi: int) -> int:
    output_subdir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    matrix = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    page_count = doc.page_count

    for page_index in range(page_count):
        page = doc.load_page(page_index)
        out_path = output_subdir / f"{source_name}_page{page_index + 1:03d}.png"
        page.get_pixmap(matrix=matrix, alpha=False).save(out_path)

    doc.close()
    return page_count


def main() -> None:
    args = parse_args()

    if args.dpi < 300:
        raise SystemExit("DPI must be at least 300 for Step 1.1.")

    pdf_files = list_pdfs(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    total_pages = 0

    for index, pdf_path in enumerate(pdf_files, start=1):
        source_name = f"source{index}"
        source_dir = args.output_dir / source_name

        pages_written = render_pdf_to_pngs(
            pdf_path=pdf_path,
            source_name=source_name,
            output_subdir=source_dir,
            dpi=args.dpi,
        )
        total_pages += pages_written

        manifest.append(
            {
                "source": source_name,
                "pdf_file": str(pdf_path),
                "output_folder": str(source_dir),
                "pages_written": pages_written,
                "name_pattern": f"{source_name}_page###.png",
            }
        )

        print(f"[{source_name}] {pdf_path.name}: {pages_written} pages")

    manifest_path = args.output_dir / "conversion_manifest.json"
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "dpi": args.dpi,
        "sources": len(pdf_files),
        "total_pages": total_pages,
        "files": manifest,
    }
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Converted {len(pdf_files)} PDFs into {total_pages} PNG pages.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
