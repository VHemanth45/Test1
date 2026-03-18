"""Step 1: Convert PDFs to high-resolution PNG page images."""

from pathlib import Path

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


def pdf_to_images(input_dir="Handwritten", output_dir="Data/page_images", dpi=300):
    """Convert all PDFs in input_dir to PNG images in output_dir."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {input_dir}")

    scale = dpi / 72.0
    total = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        src = f"source{idx}"
        src_dir = output_dir / src
        src_dir.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open(pdf_path)
        mat = pymupdf.Matrix(scale, scale)
        for i in range(doc.page_count):
            doc[i].get_pixmap(matrix=mat, alpha=False).save(
                src_dir / f"{src}_page{i+1:03d}.png"
            )
        total += doc.page_count
        print(f"[{src}] {pdf_path.name}: {doc.page_count} pages")
        doc.close()

    print(f"\nDone. {len(pdf_files)} PDFs → {total} PNG pages in {output_dir}")


if __name__ == "__main__":
    pdf_to_images()
