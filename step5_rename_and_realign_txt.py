"""Step 4: Rename line crops sequentially and align with transcript text."""

import csv, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
LINE_CROPS_DIR = ROOT / "Data" / "line_crops"
TRANSCRIPTS_DIR = ROOT / "Transcripts"
CSV_OUTPUTS = [ROOT / "line_alignment.csv", ROOT / "Data" / "line_alignment.csv"]

LINE_RE = re.compile(r"^(?P<page>.+)_line_(?P<line>\d+)\.png$", re.IGNORECASE)

SOURCE_META = {
    "Buendia - Instruccion transcription":          ("source1", "Buendia - Instruccion transcription.txt"),
    "Covarrubias - Tesoro lengua transcription":    ("source2", "Covarrubias - Tesoro lengua transcription.txt"),
    "Guardiola - Tratado nobleza transcription":    ("source3", "Guardiola - Tratado nobleza transcription.txt"),
    "PORCONES.228.38 – 1646":                      ("source4", "PORCONES.228.38 - 1646 transcription.txt"),
    "PORCONES.23.5 - 1628":                        ("source5", "PORCONES.23.5 - 1628 transcription.txt"),
    "PORCONES.748.6 – 1650":                       ("source6", "PORCONES.748.6 – 1650 Transcription.txt"),
}


def natural_sort_key(s):
    """Sort key that handles embedded numbers naturally."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def get_sorted_crops(folder):
    """Parse and sort line crop files by page then line number."""
    crops = []
    for f in folder.glob("*.png"):
        m = LINE_RE.match(f.name)
        if m:
            crops.append((f, m.group("page"), int(m.group("line"))))
    crops.sort(key=lambda x: (natural_sort_key(x[1]), x[2]))
    return crops


def rename_sequentially(folder):
    """Rename line crops to sequential numbering within each page."""
    crops = get_sorted_crops(folder)
    by_page = defaultdict(list)
    for path, page, _ in crops:
        by_page[page].append(path)

    # Two-pass rename via temp files to avoid collisions
    renames = []
    for page in sorted(by_page, key=natural_sort_key):
        for idx, old in enumerate(by_page[page], 1):
            new = folder / f"{page}_line_{idx:03d}.png"
            if old != new:
                renames.append((old, new))

    if renames:
        tmps = []
        for i, (old, new) in enumerate(renames):
            tmp = folder / f".__tmp_{i:06d}.png"
            old.rename(tmp)
            tmps.append((tmp, new))
        for tmp, new in tmps:
            tmp.rename(new)


def main():
    fields = ["source", "source_id", "page", "line", "image_path", "ground_truth_text", "status"]
    all_rows = []

    for name, (src_id, txt_file) in SOURCE_META.items():
        crops_dir = LINE_CROPS_DIR / name
        txt_path = TRANSCRIPTS_DIR / txt_file

        if not crops_dir.exists() or not txt_path.exists():
            print(f"  SKIP {name}: missing crops or transcript")
            continue

        rename_sequentially(crops_dir)
        crops = get_sorted_crops(crops_dir)
        lines = [l.strip() for l in txt_path.read_text("utf-8").splitlines() if l.strip()]

        n = min(len(crops), len(lines))
        for i in range(n):
            path, page, line_num = crops[i]
            all_rows.append({"source": name, "source_id": src_id, "page": page,
                             "line": i+1, "image_path": str(path.resolve()),
                             "ground_truth_text": lines[i], "status": "aligned"})

        for i in range(n, len(crops)):
            path, page, _ = crops[i]
            all_rows.append({"source": name, "source_id": src_id, "page": page,
                             "line": i+1, "image_path": str(path.resolve()),
                             "ground_truth_text": "", "status": "unaligned_excess_image"})

        for i in range(n, len(lines)):
            all_rows.append({"source": name, "source_id": src_id, "page": "N/A",
                             "line": i+1, "image_path": "",
                             "ground_truth_text": lines[i], "status": "unaligned_excess_transcript"})

        print(f"  {name}: {len(crops)} crops, {len(lines)} transcript lines, {n} aligned")

    for out in CSV_OUTPUTS:
        with open(out, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
            csv.DictWriter(f, fieldnames=fields).writerows(all_rows)

    print(f"\nDone. {len(all_rows)} rows written to {CSV_OUTPUTS[0].name}")


if __name__ == "__main__":
    main()
