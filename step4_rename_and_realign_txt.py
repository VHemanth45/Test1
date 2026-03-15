from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
LINE_CROPS_DIR = DATA / "line_crops"
TRANSCRIPTS_DIR = ROOT / "Transcripts"
CSV_OUTPUT = ROOT / "line_alignment.csv"
CSV_OUTPUT_DATA = DATA / "line_alignment.csv"

SOURCE_META = {
    "Buendia - Instruccion transcription": {
        "source_id": "source1",
        "transcript_txt": "Buendia - Instruccion transcription.txt",
    },
    "Covarrubias - Tesoro lengua transcription": {
        "source_id": "source2",
        "transcript_txt": "Covarrubias - Tesoro lengua transcription.txt",
    },
    "Guardiola - Tratado nobleza transcription": {
        "source_id": "source3",
        "transcript_txt": "Guardiola - Tratado nobleza transcription.txt",
    },
    "PORCONES.228.38 – 1646": {
        "source_id": "source4",
        "transcript_txt": "PORCONES.228.38 - 1646 transcription.txt",
    },
    "PORCONES.23.5 - 1628": {
        "source_id": "source5",
        "transcript_txt": "PORCONES.23.5 - 1628 transcription.txt",
    },
    "PORCONES.748.6 – 1650": {
        "source_id": "source6",
        "transcript_txt": "PORCONES.748.6 – 1650 Transcription.txt",
    },
}

LINE_FILE_RE = re.compile(r"^(?P<page>.+)_line_(?P<line>\d+)\.png$", re.IGNORECASE)
PAGE_TOKEN_RE = re.compile(r"\d+\.\d+|\d+|[A-Za-z]+")


def page_token_key(token: str) -> Tuple[int, int, int | str]:
    token_lower = token.lower()

    if re.fullmatch(r"\d+\.\d+", token_lower):
        major, minor = token_lower.split(".", 1)
        return (int(major), 0, int(minor))

    if re.fullmatch(r"\d+", token_lower):
        return (int(token_lower), 1, 0)

    if re.fullmatch(r"\d+[a-z]+", token_lower):
        n = re.match(r"(\d+)([a-z]+)", token_lower)
        assert n is not None
        base = int(n.group(1))
        suffix = n.group(2)
        suffix_rank = {"left": 0, "right": 1}.get(suffix, 2)
        return (base, 2, f"{suffix_rank}:{suffix}")

    return (10**9, 3, token_lower)


def page_sort_key(page_label: str) -> Tuple[Tuple[int, int, int | str], ...]:
    parts = PAGE_TOKEN_RE.findall(page_label)
    if not parts:
        return ((10**9, 3, page_label.lower()),)
    return tuple(page_token_key(p) for p in parts)


def parse_line_file_name(path: Path) -> Tuple[str, int] | None:
    match = LINE_FILE_RE.match(path.name)
    if not match:
        return None
    return match.group("page"), int(match.group("line"))


def parse_transcript_txt(txt_path: Path) -> List[str]:
    lines: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                lines.append(stripped)
    return lines


def iter_line_crops(folder: Path) -> List[Tuple[Path, str, int]]:
    parsed: List[Tuple[Path, str, int]] = []
    for image_path in folder.glob("*.png"):
        parsed_name = parse_line_file_name(image_path)
        if parsed_name is None:
            continue
        page, line = parsed_name
        parsed.append((image_path, page, line))

    parsed.sort(key=lambda x: (page_sort_key(x[1]), x[2], x[0].name.lower()))
    return parsed


def rename_line_crops_in_order(folder: Path) -> Dict[Path, Path]:
    parsed = iter_line_crops(folder)
    if not parsed:
        return {}

    by_page: Dict[str, List[Path]] = {}
    for image_path, page, _line in parsed:
        by_page.setdefault(page, []).append(image_path)

    old_to_new: Dict[Path, Path] = {}
    for page in sorted(by_page.keys(), key=page_sort_key):
        for idx, old_path in enumerate(by_page[page], start=1):
            new_name = f"{page}_line_{idx:03d}.png"
            old_to_new[old_path] = folder / new_name

    changed = {k: v for k, v in old_to_new.items() if k != v}
    if not changed:
        return old_to_new

    tmp_map: Dict[Path, Path] = {}
    counter = 1
    for old_path in changed:
        tmp_path = folder / f".__tmp_realign__{counter:06d}.png"
        counter += 1
        old_path.rename(tmp_path)
        tmp_map[tmp_path] = old_to_new[old_path]

    for tmp_path, final_path in tmp_map.items():
        tmp_path.rename(final_path)

    return old_to_new


def build_rows_for_source(source_name: str, source_id: str, transcript_lines: List[str], crops: Iterable[Tuple[Path, str, int]]) -> List[dict]:
    crops_list = list(crops)
    rows: List[dict] = []

    n_aligned = min(len(crops_list), len(transcript_lines))

    for i in range(n_aligned):
        image_path, page, line_num = crops_list[i]
        rows.append(
            {
                "source": source_name,
                "source_id": source_id,
                "page": page,
                "line": line_num,
                "image_path": str(image_path.resolve()),
                "ground_truth_text": transcript_lines[i],
                "status": "aligned",
            }
        )

    for i in range(n_aligned, len(crops_list)):
        image_path, page, line_num = crops_list[i]
        rows.append(
            {
                "source": source_name,
                "source_id": source_id,
                "page": page,
                "line": line_num,
                "image_path": str(image_path.resolve()),
                "ground_truth_text": "",
                "status": "unaligned_excess_image",
            }
        )

    for i in range(n_aligned, len(transcript_lines)):
        rows.append(
            {
                "source": source_name,
                "source_id": source_id,
                "page": "N/A",
                "line": i + 1,
                "image_path": "",
                "ground_truth_text": transcript_lines[i],
                "status": "unaligned_excess_transcript",
            }
        )

    return rows


def main() -> None:
    all_rows: List[dict] = []

    for source_name, meta in SOURCE_META.items():
        crops_dir = LINE_CROPS_DIR / source_name
        transcript_path = TRANSCRIPTS_DIR / meta["transcript_txt"]

        if not crops_dir.exists():
            raise FileNotFoundError(f"Missing line-crops folder: {crops_dir}")
        if not transcript_path.exists():
            raise FileNotFoundError(f"Missing transcript txt: {transcript_path}")

        rename_line_crops_in_order(crops_dir)

        sorted_crops = iter_line_crops(crops_dir)
        normalized = [(path, page, idx + 1) for idx, (path, page, _old_line) in enumerate(sorted_crops)]

        transcript_lines = parse_transcript_txt(transcript_path)

        source_rows = build_rows_for_source(
            source_name=source_name,
            source_id=meta["source_id"],
            transcript_lines=transcript_lines,
            crops=normalized,
        )
        all_rows.extend(source_rows)

        print(
            f"{source_name}: crops={len(normalized)} transcript={len(transcript_lines)} "
            f"aligned={min(len(normalized), len(transcript_lines))}"
        )

    fieldnames = [
        "source",
        "source_id",
        "page",
        "line",
        "image_path",
        "ground_truth_text",
        "status",
    ]

    for target in [CSV_OUTPUT, CSV_OUTPUT_DATA]:
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {CSV_OUTPUT} and {CSV_OUTPUT_DATA}")


if __name__ == "__main__":
    main()
