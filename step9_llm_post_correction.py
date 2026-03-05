"""
Phase 9 — LLM Post-Correction (Gemini)

Supports:
    - TXT in -> TXT out (one OCR line per input line)
    - CSV in -> CSV out (correct a chosen text column)
    - Chunked correction with overlap (default: 8 lines, overlap 2)

Usage:
  export GEMINI_API_KEY="your_key"
  python step9_llm_post_correction.py \
      --input Data/trocr_output_lines.txt \
      --output Data/trocr_output_lines_corrected.txt

    python step9_llm_post_correction.py \
            --input runs/trocr_finetune/predictions.csv \
            --output runs/trocr_finetune/predictions_corrected.csv \
            --text-column prediction_text
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from google import genai


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9: Gemini post-correction for OCR lines.")
    p.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default="ollama",
        help="LLM backend provider (default: ollama).",
    )
    p.add_argument("--input", type=Path, required=True, help="Input TXT or CSV")
    p.add_argument("--output", type=Path, required=True, help="Output TXT or CSV")
    p.add_argument(
        "--format",
        choices=["auto", "txt", "csv"],
        default="auto",
        help="Input/output format (default: auto from input suffix).",
    )
    p.add_argument(
        "--text-column",
        default="ocr_text",
        help="CSV mode: column containing raw OCR text.",
    )
    p.add_argument(
        "--output-column",
        default="llm_corrected_text",
        help="CSV mode: column to write corrected text.",
    )
    p.add_argument(
        "--group-column",
        default="source",
        help="CSV mode: chunk within each group value (default: source). Use empty string to disable.",
    )
    p.add_argument(
        "--group-value",
        default="",
        help="CSV mode: process only rows where group-column equals this value (e.g., source1).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Lines per chunk for LLM correction (recommended 5-10).",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="Overlapping lines between consecutive chunks.",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (used if GEMINI_API_KEY is not already exported).",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=15.0,
        help="Seconds to sleep between API requests (default: 15).",
    )
    p.add_argument("--model", default="phi2", help="Model name (e.g., gemini-2.0-flash or phi2).")
    p.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint URL.",
    )
    p.add_argument(
        "--ollama-timeout",
        type=int,
        default=600,
        help="Ollama request timeout in seconds (default: 600).",
    )
    p.add_argument(
        "--ollama-retries",
        type=int,
        default=2,
        help="Number of retries for Ollama request failures (default: 2).",
    )
    return p.parse_args()


PROMPT_NOTES = """
NOTES: u and v are used interchangeably check against dictionary? f and s are used interchangeably check against dictionary? accents are inconsistent should be ignored (except ñ) some letters have horizontal “cap” tends to mean n follows, or ue after capped q some line end hyphens not present leave words split for now, can decide later ç old spelling is always modern z teach AI to always interpret ç as z
Notes: i en lugar de j (Iacinto, Iuan)

u en lugar de v en lugar de b (Rauasco, auer, auia, etc.)

Las vocales con acentos nasales = vocal + n (cuéta --> cuenta)

Z en lugar de c (vezes, dize, etc.)

Que (la palabra completa) en lugar de q con acento

Ç = z (cobrança = cobranza)

Azul = otros casos de la misma pregunta
""".strip()


def build_prompt(ocr_chunk: str) -> str:
    return (
        "You are correcting OCR text from Early Modern Spanish documents.\\n"
        "Rules:\\n"
        "- Fix only likely OCR character/reading errors.\\n"
        "- Preserve archaic spelling and punctuation style.\\n"
        "- Do not modernize vocabulary or rewrite meaning.\\n"
        "- Return only corrected text lines.\\n"
        "- Keep EXACTLY the same number of lines as input.\\n"
        "- No explanations, no markdown, no numbering.\\n\\n"
        f"{PROMPT_NOTES}\\n\\n"
        f"OCR chunk:\\n{ocr_chunk}"
    )


def gemini_correct_chunk(client: genai.Client, model: str, text: str) -> str:
    try:
        response = client.models.generate_content(
            model=model,
            contents=build_prompt(text),
            config={"temperature": 0.0},
        )
    except Exception as e:
        raise RuntimeError(f"Gemini client request failed: {e}") from e

    corrected = (getattr(response, "text", "") or "").strip()
    return corrected or text


def ollama_correct_chunk(
    ollama_url: str,
    model: str,
    text: str,
    timeout_seconds: int,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "prompt": build_prompt(text),
        "stream": False,
        "options": {"temperature": 0.0},
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                corrected = (data.get("response") or "").strip()
                return corrected or text
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"Ollama HTTP {e.code}: {body}")
        except Exception as e:
            last_error = RuntimeError(f"Ollama request failed: {e}")

        if attempt < retries:
            time.sleep(3)

    raise RuntimeError(str(last_error) if last_error else "Ollama request failed")


def _normalize_output_lines(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+[\.)]\s+", stripped):
            cleaned.append(re.sub(r"^\d+[\.)]\s+", "", stripped))
        elif stripped.startswith("- "):
            cleaned.append(stripped[2:])
        else:
            cleaned.append(line)
    return cleaned


def _chunk_ranges(n: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if n == 0:
        return []
    if chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if overlap < 0:
        raise SystemExit("--overlap must be >= 0")
    if overlap >= chunk_size:
        raise SystemExit("--overlap must be smaller than --chunk-size")

    ranges: list[tuple[int, int]] = []
    step = chunk_size - overlap
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        ranges.append((start, end))
        if end == n:
            break
        start += step
    return ranges


def correct_lines_in_chunks(
    lines: list[str],
    correct_chunk: Callable[[str], str],
    chunk_size: int,
    overlap: int,
    sleep_seconds: float,
) -> tuple[list[str], list[tuple[int, int]]]:
    ranges = _chunk_ranges(len(lines), chunk_size, overlap)
    corrected_chunks: list[list[str]] = []
    flagged: list[tuple[int, int]] = []

    for idx, (start, end) in enumerate(ranges):
        original_chunk = lines[start:end]
        chunk_text = "\n".join(original_chunk)
        response_text = correct_chunk(chunk_text)
        out_lines = _normalize_output_lines(response_text)

        if len(out_lines) != len(original_chunk):
            flagged.append((start, end))
            out_lines = original_chunk

        corrected_chunks.append(out_lines)

        if sleep_seconds > 0 and idx < len(ranges) - 1:
            time.sleep(sleep_seconds)

    if not corrected_chunks:
        return [], flagged

    merged: list[str] = corrected_chunks[0]
    for i in range(1, len(corrected_chunks)):
        start, end = ranges[i]
        prev_start, _ = ranges[i - 1]
        actual_overlap = max(0, (prev_start + chunk_size) - start)
        if actual_overlap > 0:
            merged.extend(corrected_chunks[i][actual_overlap:])
        else:
            merged.extend(corrected_chunks[i])

    return merged[: len(lines)], flagged


def resolve_format(input_path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    return "csv" if input_path.suffix.lower() == ".csv" else "txt"


def load_api_key(env_file: Path) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key.strip().strip('"').strip("'")

    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "GEMINI_API_KEY":
            return value.strip().strip('"').strip("'")
    return None


def process_txt(
    input_path: Path,
    output_path: Path,
    correct_chunk: Callable[[str], str],
    chunk_size: int,
    overlap: int,
    sleep_seconds: float,
) -> tuple[int, int]:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    corrected, flagged = correct_lines_in_chunks(
        lines,
        correct_chunk,
        chunk_size,
        overlap,
        sleep_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(corrected) + "\n", encoding="utf-8")
    return len(corrected), len(flagged)


def process_csv(
    input_path: Path,
    output_path: Path,
    correct_chunk: Callable[[str], str],
    text_column: str,
    output_column: str,
    group_column: str,
    group_value: str,
    chunk_size: int,
    overlap: int,
    sleep_seconds: float,
) -> tuple[int, int]:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not fieldnames:
        raise SystemExit("CSV has no header row.")
    if text_column not in fieldnames:
        raise SystemExit(
            f"CSV column '{text_column}' not found. Available columns: {', '.join(fieldnames)}"
        )
    if output_column not in fieldnames:
        fieldnames.append(output_column)
    if group_value and (not group_column or group_column not in fieldnames):
        raise SystemExit(
            f"--group-value requires a valid --group-column. Available columns: {', '.join(fieldnames)}"
        )

    if group_value:
        target = group_value.strip().lower()
        rows = [
            row
            for row in rows
            if str(row.get(group_column, "")).strip().lower() == target
        ]

    if not rows:
        raise SystemExit("No rows matched the selected filter/group.")

    corrected_lines = ["" for _ in rows]
    flagged_total = 0

    if group_column and group_column in fieldnames:
        groups: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            key = (row.get(group_column) or "__EMPTY__").strip()
            groups.setdefault(key, []).append(idx)
    else:
        groups = {"__ALL__": list(range(len(rows)))}

    for _, idxs in groups.items():
        raw_lines = [(rows[i].get(text_column) or "") for i in idxs]
        corrected_group, flagged = correct_lines_in_chunks(
            raw_lines,
            correct_chunk,
            chunk_size,
            overlap,
            sleep_seconds,
        )
        flagged_total += len(flagged)
        for local_i, row_i in enumerate(idxs):
            corrected_lines[row_i] = corrected_group[local_i] if local_i < len(corrected_group) else ""

    count = 0
    for row, corrected in zip(rows, corrected_lines):
        row[output_column] = corrected
        count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return count, flagged_total


def main() -> None:
    args = parse_args()
    if args.provider == "gemini":
        api_key = load_api_key(args.env_file)
        if not api_key:
            raise SystemExit(
                f"Missing GEMINI_API_KEY. Export it or set it in {args.env_file}."
            )
        client = genai.Client(api_key=api_key)
        correct_chunk = lambda text: gemini_correct_chunk(client, args.model, text)
    else:
        correct_chunk = lambda text: ollama_correct_chunk(
            args.ollama_url,
            args.model,
            text,
            args.ollama_timeout,
            args.ollama_retries,
        )

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    mode = resolve_format(args.input, args.format)
    if mode == "txt":
        count, flagged = process_txt(
            args.input,
            args.output,
            correct_chunk,
            args.chunk_size,
            args.overlap,
            args.sleep_seconds,
        )
    else:
        count, flagged = process_csv(
            args.input,
            args.output,
            correct_chunk,
            args.text_column,
            args.output_column,
            args.group_column,
            args.group_value,
            args.chunk_size,
            args.overlap,
            args.sleep_seconds,
        )

    print(f"Wrote {count} corrected rows to: {args.output}")
    print(f"Flagged chunks for manual review: {flagged}")


if __name__ == "__main__":
    main()
