"""
Step 3 — Line Segmentation Pipeline  (Seam-Carving edition)
=============================================================
Phase 3 of FinalPlan.md: Preprocess → Line Segment → Align with Transcripts → CSV

Pipeline per image:
  1. Deskew (on grayscale — projection-profile variance maximisation)
  2. Denoise / Remove Bleed-Through (median filter + morphological ops)
  3. Binarize (Sauvola adaptive thresholding)
  4. Line Segmentation (seam carving — constrained DP between projection-
     profile valleys, energy = ink density + distance transform)
  5. Save individual line crop images (per-column seam boundaries → white-
     padded rectangular strips)

Then:
  6. Parse transcripts (.docx → ordered lines)
  7. Sequential alignment (all image lines in reading order ↔ transcript lines)
  8. Output CSV: source, page, line, image_path, ground_truth_text

Input:  Data/needed to linecrop/<source_folder>/<page>.png
Output: Data/line_crops/<source_folder>/<page>_line_NNN.png
        Data/line_alignment.csv
"""

import os
import csv
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d, distance_transform_edt
from scipy.signal import find_peaks

try:
    import docx
except ImportError:
    raise ImportError("python-docx is required: uv add python-docx")

try:
    from skimage.filters import threshold_sauvola
    from skimage.transform import rotate as sk_rotate
except ImportError:
    raise ImportError("scikit-image is required: uv add scikit-image")


# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
INPUT_DIR = DATA / "needed to linecrop"
OUTPUT_DIR = DATA / "line_crops"
TRANSCRIPT_DIR = ROOT / "Transcripts"
CSV_OUTPUT = DATA / "line_alignment.csv"
DEBUG_DIR = DATA / "line_crops_debug"  # intermediate images for inspection

# ── Source configuration ─────────────────────────────────────────────────────
# Maps each subfolder in "needed to linecrop" to:
#   - transcript: matching .docx filename in Transcripts/
#   - source_id: the sourceN identifier used elsewhere in the project
#   - reading_order: ordered list of image filenames (left col before right col)

SOURCE_CONFIG = {
    "Buendia - Instruccion transcription": {
        "transcript": "Buendia - Instruccion transcription.docx",
        "source_id": "source1",
        "reading_order": ["2.png", "3.png", "4.png"],
    },
    "Covarrubias - Tesoro lengua transcription": {
        "transcript": "Covarrubias - Tesoro lengua transcription.docx",
        "source_id": "source2",
        "reading_order": ["7.png", "8.png", "9.png"],
    },
    "Guardiola - Tratado nobleza transcription": {
        "transcript": "Guardiola - Tratado nobleza transcription.docx",
        "source_id": "source3",
        "reading_order": ["12.png", "13.png", "14.png"],
    },
    "PORCONES.228.38 – 1646": {
        "transcript": "PORCONES.228.38 - 1646 transcription.docx",
        "source_id": "source4",
        "reading_order": ["1.png", "2.png", "3.png", "4.png", "5.png"],
    },
    "PORCONES.23.5 - 1628": {
        "transcript": "PORCONES.23.5 - 1628 transcription.docx",
        "source_id": "source5",
        "reading_order": [
            "1.png",
            "2left.png", "2right.png",
            "3left.png", "3right.png",
            "4left.png",
        ],
    },
    "PORCONES.748.6 – 1650": {
        "transcript": "PORCONES.748.6 – 1650 Transcription.docx",
        "source_id": "source6",
        "reading_order": ["1.png", "2.png", "3.png", "4.png"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DESKEW
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_skew_angle(gray: np.ndarray, angle_range: float = 5.0) -> float:
    """
    Estimate skew angle using horizontal projection profile variance.
    Tries angles in [-angle_range, +angle_range] degrees and picks the one
    that maximises the variance of the horizontal projection (= sharpest
    peaks when lines are perfectly horizontal).
    """
    best_angle = 0.0
    best_variance = 0.0

    # Two-pass: coarse (0.5° steps) then fine (0.1° steps)
    for coarse in np.arange(-angle_range, angle_range + 0.5, 0.5):
        rotated = sk_rotate(gray, coarse, resize=True, mode="constant",
                            cval=255, preserve_range=True).astype(np.uint8)
        # Binarize quickly with Otsu for projection
        _, bw = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj = np.sum(bw, axis=1)
        var = np.var(proj)
        if var > best_variance:
            best_variance = var
            best_angle = coarse

    # Fine pass around the best coarse angle
    fine_best = best_angle
    for fine in np.arange(best_angle - 0.5, best_angle + 0.55, 0.1):
        rotated = sk_rotate(gray, fine, resize=True, mode="constant",
                            cval=255, preserve_range=True).astype(np.uint8)
        _, bw = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj = np.sum(bw, axis=1)
        var = np.var(proj)
        if var > best_variance:
            best_variance = var
            fine_best = fine

    return fine_best


def deskew(gray: np.ndarray) -> np.ndarray:
    """Deskew a grayscale image. Returns corrected grayscale."""
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.05:
        log.debug("  Skew angle ≈ 0°, skipping rotation")
        return gray
    log.info(f"  Deskew angle: {angle:.2f}°")
    corrected = sk_rotate(gray, angle, resize=True, mode="constant",
                          cval=255, preserve_range=True).astype(np.uint8)
    return corrected


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DENOISE / REMOVE BLEED-THROUGH
# ═══════════════════════════════════════════════════════════════════════════════

def denoise_and_remove_bleed(gray: np.ndarray) -> np.ndarray:
    """
    Remove bleed-through and noise from a grayscale document image.

    Strategy:
      1. Median filter (size=3) — removes salt-and-pepper noise while
         preserving edges (important for thin strokes in historical fonts).
      2. Background estimation via large morphological CLOSING — the closed
         image approximates the paper tone (bright background). Dividing the
         original by this normalises uneven illumination and suppresses
         bleed-through (which is lighter than foreground ink).
      3. Light morphological closing (small kernel) to heal thin stroke breaks.
    """
    # Step 1: Light median filter for impulse noise
    denoised = median_filter(gray, size=3).astype(np.uint8)

    # Step 2: Background estimation (bleed-through suppression)
    # Large kernel CLOSING estimates the paper background (bright parts)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_bg)

    # Divide-based normalisation: result = (img / background) * 255
    # This evens out illumination and suppresses light bleed-through
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.where(
            background > 0,
            np.clip((denoised.astype(np.float32) / background.astype(np.float32)) * 255, 0, 255),
            255,
        ).astype(np.uint8)

    # Step 3: Light morphological closing to heal thin stroke breaks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    result = cv2.morphologyEx(normalised, cv2.MORPH_CLOSE, kernel_close)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BINARIZE (Sauvola adaptive thresholding)
# ═══════════════════════════════════════════════════════════════════════════════

def binarize(gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """
    Sauvola binarization — well-suited for historical documents with
    uneven illumination, foxing, and varying ink density.

    Returns a binary image: 255 = background (white), 0 = foreground (ink).
    """
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = (gray > thresh).astype(np.uint8) * 255  # 255=bg, 0=fg
    return binary


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LINE SEGMENTATION (seam carving)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LineRegion:
    """A detected text line region with seam boundaries."""
    y_start: int          # bounding-box top (min of upper seam)
    y_end: int            # bounding-box bottom (max of lower seam)
    line_num: int = 0
    seam_above: Optional[np.ndarray] = field(default=None, repr=False)
    seam_below: Optional[np.ndarray] = field(default=None, repr=False)


def _find_projection_peaks(
    binary: np.ndarray,
    smooth_sigma: float = 2.0,
    peak_height_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Find peak positions (line centres) in the horizontal projection profile.

    Uses autocorrelation for robust line-spacing estimation, then:
      1. Estimate dominant line spacing via autocorrelation of projection
      2. Detect peaks with proper distance constraint derived from (1)
      3. Conservative gap-filling for large inter-peak gaps

    Returns:
        peaks, valleys, est_line_h, image_width
    """
    h, w = binary.shape[:2]
    ink = (binary < 128).astype(np.float32)
    proj = np.sum(ink, axis=1)
    proj_smooth = gaussian_filter1d(proj, sigma=smooth_sigma)

    nz = proj_smooth[proj_smooth > 0]
    if len(nz) == 0:
        return np.array([]), np.array([]), 0.0, 0

    med_nz = float(np.median(nz))

    # ── Phase 1: Estimate line spacing via autocorrelation ────────────────
    # Autocorrelation finds the dominant periodic spacing (= line height)
    # which is robust against noise, decorations, and varying ink density.
    centered = proj_smooth - np.mean(proj_smooth)
    std_c = float(np.std(centered))
    est_line_h = max(15, h // 30)  # fallback

    if std_c > 1e-6:
        autocorr = np.correlate(centered, centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]  # keep positive lags
        autocorr = autocorr / (autocorr[0] + 1e-10)  # normalise

        min_lag = max(10, h // 100)
        max_lag = min(len(autocorr), h // 3)
        ac_region = autocorr[min_lag:max_lag]

        if len(ac_region) > 5:
            ac_peaks, _ = find_peaks(ac_region, height=0.05, prominence=0.03)
            if len(ac_peaks) > 0:
                est_line_h = float(ac_peaks[0] + min_lag)

    log.info(f"    est_line_h = {est_line_h:.0f}px (autocorrelation)")

    # ── Phase 2: Peak detection with proper distance constraint ──────────
    min_peak_dist = max(8, int(est_line_h * 0.45))

    peaks, props = find_peaks(
        proj_smooth,
        distance=min_peak_dist,
        prominence=med_nz * 0.08,
        height=med_nz * 0.08,
    )

    if len(peaks) == 0:
        return np.array([]), np.array([]), est_line_h, 0

    # Filter very weak peaks
    if len(peaks) >= 3:
        med_ph = float(np.median(props["peak_heights"]))
        keep = props["peak_heights"] >= med_ph * peak_height_frac
        peaks = peaks[keep]

    # ── Phase 3: Conservative gap-filling for missed peaks ───────────────
    if len(peaks) >= 2:
        spacings = np.diff(peaks)
        med_spacing = float(np.median(spacings))
        extra_peaks: list[int] = []

        for i in range(len(peaks) - 1):
            gap = int(peaks[i + 1] - peaks[i])
            if gap > med_spacing * 2.5:  # only truly large gaps
                region = proj_smooth[peaks[i]:peaks[i + 1]]
                sub_dist = max(5, int(med_spacing * 0.4))
                sub_peaks, _ = find_peaks(
                    region,
                    distance=sub_dist,
                    height=med_nz * 0.12,
                )
                for sp in sub_peaks:
                    gp = int(peaks[i]) + sp
                    if (gp - peaks[i] > med_spacing * 0.4 and
                            peaks[i + 1] - gp > med_spacing * 0.4):
                        extra_peaks.append(gp)

        if extra_peaks:
            peaks = np.sort(np.concatenate(
                [peaks, np.array(extra_peaks, dtype=peaks.dtype)]
            ))
            log.info(f"    Gap-fill added {len(extra_peaks)} peak(s)")

    # Compute valleys between consecutive peaks
    valleys = np.empty(len(peaks) - 1, dtype=np.intp)
    for i in range(len(peaks) - 1):
        seg = proj_smooth[peaks[i] : peaks[i + 1]]
        valleys[i] = peaks[i] + int(np.argmin(seg))

    return peaks, valleys, est_line_h, w


def _carve_seam(energy: np.ndarray, seed_row: int, band_half: int, h: int, w: int) -> np.ndarray:
    """
    Carve a single horizontal seam through *energy* using band-constrained DP.

    The seam is confined to rows [seed_row - band_half, seed_row + band_half]
    to prevent it from drifting across line boundaries.

    Returns a 1-D array of length *w* giving the row index at each column.
    """
    r_min = max(0, seed_row - band_half)
    r_max = min(h, seed_row + band_half + 1)
    band = energy[r_min:r_max, :].copy()
    bh, bw = band.shape

    # Forward pass (left → right cumulative cost)
    M = np.full_like(band, np.inf)
    M[:, 0] = band[:, 0]
    for c in range(1, bw):
        centre = M[:, c - 1]
        up   = np.empty(bh); up[0] = np.inf; up[1:] = M[:-1, c - 1]
        down = np.empty(bh); down[-1] = np.inf; down[:-1] = M[1:, c - 1]
        M[:, c] = band[:, c] + np.minimum(np.minimum(centre, up), down)

    # Backtrack (right → left)
    seam_band = np.empty(bw, dtype=np.intp)
    seam_band[-1] = int(np.argmin(M[:, -1]))
    for c in range(bw - 2, -1, -1):
        r = seam_band[c + 1]
        best = r
        best_cost = M[r, c]
        if r > 0 and M[r - 1, c] < best_cost:
            best, best_cost = r - 1, M[r - 1, c]
        if r < bh - 1 and M[r + 1, c] < best_cost:
            best = r + 1
        seam_band[c] = best

    return seam_band + r_min  # map back to full-image row indices


def segment_lines(
    binary: np.ndarray,
    min_line_height_frac: float = 0.3,
    smooth_sigma: float = 2.0,
    min_ink_width_frac: float = 0.01,
) -> list[LineRegion]:
    """
    Segment text lines using **seam carving**.

    1. Detect line centres via horizontal projection peaks.
    2. Find valleys (gaps) between consecutive peaks.
    3. Compute an energy map:  ink_density × 5 + distance_transform(ink).
    4. Carve a constrained seam through each valley (±band around the valley)
       using left-to-right DP that favours cutting through whitespace.
    5. Build line regions from consecutive seams; first/last lines use the
       ink extent as boundary.
    6. Filter out regions that are too thin or contain negligible ink.

    Returns:
        List of LineRegion, each carrying its upper and lower seam arrays.
    """
    h, w = binary.shape[:2]

    peaks, valleys, est_line_h, _ = _find_projection_peaks(binary, smooth_sigma)
    if len(peaks) == 0:
        log.warning("  No line peaks found in projection profile!")
        return []

    # ── Energy map ────────────────────────────────────────────────────────
    ink = (binary < 128).astype(np.float32)
    energy = ink * 5.0 + np.asarray(distance_transform_edt(ink), dtype=np.float32)

    band_half = int(max(8, est_line_h // 2))

    # ── Carve seams at every valley ───────────────────────────────────────
    seams: list[np.ndarray] = []
    for vy in valleys:
        seams.append(_carve_seam(energy, int(vy), band_half, h, w))

    # ── Top / bottom boundary seams (flat, at the ink extent) ─────────────
    proj = np.sum(ink, axis=1)
    ink_rows = np.where(proj > proj.max() * 0.02)[0]
    if len(ink_rows) == 0:
        log.warning("  No significant ink rows found!")
        return []

    top_boundary = max(0, int(ink_rows[0]) - 2)
    bot_boundary = min(h, int(ink_rows[-1]) + 2)
    top_seam = np.full(w, top_boundary, dtype=np.intp)
    bot_seam = np.full(w, bot_boundary, dtype=np.intp)

    all_seams = [top_seam] + seams + [bot_seam]

    # ── Build line regions from consecutive seams ─────────────────────────
    min_line_h = int(est_line_h * min_line_height_frac)
    lines: list[LineRegion] = []

    for i in range(len(all_seams) - 1):
        sa = all_seams[i]
        sb = all_seams[i + 1]

        med_top = int(np.median(sa))
        med_bot = int(np.median(sb))
        if med_bot - med_top < min_line_h:
            continue

        # Check region has meaningful ink
        y1, y2 = max(0, med_top), min(h, med_bot)
        region_proj = np.sum(ink[y1:y2, :], axis=1)
        if np.max(region_proj) < w * min_ink_width_frac:
            continue

        lines.append(LineRegion(
            y_start=int(np.min(sa)),
            y_end=int(np.max(sb)),
            seam_above=sa,
            seam_below=sb,
        ))

    # ── Post-process: split oversized regions that contain >1 line ────────
    if len(lines) >= 3:
        heights = [
            int(np.median(r.seam_below)) - int(np.median(r.seam_above))
            for r in lines
            if r.seam_above is not None and r.seam_below is not None
        ]
        if heights:
            median_h = float(np.median(heights))
            split_lines: list[LineRegion] = []

            for region in lines:
                if region.seam_above is None or region.seam_below is None:
                    split_lines.append(region)
                    continue

                reg_h = int(np.median(region.seam_below)) - int(np.median(region.seam_above))
                if reg_h > median_h * 2.0:
                    # Try to find sub-lines within this oversized region
                    y1 = max(0, int(np.min(region.seam_above)))
                    y2 = min(h, int(np.max(region.seam_below)))
                    sub_bin = binary[y1:y2, :]
                    sub_ink = (sub_bin < 128).astype(np.float32)
                    sub_proj = np.sum(sub_ink, axis=1)
                    sub_smooth = gaussian_filter1d(sub_proj, sigma=1.0)

                    sub_nz = sub_smooth[sub_smooth > 0]
                    did_split = False
                    if len(sub_nz) > 0:
                        sub_med = float(np.median(sub_nz))
                        sub_dist = max(5, int(median_h * 0.4))
                        sub_peaks, _ = find_peaks(
                            sub_smooth,
                            distance=sub_dist,
                            height=sub_med * 0.10,
                        )
                        if len(sub_peaks) > 1:
                            sub_valleys: list[int] = []
                            for k in range(len(sub_peaks) - 1):
                                seg = sub_smooth[sub_peaks[k]:sub_peaks[k + 1]]
                                sub_valleys.append(
                                    sub_peaks[k] + int(np.argmin(seg))
                                )

                            sub_band = max(5, int(median_h // 4))
                            carved: list[np.ndarray] = []
                            for sv in sub_valleys:
                                global_seed = y1 + sv
                                carved.append(
                                    _carve_seam(energy, global_seed, sub_band, h, w)
                                )

                            all_sub = [region.seam_above] + carved + [region.seam_below]
                            for j in range(len(all_sub) - 1):
                                sa2 = all_sub[j]
                                sb2 = all_sub[j + 1]
                                if int(np.median(sb2)) - int(np.median(sa2)) >= min_line_h:
                                    split_lines.append(LineRegion(
                                        y_start=int(np.min(sa2)),
                                        y_end=int(np.max(sb2)),
                                        seam_above=sa2,
                                        seam_below=sb2,
                                    ))
                                    did_split = True

                            if did_split:
                                n_parts = sum(
                                    1 for j in range(len(all_sub) - 1)
                                    if int(np.median(all_sub[j + 1])) - int(np.median(all_sub[j])) >= min_line_h
                                )
                                log.info(
                                    f"    Split oversized region ({reg_h}px) "
                                    f"into {n_parts} sub-lines"
                                )
                                continue

                    if not did_split:
                        split_lines.append(region)
                else:
                    split_lines.append(region)

            lines = split_lines

    # Number the lines
    for i, region in enumerate(lines, 1):
        region.line_num = i

    return lines


def crop_line(
    image: np.ndarray,
    region: LineRegion,
    pad_y: int = 2,
) -> np.ndarray:
    """
    Crop a single text line using its seam boundaries.

    For each column the crop spans from seam_above[x]−pad to seam_below[x]+pad.
    Pixels outside the seams are set to white (255) so the result is a clean
    rectangular strip without fragments of neighbouring lines.

    Falls back to a simple rectangular crop if seams are not available.
    """
    h, w = image.shape[:2]

    if region.seam_above is None or region.seam_below is None:
        # Fallback: simple rectangular crop
        y1 = max(0, region.y_start - pad_y)
        y2 = min(h, region.y_end + pad_y)
        return image[y1:y2, :]

    sa = region.seam_above
    sb = region.seam_below

    # Bounding box of the crop
    y_min = max(0, int(np.min(sa)) - pad_y)
    y_max = min(h, int(np.max(sb)) + pad_y)
    crop_h = y_max - y_min
    if crop_h <= 0:
        return image[0:1, :]  # degenerate — 1-px strip

    # Build white canvas, copy only the per-column seam-bounded region
    crop = np.full((crop_h, w), 255, dtype=np.uint8)
    for x in range(w):
        yt = max(0, int(sa[x]) - pad_y) - y_min
        yb = min(crop_h, int(sb[x]) + pad_y - y_min)
        if yb > yt:
            src_top = y_min + yt
            src_bot = y_min + yb
            crop[yt:yb, x] = image[src_top:src_bot, x]

    return crop


def compute_ink_ratio(line_img: np.ndarray) -> float:
    """
    Convert a crop to black/white and compute dark-pixel ratio.

    dark_ratio = (# pixels classified as dark ink) / (total pixels)
    """
    if line_img.size == 0:
        return 0.0

    if len(line_img.shape) == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_pixels = int(np.count_nonzero(bw == 0))
    total_pixels = int(bw.size)
    if total_pixels == 0:
        return 0.0
    return dark_pixels / total_pixels


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRANSCRIPT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_transcript(docx_path: Path) -> list[str]:
    """
    Parse a .docx transcript into an ordered list of text lines.
    Each line corresponds to one physical printed line in the source.

    Extracts all text from paragraphs, splits on newlines,
    and returns non-empty lines in order.
    """
    doc = docx.Document(str(docx_path))
    all_lines: list[str] = []
    for para in doc.paragraphs:
        lines = para.text.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped:
                all_lines.append(stripped)
    return all_lines


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FULL PREPROCESSING PIPELINE (per image)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_and_segment(
    image_path: Path,
    output_dir: Path,
    debug_dir: Path | None = None,
    page_label: str = "page",
) -> list[dict]:
    """
    Run the full preprocessing + line segmentation pipeline on one image.

    Returns a list of dicts:
      {line_num, image_path, y_start, y_end, page_label}
    """
    log.info(f"Processing: {image_path.name}")

    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        log.error(f"  Failed to read image: {image_path}")
        return []

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    log.info(f"  Image size: {gray.shape[1]}×{gray.shape[0]} px")

    # ── Step 1: Deskew ────────────────────────────────────────────────────
    deskewed = deskew(gray)

    # ── Step 2: Denoise / Remove Bleed-Through ───────────────────────────
    denoised = denoise_and_remove_bleed(deskewed)

    # ── Step 3: Binarize ─────────────────────────────────────────────────
    binary = binarize(denoised)

    # Save debug images if requested
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        cv2.imwrite(str(debug_dir / f"{stem}_1_deskewed.png"), deskewed)
        cv2.imwrite(str(debug_dir / f"{stem}_2_denoised.png"), denoised)
        cv2.imwrite(str(debug_dir / f"{stem}_3_binary.png"), binary)

    # ── Step 4: Line Segmentation ────────────────────────────────────────
    lines = segment_lines(binary, smooth_sigma=2.0)
    log.info(f"  Detected {len(lines)} lines")

    # ── Step 5: Save line crops ──────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for idx, region in enumerate(lines, 1):
        # Crop from the DESKEWED grayscale (not binary) for best quality
        line_img = crop_line(deskewed, region, pad_y=2)
        ink_ratio = compute_ink_ratio(line_img)

        fname = f"{image_path.stem}_line_{idx:03d}.png"
        out_path = output_dir / fname
        cv2.imwrite(str(out_path), line_img)

        results.append({
            "line_num": idx,
            "image_path": str(out_path),
            "y_start": region.y_start,
            "y_end": region.y_end,
            "page_label": page_label,
            "ink_ratio": ink_ratio,
        })

    # Save debug: annotated binary with seam boundaries
    if debug_dir is not None:
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for region in lines:
            if region.seam_above is not None:
                for x in range(debug_img.shape[1] - 1):
                    cv2.line(debug_img, (x, int(region.seam_above[x])),
                             (x + 1, int(region.seam_above[x + 1])), (0, 255, 0), 1)
            if region.seam_below is not None:
                for x in range(debug_img.shape[1] - 1):
                    cv2.line(debug_img, (x, int(region.seam_below[x])),
                             (x + 1, int(region.seam_below[x + 1])), (0, 0, 255), 1)
            cv2.putText(debug_img, str(region.line_num),
                        (5, region.y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_4_lines.png"), debug_img)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(save_debug: bool = True):
    """Run the full line segmentation + alignment pipeline."""

    log.info("=" * 70)
    log.info("Step 3 — Line Segmentation Pipeline")
    log.info("=" * 70)

    all_csv_rows: list[dict] = []
    alignment_stats: dict = {}

    for folder_name, config in SOURCE_CONFIG.items():
        source_id = config["source_id"]
        reading_order = config["reading_order"]
        transcript_file = TRANSCRIPT_DIR / config["transcript"]

        log.info("")
        log.info(f"{'─' * 60}")
        log.info(f"Source: {folder_name}  ({source_id})")
        log.info(f"{'─' * 60}")

        input_folder = INPUT_DIR / folder_name
        output_folder = OUTPUT_DIR / folder_name
        debug_folder = DEBUG_DIR / folder_name if save_debug else None

        # ── Check inputs exist ────────────────────────────────────────────
        if not input_folder.exists():
            log.error(f"  Input folder not found: {input_folder}")
            continue
        if not transcript_file.exists():
            log.error(f"  Transcript not found: {transcript_file}")
            continue

        # ── Parse transcript ──────────────────────────────────────────────
        transcript_lines = parse_transcript(transcript_file)
        log.info(f"  Transcript: {len(transcript_lines)} lines")

        # ── Process each image in reading order ──────────────────────────
        all_image_lines: list[dict] = []

        for img_name in reading_order:
            img_path = input_folder / img_name
            if not img_path.exists():
                log.warning(f"  Image not found (skipping): {img_path}")
                continue

            page_label = img_path.stem  # e.g. "2", "3left"
            results = preprocess_and_segment(
                image_path=img_path,
                output_dir=output_folder,
                debug_dir=debug_folder,
                page_label=page_label,
            )
            all_image_lines.extend(results)

        total_image_lines = len(all_image_lines)
        log.info(f"  Total segmented lines (before filter): {total_image_lines}")

        # ── Dynamic blank-crop threshold (min ink_ratio + 0.5%) ──────
        if all_image_lines:
            ink_ratios = [r["ink_ratio"] for r in all_image_lines]
            min_ink = min(ink_ratios)
            dyn_thresh = min_ink + 0.005  # add 0.5%
            log.info(
                f"  Ink ratios — min={min_ink:.4%}, "
                f"dynamic threshold={dyn_thresh:.4%}"
            )

            kept: list[dict] = []
            for r in all_image_lines:
                if r["ink_ratio"] < dyn_thresh:
                    crop_path = Path(r["image_path"])
                    if crop_path.exists():
                        crop_path.unlink()
                    log.info(
                        f"    Discarded: {Path(r['image_path']).name} "
                        f"(ink={r['ink_ratio']:.4%} < {dyn_thresh:.4%})"
                    )
                else:
                    kept.append(r)

            removed = total_image_lines - len(kept)
            if removed > 0:
                log.info(f"  Filtered {removed} blank-like crop(s)")
                # Renumber kept lines sequentially
                for i, r in enumerate(kept, 1):
                    r["line_num"] = i

            all_image_lines = kept
            total_image_lines = len(all_image_lines)

        log.info(f"  Lines after filter: {total_image_lines}")
        log.info(f"  Transcript lines:   {len(transcript_lines)}")

        # ── Alignment ────────────────────────────────────────────────────
        diff = total_image_lines - len(transcript_lines)
        if diff != 0:
            log.warning(
                f"  ⚠ LINE COUNT MISMATCH: {total_image_lines} image lines "
                f"vs {len(transcript_lines)} transcript lines (diff={diff:+d})"
            )
            log.warning(
                "  Will align up to min(image_lines, transcript_lines). "
                "Excess lines will be marked as UNALIGNED."
            )

        n_aligned = min(total_image_lines, len(transcript_lines))

        for i in range(n_aligned):
            img_info = all_image_lines[i]
            all_csv_rows.append({
                "source": folder_name,
                "source_id": source_id,
                "page": img_info["page_label"],
                "line": img_info["line_num"],
                "image_path": img_info["image_path"],
                "ground_truth_text": transcript_lines[i],
                "status": "aligned",
            })

        # Mark excess image lines as unaligned
        for i in range(n_aligned, total_image_lines):
            img_info = all_image_lines[i]
            all_csv_rows.append({
                "source": folder_name,
                "source_id": source_id,
                "page": img_info["page_label"],
                "line": img_info["line_num"],
                "image_path": img_info["image_path"],
                "ground_truth_text": "",
                "status": "unaligned_excess_image",
            })

        # Mark excess transcript lines
        for i in range(n_aligned, len(transcript_lines)):
            all_csv_rows.append({
                "source": folder_name,
                "source_id": source_id,
                "page": "N/A",
                "line": i + 1,
                "image_path": "",
                "ground_truth_text": transcript_lines[i],
                "status": "unaligned_excess_transcript",
            })

        alignment_stats[folder_name] = {
            "source_id": source_id,
            "image_lines": total_image_lines,
            "transcript_lines": len(transcript_lines),
            "aligned": n_aligned,
            "diff": diff,
        }

    # ── Write CSV ────────────────────────────────────────────────────────
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source", "source_id", "page", "line",
        "image_path", "ground_truth_text", "status",
    ]
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    log.info("")
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 70)
    log.info(f"CSV written to: {CSV_OUTPUT}")
    log.info(f"Total rows: {len(all_csv_rows)}")
    log.info(f"Line crops in: {OUTPUT_DIR}")
    if save_debug:
        log.info(f"Debug images in: {DEBUG_DIR}")

    # ── Summary table ────────────────────────────────────────────────────
    log.info("")
    log.info(f"{'Source':<45} {'Img Lines':>10} {'Txt Lines':>10} {'Aligned':>8} {'Diff':>6}")
    log.info("─" * 85)
    for name, stats in alignment_stats.items():
        marker = " ⚠" if stats["diff"] != 0 else " ✓"
        log.info(
            f"{name:<45} {stats['image_lines']:>10} "
            f"{stats['transcript_lines']:>10} {stats['aligned']:>8} "
            f"{stats['diff']:>+6}{marker}"
        )

    # ── Save alignment stats as JSON ─────────────────────────────────────
    stats_path = DATA / "line_alignment_stats.json"
    with open(stats_path, "w") as f:
        json.dump(alignment_stats, f, indent=2)
    log.info(f"\nAlignment stats: {stats_path}")

    return alignment_stats


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_pipeline(save_debug=True)
