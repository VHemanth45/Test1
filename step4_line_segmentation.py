"""Step 5: Line Segmentation — Deskew, denoise, binarize, seam-carve lines, align with transcripts."""

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d, distance_transform_edt
from scipy.signal import find_peaks
from skimage.filters import threshold_sauvola
from skimage.transform import rotate as sk_rotate
import docx

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
INPUT_DIR = DATA / "needed to linecrop"
OUTPUT_DIR = DATA / "line_crops"
TRANSCRIPT_DIR = ROOT / "Transcripts"
CSV_OUTPUT = DATA / "line_alignment.csv"

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
        "reading_order": ["1.png", "2left.png", "2right.png", "3left.png", "3right.png", "4left.png"],
    },
    "PORCONES.748.6 – 1650": {
        "transcript": "PORCONES.748.6 – 1650 Transcription.docx",
        "source_id": "source6",
        "reading_order": ["1.png", "2.png", "3.png", "4.png"],
    },
}


# ── 1. Deskew ────────────────────────────────────────────────────────────────

def deskew(gray):
    """Deskew via projection-profile variance maximisation (coarse + fine pass)."""
    best_angle, best_var = 0.0, 0.0

    for angle in np.arange(-5.0, 5.5, 0.5):  # coarse
        rot = sk_rotate(gray, angle, resize=True, cval=255, preserve_range=True).astype(np.uint8)
        _, bw = cv2.threshold(rot, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        v = np.var(np.sum(bw, axis=1))
        if v > best_var:
            best_var, best_angle = v, angle

    fine_best = best_angle
    for angle in np.arange(best_angle - 0.5, best_angle + 0.55, 0.1):  # fine
        rot = sk_rotate(gray, angle, resize=True, cval=255, preserve_range=True).astype(np.uint8)
        _, bw = cv2.threshold(rot, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        v = np.var(np.sum(bw, axis=1))
        if v > best_var:
            best_var, fine_best = v, angle

    if abs(fine_best) < 0.05:
        return gray
    return sk_rotate(gray, fine_best, resize=True, cval=255, preserve_range=True).astype(np.uint8)


# ── 2. Denoise ───────────────────────────────────────────────────────────────

def denoise(gray):
    """Median filter + background normalisation + light morphological closing."""
    denoised = median_filter(gray, size=3).astype(np.uint8)

    # Background estimation via large closing
    bg = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(bg > 0, np.clip(denoised.astype(np.float32) / bg * 255, 0, 255), 255).astype(np.uint8)

    return cv2.morphologyEx(norm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))


# ── 3. Binarize ──────────────────────────────────────────────────────────────

def binarize(gray, window_size=25, k=0.2):
    """Sauvola adaptive thresholding. Returns 255=bg, 0=ink."""
    ws = window_size if window_size % 2 == 1 else window_size + 1
    return ((gray > threshold_sauvola(gray, window_size=ws, k=k)).astype(np.uint8) * 255)


# ── 4. Line Segmentation (seam carving) ──────────────────────────────────────

def _find_peaks_and_valleys(binary, sigma=2.0):
    """Find line-centre peaks and inter-line valleys from horizontal projection."""
    h, w = binary.shape
    ink = (binary < 128).astype(np.float32)
    proj = gaussian_filter1d(np.sum(ink, axis=1), sigma=sigma)

    nz = proj[proj > 0]
    if len(nz) == 0:
        return np.array([]), np.array([]), 0

    med_nz = float(np.median(nz))

    # Estimate line spacing via autocorrelation
    centered = proj - np.mean(proj)
    est_h = max(15, h // 30)
    if np.std(centered) > 1e-6:
        ac = np.correlate(centered, centered, mode="full")
        ac = ac[len(ac) // 2:]
        ac /= (ac[0] + 1e-10)
        lo, hi = max(10, h // 100), min(len(ac), h // 3)
        ac_peaks, _ = find_peaks(ac[lo:hi], height=0.05, prominence=0.03)
        if len(ac_peaks) > 0:
            est_h = float(ac_peaks[0] + lo)

    # Detect peaks
    peaks, props = find_peaks(proj, distance=max(8, int(est_h * 0.45)),
                              prominence=med_nz * 0.08, height=med_nz * 0.08)
    if len(peaks) == 0:
        return np.array([]), np.array([]), est_h

    # Filter weak peaks
    if len(peaks) >= 3:
        peaks = peaks[props["peak_heights"] >= np.median(props["peak_heights"]) * 0.20]

    # Gap-fill for large inter-peak gaps
    if len(peaks) >= 2:
        med_sp = float(np.median(np.diff(peaks)))
        extras = []
        for i in range(len(peaks) - 1):
            if peaks[i + 1] - peaks[i] > med_sp * 2.5:
                sub, _ = find_peaks(proj[peaks[i]:peaks[i + 1]],
                                    distance=max(5, int(med_sp * 0.4)), height=med_nz * 0.12)
                for s in sub:
                    gp = int(peaks[i]) + s
                    if gp - peaks[i] > med_sp * 0.4 and peaks[i + 1] - gp > med_sp * 0.4:
                        extras.append(gp)
        if extras:
            peaks = np.sort(np.concatenate([peaks, np.array(extras, dtype=peaks.dtype)]))

    # Valleys between peaks
    valleys = np.array([peaks[i] + np.argmin(proj[peaks[i]:peaks[i + 1]])
                        for i in range(len(peaks) - 1)], dtype=np.intp)

    return peaks, valleys, est_h


def _carve_seam(energy, seed_row, band_half, h, w):
    """Carve a horizontal seam via band-constrained DP."""
    r_min, r_max = max(0, seed_row - band_half), min(h, seed_row + band_half + 1)
    band = energy[r_min:r_max, :].copy()
    bh, bw = band.shape

    M = np.full_like(band, np.inf)
    M[:, 0] = band[:, 0]
    for c in range(1, bw):
        centre = M[:, c - 1]
        up = np.empty(bh); up[0] = np.inf; up[1:] = M[:-1, c - 1]
        down = np.empty(bh); down[-1] = np.inf; down[:-1] = M[1:, c - 1]
        M[:, c] = band[:, c] + np.minimum(np.minimum(centre, up), down)

    seam = np.empty(bw, dtype=np.intp)
    seam[-1] = int(np.argmin(M[:, -1]))
    for c in range(bw - 2, -1, -1):
        r = seam[c + 1]
        best, best_c = r, M[r, c]
        if r > 0 and M[r - 1, c] < best_c:
            best, best_c = r - 1, M[r - 1, c]
        if r < bh - 1 and M[r + 1, c] < best_c:
            best = r + 1
        seam[c] = best

    return seam + r_min


def segment_lines(binary, sigma=2.0, min_h_frac=0.3):
    """Segment text lines using seam carving. Returns list of (seam_above, seam_below)."""
    h, w = binary.shape
    peaks, valleys, est_h = _find_peaks_and_valleys(binary, sigma)
    if len(peaks) == 0:
        return []

    ink = (binary < 128).astype(np.float32)
    energy = ink * 5.0 + distance_transform_edt(ink).astype(np.float32)
    band_half = int(max(8, est_h / 2))

    # Carve seams at valleys
    seams = [_carve_seam(energy, int(v), band_half, h, w) for v in valleys]

    # Top/bottom boundaries
    proj = np.sum(ink, axis=1)
    ink_rows = np.where(proj > proj.max() * 0.02)[0]
    if len(ink_rows) == 0:
        return []
    top = np.full(w, max(0, int(ink_rows[0]) - 2), dtype=np.intp)
    bot = np.full(w, min(h, int(ink_rows[-1]) + 2), dtype=np.intp)
    all_seams = [top] + seams + [bot]

    # Build line regions, filter by min height and ink content
    min_h = int(est_h * min_h_frac)
    lines = []
    for i in range(len(all_seams) - 1):
        sa, sb = all_seams[i], all_seams[i + 1]
        if int(np.median(sb)) - int(np.median(sa)) < min_h:
            continue
        y1, y2 = max(0, int(np.median(sa))), min(h, int(np.median(sb)))
        if np.max(np.sum(ink[y1:y2, :], axis=1)) < w * 0.01:
            continue
        lines.append((sa, sb))

    # Split oversized regions
    if len(lines) >= 3:
        med_line_h = float(np.median([int(np.median(sb)) - int(np.median(sa)) for sa, sb in lines]))
        split = []
        for sa, sb in lines:
            rh = int(np.median(sb)) - int(np.median(sa))
            if rh > med_line_h * 2.0:
                y1, y2 = max(0, int(np.min(sa))), min(h, int(np.max(sb)))
                sub_proj = gaussian_filter1d(np.sum((binary[y1:y2, :] < 128).astype(np.float32), axis=1), 1.0)
                sub_nz = sub_proj[sub_proj > 0]
                if len(sub_nz) > 0:
                    sp, _ = find_peaks(sub_proj, distance=max(5, int(med_line_h * 0.4)),
                                       height=float(np.median(sub_nz)) * 0.10)
                    if len(sp) > 1:
                        sv = [sp[k] + np.argmin(sub_proj[sp[k]:sp[k + 1]]) for k in range(len(sp) - 1)]
                        sub_seams = [sa] + [_carve_seam(energy, y1 + v, max(5, int(med_line_h // 4)), h, w) for v in sv] + [sb]
                        added = False
                        for j in range(len(sub_seams) - 1):
                            if int(np.median(sub_seams[j + 1])) - int(np.median(sub_seams[j])) >= min_h:
                                split.append((sub_seams[j], sub_seams[j + 1]))
                                added = True
                        if added:
                            continue
            split.append((sa, sb))
        lines = split

    return lines


def crop_line(image, sa, sb, pad=2):
    """Crop a line using seam boundaries, whiting out pixels outside seams."""
    h, w = image.shape[:2]
    y_min, y_max = max(0, int(np.min(sa)) - pad), min(h, int(np.max(sb)) + pad)
    crop_h = y_max - y_min
    if crop_h <= 0:
        return image[0:1, :]

    crop = np.full((crop_h, w), 255, dtype=np.uint8)
    for x in range(w):
        yt = max(0, int(sa[x]) - pad) - y_min
        yb = min(crop_h, int(sb[x]) + pad - y_min)
        if yb > yt:
            crop[yt:yb, x] = image[y_min + yt:y_min + yb, x]
    return crop


# ── 5. Transcript parser ─────────────────────────────────────────────────────

def parse_transcript(docx_path):
    """Parse .docx transcript into list of non-empty text lines."""
    lines = []
    for para in docx.Document(str(docx_path)).paragraphs:
        for line in para.text.split("\n"):
            s = line.strip()
            if s:
                lines.append(s)
    return lines


# ── 6. Full pipeline ─────────────────────────────────────────────────────────

def process_image(image_path, output_dir):
    """Preprocess + segment one image → saved line crops. Returns list of output paths."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Pipeline: deskew → denoise → binarize → segment
    deskewed = deskew(gray)
    binary = binarize(denoise(deskewed))
    lines = segment_lines(binary)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, (sa, sb) in enumerate(lines, 1):
        line_img = crop_line(deskewed, sa, sb)
        fname = f"{image_path.stem}_line_{idx:03d}.png"
        out = output_dir / fname
        cv2.imwrite(str(out), line_img)
        results.append(str(out))

    print(f"  {image_path.name}: {len(results)} lines")
    return results


def run_pipeline():
    """Run line segmentation + transcript alignment for all sources."""
    all_rows = []
    fields = ["source", "source_id", "page", "line", "image_path", "ground_truth_text", "status"]

    for folder, cfg in SOURCE_CONFIG.items():
        src_id = cfg["source_id"]
        in_dir = INPUT_DIR / folder
        out_dir = OUTPUT_DIR / folder
        txt_path = TRANSCRIPT_DIR / cfg["transcript"]

        if not in_dir.exists() or not txt_path.exists():
            print(f"  SKIP {folder}: missing input or transcript")
            continue

        # Process images in reading order
        print(f"\n{folder} ({src_id})")
        all_paths = []
        page_labels = []
        for img_name in cfg["reading_order"]:
            img_path = in_dir / img_name
            if not img_path.exists():
                continue
            paths = process_image(img_path, out_dir)
            all_paths.extend(paths)
            page_labels.extend([img_path.stem] * len(paths))

        transcript = parse_transcript(txt_path)
        n = min(len(all_paths), len(transcript))

        for i in range(n):
            all_rows.append({"source": folder, "source_id": src_id, "page": page_labels[i],
                             "line": i + 1, "image_path": all_paths[i],
                             "ground_truth_text": transcript[i], "status": "aligned"})
        for i in range(n, len(all_paths)):
            all_rows.append({"source": folder, "source_id": src_id, "page": page_labels[i],
                             "line": i + 1, "image_path": all_paths[i],
                             "ground_truth_text": "", "status": "unaligned_excess_image"})
        for i in range(n, len(transcript)):
            all_rows.append({"source": folder, "source_id": src_id, "page": "N/A",
                             "line": i + 1, "image_path": "",
                             "ground_truth_text": transcript[i], "status": "unaligned_excess_transcript"})

        print(f"  Aligned: {n}/{len(all_paths)} images, {n}/{len(transcript)} transcript lines")

    # Write CSV
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} rows → {CSV_OUTPUT}")


if __name__ == "__main__":
    run_pipeline()
