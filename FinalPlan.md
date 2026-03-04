## Complete Updated Plan — RenAIssance OCR Pipeline

---

## Overview

```
PDFs → Images → Label → YOLO → Crop → Align → Augment → Split → Tesseract → TrOCR → LLM
```

---

## PHASE 1 — Data Preparation

### Step 1.1 — Convert PDFs to Images
- Convert all 6 printed PDFs to high resolution PNG images
- Use 300 DPI minimum
- Store each PDF's pages in its own named subfolder
- Name pages systematically (source1_page001.png, source1_page002.png etc.)

**Output:** ~180 page images organized by source

---

### Step 1.2 — Annotate in Label Studio
- Install and open Label Studio locally
- Upload all 180 page images
- Create one single label class called **main_text**
- For every page, draw a tight bounding box around the main text body only
- For two column pages draw TWO separate boxes — one per column
- Leave everything else completely unannotated — headers, page numbers, marginalia, decorative elements are all ignored
- Be consistent across all pages

**Tips during annotation:**
- Include decorative large initials if they are part of the main text flow
- Make boxes tight but do not clip the first or last line
- For two column pages make sure left box comes before right box

**Output:** ~180 annotated pages with main_text bounding boxes

---

### Step 1.3 — Export in YOLO Format
- Export your annotations from Label Studio in YOLOv8 format
- This gives you image files and corresponding .txt label files
- Each .txt file contains normalized bounding box coordinates

**Output folder structure:**
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**Split rule during export:**
- 80% of pages go to train
- 20% of pages go to val
- Make sure BOTH splits contain single column AND two column pages
- Make sure BOTH splits contain pages from ALL 6 sources

---

## PHASE 2 — YOLO Fine-tuning

### Step 2.1 — Choose Model
- Use **YOLOv8n** (nano variant) — fast, lightweight, sufficient for one class detection
- Start from pretrained COCO weights — do not train from scratch
- This means the model already knows how to detect objects, you just teach it to find main text

---

### Step 2.2 — Fine-tune the Model
- Train for 50 to 100 epochs
- Monitor validation mAP after every epoch
- Stop training when validation mAP stops improving
- Save the best performing checkpoint

---

### Step 2.3 — Evaluate YOLO Performance
Run on your validation pages and check:

| Metric | What it Means |
|---|---|
| mAP@50 | Box accuracy at 50% overlap threshold |
| Precision | How many detected boxes were correct |
| Recall | How many actual text regions were found |

**Also visually inspect 3-4 pages from each source:**
- Are boxes tight around main text?
- Are headers excluded?
- Are marginalia excluded?
- Are two column pages giving two separate boxes?

If something looks wrong — go back to annotation, fix the problematic pages, retrain.

---

### Step 2.4 — Run on Full Dataset
- Run your fine-tuned YOLO model on all 180 pages
- It outputs bounding box coordinates for main text regions on every page
- Save the bounding box coordinates in a manifest file alongside each page

---

### Step 2.5 — Crop Main Text Regions
- Using the bounding boxes from Step 2.4, crop each detected region from its page image
- Save each crop as a separate PNG file
- Name crops to trace back to their source page

**Output:** Cropped main text region images, one per detected text block

---

## PHASE 3 — Line Segmentation

### Step 3.1 — Split Crops into Individual Lines
- Take each cropped text region from Phase 2
- Use horizontal projection profiling to find gaps between lines
- Split at those gaps to produce individual line strip images
- Each line strip is one image file

**Naming convention:**
```
source1_page001_col1_line001.png
source1_page001_col1_line002.png
source1_page001_col2_line001.png
```

**Output:** Individual line crop images for all 180 pages

---

### Step 3.2 — Verify Line Segmentation
- Manually check 2-3 pages from each source
- Are lines being split cleanly?
- Are any lines being merged together?
- Are any lines being cut in half?

If line segmentation is poor, adjust the projection profile sensitivity before proceeding.

---

## PHASE 4 — Ground Truth Alignment

### Step 4.1 — Parse the Transcript First
Before touching any images, clean and structure each transcript.

**Remove completely:**
- Entire NOTES section at the top
- Page marker lines themselves (PDF p2, PDF p3, PDF p4, etc.)
- Blank lines between paragraphs

**Keep exactly as written:**
- All actual text content
- Existing line breaks in the text (alignment anchors)
- Archaic spellings, accents, and hyphens

**Target parsed structure:**
```
Page 2 → [line1, line2, line3, ...]
Page 3 → [line1, line2, line3, ...]
Page 4 → [line1, line2, line3, ...]
```

---

### Step 4.2 — Identify Alignable Pages
- Use transcript page markers to list exactly which PDF pages are covered
- Cross-check that line crop images exist for those same page numbers
- Treat only this overlap as alignable data

---

### Step 4.3 — Check Line Count Risk Before Matching
Transcript lines and segmented crops may not be perfectly one-to-one.

For each alignable page, record:
- Number of segmented line crops
- Number of transcript lines

If counts are close, proceed with sequential alignment. If counts differ significantly, inspect segmentation or page parsing first.

---

### Step 4.4 — Sequential Alignment (Page by Page)
For each alignable page:
- List transcript lines for that page
- List line crop images in reading order (top→bottom, left column before right column)
- Match sequentially one-to-one

```
Line crop 001  →  "Al"
Line crop 002  →  "INFINITAMENTE AMABLE"
Line crop 003  →  "NIÑO JESUS."
Line crop 004  →  "A Vos, Dulcissimo Niño"
```

Store each aligned pair with:
- Image path
- Ground-truth text
- Source name
- Page number
- Line number

---

### Step 4.5 — Handle Transcript-Specific Special Cases

| Note in Transcript | Alignment/Evaluation Handling |
|---|---|
| u and v used interchangeably | Accept both during evaluation |
| f and s used interchangeably | Accept both during evaluation |
| Accents are inconsistent | Ignore accent mismatches except ñ |
| Some line-end hyphens not present | Treat crop/transcript hyphen mismatch as acceptable |
| ç always means modern z | Keep note for LLM correction stage |

Do not normalize or modernize the transcript text during alignment.

---

### Step 4.6 — Manual Verification Gate
- Visually inspect first 10 aligned pairs per source
- Compare line image and text side by side

Watch for:
- One image line matched to two lines of text (merged segmentation)
- Truncated image text matched to full words (over-aggressive split)
- Completely different content (reading-order issue, often column order)
- Line-end hyphen mismatch (expected in this dataset, not an error)

If more than 3/10 pairs are wrong for a source, stop and fix segmentation or ordering before continuing.

---

### Step 4.7 — Output Artifact
Produce one paired dataset file (CSV or JSON), e.g.:

```
source          | page | line | image_path                        | ground_truth_text
----------------|------|------|-----------------------------------|------------------
buendia_instruc | 2    | 001  | crops/buendia_p002_line001.png   | "Al"
buendia_instruc | 2    | 002  | crops/buendia_p002_line002.png   | "INFINITAMENTE AMABLE"
buendia_instruc | 2    | 003  | crops/buendia_p002_line003.png   | "NIÑO JESUS."
```

This artifact is the single source of truth for augmentation, training, and evaluation.

---

## PHASE 5 — Augmentation

### Step 5.1 — Apply Augmentation to Training Pairs Only
Take each aligned training line image and generate 5 to 8 variations:

| Augmentation | Why |
|---|---|
| Slight rotation ±2° | Simulates tilted scanning |
| Mild gaussian blur | Simulates focus issues |
| Brightness variation | Simulates ink fading |
| Subtle noise | Simulates paper texture |
| Slight elastic distortion | Simulates paper warping |

**Important rules:**
- Apply distortions to images only — text labels stay exactly the same
- Keep distortions subtle — text must still be readable by a human
- Never augment validation pairs — those stay completely pristine

**Output:** Training set multiplied 5-8x in size

---

## PHASE 6 — Train / Val Split

### Step 6.1 — Split Aligned Pairs by Source
- Assign 4 sources worth of aligned pairs → **Training set**
- Assign 2 sources worth of aligned pairs → **Validation set**
- Split by source, NOT by line — this tests real generalization to unseen books

**Final dataset summary:**
```
Training:   ~4 sources × ~3 pages × lines per page × 5-8 augmentations
Validation: ~2 sources × ~3 pages × lines per page (no augmentation)
```

---

## PHASE 7 — Tesseract Baseline

### Step 7.1 — Run Tesseract on Validation Pages
- Run Tesseract with Spanish language settings on your cropped text region images
- No training, no fine-tuning — completely out of the box
- Use PSM 6 (single uniform block of text) for your cropped regions

### Step 7.2 — Calculate Baseline Metrics
Compare Tesseract output against ground truth transcripts:

| Metric | Description |
|---|---|
| **CER** | Character Error Rate — primary metric |
| **WER** | Word Error Rate |
| **chrF** | Character n-gram F-score — good for archaic text |

**Write these numbers down — this is the floor you must beat with TrOCR.**

---

## PHASE 8 — TrOCR Fine-tuning

### Step 8.1 — Start from Pretrained TrOCR
- Use **microsoft/trocr-large-printed** as starting point
- Already pretrained on historical printed documents
- Fine-tune only on your aligned training pairs

### Step 8.2 — Train
- Feed line image crops + ground truth text pairs into TrOCR
- Train for 20 epochs initially
- Check validation CER after every epoch
- Stop when validation CER stops improving

### Step 8.3 — Watch for These Problems

| Problem | Sign | Fix |
|---|---|---|
| Overfitting | Val CER rises while train loss falls | Add more augmentation |
| Underfitting | Both CER stay high | Train more epochs |
| Source bias | Works on 4 sources, fails on 2 | Rebalance training data |

### Step 8.4 — Evaluate and Iterate
- Calculate CER, WER, chrF on validation set
- Look at actual errors — find patterns:
  - Confusing long-s (ſ) with f?
  - Failing on a specific source?
  - Struggling with line beginnings or endings?
- Fix the identified issue and retrain
- Repeat 2 to 3 cycles until CER plateaus

---

## PHASE 9 — LLM Post-Correction

### Step 9.1 — Send Raw TrOCR Output to LLM
- Take raw text output from your best TrOCR checkpoint
- Send to LLM API (Claude or GPT-4o)

### Step 9.2 — Prompt Design
The prompt must tell the LLM to:
- Fix obvious OCR character errors (0→o, 1→l, rn→m, f/ſ confusion)
- Preserve ALL archaic spellings exactly as they appear
- Preserve original punctuation style
- Not modernize any vocabulary
- Return only the corrected text — nothing else

### Step 9.3 — Evaluate Improvement
- Calculate CER, WER, chrF after LLM correction
- Compare against Phase 8 TrOCR only results

---

## PHASE 10 — Final Evaluation

### Fill This Table With Real Numbers

| Stage | CER | WER | chrF |
|---|---|---|---|
| Tesseract Baseline | ? | ? | ? |
| TrOCR Fine-tuned | ? | ? | ? |
| TrOCR + LLM Correction | ? | ? | ? |

This table proves each phase added value and is your core result.

---

## Complete Progress Tracker

```
✅ PDF → Images
⬜ Annotate in Label Studio        ← START HERE
⬜ Export YOLO format
⬜ Fine-tune YOLOv8n
⬜ Evaluate YOLO
⬜ Run on full dataset
⬜ Crop main text regions
⬜ Line segmentation
⬜ Verify line segmentation
⬜ Ground truth alignment
⬜ Verify alignment manually
⬜ Augmentation
⬜ Train/Val split
⬜ Tesseract baseline
⬜ TrOCR fine-tuning
⬜ Evaluate and iterate
⬜ LLM post-correction
⬜ Final evaluation table
```

---

This is your complete plan from start to finish. Start with Label Studio annotation — everything else follows from that. Shall we go through any specific phase in more detail?
