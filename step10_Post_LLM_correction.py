import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError
import json
import time
from tqdm import tqdm
import jiwer
import nltk
from nltk.translate.chrf_score import sentence_chrf
from dotenv import load_dotenv
import os

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_CSV = "kraken_eval_results.csv"
OUTPUT_CSV = "llm_corrected_results_final.csv"

# OPTIMIZED FOR FREE TIER LIMITS
CHUNK_SIZE = 35 # Processes 35 lines at a time (Only takes 9 requests total for 296 lines)
MODEL_ID = 'gemini-2.5-flash'
REQUEST_DELAY = 8.0 # Waits 8 seconds between chunks to guarantee we never hit the RPM ceiling

client = genai.Client(api_key=GEMINI_API_KEY)

# THE STRICT, NO-HALLUCINATION PROMPT
SYSTEM_PROMPT = """
You are a highly strict OCR correction engine for historical Spanish.
You will receive a JSON array of OCR-detected text strings.
Return a JSON array of exactly the same length containing the corrected strings.

CRITICAL DIRECTIVES:
1. NO HALLUCINATIONS: Do NOT rewrite sentences or guess entire phrases based on context. You must ONLY correct character misrecognitions (e.g., swapping 'f'/'s', 'I'/'l', 'N'/'M'). If a line is garbled, output the phonetically/visually closest words and nothing else.
2. NO MODERNIZATION OF ACCENTS: DO NOT add modern tildes (á, é, í, ó, ú) to the text. Keep the exact accents as they appear in the OCR. The ONLY accent you may add/preserve is 'ñ'. Keep historical spelling exactly as it appears (e.g., leave "felizidad" as "felizidad", "assi" as "assi", do NOT modernize them to "felicidad" or "así").
3. NO WORD COMPLETION: If a string ends in a fragment or a hyphen (e.g., "asi", "assis-", "favo-", "crian-"), LEAVE IT CUT OFF exactly as it is. DO NOT complete the word across lines (e.g., do NOT write "asistir" or "favorecer").
4. ALLOWED HISTORICAL FIXES ONLY:
   - 'ç' -> 'z'
   - 'i' (as j) -> 'j'
   - 'u' (as v/b) -> 'v/b'
   - 'z' (as c) -> 'c'
   - Expand standalone 'q' -> 'que'
   - Restore missing 'n'/'m' for nasal caps.
5. SPECIAL LIGATURES: Replace out-of-place 'ee', 'de', 'es' with '&', and 'S.' or 'E.' with '§' in Latin contexts.
6. PRESERVE PUNCTUATION: Keep all commas, periods, and colons exactly where they are.

You must output a JSON array of strings exactly mapping to the input strings. Do NOT combine lines.
"""

def process_chunk_with_llm(chunk_lines, max_retries=5):
    prompt_input = json.dumps(chunk_lines, ensure_ascii=False)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=f"{SYSTEM_PROMPT}\n\nInput JSON Array:\n{prompt_input}",
                config=types.GenerateContentConfig(
                    temperature=0.0, # Forces deterministic, uncreative output
                    response_mime_type="application/json"
                )
            )

            corrected_chunk = json.loads(response.text)
            if isinstance(corrected_chunk, list) and len(corrected_chunk) == len(chunk_lines):
                return corrected_chunk

        except APIError as e:
            if e.code == 429: # Resource Exhausted / Rate Limit
                wait_time = (2 ** attempt) + 10
                print(f"\n[Rate Limit 429] Sleeping for {wait_time}s... (Attempt {attempt+1})")
                time.sleep(wait_time)
            else:
                print(f"\n[API Error] {e.message}. Retrying...")
                time.sleep(5)

        except Exception as e:
            print(f"\n[Error] {e}. Retrying...")
            time.sleep(5)

    print("\n[Fallback] Max retries reached for chunk. Returning original OCR text.")
    return chunk_lines

def main():
    print("Loading aligned data...")
    df = pd.read_csv(INPUT_CSV).dropna(subset=['ground_truth_text', 'crnn_prediction']).copy()

    # Sort to ensure contiguous lines stay together
    df = df.sort_values('image_path').reset_index(drop=True)

    ocr_lines = df['crnn_prediction'].tolist()
    all_corrected_lines = []

    total_chunks = (len(ocr_lines) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Processing {len(df)} lines in {total_chunks} chunks using {MODEL_ID}...")

    for i in tqdm(range(0, len(ocr_lines), CHUNK_SIZE)):
        chunk = ocr_lines[i:i + CHUNK_SIZE]
        all_corrected_lines.extend(process_chunk_with_llm(chunk))
        time.sleep(REQUEST_DELAY)

    df['llm_correction'] = all_corrected_lines

    print("\nCalculating Metrics (CER, WER, chrF)...")
    gt = df['ground_truth_text'].tolist()
    ocr = df['crnn_prediction'].tolist()
    llm = df['llm_correction'].tolist()

    # Calculate Line-by-Line Metrics
    df['cer_before'] = [jiwer.cer(g, o) if g.strip() else 0 for g, o in zip(gt, ocr)]
    df['cer_after'] = [jiwer.cer(g, l) if g.strip() else 0 for g, l in zip(gt, llm)]

    df['wer_before'] = [jiwer.wer(g, o) if g.strip() else 0 for g, o in zip(gt, ocr)]
    df['wer_after'] = [jiwer.wer(g, l) if g.strip() else 0 for g, l in zip(gt, llm)]

    df['chrf_before'] = [sentence_chrf(g, o) for g, o in zip(gt, ocr)]
    df['chrf_after'] = [sentence_chrf(g, l) for g, l in zip(gt, llm)]

    # Global Corpus Metrics Outputs
    print("\n" + "="*40)
    print("         PERFORMANCE COMPARISON         ")
    print("="*40)
    print(f"OCR Global CER:  {jiwer.cer(gt, ocr):.4f}")
    print(f"LLM Global CER:  {jiwer.cer(gt, llm):.4f}")
    print("-" * 40)
    print(f"OCR Global WER:  {jiwer.wer(gt, ocr):.4f}")
    print(f"LLM Global WER:  {jiwer.wer(gt, llm):.4f}")
    print("="*40)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()