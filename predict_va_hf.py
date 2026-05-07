from __future__ import annotations

import os
import json
import re
import time
import logging
import pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient

# ─────────────
# CONFIGURATION
# ─────────────
HF_TOKEN = os.getenv("HF_TOKEN")
#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
INPUT_CSV = "metadata_2014.csv"
OUTPUT_XLSX = "valence_arousal_predictions.xlsx"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0            # low temperature → more deterministic
RETRY_LIMIT = 3            # retries on transient errors
RETRY_DELAY = 5            # seconds between retries
LIMIT = 50                 # For testing purposes: limit to process the first N rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ──────────────
# PROMPT BUILDER
# ──────────────
SYSTEM_PROMPT = (
    "You are a music emotion expert. "
    "Given song metadata, predict the emotional dimensions valence and arousal "
    "on a scale from 1.0 (very low) to 9.0 (very high). "
    "Valence represents the positivity/negativity of the emotion (1=very negative, 9=very positive). "
    "Arousal represents the energy/intensity level (1=very calm, 9=very energetic). "
    "Respond ONLY with a valid JSON object. "
    "Do NOT include any explanation, text, or formatting outside JSON. "
    'Example: {"valence": 5.0, "arousal": 6.0}'
)


def build_user_prompt(row):
    lastfm = str(row.get("last.fm labels", "")).strip()
    # Trim last.fm labels to avoid token overflow (keep first ~200 chars)
    if len(lastfm) > 200:
        lastfm = lastfm[:200] + "..."

    return (
        f"Artist: {row.get('Artist', 'Unknown')}\n"
        f"Album: {row.get('Album', 'Unknown')}\n"
        f"Track: {row.get('Track', 'Unknown')}\n"
        f"Genre: {row.get('Genre', 'Unknown')}\n"
        f"Last.fm tags: {lastfm if lastfm and lastfm != 'nan' else 'N/A'}\n\n"
        "Based on the metadata above, predict valence and arousal."
    )

# ───────────────
# RESPONSE PARSER
# ───────────────
def parse_response(text):
    text = text.strip()
    # Try direct JSON parse first
    try:
        obj = json.loads(text)
        valence = float(obj["valence"])
        arousal = float(obj["arousal"])
        return clamp(valence), clamp(arousal)
    except Exception:
        pass

    # Fallback: regex extraction
    v_match = re.search(r'"valence"\s*:\s*([0-9]+\.?[0-9]*)', text)
    a_match = re.search(r'"arousal"\s*:\s*([0-9]+\.?[0-9]*)', text)
    if v_match and a_match:
        return clamp(float(v_match.group(1))), clamp(float(a_match.group(1)))

    log.warning("Could not parse response: %s", text[:200])
    return None, None


def clamp(value: float, lo: float = 1.0, hi: float = 9.0) -> float:
    return max(lo, min(hi, value))

# ──────────
# PREDICTION
# ──────────
# Call the LLM and return (valence, arousal) for one row.
def predict_row(client, row):
    user_msg = build_user_prompt(row)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    # log.info("Messages: %s", messages)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat_completion(
                messages=messages,
                model=MODEL_ID,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            reply = response.choices[0].message.content
            # log.info("Reply: %s", reply)
            valence, arousal = parse_response(reply)
            return valence, arousal

        except Exception as exc:
            log.warning("Attempt %d/%d failed for id=%s: %s",
                        attempt, RETRY_LIMIT, row.get("Id", "?"), exc)
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

    return None, None   # all retries exhausted

def load_dataset(filename: str) -> pd.DataFrame:
    df_raw = pd.read_csv(
        filename,
        header=None,
        skiprows=1,   # skip the original header row
        engine="python",
        encoding="utf-8",
    )
    df = pd.DataFrame()
    df["Id"] = df_raw[0]
    df["Artist"] = df_raw[1]
    df["Album"] = df_raw[2]
    df["Track"] = df_raw[3]
    df["Genre"] = df_raw[4]
    df["segment start"] = df_raw[5]
    df["segment end"] = df_raw[6]

    # Merge all tag columns (index 7 onwards) into one space-separated string
    tag_cols = list(range(7, df_raw.shape[1]))
    df["last.fm labels"] = df_raw[tag_cols].apply(
        lambda row: " ".join(
            str(v) for v in row if pd.notna(v) and str(v).strip() not in ("", "nan")
        ),
        axis=1,
    )
    return df


def resolve_output_file(filename):
    # If the output filename already exists, ask the user for a new name.
    while os.path.exists(filename):
        log.warning("Output filename '%s' already exists.", filename)
        print()
        user_input = input("Enter a new filename (or press Enter to overwrite): ").strip()
        print()
        if user_input == "":
            log.info("Overwriting '%s'.", filename)
            break
        # Ensure it ends with .xlsx
        if not user_input.endswith(".xlsx"):
            user_input += ".xlsx"
        filename = user_input
    return filename

# ────
# MAIN
# ────
def main():
    log.info("Loading dataset from %s", INPUT_CSV)
    df = load_dataset(INPUT_CSV)
    output_file = resolve_output_file(OUTPUT_XLSX)

    if LIMIT:
        df = df.head(LIMIT)

    # Validate that env variable is set
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set.")

    client = InferenceClient(token=HF_TOKEN)

    valences, arousals = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        valence, arousal = predict_row(client, row)
        valences.append(valence)
        arousals.append(arousal)

    # Attach predictions
    df["predicted_valence"] = valences
    df["predicted_arousal"] = arousals

    # Summary
    num_failures = df["predicted_valence"].isna().sum()
    log.info("Done. Predictions: %d succeeded, %d failed.", len(df) - num_failures, num_failures)

    # Save to Excel
    df.to_excel(output_file, index=False)
    log.info("Results saved to: %s", output_file)


if __name__ == "__main__":
    main()
