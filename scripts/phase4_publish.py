"""
Phase 4: Publish to HuggingFace as a public dataset.

Steps:
  1. Load all passed segments from segments.jsonl
  2. Build a HuggingFace Dataset with audio + metadata columns
  3. Push to hub as public dataset: Itsharshi/indian-tts-dataset
  4. Write dataset/metadata.csv for GitHub
"""

import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).parent.parent
load_dotenv(BASE / ".env")

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO_ID = "Itsharshi/indian-tts-dataset"

SEGMENTS_JSONL = BASE / "metadata" / "segments.jsonl"
DATASET_DIR = BASE / "dataset"
DATASET_DIR.mkdir(exist_ok=True)


def load_passed_segments():
    records = []
    with open(SEGMENTS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            if r["qc_status"] == "passed":
                records.append(r)
    print(f"Loaded {len(records)} passed segments")
    return records


def build_and_push(records):
    from datasets import Dataset, Audio, Features, Value
    import datasets as ds_lib

    print("Building HuggingFace dataset...")

    rows = []
    for r in records:
        wav_path = BASE / r["audio_path"]
        rows.append({
            "segment_id": r["segment_id"],
            "audio": str(wav_path),
            "transcript": r["transcript"],
            "language": r["language"],
            "language_code": r["language_code"],
            "emotion_tag": r["emotion_tag"],
            "source_channel": r["channel"],
            "style": r["style"],
            "duration_s": round(r["duration_s"], 2),
            "mean_dbfs": r.get("mean_dbfs", None),
        })

    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Dataset built: {len(dataset)} rows")
    print(f"Features: {dataset.features}")

    print(f"\nPushing to HuggingFace hub: {HF_REPO_ID} ...")
    dataset.push_to_hub(
        HF_REPO_ID,
        token=HF_TOKEN,
        private=False,
        commit_message="Add Indian TTS dataset: 143 segments, 60 min, Hindi + Indian English",
    )
    print(f"✓ Dataset published at: https://huggingface.co/datasets/{HF_REPO_ID}")
    return dataset


def write_csv(records):
    """Write a flat CSV for GitHub reference."""
    import csv
    csv_path = DATASET_DIR / "metadata.csv"
    fields = ["segment_id", "language", "language_code", "channel", "style",
              "emotion_tag", "duration_s", "mean_dbfs", "qc_status", "transcript"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"✓ CSV saved to: {csv_path}")


def write_dataset_card(records):
    """Write README.md as HuggingFace dataset card."""
    en_segs = [r for r in records if r["language"] == "indian_english"]
    hi_segs = [r for r in records if r["language"] == "hindi"]
    en_dur = sum(r["duration_s"] for r in en_segs) / 60
    hi_dur = sum(r["duration_s"] for r in hi_segs) / 60

    from collections import Counter
    emotion_dist = Counter(r["emotion_tag"] for r in records)
    emotion_table = "\n".join(
        f"| {e} | {c} |" for e, c in emotion_dist.most_common()
    )

    sources_en = sorted(set(r["channel"] for r in en_segs))
    sources_hi = sorted(set(r["channel"] for r in hi_segs))

    card = f"""---
language:
- en
- hi
tags:
- audio
- speech
- tts
- text-to-speech
- indian-english
- hindi
- emotion
license: cc-by-4.0
task_categories:
- text-to-speech
pretty_name: Indian TTS Dataset (Hindi + Indian English)
size_categories:
- 1K<n<10K
---

# Indian TTS Dataset — Hindi + Indian English

A curated, single-speaker TTS training dataset with **{len(records)} segments** (~**{en_dur+hi_dur:.0f} minutes** total) sourced from YouTube, transcribed using [Sarvam AI](https://docs.sarvam.ai) ASR, and annotated with emotion/style tags.

## Dataset Summary

| Split | Segments | Duration |
|-------|----------|----------|
| Indian English | {len(en_segs)} | {en_dur:.1f} min |
| Hindi | {len(hi_segs)} | {hi_dur:.1f} min |
| **Total** | **{len(records)}** | **{en_dur+hi_dur:.1f} min** |

## Audio Specs
- Sample rate: **16 kHz**
- Channels: **Mono**
- Format: WAV (16-bit PCM)
- Segment length: 20–28 seconds

## Columns

| Column | Description |
|--------|-------------|
| `audio` | Audio segment (16kHz mono WAV) |
| `transcript` | ASR transcript (Sarvam saarika:v2.5) |
| `language` | `indian_english` or `hindi` |
| `language_code` | `en-IN` or `hi-IN` |
| `emotion_tag` | Emotion/style label |
| `source_channel` | YouTube channel name |
| `style` | Content style (news_formal, conversational_podcast, etc.) |
| `duration_s` | Segment duration in seconds |
| `mean_dbfs` | Mean audio level in dBFS |

## Emotion Distribution

| Emotion | Count |
|---------|-------|
{emotion_table}

## Sources

**Indian English:** {', '.join(sources_en)}

**Hindi:** {', '.join(sources_hi)}

## Pipeline

1. Audio downloaded via `yt-dlp`, converted to 16kHz mono WAV via `ffmpeg`
2. Silence-based segmentation into 20–28s chunks
3. Transcription via [Sarvam AI](https://sarvam.ai) `saarika:v2.5` ASR
4. Automated QC: SNR check, transcript length, ASR artifact detection
5. Emotion tagging: rule-based classifier on transcript + source style

See the full pipeline code at: https://github.com/itsharshi/indian-tts-dataset

## License

Audio sourced from YouTube under fair use for research/educational purposes.
Dataset annotations (transcripts, tags) are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
"""
    card_path = BASE / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"✓ Dataset card written: {card_path}")
    return card_path


def push_dataset_card():
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(BASE / "README.md"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Add dataset card",
    )
    print(f"✓ Dataset card pushed to HuggingFace")


def main():
    records = load_passed_segments()

    # Write CSV for GitHub
    write_csv(records)

    # Write dataset card
    write_dataset_card(records)

    # Build + push HF dataset
    build_and_push(records)

    # Push dataset card
    push_dataset_card()

    print(f"\n{'='*60}")
    print("PHASE 4 COMPLETE")
    print("="*60)
    print(f"  HuggingFace: https://huggingface.co/datasets/{HF_REPO_ID}")
    print(f"  GitHub:      https://github.com/itsharshi/indian-tts-dataset")


if __name__ == "__main__":
    main()
