"""
Phase 2: Segment processed WAVs into ~25s chunks, then transcribe each via Sarvam ASR.

Strategy (token-efficient):
  - Use ffmpeg silence detection to split on natural pauses (avoids mid-word cuts)
  - Target 20-28s per segment (fits within Sarvam's 30s hard limit with margin)
  - Only process enough audio per source to hit 60-min total target
  - Skip sources already transcribed (idempotent re-runs)
  - Save segments + transcripts to metadata/segments.jsonl

API usage estimate:
  - 120 segments × 1 ASR call = 120 calls total
  - Each call transcribes ~25s of audio
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE = Path(__file__).parent.parent
load_dotenv(BASE / ".env")

PROCESSED_DIR = BASE / "audio" / "processed"
SEGMENTS_DIR = BASE / "audio" / "segments"
METADATA_DIR = BASE / "metadata"
SEGMENTS_JSONL = METADATA_DIR / "segments.jsonl"
DOWNLOAD_REPORT = METADATA_DIR / "download_report.json"

SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

SARVAM_API_KEY = os.environ["SARVAM_API_KEY"]
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# Targeting 60 min total = 3600s. Split evenly: 30 min English, 30 min Hindi.
TARGET_SECONDS_PER_LANG = 1800

# Segment duration targets
SEG_TARGET_S = 25      # aim for this
SEG_MAX_S = 28         # hard cap before Sarvam's 30s limit
SEG_MIN_S = 8          # discard segments shorter than this

# Silence detection thresholds for natural split points
SILENCE_THRESH_DB = "-35dB"
SILENCE_MIN_DUR = "0.4"        # minimum silence length to split on (seconds)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


# ── silence-based segmentation ────────────────────────────────────────────────

def detect_silences(wav_path: Path) -> list[tuple[float, float]]:
    """Return list of (silence_start, silence_end) in seconds."""
    result = run([
        "ffmpeg", "-i", str(wav_path),
        "-af", f"silencedetect=noise={SILENCE_THRESH_DB}:d={SILENCE_MIN_DUR}",
        "-f", "null", "-"
    ])
    silences = []
    start = None
    for line in result.stderr.splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            start = float(m.group(1))
        m = re.search(r"silence_end: ([\d.]+)", line)
        if m and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    return silences


def compute_split_points(silences: list[tuple[float, float]], total_dur: float, limit_s: float) -> list[float]:
    """
    Given silence intervals, compute cut points that keep segments ~SEG_TARGET_S
    and don't exceed limit_s (both SEG_MAX_S and 30s Sarvam limit).
    Returns list of split timestamps (the end of each segment).
    """
    cut_points = [0.0]
    pos = 0.0

    while pos < limit_s:
        # Look for a silence midpoint between pos+SEG_TARGET_S and pos+SEG_MAX_S
        window_start = pos + SEG_TARGET_S
        window_end = min(pos + SEG_MAX_S, limit_s)

        best_cut = None
        for s_start, s_end in silences:
            midpoint = (s_start + s_end) / 2
            if window_start <= midpoint <= window_end:
                best_cut = midpoint
                break

        if best_cut is None:
            # No silence found in window — hard cut at SEG_TARGET_S
            best_cut = min(pos + SEG_TARGET_S, limit_s)

        cut_points.append(best_cut)
        pos = best_cut

    return cut_points


def extract_segment(wav_path: Path, start: float, end: float, out_path: Path) -> bool:
    """Extract a slice from wav_path to out_path."""
    duration = end - start
    if duration < SEG_MIN_S:
        return False
    result = run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", str(wav_path),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(out_path)
    ])
    return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


# ── ASR ───────────────────────────────────────────────────────────────────────

def transcribe(wav_path: Path, language_code: str, retries: int = 3) -> str | None:
    """Call Sarvam STT on a ≤30s WAV. Returns transcript string or None on failure."""
    for attempt in range(retries):
        try:
            with open(wav_path, "rb") as f:
                resp = requests.post(
                    SARVAM_STT_URL,
                    headers={"api-subscription-key": SARVAM_API_KEY},
                    files={"file": (wav_path.name, f, "audio/wav")},
                    data={
                        "language_code": language_code,
                        "model": "saarika:v2.5",
                    },
                    timeout=30,
                )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("transcript", "").strip()
            elif resp.status_code == 429:
                wait = 2 ** attempt
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ASR error {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.RequestException as e:
            print(f"    Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def load_existing_segment_ids() -> set[str]:
    """Return set of segment IDs already in segments.jsonl (for idempotency)."""
    ids = set()
    if SEGMENTS_JSONL.exists():
        with open(SEGMENTS_JSONL) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["segment_id"])
                except Exception:
                    pass
    return ids


def process_source(entry: dict, lang: str, lang_code: str,
                   budget_remaining: float, existing_ids: set[str],
                   out_file) -> float:
    """
    Segment and transcribe one source. Returns seconds of audio successfully transcribed.
    Writes each segment as a JSONL line to out_file.
    """
    video_id = entry["id"]
    wav_path = PROCESSED_DIR / f"{video_id}.wav"

    if not wav_path.exists():
        print(f"  [{video_id}] WAV not found, skipping")
        return 0.0

    # How much of this file do we need?
    dl_limit = entry.get("download_limit_mins") or 999
    limit_s = min(budget_remaining, dl_limit * 60)
    if limit_s <= 0:
        print(f"  [{video_id}] Budget exhausted, skipping")
        return 0.0

    print(f"\n[{video_id}] {entry['channel']} ({lang})")
    print(f"  Processing up to {limit_s/60:.1f} min")

    # Detect silences
    silences = detect_silences(wav_path)
    print(f"  Found {len(silences)} silence points")

    # Get total duration
    probe = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)])
    total_dur = float(probe.stdout.strip() or "0")

    # Compute split points
    cut_points = compute_split_points(silences, total_dur, min(limit_s, total_dur))

    transcribed_s = 0.0
    seg_count = 0

    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        duration = end - start

        if duration < SEG_MIN_S:
            continue

        seg_id = f"{video_id}_seg{i:03d}"
        if seg_id in existing_ids:
            print(f"  [{seg_id}] already done, skipping")
            transcribed_s += duration
            seg_count += 1
            continue

        seg_path = SEGMENTS_DIR / f"{seg_id}.wav"

        # Extract
        ok = extract_segment(wav_path, start, end, seg_path)
        if not ok:
            print(f"  [{seg_id}] extraction failed")
            continue

        # Transcribe
        print(f"  [{seg_id}] {start:.1f}-{end:.1f}s ({duration:.1f}s) → ASR...", end=" ", flush=True)
        transcript = transcribe(seg_path, lang_code)

        if not transcript:
            print("FAILED")
            seg_path.unlink(missing_ok=True)
            continue

        print(f"OK ({len(transcript)} chars)")

        record = {
            "segment_id": seg_id,
            "source_id": video_id,
            "language": lang,
            "language_code": lang_code,
            "channel": entry["channel"],
            "style": entry["style"],
            "start_s": round(start, 2),
            "end_s": round(end, 2),
            "duration_s": round(duration, 2),
            "audio_path": str(seg_path.relative_to(BASE)),
            "transcript": transcript,
            "emotion_tag": None,  # filled in Phase 3
            "qc_status": "pending",
        }

        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        existing_ids.add(seg_id)

        transcribed_s += duration
        seg_count += 1

        # Throttle slightly to be kind to the API
        time.sleep(0.3)

    print(f"  Done: {seg_count} segments, {transcribed_s/60:.1f} min transcribed")
    return transcribed_s


def main():
    with open(DOWNLOAD_REPORT) as f:
        reports = {r["id"]: r for r in json.load(f) if r["status"] == "ready"}

    with open(BASE / "sources.json") as f:
        sources = json.load(f)

    existing_ids = load_existing_segment_ids()
    print(f"Already have {len(existing_ids)} segments, resuming...")

    # Map source entries by id
    source_map = {}
    for lang, entries in sources.items():
        for e in entries:
            source_map[e["id"]] = (e, lang)

    # Determine language code mapping
    lang_codes = {
        "indian_english": "en-IN",
        "hindi": "hi-IN",
    }

    # Count already-done seconds per language
    done_by_lang = {"indian_english": 0.0, "hindi": 0.0}
    if SEGMENTS_JSONL.exists():
        with open(SEGMENTS_JSONL) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_by_lang[r["language"]] += r["duration_s"]
                except Exception:
                    pass

    print(f"Progress: EN={done_by_lang['indian_english']/60:.1f}min, HI={done_by_lang['hindi']/60:.1f}min")

    with open(SEGMENTS_JSONL, "a") as out_file:
        for lang, entries in sources.items():
            lang_code = lang_codes[lang]
            budget = TARGET_SECONDS_PER_LANG - done_by_lang[lang]
            print(f"\n{'='*60}")
            print(f"Language: {lang} | Remaining budget: {budget/60:.1f} min")
            print("="*60)

            for entry in entries:
                if budget <= 0:
                    print(f"  {lang} target reached, stopping")
                    break
                if entry["id"] not in reports:
                    print(f"  [{entry['id']}] not in download report, skipping")
                    continue
                spent = process_source(
                    entry, lang, lang_code, budget, existing_ids, out_file
                )
                budget -= spent

    # Final summary
    total_by_lang = {"indian_english": 0.0, "hindi": 0.0}
    total_segs = {"indian_english": 0, "hindi": 0}
    with open(SEGMENTS_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
                total_by_lang[r["language"]] += r["duration_s"]
                total_segs[r["language"]] += 1
            except Exception:
                pass

    print(f"\n{'='*60}")
    print("PHASE 2 COMPLETE")
    print("="*60)
    for lang in ["indian_english", "hindi"]:
        mins = total_by_lang[lang] / 60
        segs = total_segs[lang]
        target_ok = "✓" if mins >= 30 else f"✗ ({30-mins:.1f} min short)"
        print(f"  {lang}: {segs} segments, {mins:.1f} min {target_ok}")
    total_mins = sum(total_by_lang.values()) / 60
    print(f"  TOTAL: {total_mins:.1f} min")
    print(f"\nSegments saved to: {SEGMENTS_JSONL}")


if __name__ == "__main__":
    main()
