"""
Phase 1: Download YouTube audio and convert to 16kHz mono WAV.
- Reads sources.json
- Downloads audio-only via yt-dlp
- Converts to 16kHz mono WAV via ffmpeg
- Runs a basic SNR check and rejects files below threshold
- Outputs a download_report.json in metadata/
"""

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
SOURCES = BASE / "sources.json"
RAW_DIR = BASE / "audio" / "raw"
PROCESSED_DIR = BASE / "audio" / "processed"
METADATA_DIR = BASE / "metadata"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# SNR threshold in dBFS: files whose RMS is below this are rejected as too quiet/noisy
SNR_REJECT_DBFS = -45.0  # anything quieter than -45 dBFS mean is suspicious


def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def download_audio(entry: dict) -> Path | None:
    """Download audio-only from YouTube, return path to downloaded file."""
    video_id = entry["id"]
    url = entry["url"]
    out_template = str(RAW_DIR / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", out_template,
        "--no-playlist",
        "--quiet",
        "--progress",
    ]

    # Apply download time limit if specified
    limit = entry.get("download_limit_mins")
    if limit:
        cmd += ["--download-sections", f"*0-{int(limit * 60)}"]

    cmd.append(url)

    print(f"\n[{video_id}] Downloading: {url}")
    result = run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ERROR downloading {video_id}: {result.stderr[-300:]}")
        return None

    # Find the downloaded file (extension may vary)
    matches = list(RAW_DIR.glob(f"{video_id}.*"))
    if not matches:
        print(f"  ERROR: no file found after download for {video_id}")
        return None

    return matches[0]


def convert_to_wav(raw_path: Path, video_id: str) -> Path | None:
    """Convert any audio file to 16kHz mono WAV."""
    out_path = PROCESSED_DIR / f"{video_id}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_path),
        "-ar", "16000",       # 16kHz sample rate (TTS standard)
        "-ac", "1",           # mono
        "-sample_fmt", "s16", # 16-bit PCM
        str(out_path),
    ]
    print(f"  Converting to 16kHz WAV -> {out_path.name}")
    result = run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ERROR converting {video_id}: {result.stderr[-300:]}")
        return None
    return out_path


def compute_rms_dbfs(wav_path: Path) -> float:
    """Compute mean RMS level in dBFS using ffmpeg volumedetect."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(wav_path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True
    )
    # parse mean_volume from stderr
    for line in result.stderr.splitlines():
        if "mean_volume" in line:
            # e.g. "  mean_volume: -23.4 dB"
            parts = line.split("mean_volume:")
            if len(parts) == 2:
                return float(parts[1].strip().split()[0])
    return -99.0  # couldn't parse → treat as silent


def get_duration_seconds(wav_path: Path) -> float:
    """Return audio duration in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def snr_check(wav_path: Path, video_id: str) -> tuple[bool, float]:
    """Return (passed, mean_dbfs). Fails if mean level is below threshold."""
    mean_dbfs = compute_rms_dbfs(wav_path)
    passed = mean_dbfs >= SNR_REJECT_DBFS
    status = "OK" if passed else "REJECTED (too quiet/noisy)"
    print(f"  SNR check: mean_volume={mean_dbfs:.1f} dBFS → {status}")
    return passed, mean_dbfs


def process_source(entry: dict) -> dict:
    video_id = entry["id"]
    report = {
        "id": video_id,
        "url": entry["url"],
        "channel": entry["channel"],
        "style": entry["style"],
        "language": "hindi" if video_id.startswith("hi_") else "indian_english",
        "status": "pending",
        "raw_path": None,
        "wav_path": None,
        "duration_seconds": 0,
        "mean_dbfs": None,
        "notes": entry.get("notes", ""),
    }

    # Step 1: download
    raw_path = download_audio(entry)
    if not raw_path:
        report["status"] = "download_failed"
        return report
    report["raw_path"] = str(raw_path)

    # Step 2: convert
    wav_path = convert_to_wav(raw_path, video_id)
    if not wav_path:
        report["status"] = "conversion_failed"
        return report
    report["wav_path"] = str(wav_path)

    # Step 3: duration
    duration = get_duration_seconds(wav_path)
    report["duration_seconds"] = round(duration, 1)
    print(f"  Duration: {duration / 60:.1f} min")

    # Step 4: SNR check
    passed, mean_dbfs = snr_check(wav_path, video_id)
    report["mean_dbfs"] = round(mean_dbfs, 1)
    if not passed:
        report["status"] = "snr_rejected"
        return report

    report["status"] = "ready"
    return report


def main():
    with open(SOURCES) as f:
        sources = json.load(f)

    all_entries = sources["indian_english"] + sources["hindi"]

    # Allow running a single video_id for testing: python phase1_download.py en_01
    if len(sys.argv) > 1:
        target_ids = set(sys.argv[1:])
        all_entries = [e for e in all_entries if e["id"] in target_ids]
        if not all_entries:
            print(f"No matching IDs found for: {target_ids}")
            sys.exit(1)

    reports = []
    total_duration = 0.0

    for entry in all_entries:
        report = process_source(entry)
        reports.append(report)
        if report["status"] == "ready":
            total_duration += report["duration_seconds"]

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    ready = [r for r in reports if r["status"] == "ready"]
    failed = [r for r in reports if r["status"] != "ready"]
    print(f"  Ready:  {len(ready)}/{len(reports)} files")
    print(f"  Total duration of ready files: {total_duration/60:.1f} min")
    if failed:
        print(f"  Failed/rejected:")
        for r in failed:
            print(f"    [{r['id']}] {r['status']}")

    # Save report
    report_path = METADATA_DIR / "download_report.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    # Warn if we're short of 60 min
    if total_duration < 3600:
        shortage = (3600 - total_duration) / 60
        print(f"\nWARNING: {shortage:.1f} min short of 60-min target. Add more sources.")
    else:
        print(f"\nTarget met: {total_duration/3600:.2f}h >= 1h required.")


if __name__ == "__main__":
    main()
