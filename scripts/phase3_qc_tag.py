"""
Phase 3: Quality Control + Emotion Tagging

QC checks (automated):
  1. Transcript too short (<10 words) → flag as 'short_transcript'
  2. Transcript looks like ASR garbage (very high punct ratio, repeated chars) → flag
  3. Audio RMS too low (<-42 dBFS) → flag as 'low_energy'
  4. Transcript has suspicious non-target language content → flag

Emotion tagging (rule-based hybrid):
  - Uses source style as base emotion
  - Refines with transcript keyword signals
  - Tags: neutral, formal, informational, conversational, excited, emphatic,
           motivational, sad, angry, questioning, storytelling
"""

import json
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).parent.parent
SEGMENTS_JSONL = BASE / "metadata" / "segments.jsonl"
QC_REPORT = BASE / "metadata" / "qc_report.json"

# ── QC thresholds ─────────────────────────────────────────────────────────────
MIN_WORDS = 10
LOW_RMS_DBFS = -42.0
MAX_PUNCT_RATIO_EN = 0.25   # English: >25% punctuation = garbage
MAX_PUNCT_RATIO_HI = 0.55   # Hindi: Devanagari matras/danda inflate ratio naturally


# ── emotion taxonomy ──────────────────────────────────────────────────────────
# Each rule: (regex_pattern, emotion_tag)
# Evaluated in order; first match wins. Falls back to style default.
ENGLISH_RULES = [
    # questioning / analytical
    (r"\b(why|how|what if|is it|are we|isn't it|don't you think|would you say)\b", "questioning"),
    # excited / emphatic
    (r"\b(amazing|incredible|extraordinary|wow|fantastic|brilliant|remarkable|huge|massive|explosive)\b", "excited"),
    # emphatic assertion
    (r"\b(absolutely|definitely|certainly|completely|totally|exactly|precisely|clearly|obviously)\b", "emphatic"),
    # motivational / inspirational
    (r"\b(believe|dream|achieve|potential|success|inspire|courage|never give up|you can|we can|transform|change your)\b", "motivational"),
    # sad / sombre
    (r"\b(tragic|unfortunate|died|death|loss|grief|suffer|crisis|disaster|devastating|heartbreak)\b", "sad"),
    # angry / critical
    (r"\b(outrage|unacceptable|disgrace|shameful|corrupt|failure|blame|wrong|must stop|cannot allow)\b", "angry"),
    # storytelling / narrative
    (r"\b(once upon|years ago|back in|it was|she was|he was|they were|the story|began when|I remember)\b", "storytelling"),
    # formal / news
    (r"\b(according to|reported|announced|statement|government|minister|official|parliament|policy|legislation)\b", "formal"),
]

HINDI_RULES = [
    # questioning
    (r"(क्यों|कैसे|क्या|कब|कहाँ|है ना|है न|नहीं क्या)", "questioning"),
    # excited / emphatic
    (r"(बहुत ही|अद्भुत|शानदार|जबरदस्त|कमाल|अविश्वसनीय|धमाकेदार|बेहतरीन)", "excited"),
    # emphatic
    (r"(बिल्कुल|एकदम|ज़रूर|निश्चित रूप से|स्पष्ट रूप से|सच में)", "emphatic"),
    # motivational
    (r"(सपना|सफलता|आगे बढ़|हिम्मत|विश्वास|बदलाव|प्रेरण|कर सकते|मुमकिन)", "motivational"),
    # sad / sombre
    (r"(दुखद|दुर्भाग्य|मृत्यु|नुकसान|संकट|पीड़ा|आपदा|त्रासदी)", "sad"),
    # angry / critical
    (r"(गलत|भ्रष्ट|शर्मनाक|निंदनीय|बर्दाश्त नहीं|रोकना होगा)", "angry"),
    # storytelling
    (r"(एक बार|कहानी|याद है|उन दिनों|बचपन में|वो वक्त|जब मैं)", "storytelling"),
    # formal / news
    (r"(सरकार|मंत्री|संसद|नीति|रिपोर्ट|बयान|घोषणा|अधिकारी)", "formal"),
]

STYLE_TO_DEFAULT_EMOTION = {
    "conversational_podcast": "conversational",
    "motivational_speech": "motivational",
    "news_formal": "formal",
    "educational": "informational",
    "educational_informational": "informational",
}


def get_rms_dbfs(wav_path: Path) -> float:
    result = subprocess.run(
        ["ffmpeg", "-i", str(wav_path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True
    )
    for line in result.stderr.splitlines():
        if "mean_volume" in line:
            try:
                return float(line.split("mean_volume:")[1].strip().split()[0])
            except Exception:
                pass
    return -99.0


def qc_transcript(transcript: str, language: str) -> list[str]:
    """Return list of QC flag strings. Empty = passed."""
    flags = []
    words = transcript.split()

    if len(words) < MIN_WORDS:
        flags.append(f"short_transcript({len(words)} words)")

    # High punctuation ratio = likely garbage / music / noise
    # Hindi Devanagari script has naturally high non-alphanumeric ratio due to matras
    # so we use a language-specific threshold
    non_space = transcript.replace(" ", "").replace("\n", "")
    if non_space:
        if language == "hindi":
            # For Hindi: only count ASCII punctuation as suspicious (not Devanagari diacritics)
            ascii_punct = sum(1 for c in non_space if ord(c) < 128 and not c.isalnum())
            ratio = ascii_punct / len(non_space)
            threshold = MAX_PUNCT_RATIO_HI
        else:
            punct = sum(1 for c in non_space if not c.isalnum())
            ratio = punct / len(non_space)
            threshold = MAX_PUNCT_RATIO_EN
        if ratio > threshold:
            flags.append(f"high_punct_ratio({ratio:.2f})")

    # Repeated word patterns (stuttering ASR on noise)
    word_list = [w.lower() for w in words]
    if len(word_list) > 4:
        pairs = [(word_list[i], word_list[i+1]) for i in range(len(word_list)-1)]
        repeats = sum(1 for a, b in pairs if a == b)
        if repeats > 3:
            flags.append(f"repetitive_asr({repeats} repeats)")

    # English transcript in Hindi segment (wrong language ASR)
    if language == "hindi":
        # If >60% of words are ASCII-only, likely English content slipped in
        ascii_words = sum(1 for w in words if all(ord(c) < 128 for c in w))
        if len(words) > 0 and ascii_words / len(words) > 0.6:
            flags.append("possible_wrong_language(mostly_ascii)")

    return flags


def tag_emotion(transcript: str, style: str, language: str) -> str:
    """Assign an emotion/style tag based on rules then style default."""
    rules = ENGLISH_RULES if language == "indian_english" else HINDI_RULES
    text = transcript.lower()

    for pattern, emotion in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return emotion

    return STYLE_TO_DEFAULT_EMOTION.get(style, "neutral")


def process_all():
    records = []
    with open(SEGMENTS_JSONL) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Processing {len(records)} segments...")

    qc_flags_summary = {}
    emotion_counts = {}
    flagged = []
    passed = []

    updated = []
    for i, r in enumerate(records):
        seg_id = r["segment_id"]
        wav_path = BASE / r["audio_path"]

        # QC: transcript checks
        flags = qc_transcript(r["transcript"], r["language"])

        # QC: audio energy check
        rms = get_rms_dbfs(wav_path)
        r["mean_dbfs"] = round(rms, 1)
        if rms < LOW_RMS_DBFS:
            flags.append(f"low_energy({rms:.1f}dBFS)")

        # Emotion tag
        emotion = tag_emotion(r["transcript"], r["style"], r["language"])
        r["emotion_tag"] = emotion

        # QC status
        if flags:
            r["qc_status"] = "flagged"
            r["qc_flags"] = flags
            flagged.append(seg_id)
        else:
            r["qc_status"] = "passed"
            r["qc_flags"] = []
            passed.append(seg_id)

        # Track stats
        for flag in flags:
            key = flag.split("(")[0]
            qc_flags_summary[key] = qc_flags_summary.get(key, 0) + 1
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        updated.append(r)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(records)} done...")

    # Write updated JSONL
    with open(SEGMENTS_JSONL, "w") as f:
        for r in updated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write QC report
    report = {
        "total_segments": len(records),
        "passed": len(passed),
        "flagged": len(flagged),
        "flag_breakdown": qc_flags_summary,
        "emotion_distribution": emotion_counts,
        "flagged_ids": flagged,
    }
    with open(QC_REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("QC + TAGGING SUMMARY")
    print("="*60)
    print(f"  Total:   {len(records)}")
    print(f"  Passed:  {len(passed)}")
    print(f"  Flagged: {len(flagged)}")
    if qc_flags_summary:
        print(f"\n  Flag breakdown:")
        for k, v in sorted(qc_flags_summary.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    print(f"\n  Emotion distribution:")
    for e, c in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"    {e:25s}: {c:3d}  {'█' * (c // 2)}")

    # Show flagged segments for manual review
    if flagged:
        print(f"\n  Flagged segments (review these):")
        for r in updated:
            if r["qc_status"] == "flagged":
                print(f"    [{r['segment_id']}] {r['qc_flags']} | {r['transcript'][:60]!r}")

    # Duration of passed segments
    passed_dur = sum(r["duration_s"] for r in updated if r["qc_status"] == "passed")
    en_passed = sum(r["duration_s"] for r in updated if r["qc_status"] == "passed" and r["language"] == "indian_english")
    hi_passed = sum(r["duration_s"] for r in updated if r["qc_status"] == "passed" and r["language"] == "hindi")
    print(f"\n  Duration of PASSED segments:")
    print(f"    Indian English: {en_passed/60:.1f} min")
    print(f"    Hindi:          {hi_passed/60:.1f} min")
    print(f"    Total:          {passed_dur/60:.1f} min")
    print(f"\nQC report saved to: {QC_REPORT}")


if __name__ == "__main__":
    process_all()
