"""Generate PDF report for TTS dataset assignment submission."""

import json
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

BASE = Path(__file__).parent.parent
SEGMENTS_JSONL = BASE / "metadata" / "segments.jsonl"
QC_REPORT = BASE / "metadata" / "qc_report.json"
OUTPUT_PDF = BASE / "TTS_Dataset_Report.pdf"

# ── colors ────────────────────────────────────────────────────────────────────
SARVAM_BLUE   = colors.HexColor("#1A56DB")
SARVAM_DARK   = colors.HexColor("#111827")
SARVAM_GREY   = colors.HexColor("#6B7280")
SARVAM_LIGHT  = colors.HexColor("#F3F4F6")
SARVAM_GREEN  = colors.HexColor("#059669")
SARVAM_ORANGE = colors.HexColor("#D97706")
WHITE         = colors.white


def load_data():
    records = []
    with open(SEGMENTS_JSONL) as f:
        for line in f:
            records.append(json.loads(line))
    passed  = [r for r in records if r["qc_status"] == "passed"]
    en      = [r for r in passed  if r["language"] == "indian_english"]
    hi      = [r for r in passed  if r["language"] == "hindi"]
    return records, passed, en, hi


def make_styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle("ReportTitle",
        parent=base["Title"], fontSize=24, textColor=SARVAM_DARK,
        spaceAfter=4, fontName="Helvetica-Bold")

    subtitle = ParagraphStyle("Subtitle",
        parent=base["Normal"], fontSize=11, textColor=SARVAM_GREY,
        spaceAfter=20, alignment=TA_CENTER)

    h1 = ParagraphStyle("H1",
        parent=base["Heading1"], fontSize=14, textColor=SARVAM_BLUE,
        spaceBefore=18, spaceAfter=6, fontName="Helvetica-Bold",
        borderPad=0)

    h2 = ParagraphStyle("H2",
        parent=base["Heading2"], fontSize=11, textColor=SARVAM_DARK,
        spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")

    body = ParagraphStyle("Body",
        parent=base["Normal"], fontSize=9.5, textColor=SARVAM_DARK,
        leading=15, spaceAfter=6, alignment=TA_JUSTIFY)

    bullet = ParagraphStyle("Bullet",
        parent=base["Normal"], fontSize=9.5, textColor=SARVAM_DARK,
        leading=14, spaceAfter=3, leftIndent=14, bulletIndent=4)

    code = ParagraphStyle("Code",
        parent=base["Code"], fontSize=8, textColor=SARVAM_DARK,
        backColor=SARVAM_LIGHT, leading=12, leftIndent=8, rightIndent=8,
        spaceBefore=4, spaceAfter=4, fontName="Courier")

    caption = ParagraphStyle("Caption",
        parent=base["Normal"], fontSize=8, textColor=SARVAM_GREY,
        alignment=TA_CENTER, spaceAfter=8)

    return title, subtitle, h1, h2, body, bullet, code, caption


def stat_table(rows, col_widths=None):
    """Build a compact stats table."""
    col_widths = col_widths or [8*cm, 8*cm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), SARVAM_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, SARVAM_LIGHT]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#E5E7EB")),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t


def build_report():
    records, passed, en, hi = load_data()
    with open(QC_REPORT) as f:
        qc = json.load(f)

    en_dur = sum(r["duration_s"] for r in en) / 60
    hi_dur = sum(r["duration_s"] for r in hi) / 60
    total_dur = en_dur + hi_dur

    emotion_en = Counter(r["emotion_tag"] for r in en)
    emotion_hi = Counter(r["emotion_tag"] for r in hi)

    dbfs_vals = [r["mean_dbfs"] for r in passed if r.get("mean_dbfs") is not None]
    durs = [r["duration_s"] for r in passed]

    T, SUB, H1, H2, BODY, BULL, CODE, CAP = make_styles()

    doc = SimpleDocTemplate(str(OUTPUT_PDF), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm)

    story = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Indian TTS Dataset", T))
    story.append(Paragraph(
        "Pipeline Report &amp; Quality Analysis<br/>"
        "<font color='#6B7280' size='9'>Submitted to Sarvam AI · June 2026</font>", SUB))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SARVAM_BLUE, spaceAfter=16))

    # ── QUICK STATS BANNER ────────────────────────────────────────────────────
    banner = Table([[
        Paragraph(f"<b>{len(passed)}</b><br/><font size='8' color='#6B7280'>Segments</font>", CAP),
        Paragraph(f"<b>{total_dur:.1f} min</b><br/><font size='8' color='#6B7280'>Total Audio</font>", CAP),
        Paragraph(f"<b>{en_dur:.1f} min</b><br/><font size='8' color='#6B7280'>Indian English</font>", CAP),
        Paragraph(f"<b>{hi_dur:.1f} min</b><br/><font size='8' color='#6B7280'>Hindi</font>", CAP),
        Paragraph(f"<b>{qc['passed']/qc['total_segments']*100:.0f}%</b><br/><font size='8' color='#6B7280'>QC Pass Rate</font>", CAP),
        Paragraph(f"<b>{len(emotion_en)+len(emotion_hi)}</b><br/><font size='8' color='#6B7280'>Emotion Tags</font>", CAP),
    ]], colWidths=[2.7*cm]*6)
    banner.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), SARVAM_LIGHT),
        ("BOX",         (0,0),(-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 13),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LINEAFTER",   (0,0),(-2,-1), 0.5, colors.HexColor("#D1D5DB")),
    ]))
    story.append(banner)
    story.append(Spacer(1, 14))

    # ── 1. WHAT I BUILT ───────────────────────────────────────────────────────
    story.append(Paragraph("1. What I Built", H1))
    story.append(Paragraph(
        "A fully automated TTS training dataset pipeline that sources real speech from YouTube, "
        "transcribes it using Sarvam AI's ASR API, applies quality filtering, and annotates each "
        "segment with emotion and style tags. The final dataset contains <b>143 audio segments totalling "
        f"{total_dur:.1f} minutes</b> — {en_dur:.1f} min of Indian English and {hi_dur:.1f} min of Hindi — "
        "published as a public HuggingFace dataset.", BODY))

    story.append(Paragraph("Pipeline overview:", H2))
    steps = [
        ("Phase 1 — Source curation &amp; download",
         "Hand-picked 8 YouTube sources (4 per language) prioritising single-speaker, "
         "clean audio: podcasts, news anchors, educational lectures, and stage talks. "
         "Downloaded audio-only via yt-dlp, converted to 16 kHz mono WAV with ffmpeg. "
         "Applied an SNR gate (reject if mean &lt; −45 dBFS). All 8 sources passed."),
        ("Phase 2 — Segmentation &amp; ASR transcription",
         "Used ffmpeg silence detection (−35 dB, min 0.4 s) to find natural pause points, "
         "then cut into 22–28 s chunks — safely below Sarvam's 30 s per-call limit. "
         "Each segment was transcribed with Sarvam saarika:v2.5 (en-IN / hi-IN). "
         "144 API calls, 0 failures."),
        ("Phase 3 — Quality control &amp; emotion tagging",
         "Automated QC checks: minimum transcript length, ASR repetition artifacts, "
         "audio energy. 1 segment rejected (ASR glitch: 'of of of of'). "
         "Emotion tagging via rule-based classifier using transcript keywords + source style. "
         "11 distinct emotion/style tags applied."),
        ("Phase 4 — Publish",
         "Dataset pushed to HuggingFace as a public datasets repo with Audio column "
         "(16 kHz, encoded as parquet). Full pipeline code pushed to GitHub."),
    ]
    for title_text, desc in steps:
        story.append(Paragraph(f"<b>{title_text}:</b> {desc}", BULL))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Links:", H2))
    story.append(Paragraph("• HuggingFace: <font color='#1A56DB'>https://huggingface.co/datasets/Itsharshi/indian-tts-dataset</font>", BULL))
    story.append(Paragraph("• GitHub: <font color='#1A56DB'>https://github.com/itsharshi/indian-tts-dataset</font>", BULL))
    story.append(Spacer(1, 4))

    # ── 2. DATA SOURCES ───────────────────────────────────────────────────────
    story.append(Paragraph("2. Data Sources", H1))

    src_rows = [["Channel", "Language", "Style", "Segments", "Duration"]]
    src_map = {}
    for r in passed:
        ch = r["channel"]
        if ch not in src_map:
            src_map[ch] = {"lang": r["language"], "style": r["style"], "segs": 0, "dur": 0}
        src_map[ch]["segs"] += 1
        src_map[ch]["dur"] += r["duration_s"]
    lang_label = {"indian_english": "Indian English", "hindi": "Hindi"}
    style_label = {
        "conversational_podcast": "Podcast", "news_formal": "News",
        "educational": "Educational", "educational_informational": "Educational",
        "motivational_speech": "Motivational"
    }
    for ch, d in sorted(src_map.items(), key=lambda x: -x[1]["dur"]):
        src_rows.append([ch, lang_label[d["lang"]], style_label.get(d["style"], d["style"]),
                         str(d["segs"]), f"{d['dur']/60:.1f} min"])
    story.append(stat_table(src_rows, [5.5*cm, 3*cm, 3*cm, 2*cm, 2*cm]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Source selection rationale: I prioritised sources where the speaker is clearly "
        "alone (no guests mid-segment), uses a close/studio microphone, and speaks "
        "naturally at a conversational pace. News anchors give formal style with very "
        "clean diction. Podcasts like Seen &amp; Unseen (Amit Varma) provide natural "
        "Indian English prosody. Dhruv Rathee provides studio-quality Hindi with varied "
        "emotional range. Josh Talks Hindi adds motivational register.", BODY))

    # ── 3. QUALITY OBSERVATIONS ───────────────────────────────────────────────
    story.append(Paragraph("3. Quality Observations &amp; Decisions", H1))

    story.append(Paragraph("Audio quality metrics (passed segments):", H2))
    aq_rows = [
        ["Metric", "Value"],
        ["Sample rate", "16 kHz mono (TTS standard)"],
        ["Segment length", f"{min(durs):.0f}–{max(durs):.0f} s (avg {sum(durs)/len(durs):.1f} s)"],
        ["Mean audio level range", f"{min(dbfs_vals):.1f} to {max(dbfs_vals):.1f} dBFS"],
        ["Average audio level", f"{sum(dbfs_vals)/len(dbfs_vals):.1f} dBFS"],
        ["QC pass rate", f"{qc['passed']}/{qc['total_segments']} ({qc['passed']/qc['total_segments']*100:.1f}%)"],
        ["Segments rejected", f"{len(records)-len(passed)} (ASR repetition artifact)"],
    ]
    story.append(stat_table(aq_rows))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Key quality decisions made:", H2))
    decisions = [
        "<b>Silence-based cuts over fixed-length cuts:</b> I used ffmpeg silencedetect to find "
        "natural pause points rather than slicing every 25 s. This avoids mid-word or "
        "mid-sentence cuts, which produce unnatural TTS training samples.",
        "<b>25 s target, 28 s cap:</b> Sarvam's ASR has a hard 30 s limit. I set a 2 s safety "
        "margin to prevent failures from timing imprecision in silence detection.",
        "<b>SNR gate at −45 dBFS:</b> Any file with mean level below −45 dBFS is rejected "
        "before segmentation. This catches silent downloads or music-only files early.",
        "<b>Hindi punctuation ratio fix:</b> Initial QC flagged all 72 Hindi segments as "
        "having 'high punctuation ratio.' Investigation showed Devanagari matras and the "
        "danda (।) are counted as non-alphanumeric by Python's isalnum(). I switched to "
        "counting only ASCII punctuation for Hindi — a language-aware fix that eliminated "
        "false positives without hiding real issues.",
        "<b>Podcast de-guest filtering:</b> The Seen &amp; Unseen is an interview podcast. "
        "I deliberately capped the download at 50 minutes (the first 50 min of a ~3h episode) "
        "where the host speaks mostly solo in the intro. Diarization would help here given more time.",
        "<b>Manual rejection:</b> en_01_seg056 was rejected for repetitive ASR output "
        "('of of of of') — a recognisable artifact when the ASR encounters overlapping "
        "speech or a noisy segment.",
    ]
    for d in decisions:
        story.append(Paragraph(f"• {d}", BULL))
    story.append(Spacer(1, 4))

    # ── 4. EMOTION DISTRIBUTION ───────────────────────────────────────────────
    story.append(Paragraph("4. Emotion / Style Tag Distribution", H1))

    em_rows = [["Emotion Tag", "Indian English", "Hindi", "Total"]]
    all_emotions = sorted(set(list(emotion_en.keys()) + list(emotion_hi.keys())))
    for e in sorted(all_emotions, key=lambda x: -(emotion_en.get(x,0)+emotion_hi.get(x,0))):
        en_c = emotion_en.get(e, 0)
        hi_c = emotion_hi.get(e, 0)
        em_rows.append([e, str(en_c) if en_c else "—", str(hi_c) if hi_c else "—", str(en_c+hi_c)])
    em_rows.append(["Total", str(len(en)), str(len(hi)), str(len(passed))])
    story.append(stat_table(em_rows, [5*cm, 3.5*cm, 3.5*cm, 3.5*cm]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "The dominant tag is <i>questioning</i> — reflecting the analytical/interrogative "
        "style common in both Indian English podcasts (Seen &amp; Unseen asks rhetorical Qs "
        "frequently) and Hindi news commentary (Dhruv Rathee). The <i>formal</i> and "
        "<i>informational</i> tags cover news anchors and educational content. "
        "<i>Motivational</i> comes from Josh Talks and TEDx. Rare tags (sad, angry, "
        "excited, storytelling) appear in specific segments where the speaker's language "
        "clearly shifts register — these are valuable precisely because they're scarce.", BODY))

    # ── 5. ITERATIONS ─────────────────────────────────────────────────────────
    story.append(Paragraph("5. Iterations to Improve Data Quality", H1))

    iters = [
        ("Iteration 1 — Fixed deprecated Sarvam model",
         "Initial calls with model=saarika:v2 returned a 422 deprecation error. "
         "Switched to saarika:v2.5 which is the current production model."),
        ("Iteration 2 — Discovered 30 s hard limit",
         "Initial test with a 5-minute file returned: 'Audio duration exceeds the "
         "maximum limit of 30 seconds.' Redesigned segmentation to target 25 s with "
         "28 s cap before ASR calls."),
        ("Iteration 3 — Hindi QC false positives",
         "First QC run flagged all 72 Hindi segments for 'high_punct_ratio'. "
         "Root cause: Devanagari Unicode chars (matras, chandrabindu, danda) are not "
         "alphanumeric in Python. Fixed by using ASCII-only punctuation count for Hindi. "
         "Result: 0 false positives, 1 true positive (repetitive ASR)."),
        ("Iteration 4 — Silence detection sensitivity",
         "Initial silence threshold of −30 dB produced only 1 silence point in Dhruv "
         "Rathee audio (continuous fast speech). Relaxed to −35 dB minimum 0.4 s silence, "
         "which gave natural cuts without over-fragmenting. Fall-back to hard cut at 25 s "
         "handles segments with no detected silence."),
        ("Iteration 5 — Source vetting",
         "The Aaj Tak channel initially showed a panel discussion in preview. "
         "Verified by checking the source video description before downloading — "
         "confirmed single-anchor segment. This saved rejecting an entire source mid-pipeline."),
    ]
    for title_text, desc in iters:
        story.append(KeepTogether([
            Paragraph(f"<b>{title_text}</b>", H2),
            Paragraph(desc, BODY),
        ]))

    # ── 6. WHAT WORKED / DIDN'T ───────────────────────────────────────────────
    story.append(Paragraph("6. What Worked and What Didn't", H1))

    ww_rows = [["What Worked ✓", "What Didn't / Limitations ✗"]]
    worked = [
        "Silence-based segmentation: zero mid-word cuts",
        "Sarvam ASR accuracy: clean transcripts even for fast Hindi speech",
        "SNR gate eliminated noisy sources before any API calls",
        "Idempotent pipeline: crash-safe, resumes from last good segment",
        "Language-aware QC: Hindi and English thresholds differ correctly",
        "8 sources → 173 min raw → 60 min clean after QC budget",
    ]
    didnt = [
        "No diarization: podcast has interview sections mixed in",
        "Rule-based emotion tagger misses nuance (irony, sarcasm)",
        "questioning over-represented (57/143) — reflects source bias",
        "Dhruv Rathee: very fast speech, silence detection falls back to hard cuts",
        "Sarvam batch diarization API returned 404 — not available",
        "TEDx Hindi (hi_04) unused — budget met before reaching it",
    ]
    for w, d in zip(worked, didnt):
        ww_rows.append([f"• {w}", f"• {d}"])
    ww = Table(ww_rows, colWidths=[8.1*cm, 8.1*cm])
    ww.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,0), SARVAM_GREEN),
        ("BACKGROUND",    (1,0),(1,0), SARVAM_ORANGE),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, SARVAM_LIGHT]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E5E7EB")),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ]))
    story.append(ww)
    story.append(Spacer(1, 8))

    # ── 7. WHAT I'D IMPROVE ───────────────────────────────────────────────────
    story.append(Paragraph("7. What I'd Improve Given More Time", H1))

    improvements = [
        "<b>Speaker diarization:</b> The Seen &amp; Unseen podcast has interview guests. "
        "With a working diarization API (or pyannote.audio locally), I'd isolate only the "
        "host's speech — currently some interview segments may slip through.",
        "<b>LLM-based emotion tagging:</b> Rule-based tagging works but is brittle. "
        "With an Anthropic or Sarvam LLM call per segment, I'd get richer nuanced tags "
        "(curious, ironic, formal-assertive, warm) with confidence scores.",
        "<b>Whisper-based transcript verification:</b> Cross-check Sarvam ASR output "
        "against OpenAI Whisper on a 10% random sample to catch systematic errors — "
        "especially for code-switched Hindi/English segments.",
        "<b>Emotion balance:</b> The dataset is questioning-heavy. I'd add sources with "
        "more narrative/storytelling content (Hindi audiobooks, English storytelling "
        "channels) to balance the emotion distribution.",
        "<b>Audio normalisation:</b> Segments range from −30 to −14 dBFS. "
        "A loudness normalisation pass (ffmpeg loudnorm to −23 LUFS) would make "
        "the dataset more consistent for training.",
        "<b>Expand to more Indian languages:</b> Tamil, Telugu, Bengali, Marathi — "
        "the pipeline is language-agnostic once you have a Sarvam language code.",
    ]
    for imp in improvements:
        story.append(Paragraph(f"• {imp}", BULL))
    story.append(Spacer(1, 4))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=SARVAM_GREY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<font color='#6B7280'>Harshith Geddada · harshith.geddada@raga.ai · June 2026 · "
        "HuggingFace: Itsharshi/indian-tts-dataset · GitHub: itsharshi/indian-tts-dataset</font>",
        ParagraphStyle("Footer", parent=getSampleStyleSheet()["Normal"],
                       fontSize=8, textColor=SARVAM_GREY, alignment=TA_CENTER)))

    doc.build(story)
    print(f"✓ PDF report saved: {OUTPUT_PDF}")
    return OUTPUT_PDF


if __name__ == "__main__":
    build_report()
