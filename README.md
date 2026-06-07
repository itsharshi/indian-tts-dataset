---
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

A curated, single-speaker TTS training dataset with **143 segments** (~**60 minutes** total) sourced from YouTube, transcribed using [Sarvam AI](https://docs.sarvam.ai) ASR, and annotated with emotion/style tags.

## Dataset Summary

| Split | Segments | Duration |
|-------|----------|----------|
| Indian English | 71 | 29.6 min |
| Hindi | 72 | 30.3 min |
| **Total** | **143** | **59.9 min** |

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
| questioning | 56 |
| formal | 27 |
| informational | 19 |
| conversational | 17 |
| motivational | 17 |
| emphatic | 2 |
| excited | 2 |
| storytelling | 1 |
| angry | 1 |
| sad | 1 |

## Sources

**Indian English:** The Seen and the Unseen

**Hindi:** Aaj Tak, Dhruv Rathee, Josh Talks Hindi

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
