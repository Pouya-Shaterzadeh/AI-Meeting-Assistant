---
title: AI Meeting Assistant by PouyaDevA1
emoji: 📝
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
tags:
- ai
- meetings
- transcription
- nlp
- productivity
- whisper
- langchain
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/688f59b4fe95b912726282f2/U_NjwBUEE1uVbXIJ1uuPd.png
---

# AI Meeting Assistant

> Transform your meeting recordings into actionable insights with AI-powered analysis.

**Created by:** [PouyaDevA1](https://huggingface.co/PouyaDevA1) | **Free & Open Source**

## Features

- **Audio Transcription** — Whisper-large-v3-turbo for high-accuracy speech-to-text
- **Executive Summary** — Phi-3-mini generates concise, metric-focused summaries
- **Task Extraction** — LLM-powered action item detection with single-speaker awareness
- **Sentiment Analysis** — Emotion classification via DistilRoBERTa
- **Key Topic Identification** — Semantic noun-phrase pattern matching
- **Chunked Processing** — 30s windows with map-reduce for long meetings (up to 60 min)
- **Downloadable Reports** — Export meeting minutes as text files
- **Sample Audio** — Try it instantly with a built-in sample meeting

## How It Works

1. **Upload Audio** — Drop your meeting recording (WAV, MP3, M4A, etc.)
2. **AI Processing** — Transcription, summarization, sentiment, and topic extraction run in parallel
3. **Get Results** — Receive a structured meeting report
4. **Download** — Save your analysis as a text file

## AI Models

All models are **free and open-source**, served via Hugging Face Inference API:

| Model | Purpose | License |
|-------|---------|---------|
| OpenAI Whisper-large-v3-turbo | Speech-to-text | MIT |
| Microsoft Phi-3-mini-4k-instruct | Executive summary & task extraction | MIT |
| BART-large-cnn-samsum | Summarization fallback | MIT |
| Cardiff DistilRoBERTa | Sentiment analysis | Apache 2.0 |

## Technical Details

- **Framework**: Gradio 4.44.1
- **Backend**: Hugging Face Inference API (serverless)
- **Language**: Python 3.12+
- **Deployment**: Hugging Face Spaces
- **License**: MIT

## Quick Start

1. Visit the [Live App](https://huggingface.co/spaces/PouyaDevA1/ai-meeting-assistant)
2. Upload your meeting audio or try the sample
3. Click **Submit**
4. Download your meeting analysis

## Output

- **Executive Summary** — 2-3 sentence overview with key metrics
- **Action Items** — Extracted tasks (or "No actionable tasks" for single-speaker)
- **Sentiment** — Primary emotion with confidence scores
- **Key Topics** — Main discussion themes
- **Meeting Type** — Presentation, 1-on-1, or multi-participant

---

**Start analyzing your meetings today with AI-powered insights!**
