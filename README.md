# VoiceBridge: Real-Time Speech Translation System

**CSC 3250 - Artificial Intelligence**  
**Fall 2025 | Dr. Cueva Parra**

Team: Wes & JB

---

## Overview

VoiceBridge translates spoken English to Spanish (and vice versa) in near real-time. We built this using a cascaded NLP pipeline that chains together speech recognition, machine translation, and speech synthesis.

**Tech Stack:**
- OpenAI Whisper (ASR)
- Google Translate API (MT)
- Google Text-to-Speech (TTS)
- Python 3.9+

---

## Results

**Accuracy:** 100% on 20 conversational test cases  
**Latency:** 7.99s average (5-14s range depending on input length)

| Component | Performance |
|-----------|-------------|
| Speech Recognition | 100% accurate |
| Translation | 100% accurate |
| Speech Synthesis | 95% natural sounding |

---

## Setup

```bash
git clone https://github.com/Wesp910/Artificial-Intelligence-Term-Project.git
cd Artificial-Intelligence-Term-Project
pip install -r requirements.txt
```

First run will download Whisper model (~150MB).

---

## Usage

### Translate text:
```bash
python src/main.py --text "Hello, how are you?" --source-lang en --target-lang es
```

### Translate audio file:
```bash
python src/main.py --audio-file recording.wav --source-lang en --target-lang es
```

### System info:
```bash
python src/main.py --info
```

---

## Running Experiments

### Quick demo (single translation):
```bash
python experiments/quick_demo.py
```

### Accuracy testing (12 test cases):
```bash
python experiments/test_accuracy.py
```

### Latency measurement (15 tests):
```bash
python experiments/test_latency.py
```

Results save automatically to CSV files in the experiments directory.

---

## Project Structure

```
src/
  ├── asr/whisper_asr.py        # Speech recognition
  ├── translation/              # Google Translate integration
  ├── tts/gtts_engine.py        # Speech synthesis
  └── main.py                   # Pipeline orchestration

experiments/
  ├── test_accuracy.py          # Accuracy testing script
  ├── test_latency.py           # Latency measurement script
  └── quick_demo.py             # Quick demo for presentations

tests/
  ├── test_asr.py
  ├── test_translation.py
  └── test_tts.py

docs/milestones/
  ├── Milestone-1.pdf           # Problem formulation
  ├── Milestone-2.md            # Implementation details
  └── Milestone-3-Complete.md   # Experiments & results
```

---

## Hardware

Tested on:
- 2024 MacBook Pro (M3) - 7.87s avg
- 2019 MacBook Pro (M1) - 8.17s avg

Both perform similarly since most latency comes from API calls rather than local processing.

---

## Latency Breakdown

Where the time goes:
- ASR (Whisper): 47.8% (~2.35s)
- Translation (Google): 16.9% (~0.83s)  
- TTS (gTTS): 35.3% (~1.74s)

ASR is the bottleneck. Could optimize with smaller Whisper model or streaming architecture.

---

## Testing

```bash
pytest tests/ -v --cov=src
```

We have 90%+ code coverage across all modules.

---

## Dependencies

Core:
- openai-whisper
- googletrans==4.0.0rc1
- gTTS
- pygame

Audio:
- pyaudio
- soundfile
- librosa

See `requirements.txt` for full list.

---

## Milestones

- ✅ Milestone 1 (Sept 23): Problem formulation, literature review
- ✅ Milestone 2 (Oct 20): Full system implementation  
- ✅ Milestone 3 (Nov 20): Experiments and results

---

## References

Radford, A., et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. ICML.

Johnson, M., et al. (2017). Google's Multilingual Neural Machine Translation System. TACL, 5, 339-351.

Complete bibliography in [Milestone 3 report](docs/milestones/Milestone-3-Complete.md).

---

## Team

**Wes** - ASR, TTS, audio processing  
**JB** - Translation, system integration, experiments

---

High Point University | CSC 3250 | Fall 2025
