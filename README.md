# VoiceBridge: Real-Time Bilingual Speech Translation System

**CSC 3250 - Artificial Intelligence Term Project**  
**Dr. Cueva Parra | Fall 2025**

**Team Members:** Wes & JB  
**Project Status:** ✅ Milestone 3 Complete

---

## Project Overview

**VoiceBridge** is a real-time speech-to-speech translation system that enables seamless communication between English and Spanish speakers. The system uses a cascaded Natural Language Processing (NLP) architecture combining:

- **Automatic Speech Recognition (ASR)** - OpenAI Whisper
- **Machine Translation (MT)** - Google Translate API
- **Text-to-Speech (TTS)** - Google TTS (gTTS)

### Key Performance Metrics

- **Translation Accuracy:** 100% (20/20 test cases)
- **Average Latency:** 7.99 seconds
- **ASR Accuracy:** 100%
- **TTS Naturalness:** 95%

---

## Quick Start

### Installation

```bash
git clone https://github.com/Wesp910/Artificial-Intelligence-Term-Project.git
cd Artificial-Intelligence-Term-Project
pip install -r requirements.txt
```

### Basic Usage

```bash
# Translate an audio file
python src/main.py --audio-file sample.wav --source-lang en --target-lang es

# Translate text directly
python src/main.py --text "Hello, how are you?" --source-lang en --target-lang es

# Show system information
python src/main.py --info
```

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── asr/
│   │   ├── __init__.py
│   │   └── whisper_asr.py          # ASR implementation (Figure 7)
│   ├── translation/
│   │   ├── __init__.py
│   │   └── google_translate.py     # MT implementation (Figure 8)
│   ├── tts/
│   │   ├── __init__.py
│   │   └── gtts_engine.py          # TTS implementation
│   └── main.py                     # System integration (Figures 9-10)
├── tests/
│   ├── test_asr.py
│   ├── test_translation.py
│   └── test_tts.py
├── experiments/
│   ├── accuracy_test_results.csv   # 20 accuracy test cases
│   ├── latency_test_results.csv    # 15 latency measurements
│   └── figures/                    # Experimental visualizations
├── docs/
│   └── milestones/
│       ├── Milestone-1.pdf         # Problem formulation
│       ├── Milestone-2.md          # Methods & Implementation
│       └── Milestone-3-Complete.md # Experiments & Results ✅
└── data/
    └── sample_audio/               # Test audio files
```

---

## Hardware Requirements

**Tested On:**
- 2024 MacBook Pro (M3, 16GB RAM) - 7.87s avg latency
- 2019 MacBook Pro (M1, 8GB RAM) - 8.17s avg latency

**Minimum Requirements:**
- Python 3.9+
- 8GB RAM
- Internet connection (for Google Translate & gTTS APIs)
- Microphone for audio input

---

## Experimental Results

### Accuracy Assessment (20 Test Cases)

| Component | Accuracy |
|-----------|----------|
| ASR (Whisper) | 100% (20/20) |
| Translation (Google) | 100% (20/20) |
| TTS Naturalness (gTTS) | 95% (19/20) |
| **Overall System** | **100%** |

### Latency Analysis (15 Tests)

| Input Type | Mean Latency | Range |
|-----------|--------------|-------|
| Single Sentence | 5.86s | 5.2s - 6.7s |
| Multi-Sentence | 11.88s | 11.0s - 13.8s |
| Question | 6.22s | 5.1s - 7.8s |
| **Overall Average** | **7.99s** | **5.1s - 13.8s** |

### Component Breakdown

- **ASR Processing:** 47.8% of total latency (2.35s avg)
- **Translation:** 16.9% of total latency (0.83s avg)
- **TTS Synthesis:** 35.3% of total latency (1.74s avg)

See [Milestone 3 Report](docs/milestones/Milestone-3-Complete.md) for complete experimental details and analysis.

---

## Key Features

✅ **High Accuracy** - 100% translation accuracy on conversational speech  
✅ **Near Real-Time** - 8 second average latency for typical conversations  
✅ **Modular Design** - Independent ASR, MT, and TTS components  
✅ **Consumer Hardware** - Runs on standard MacBook laptops  
✅ **Bidirectional** - Supports English ↔ Spanish translation  
✅ **Optimized Pipeline** - Model pre-loading and connection pooling  
✅ **Comprehensive Testing** - 90%+ code coverage with pytest  

---

## System Architecture

```
Audio Input (English)
    ↓
[ASR Module - Whisper]
    ↓ (2.35s avg)
Transcribed Text (English)
    ↓
[MT Module - Google Translate]
    ↓ (0.83s avg)
Translated Text (Spanish)
    ↓
[TTS Module - gTTS]
    ↓ (1.74s avg)
Audio Output (Spanish)
```

See [System Architecture Diagram](docs/milestones/Milestone-3-Complete.md#system-architecture-overview) for detailed flowchart.

---

## Dependencies

### Core Libraries
- `openai-whisper>=20231117` - ASR
- `googletrans==4.0.0rc1` - Translation
- `gTTS>=2.4.0` - Text-to-Speech
- `pygame>=2.0.0` - Audio playback

### Audio Processing
- `pyaudio>=0.2.11`
- `soundfile>=0.12.1`
- `librosa>=0.10.1`

### Development
- `pytest>=7.4.0` - Testing framework
- `numpy>=1.24.0` - Data processing
- `pandas>=1.5.0` - Experimental analysis

See [requirements.txt](requirements.txt) for complete dependency list.

---

## Milestones

- ✅ **Milestone 1** (Sept 23, 2025): Problem formulation, NLP framework, literature review
- ✅ **Milestone 2** (Oct 20, 2025): Complete ASR/MT/TTS implementation, system integration
- ✅ **Milestone 3** (Nov 20, 2025): Comprehensive experiments, results analysis, final documentation

---

## Testing

Run the complete test suite:

```bash
pytest tests/ -v --cov=src
```

Run specific component tests:

```bash
pytest tests/test_asr.py -v
pytest tests/test_translation.py -v
pytest tests/test_tts.py -v
```

---

## Acknowledgments

### Technologies
- **OpenAI Whisper** (Radford et al., 2023) - Robust ASR model
- **Google Translate API** (Johnson et al., 2017) - Neural machine translation
- **gTTS** - Google Text-to-Speech synthesis

### AI Assistance
This project utilized **ChatGPT (OpenAI, 2024)** for:
- Code optimization strategies and patterns
- Error handling recommendations
- Experimental design guidance
- Documentation improvements

All core algorithms, architecture decisions, and experimental analyses were performed by the project team. See [AI Assistance Acknowledgment](docs/milestones/Milestone-3-Complete.md#ai-assistance-acknowledgment) for details.

---

## Team Contributions

**Wes:**
- ASR module development (Whisper integration)
- TTS implementation (gTTS + pygame)
- M1 MacBook Pro testing
- Audio processing pipeline

**JB:**
- MT module development (Google Translate)
- System integration and orchestration
- M3 MacBook Pro testing
- Statistical analysis and visualization

---

## References

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML 2023*.

Johnson, M., et al. (2017). Google's Multilingual Neural Machine Translation System. *TACL, 5*, 339-351.

See [Complete References](docs/milestones/Milestone-3-Complete.md#references) for full bibliography.

---

## License

This project is for educational purposes as part of CSC 3250 coursework at Villanova University.

---

## Contact

For questions about this project, please contact the team members through the course instructor, Dr. Cueva Parra.

**Repository:** https://github.com/Wesp910/Artificial-Intelligence-Term-Project  
**Last Updated:** November 20, 2025