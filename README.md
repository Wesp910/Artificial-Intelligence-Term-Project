# Real-Time Speech-to-Speech Translation System

CSC 3250 - Artificial Intelligence Term Project
Dr. Cueva Parra
Fall 2025

## Project Overview

This project implements a real-time speech-to-speech translation system that enables seamless communication between English and Spanish speakers. The system uses a cascaded approach combining Automatic Speech Recognition (ASR), Machine Translation (MT), and Text-to-Speech (TTS) technologies.

## Team Members
- Wes
- JB

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── asr/
│   │   ├── __init__.py
│   │   └── whisper_asr.py
│   ├── translation/
│   │   ├── __init__.py
│   │   └── google_translate.py
│   ├── tts/
│   │   ├── __init__.py
│   │   └── gtts_engine.py
│   └── main.py
├── data/
│   └── sample_audio/
├── tests/
│   ├── __init__.py
│   ├── test_asr.py
│   ├── test_translation.py
│   └── test_tts.py
└── docs/
    └── milestones/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Wesp910/Artificial-Intelligence-Term-Project.git
cd Artificial-Intelligence-Term-Project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main translation system:
```bash
python src/main.py
```

## Milestones

- [x] Milestone 1: Problem formulation, literature review, and project planning
- [x] Milestone 2: Methods implementation and initial code development
- [ ] Milestone 3: Experiments, results, and analysis
