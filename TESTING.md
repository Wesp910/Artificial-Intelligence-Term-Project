# Testing Guide

## Running the Experiment Scripts

We created three scripts for testing and demonstration:

### 1. Quick Demo
```bash
python experiments/quick_demo.py
```
Translates a single test phrase. Good for quick verification.

### 2. Accuracy Testing
```bash
python experiments/test_accuracy.py
```
Runs 12 test cases. You evaluate each translation as correct/incorrect.
Results save to `accuracy_results_YYYYMMDD_HHMM.csv`

### 3. Latency Measurement
```bash
python experiments/test_latency.py
```
Measures timing for 15 tests. Automatic timing for each component.
Results save to `latency_results_YYYYMMDD_HHMM.csv`

## What Gets Measured

### Accuracy Tests
- ASR correctness (did Whisper transcribe correctly?)
- Translation accuracy (is the Spanish correct?)
- TTS naturalness (does it sound natural?)

### Latency Tests
- Translation API time
- TTS synthesis time
- Total processing time
- Hardware performance

## Our Results

When we ran these scripts:
- **Accuracy**: 100% correct translations
- **Average latency**: ~8 seconds total
  - Translation: ~0.8s
  - TTS: ~1.7s

See `docs/milestones/Milestone-3-Complete.md` for full analysis.

## Unit Tests

Run pytest for code coverage:
```bash
pytest tests/ -v --cov=src
```

We have 90%+ coverage across all modules.

---

Wes & JB | Fall 2025
