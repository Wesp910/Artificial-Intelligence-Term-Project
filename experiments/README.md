# VoiceBridge Experiments

This directory contains experimental scripts and results for the VoiceBridge speech translation system.

## Test Scripts

### Quick Demo
Run a single translation test:
```bash
python experiments/quick_demo.py
```

### Accuracy Testing
Test translation accuracy across 12 test cases:
```bash
python experiments/test_accuracy.py
```

You'll evaluate each translation as correct/incorrect. Results saved to CSV.

### Latency Testing
Measure processing time across 15 test cases:
```bash
python experiments/test_latency.py
```

Automatic timing for each component. Results saved to CSV.

## Experimental Results

Results from our testing sessions are stored as CSV files:
- `accuracy_results_YYYYMMDD_HHMM.csv`
- `latency_results_YYYYMMDD_HHMM.csv`

## Test Environment

- **Hardware**: M1 MacBook Pro (2019), M3 MacBook Pro (2024)
- **Audio Input**: Built-in MacBook microphones
- **Network**: Standard WiFi connection for API calls

## Running Experiments

All scripts use relative paths and work automatically from the project root:

```bash
cd Artificial-Intelligence-Term-Project
python experiments/quick_demo.py
```

No path configuration needed.
