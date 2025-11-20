# Experiments

Scripts for testing VoiceBridge performance.

## Scripts

**quick_demo.py** - Single translation demo  
**test_accuracy.py** - Tests 12 translations, you evaluate correctness  
**test_latency.py** - Measures timing for 15 tests automatically

## How to Run

```bash
python experiments/quick_demo.py
python experiments/test_accuracy.py
python experiments/test_latency.py
```

All scripts work from project root, no path setup needed.

## Results

Scripts save results to CSV files:
- accuracy_results_YYYYMMDD_HHMM.csv
- latency_results_YYYYMMDD_HHMM.csv

## Test Environment

We tested on:
- M1 MacBook Pro (2019) - Wes
- M3 MacBook Pro (2024) - JB

Both using built-in mics and WiFi for API calls.

## Our Results

- **Accuracy**: 100% (12/12 correct)
- **Average latency**: ~8 seconds

Detailed analysis in Milestone 3 report.
