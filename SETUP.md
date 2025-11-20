# Setup Instructions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Wesp910/Artificial-Intelligence-Term-Project.git
cd Artificial-Intelligence-Term-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. First run downloads Whisper model (~150MB):
```bash
python -c "import whisper; whisper.load_model('base')"
```

## Verify Installation

Run quick test:
```bash
python experiments/quick_demo.py
```

Should output a translation from English to Spanish.

## Requirements

- Python 3.9+
- 8GB RAM minimum
- Internet connection (for Google Translate and gTTS)
- Microphone if using audio input

## Tested On

- 2024 MacBook Pro (M3, 16GB RAM)
- 2019 MacBook Pro (M1, 8GB RAM)

Both work fine. Latency is similar since most processing happens via API calls.

## Troubleshooting

**Import errors:**
```bash
pip install --upgrade -r requirements.txt
```

**Whisper model not found:**
```bash
python -c "import whisper; whisper.load_model('base')"
```

**Audio issues:**
Make sure pygame is installed:
```bash
pip install pygame
```
