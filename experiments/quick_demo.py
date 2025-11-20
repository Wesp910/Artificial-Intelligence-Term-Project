#!/usr/bin/env python3
"""Quick demonstration script for VoiceBridge."""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from main import SpeechTranslationSystem

print("\nInitializing VoiceBridge...")
system = SpeechTranslationSystem(source_lang="en", target_lang="es")
print("Ready!\n")

test_phrase = "Hello Professor Parra, this is our translation system."

print(f"English: {test_phrase}")
print("\nTranslating...")

start = time.perf_counter()
result = system.translate_text_direct(test_phrase, play_output=True)
end = time.perf_counter()

if result['success']:
    print(f"\nSpanish: {result['translated_text']}")
    print(f"Processing time: {end-start:.2f} seconds")
    print("\n[Audio output played]\n")
else:
    print(f"\nError: {result.get('error')}\n")
