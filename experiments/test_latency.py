#!/usr/bin/env python3
"""Latency testing script for VoiceBridge translation system."""

import sys
from pathlib import Path
import time
from datetime import datetime
import platform
import csv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from main import SpeechTranslationSystem

print("\n" + "="*80)
print("VOICEBRIDGE LATENCY EXPERIMENT")
print("="*80)

tests = [
    ("Single", "Hello, how are you?"),
    ("Single", "What time is it?"),
    ("Single", "Thank you very much."),
    ("Single", "Where is the library?"),
    ("Single", "I need to see a doctor."),
    ("Multi", "I have a meeting tomorrow. Can we reschedule?"),
    ("Multi", "Please turn left. The building is on your right."),
    ("Multi", "I need help. It won't turn on."),
    ("Multi", "I enjoyed our conversation. Let's meet again next week."),
    ("Multi", "I would like an appointment. Is Tuesday available?"),
    ("Question", "Where is the hospital?"),
    ("Question", "Do you accept credit cards?"),
    ("Question", "Is there a pharmacy nearby?"),
    ("Question", "Could you speak more slowly?"),
    ("Question", "How much does this cost?"),
]

hardware = f"{platform.machine()} MacBook Pro"
print(f"Hardware: {hardware}")

print("\nInitializing system...")
system = SpeechTranslationSystem(source_lang="en", target_lang="es")
print("Ready!\n")

results = []

for i, (category, text) in enumerate(tests, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/{len(tests)}: {text}")
    print('='*80)
    
    trans_start = time.perf_counter()
    trans_result = system.translator.translate_text(text)
    trans_end = time.perf_counter()
    trans_time = trans_end - trans_start
    
    tts_start = time.perf_counter()
    tts_result = system.tts.synthesize_speech(trans_result['translated_text'])
    tts_end = time.perf_counter()
    tts_time = tts_end - tts_start
    
    total = trans_time + tts_time
    
    print(f"Spanish: {trans_result['translated_text']}")
    print(f"\nTiming:")
    print(f"  Translation: {trans_time:.3f}s")
    print(f"  TTS: {tts_time:.3f}s")
    print(f"  TOTAL: {total:.3f}s")
    
    results.append({
        'test_id': i,
        'category': category,
        'text': text,
        'translation_time': round(trans_time, 3),
        'tts_time': round(tts_time, 3),
        'total_time': round(total, 3),
        'hardware': hardware
    })

avg_total = sum(r['total_time'] for r in results) / len(results)
min_time = min(r['total_time'] for r in results)
max_time = max(r['total_time'] for r in results)

print("\n" + "="*80)
print("LATENCY RESULTS")
print("="*80)
print(f"Average: {avg_total:.3f}s")
print(f"Range: {min_time:.3f}s - {max_time:.3f}s")
print(f"Hardware: {hardware}")
print("="*80)

filename = f"latency_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
with open(filename, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['test_id', 'category', 'text', 'translation_time', 'tts_time', 'total_time', 'hardware'])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✓ Results saved to: {filename}\n")
