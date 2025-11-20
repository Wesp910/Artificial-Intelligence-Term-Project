#!/usr/bin/env python3
"""Accuracy testing script for VoiceBridge translation system."""

import sys
from pathlib import Path
from datetime import datetime
import csv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from main import SpeechTranslationSystem

print("\n" + "="*80)
print("VOICEBRIDGE ACCURACY EXPERIMENT")
print("="*80)

tests = [
    ("Simple", "Hello, how are you today?"),
    ("Simple", "The weather is beautiful today."),
    ("Simple", "My name is John Smith."),
    ("Question", "Where is the nearest hospital?"),
    ("Question", "What time does the store close?"),
    ("Question", "How much does this cost?"),
    ("Question", "Is there a pharmacy nearby?"),
    ("Complex", "I would like to schedule an appointment for next Tuesday."),
    ("Complex", "I'm experiencing technical difficulties with my internet."),
    ("Multi", "I need help with my computer. It won't turn on."),
    ("Multi", "Thank you for your help. I really appreciate it."),
    ("Multi", "I have a meeting tomorrow. Can we reschedule?"),
]

print("\nInitializing system...")
system = SpeechTranslationSystem(source_lang="en", target_lang="es")
print("Ready!\n")

results = []
correct_count = 0

for i, (category, text) in enumerate(tests, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/{len(tests)}")
    print('='*80)
    print(f"Type: {category}")
    print(f"English: {text}")
    
    result = system.translate_text_direct(text, play_output=False)
    
    if result['success']:
        print(f"Spanish: {result['translated_text']}")
        correct = input("\n  Is translation correct? (y/n): ").strip().lower() == 'y'
        
        if correct:
            correct_count += 1
            print("  ✓ CORRECT")
        else:
            print("  ✗ INCORRECT")
        
        results.append({
            'test_id': i,
            'category': category,
            'english': text,
            'spanish': result['translated_text'],
            'correct': correct
        })
    else:
        print(f"ERROR: {result.get('error')}")
        results.append({
            'test_id': i,
            'category': category,
            'english': text,
            'spanish': 'ERROR',
            'correct': False
        })

print("\n" + "="*80)
print("ACCURACY RESULTS")
print("="*80)
print(f"Correct: {correct_count}/{len(tests)}")
print(f"Accuracy: {correct_count/len(tests)*100:.1f}%")
print("="*80)

filename = f"accuracy_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
with open(filename, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['test_id', 'category', 'english', 'spanish', 'correct'])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✓ Results saved to: {filename}\n")
