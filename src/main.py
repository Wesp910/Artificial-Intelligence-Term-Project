"""Main application for Real-Time Speech-to-Speech Translation System."""

import logging
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from asr.whisper_asr import WhisperASR
from translation.google_translate import GoogleTranslator
from tts.gtts_engine import GTTSEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpeechTranslationSystem:
    """Main system orchestrating ASR, MT, and TTS components."""
    
    def __init__(self, 
                 asr_model: str = "base",
                 source_lang: str = "en", 
                 target_lang: str = "es"):
        """
        Initialize the speech translation system.
        
        Args:
            asr_model: Whisper model size
            source_lang: Source language code
            target_lang: Target language code
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        logger.info("Initializing Speech Translation System")
        logger.info(f"ASR Model: {asr_model}")
        logger.info(f"Translation: {source_lang} -> {target_lang}")
        
        # Initialize components
        self.asr = WhisperASR(model_name=asr_model, language=source_lang)
        self.translator = GoogleTranslator(source_lang=source_lang, target_lang=target_lang)
        self.tts = GTTSEngine(language=target_lang)
        
        logger.info("System initialization completed")
    
    def translate_audio_file(self, audio_file: str, play_output: bool = True) -> Dict[str, Any]:
        """
        Translate audio file from source to target language.
        
        Args:
            audio_file: Path to audio file
            play_output: Whether to play the translated audio
            
        Returns:
            Dict containing full translation pipeline results
        """
        logger.info(f"Processing audio file: {audio_file}")
        
        # Step 1: ASR - Convert speech to text
        logger.info("Step 1: Automatic Speech Recognition")
        asr_result = self.asr.transcribe_audio(audio_file)
        
        if 'error' in asr_result:
            logger.error(f"ASR failed: {asr_result['error']}")
            return {'success': False, 'error': f"ASR failed: {asr_result['error']}"}
        
        original_text = asr_result['text']
        logger.info(f"ASR Result: '{original_text}'")
        
        if not original_text:
            logger.warning("No speech detected in audio")
            return {'success': False, 'error': 'No speech detected'}
        
        # Step 2: MT - Translate text
        logger.info("Step 2: Machine Translation")
        translation_result = self.translator.translate_text(original_text)
        
        if 'error' in translation_result:
            logger.error(f"Translation failed: {translation_result['error']}")
            return {'success': False, 'error': f"Translation failed: {translation_result['error']}"}
        
        translated_text = translation_result['translated_text']
        logger.info(f"Translation Result: '{translated_text}'")
        
        # Step 3: TTS - Convert translated text to speech
        logger.info("Step 3: Text-to-Speech Synthesis")
        
        if play_output:
            tts_result = self.tts.synthesize_and_play(translated_text)
        else:
            tts_result = self.tts.synthesize_speech(translated_text)
        
        if not tts_result['success']:
            logger.error(f"TTS failed: {tts_result.get('error', 'Unknown error')}")
            return {'success': False, 'error': f"TTS failed: {tts_result.get('error')}"}
        
        logger.info("Translation pipeline completed successfully")
        
        # Compile results
        return {
            'success': True,
            'source_language': self.source_lang,
            'target_language': self.target_lang,
            'original_text': original_text,
            'translated_text': translated_text,
            'asr_result': asr_result,
            'translation_result': translation_result,
            'tts_result': tts_result,
            'audio_file': audio_file
        }
    
    def translate_text_direct(self, text: str, play_output: bool = True) -> Dict[str, Any]:
        """
        Translate text directly (skip ASR step).
        
        Args:
            text: Text to translate
            play_output: Whether to play the translated audio
            
        Returns:
            Dict containing translation results
        """
        logger.info(f"Direct text translation: '{text}'")
        
        # Step 1: MT - Translate text
        logger.info("Step 1: Machine Translation")
        translation_result = self.translator.translate_text(text)
        
        if 'error' in translation_result:
            logger.error(f"Translation failed: {translation_result['error']}")
            return {'success': False, 'error': f"Translation failed: {translation_result['error']}"}
        
        translated_text = translation_result['translated_text']
        logger.info(f"Translation Result: '{translated_text}'")
        
        # Step 2: TTS - Convert translated text to speech
        logger.info("Step 2: Text-to-Speech Synthesis")
        
        if play_output:
            tts_result = self.tts.synthesize_and_play(translated_text)
        else:
            tts_result = self.tts.synthesize_speech(translated_text)
        
        if not tts_result['success']:
            logger.error(f"TTS failed: {tts_result.get('error', 'Unknown error')}")
            return {'success': False, 'error': f"TTS failed: {tts_result.get('error')}"}
        
        logger.info("Text translation completed successfully")
        
        return {
            'success': True,
            'source_language': self.source_lang,
            'target_language': self.target_lang,
            'original_text': text,
            'translated_text': translated_text,
            'translation_result': translation_result,
            'tts_result': tts_result
        }
    
    def swap_languages(self) -> None:
        """
        Swap source and target languages.
        """
        logger.info(f"Swapping languages: {self.source_lang} <-> {self.target_lang}")
        
        self.source_lang, self.target_lang = self.target_lang, self.source_lang
        
        # Update component languages
        self.asr.set_language(self.source_lang)
        self.translator.set_languages(self.source_lang, self.target_lang)
        self.tts.set_language(self.target_lang)
        
        logger.info(f"Languages swapped: {self.source_lang} -> {self.target_lang}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and status.
        
        Returns:
            Dict containing system information
        """
        return {
            'asr_model': self.asr.model_name,
            'source_language': self.source_lang,
            'target_language': self.target_lang,
            'supported_asr_languages': self.asr.get_supported_languages(),
            'supported_translation_languages': self.translator.get_supported_languages(),
            'supported_tts_languages': self.tts.get_supported_languages()
        }


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Real-Time Speech-to-Speech Translation System"
    )
    parser.add_argument(
        "--audio-file", "-a", 
        type=str, 
        help="Audio file to translate"
    )
    parser.add_argument(
        "--text", "-t", 
        type=str, 
        help="Text to translate directly"
    )
    parser.add_argument(
        "--source-lang", "-s", 
        type=str, 
        default="en", 
        help="Source language code (default: en)"
    )
    parser.add_argument(
        "--target-lang", "-d", 
        type=str, 
        default="es", 
        help="Target language code (default: es)"
    )
    parser.add_argument(
        "--asr-model", "-m", 
        type=str, 
        default="base", 
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--no-audio", 
        action="store_true", 
        help="Don't play audio output"
    )
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Show system information"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = SpeechTranslationSystem(
            asr_model=args.asr_model,
            source_lang=args.source_lang,
            target_lang=args.target_lang
        )
        
        if args.info:
            info = system.get_system_info()
            print("\n=== System Information ===")
            for key, value in info.items():
                print(f"{key}: {value}")
            return
        
        play_audio = not args.no_audio
        
        if args.audio_file:
            if not os.path.exists(args.audio_file):
                logger.error(f"Audio file not found: {args.audio_file}")
                sys.exit(1)
                
            result = system.translate_audio_file(args.audio_file, play_output=play_audio)
            
        elif args.text:
            result = system.translate_text_direct(args.text, play_output=play_audio)
            
        else:
            logger.error("Please provide either --audio-file or --text")
            parser.print_help()
            sys.exit(1)
        
        if result['success']:
            print("\n=== Translation Results ===")
            print(f"Original ({result['source_language']}): {result['original_text']}")
            print(f"Translated ({result['target_language']}): {result['translated_text']}")
        else:
            logger.error(f"Translation failed: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Translation interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
