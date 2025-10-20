"""Google Text-to-Speech (gTTS) implementation."""

from gtts import gTTS
import pygame
import io
import tempfile
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GTTSEngine:
    """Google Text-to-Speech engine implementation.
    
    This class provides text-to-speech functionality using Google's TTS service,
    with support for multiple languages and playback options.
    """
    
    def __init__(self, language: str = 'en', slow: bool = False):
        """
        Initialize the gTTS engine.
        
        Args:
            language: Language code for speech synthesis
            slow: Whether to use slow speech rate
        """
        self.language = language
        self.slow = slow
        self._init_pygame()
        
    def _init_pygame(self) -> None:
        """
        Initialize pygame mixer for audio playback.
        """
        try:
            pygame.mixer.init()
            logger.info("Pygame mixer initialized for audio playback")
        except Exception as e:
            logger.warning(f"Failed to initialize pygame mixer: {e}")
    
    def synthesize_speech(self, text: str, 
                         language: Optional[str] = None, 
                         slow: Optional[bool] = None) -> Dict[str, Any]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            language: Language code (overrides instance default)
            slow: Speech rate (overrides instance default)
            
        Returns:
            Dict containing synthesis results and audio data
        """
        if not text or not text.strip():
            return {
                'text': text,
                'language': language or self.language,
                'success': False,
                'error': 'Empty text provided'
            }
        
        lang = language or self.language
        speech_slow = slow if slow is not None else self.slow
        
        try:
            logger.info(f"Synthesizing speech for text in {lang}")
            
            # Create gTTS object
            tts = gTTS(
                text=text.strip(),
                lang=lang,
                slow=speech_slow
            )
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                temp_path = tmp_file.name
                
            # Save audio to temporary file
            tts.save(temp_path)
            
            result = {
                'text': text,
                'language': lang,
                'audio_file': temp_path,
                'success': True,
                'duration': self._estimate_duration(text)
            }
            
            logger.info("Speech synthesis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return {
                'text': text,
                'language': lang,
                'success': False,
                'error': str(e)
            }
    
    def synthesize_and_play(self, text: str, 
                           language: Optional[str] = None, 
                           slow: Optional[bool] = None) -> Dict[str, Any]:
        """
        Synthesize speech and play it immediately.
        
        Args:
            text: Text to convert to speech
            language: Language code (overrides instance default)
            slow: Speech rate (overrides instance default)
            
        Returns:
            Dict containing synthesis and playback results
        """
        synthesis_result = self.synthesize_speech(text, language, slow)
        
        if not synthesis_result['success']:
            return synthesis_result
        
        # Play the synthesized audio
        playback_result = self.play_audio(synthesis_result['audio_file'])
        
        # Combine results
        result = {**synthesis_result, **playback_result}
        
        # Clean up temporary file
        self._cleanup_temp_file(synthesis_result['audio_file'])
        
        return result
    
    def play_audio(self, audio_file: str) -> Dict[str, Any]:
        """
        Play audio file using pygame.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dict containing playback results
        """
        try:
            logger.info(f"Playing audio file: {audio_file}")
            
            # Load and play audio
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            logger.info("Audio playback completed")
            return {
                'playback_success': True,
                'played_file': audio_file
            }
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return {
                'playback_success': False,
                'playback_error': str(e)
            }
    
    def synthesize_to_bytes(self, text: str, 
                           language: Optional[str] = None, 
                           slow: Optional[bool] = None) -> Dict[str, Any]:
        """
        Synthesize speech and return audio as bytes.
        
        Args:
            text: Text to convert to speech
            language: Language code (overrides instance default)
            slow: Speech rate (overrides instance default)
            
        Returns:
            Dict containing synthesis results with audio bytes
        """
        if not text or not text.strip():
            return {
                'text': text,
                'language': language or self.language,
                'success': False,
                'error': 'Empty text provided'
            }
        
        lang = language or self.language
        speech_slow = slow if slow is not None else self.slow
        
        try:
            logger.info(f"Synthesizing speech to bytes for text in {lang}")
            
            # Create gTTS object
            tts = gTTS(
                text=text.strip(),
                lang=lang,
                slow=speech_slow
            )
            
            # Create bytes buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            result = {
                'text': text,
                'language': lang,
                'audio_bytes': audio_buffer.getvalue(),
                'success': True,
                'duration': self._estimate_duration(text)
            }
            
            logger.info("Speech synthesis to bytes completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Speech synthesis to bytes failed: {e}")
            return {
                'text': text,
                'language': lang,
                'success': False,
                'error': str(e)
            }
    
    def _estimate_duration(self, text: str) -> float:
        """
        Estimate audio duration based on text length.
        
        Args:
            text: Input text
            
        Returns:
            Estimated duration in seconds
        """
        # Rough estimation: ~150 words per minute, ~5 characters per word
        words = len(text.split())
        chars_per_word = 5
        words_per_minute = 150 if not self.slow else 75
        
        estimated_minutes = words / words_per_minute
        return estimated_minutes * 60
    
    def _cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary audio file.
        
        Args:
            file_path: Path to temporary file
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path}: {e}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get supported languages for gTTS.
        
        Returns:
            Dict mapping language codes to language names
        """
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish'
        }
    
    def set_language(self, language: str) -> None:
        """
        Set the default language for synthesis.
        
        Args:
            language: Language code
        """
        supported_langs = self.get_supported_languages()
        if language in supported_langs:
            self.language = language
            logger.info(f"TTS language set to: {language} ({supported_langs[language]})")
        else:
            logger.warning(f"Unsupported TTS language: {language}")
            raise ValueError(f"Unsupported TTS language: {language}")
    
    def set_speed(self, slow: bool) -> None:
        """
        Set the speech speed.
        
        Args:
            slow: Whether to use slow speech rate
        """
        self.slow = slow
        speed_desc = "slow" if slow else "normal"
        logger.info(f"TTS speed set to: {speed_desc}")
