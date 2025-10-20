"""Whisper-based Automatic Speech Recognition implementation."""

import whisper
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WhisperASR:
    """OpenAI Whisper ASR implementation for real-time speech recognition.
    
    This class provides a wrapper around OpenAI's Whisper model for converting
    speech audio to text. It supports multiple model sizes and languages.
    """
    
    def __init__(self, model_name: str = "base", language: Optional[str] = None):
        """
        Initialize the Whisper ASR system.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            language: Target language code (e.g., 'en' for English, 'es' for Spanish)
        """
        self.model_name = model_name
        self.language = language
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict containing transcription results including text and metadata
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            options = {}
            if self.language:
                options['language'] = self.language
                
            result = self.model.transcribe(audio_path, **options)
            
            logger.info("Transcription completed successfully")
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'duration': len(result.get('segments', [])) * 30  # Approximate
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': [],
                'duration': 0,
                'error': str(e)
            }
    
    def transcribe_numpy(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe numpy audio array to text.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Dict containing transcription results
        """
        try:
            logger.info("Transcribing numpy audio data")
            
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            options = {}
            if self.language:
                options['language'] = self.language
                
            result = self.model.transcribe(audio_data, **options)
            
            logger.info("Transcription completed successfully")
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'confidence': self._calculate_confidence(result)
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate average confidence score from segments.
        
        Args:
            result: Whisper transcription result
            
        Returns:
            Average confidence score (0-1)
        """
        segments = result.get('segments', [])
        if not segments:
            return 0.0
            
        confidences = []
        for segment in segments:
            # Whisper doesn't directly provide confidence, so we estimate
            # based on segment properties
            avg_logprob = segment.get('avg_logprob', -1.0)
            no_speech_prob = segment.get('no_speech_prob', 1.0)
            
            # Convert log probability to confidence estimate
            confidence = max(0.0, min(1.0, np.exp(avg_logprob) * (1 - no_speech_prob)))
            confidences.append(confidence)
            
        return np.mean(confidences) if confidences else 0.0
    
    def get_supported_languages(self) -> list:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return list(whisper.tokenizer.LANGUAGES.keys())
    
    def set_language(self, language: str) -> None:
        """
        Set the target language for transcription.
        
        Args:
            language: Language code (e.g., 'en', 'es')
        """
        if language in self.get_supported_languages():
            self.language = language
            logger.info(f"Language set to: {language}")
        else:
            logger.warning(f"Unsupported language: {language}")
            raise ValueError(f"Unsupported language: {language}")
