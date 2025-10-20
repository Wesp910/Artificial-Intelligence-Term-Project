"""Google Translate API implementation for machine translation."""

from googletrans import Translator
from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class GoogleTranslator:
    """Google Translate API wrapper for machine translation.
    
    This class provides translation capabilities between different languages
    using Google's translation service.
    """
    
    def __init__(self, source_lang: str = 'auto', target_lang: str = 'en'):
        """
        Initialize the Google Translator.
        
        Args:
            source_lang: Source language code ('auto' for auto-detection)
            target_lang: Target language code
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator = Translator()
        
    def translate_text(self, text: str, 
                      source_lang: Optional[str] = None, 
                      target_lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language (overrides instance default)
            target_lang: Target language (overrides instance default)
            
        Returns:
            Dict containing translation results
        """
        if not text or not text.strip():
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': 'unknown',
                'target_language': target_lang or self.target_lang,
                'confidence': 0.0
            }
        
        src_lang = source_lang or self.source_lang
        tgt_lang = target_lang or self.target_lang
        
        try:
            logger.info(f"Translating text from {src_lang} to {tgt_lang}")
            
            # Add retry logic for API reliability
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.translator.translate(
                        text, 
                        src=src_lang, 
                        dest=tgt_lang
                    )
                    
                    translation_result = {
                        'original_text': text,
                        'translated_text': result.text,
                        'source_language': result.src,
                        'target_language': tgt_lang,
                        'confidence': getattr(result, 'confidence', 0.8)  # Default confidence
                    }
                    
                    logger.info("Translation completed successfully")
                    return translation_result
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Translation attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(1)  # Brief delay before retry
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'original_text': text,
                'translated_text': text,  # Return original text on failure
                'source_language': 'unknown',
                'target_language': tgt_lang,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing language detection results
        """
        try:
            logger.info("Detecting language")
            detection = self.translator.detect(text)
            
            return {
                'language': detection.lang,
                'confidence': detection.confidence,
                'text': text
            }
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'text': text,
                'error': str(e)
            }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get dictionary of supported languages.
        
        Returns:
            Dict mapping language codes to language names
        """
        try:
            # Common language mappings for our use case
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
                'hi': 'Hindi'
            }
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return {}
    
    def set_languages(self, source_lang: str, target_lang: str) -> None:
        """
        Set source and target languages.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        logger.info(f"Languages set: {source_lang} -> {target_lang}")
    
    def swap_languages(self) -> None:
        """
        Swap source and target languages.
        """
        self.source_lang, self.target_lang = self.target_lang, self.source_lang
        logger.info(f"Languages swapped: {self.source_lang} -> {self.target_lang}")
