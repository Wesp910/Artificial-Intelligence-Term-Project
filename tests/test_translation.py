"""Tests for Machine Translation module."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from translation.google_translate import GoogleTranslator


class TestGoogleTranslator:
    """Test cases for GoogleTranslator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('googletrans.Translator'):
            self.translator = GoogleTranslator()
    
    def test_initialization(self):
        """Test translator initialization."""
        assert self.translator.source_lang == 'auto'
        assert self.translator.target_lang == 'en'
    
    def test_initialization_with_languages(self):
        """Test translator initialization with specific languages."""
        with patch('googletrans.Translator'):
            translator = GoogleTranslator(source_lang='en', target_lang='es')
            assert translator.source_lang == 'en'
            assert translator.target_lang == 'es'
    
    @patch('googletrans.Translator')
    def test_translate_text_success(self, mock_translator_class):
        """Test successful text translation."""
        # Mock Google Translate response
        mock_result = Mock()
        mock_result.text = "Hola, ¿cómo estás?"
        mock_result.src = "en"
        mock_result.confidence = 0.95
        
        mock_translator = Mock()
        mock_translator.translate.return_value = mock_result
        mock_translator_class.return_value = mock_translator
        
        translator = GoogleTranslator(source_lang='en', target_lang='es')
        result = translator.translate_text("Hello, how are you?")
        
        assert result['original_text'] == "Hello, how are you?"
        assert result['translated_text'] == "Hola, ¿cómo estás?"
        assert result['source_language'] == "en"
        assert result['target_language'] == "es"
        assert result['confidence'] == 0.95
    
    @patch('googletrans.Translator')
    def test_translate_text_empty(self, mock_translator_class):
        """Test translation with empty text."""
        translator = GoogleTranslator()
        result = translator.translate_text("")
        
        assert result['original_text'] == ""
        assert result['translated_text'] == ""
        assert result['confidence'] == 0.0
    
    @patch('googletrans.Translator')
    def test_translate_text_error(self, mock_translator_class):
        """Test translation error handling."""
        mock_translator = Mock()
        mock_translator.translate.side_effect = Exception("Translation failed")
        mock_translator_class.return_value = mock_translator
        
        translator = GoogleTranslator()
        result = translator.translate_text("Hello")
        
        assert 'error' in result
        assert result['confidence'] == 0.0
        assert result['translated_text'] == "Hello"  # Returns original on error
    
    @patch('googletrans.Translator')
    def test_translate_text_with_retry(self, mock_translator_class):
        """Test translation with retry logic."""
        mock_result = Mock()
        mock_result.text = "Translated text"
        mock_result.src = "en"
        
        mock_translator = Mock()
        # First call fails, second succeeds
        mock_translator.translate.side_effect = [Exception("Temp failure"), mock_result]
        mock_translator_class.return_value = mock_translator
        
        translator = GoogleTranslator()
        result = translator.translate_text("Test")
        
        assert result['translated_text'] == "Translated text"
        assert mock_translator.translate.call_count == 2
    
    @patch('googletrans.Translator')
    def test_detect_language_success(self, mock_translator_class):
        """Test successful language detection."""
        mock_detection = Mock()
        mock_detection.lang = "en"
        mock_detection.confidence = 0.98
        
        mock_translator = Mock()
        mock_translator.detect.return_value = mock_detection
        mock_translator_class.return_value = mock_translator
        
        translator = GoogleTranslator()
        result = translator.detect_language("Hello world")
        
        assert result['language'] == "en"
        assert result['confidence'] == 0.98
        assert result['text'] == "Hello world"
    
    @patch('googletrans.Translator')
    def test_detect_language_error(self, mock_translator_class):
        """Test language detection error handling."""
        mock_translator = Mock()
        mock_translator.detect.side_effect = Exception("Detection failed")
        mock_translator_class.return_value = mock_translator
        
        translator = GoogleTranslator()
        result = translator.detect_language("Test text")
        
        assert result['language'] == 'unknown'
        assert result['confidence'] == 0.0
        assert 'error' in result
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.translator.get_supported_languages()
        assert isinstance(languages, dict)
        assert 'en' in languages
        assert 'es' in languages
        assert languages['en'] == 'English'
        assert languages['es'] == 'Spanish'
    
    def test_set_languages(self):
        """Test setting source and target languages."""
        self.translator.set_languages('fr', 'de')
        assert self.translator.source_lang == 'fr'
        assert self.translator.target_lang == 'de'
    
    def test_swap_languages(self):
        """Test swapping source and target languages."""
        self.translator.set_languages('en', 'es')
        self.translator.swap_languages()
        assert self.translator.source_lang == 'es'
        assert self.translator.target_lang == 'en'
