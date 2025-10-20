"""Tests for Automatic Speech Recognition module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from asr.whisper_asr import WhisperASR


class TestWhisperASR:
    """Test cases for WhisperASR class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            self.asr = WhisperASR(model_name="base")
    
    def test_initialization(self):
        """Test ASR initialization."""
        assert self.asr.model_name == "base"
        assert self.asr.language is None
        assert self.asr.model is not None
    
    def test_initialization_with_language(self):
        """Test ASR initialization with specific language."""
        with patch('whisper.load_model'):
            asr = WhisperASR(model_name="small", language="en")
            assert asr.language == "en"
    
    @patch('whisper.load_model')
    def test_transcribe_audio_success(self, mock_load):
        """Test successful audio transcription."""
        # Mock Whisper model response
        mock_model = Mock()
        mock_result = {
            'text': ' Hello, how are you? ',
            'language': 'en',
            'segments': [
                {
                    'avg_logprob': -0.3,
                    'no_speech_prob': 0.1
                }
            ]
        }
        mock_model.transcribe.return_value = mock_result
        mock_load.return_value = mock_model
        
        asr = WhisperASR()
        result = asr.transcribe_audio("test_audio.wav")
        
        assert result['text'] == 'Hello, how are you?'
        assert result['language'] == 'en'
        assert 'segments' in result
        assert 'duration' in result
    
    @patch('whisper.load_model')
    def test_transcribe_audio_error(self, mock_load):
        """Test audio transcription error handling."""
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_load.return_value = mock_model
        
        asr = WhisperASR()
        result = asr.transcribe_audio("nonexistent.wav")
        
        assert result['text'] == ''
        assert 'error' in result
        assert "Transcription failed" in result['error']
    
    def test_transcribe_numpy_array(self):
        """Test numpy array transcription."""
        # Create mock audio data
        audio_data = np.random.rand(16000).astype(np.float32)
        
        # Mock the transcription result
        mock_result = {
            'text': 'Test transcription',
            'language': 'en',
            'segments': []
        }
        
        with patch.object(self.asr.model, 'transcribe', return_value=mock_result):
            result = self.asr.transcribe_numpy(audio_data)
            
            assert result['text'] == 'Test transcription'
            assert result['language'] == 'en'
            assert 'confidence' in result
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # Mock result with segments
        result = {
            'segments': [
                {'avg_logprob': -0.2, 'no_speech_prob': 0.1},
                {'avg_logprob': -0.3, 'no_speech_prob': 0.05}
            ]
        }
        
        confidence = self.asr._calculate_confidence(result)
        assert 0.0 <= confidence <= 1.0
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        with patch('whisper.tokenizer.LANGUAGES', {'en': 'English', 'es': 'Spanish'}):
            languages = self.asr.get_supported_languages()
            assert 'en' in languages
            assert 'es' in languages
    
    def test_set_language_valid(self):
        """Test setting valid language."""
        with patch.object(self.asr, 'get_supported_languages', return_value=['en', 'es']):
            self.asr.set_language('es')
            assert self.asr.language == 'es'
    
    def test_set_language_invalid(self):
        """Test setting invalid language."""
        with patch.object(self.asr, 'get_supported_languages', return_value=['en', 'es']):
            with pytest.raises(ValueError):
                self.asr.set_language('invalid')
