"""Tests for Text-to-Speech module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from tts.gtts_engine import GTTSEngine


class TestGTTSEngine:
    """Test cases for GTTSEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('pygame.mixer.init'):
            self.tts = GTTSEngine()
    
    def test_initialization(self):
        """Test TTS engine initialization."""
        assert self.tts.language == 'en'
        assert self.tts.slow is False
    
    def test_initialization_with_params(self):
        """Test TTS initialization with parameters."""
        with patch('pygame.mixer.init'):
            tts = GTTSEngine(language='es', slow=True)
            assert tts.language == 'es'
            assert tts.slow is True
    
    @patch('tempfile.NamedTemporaryFile')
    @patch('gtts.gTTS')
    def test_synthesize_speech_success(self, mock_gtts, mock_temp_file):
        """Test successful speech synthesis."""
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = '/tmp/test_audio.mp3'
        mock_temp_file.return_value.__enter__.return_value = mock_file
        
        # Mock gTTS
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        result = self.tts.synthesize_speech("Hello world")
        
        assert result['success'] is True
        assert result['text'] == "Hello world"
        assert result['language'] == 'en'
        assert 'audio_file' in result
        assert 'duration' in result
        
        # Verify gTTS was called correctly
        mock_gtts.assert_called_once_with(
            text="Hello world",
            lang='en',
            slow=False
        )
        mock_tts_instance.save.assert_called_once()
    
    def test_synthesize_speech_empty_text(self):
        """Test synthesis with empty text."""
        result = self.tts.synthesize_speech("")
        
        assert result['success'] is False
        assert 'error' in result
        assert result['error'] == 'Empty text provided'
    
    @patch('gtts.gTTS')
    def test_synthesize_speech_error(self, mock_gtts):
        """Test synthesis error handling."""
        mock_gtts.side_effect = Exception("TTS failed")
        
        result = self.tts.synthesize_speech("Test text")
        
        assert result['success'] is False
        assert 'error' in result
        assert "TTS failed" in result['error']
    
    @patch.object(GTTSEngine, 'synthesize_speech')
    @patch.object(GTTSEngine, 'play_audio')
    @patch.object(GTTSEngine, '_cleanup_temp_file')
    def test_synthesize_and_play_success(self, mock_cleanup, mock_play, mock_synth):
        """Test synthesis and immediate playback."""
        # Mock synthesis result
        synth_result = {
            'success': True,
            'text': 'Test',
            'audio_file': '/tmp/test.mp3'
        }
        mock_synth.return_value = synth_result
        
        # Mock playback result
        play_result = {
            'playback_success': True,
            'played_file': '/tmp/test.mp3'
        }
        mock_play.return_value = play_result
        
        result = self.tts.synthesize_and_play("Test")
        
        assert result['success'] is True
        assert result['playback_success'] is True
        mock_cleanup.assert_called_once_with('/tmp/test.mp3')
    
    @patch('pygame.mixer.music.load')
    @patch('pygame.mixer.music.play')
    @patch('pygame.mixer.music.get_busy')
    @patch('pygame.time.Clock')
    def test_play_audio_success(self, mock_clock, mock_get_busy, mock_play, mock_load):
        """Test successful audio playback."""
        # Mock pygame methods
        mock_get_busy.side_effect = [True, True, False]  # Simulate playback ending
        mock_clock_instance = Mock()
        mock_clock.return_value = mock_clock_instance
        
        result = self.tts.play_audio('/tmp/test.mp3')
        
        assert result['playback_success'] is True
        assert result['played_file'] == '/tmp/test.mp3'
        
        mock_load.assert_called_once_with('/tmp/test.mp3')
        mock_play.assert_called_once()
    
    @patch('pygame.mixer.music.load')
    def test_play_audio_error(self, mock_load):
        """Test audio playback error handling."""
        mock_load.side_effect = Exception("Playback failed")
        
        result = self.tts.play_audio('/tmp/test.mp3')
        
        assert result['playback_success'] is False
        assert 'playback_error' in result
        assert "Playback failed" in result['playback_error']
    
    @patch('io.BytesIO')
    @patch('gtts.gTTS')
    def test_synthesize_to_bytes_success(self, mock_gtts, mock_bytesio):
        """Test synthesis to bytes buffer."""
        # Mock BytesIO
        mock_buffer = Mock()
        mock_buffer.getvalue.return_value = b'audio_data'
        mock_bytesio.return_value = mock_buffer
        
        # Mock gTTS
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance
        
        result = self.tts.synthesize_to_bytes("Test text")
        
        assert result['success'] is True
        assert result['audio_bytes'] == b'audio_data'
        assert result['text'] == "Test text"
        
        mock_tts_instance.write_to_fp.assert_called_once()
    
    def test_estimate_duration(self):
        """Test duration estimation."""
        # Test with normal speed
        duration = self.tts._estimate_duration("This is a test sentence with multiple words.")
        assert duration > 0
        
        # Test with slow speed
        self.tts.slow = True
        slow_duration = self.tts._estimate_duration("This is a test sentence with multiple words.")
        assert slow_duration > duration
    
    @patch('os.path.exists')
    @patch('os.unlink')
    def test_cleanup_temp_file_success(self, mock_unlink, mock_exists):
        """Test successful temp file cleanup."""
        mock_exists.return_value = True
        
        self.tts._cleanup_temp_file('/tmp/test.mp3')
        
        mock_unlink.assert_called_once_with('/tmp/test.mp3')
    
    @patch('os.path.exists')
    def test_cleanup_temp_file_not_exists(self, mock_exists):
        """Test cleanup when file doesn't exist."""
        mock_exists.return_value = False
        
        # Should not raise exception
        self.tts._cleanup_temp_file('/tmp/nonexistent.mp3')
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.tts.get_supported_languages()
        assert isinstance(languages, dict)
        assert 'en' in languages
        assert 'es' in languages
        assert languages['en'] == 'English'
        assert languages['es'] == 'Spanish'
    
    def test_set_language_valid(self):
        """Test setting valid language."""
        self.tts.set_language('es')
        assert self.tts.language == 'es'
    
    def test_set_language_invalid(self):
        """Test setting invalid language."""
        with pytest.raises(ValueError):
            self.tts.set_language('invalid')
    
    def test_set_speed(self):
        """Test setting speech speed."""
        self.tts.set_speed(True)
        assert self.tts.slow is True
        
        self.tts.set_speed(False)
        assert self.tts.slow is False
