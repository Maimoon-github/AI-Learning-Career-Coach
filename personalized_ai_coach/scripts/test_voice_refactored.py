import asyncio
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.voice_interface.stt_service import STTService
from src.services.voice_interface.tts_service import TTSService
from src.services.voice_interface.audio_types import AudioChunk

async def test_stt_refactored_logic():
    print("Testing STT Refactored Logic (Mocked Model)...")
    
    with patch("src.services.voice_interface.stt_service.WhisperModel") as MockWhisper:
        mock_model = MockWhisper.return_value
        # Whisper transcribe returns (segments, info)
        mock_model.transcribe.return_value = ([MagicMock(text="Test transcript.")], None)
        
        stt = STTService(model_size="tiny")
        # Mock VAD frames
        # Silence (5) -> Speech (10) -> Silence (30)
        stt.vad.is_speech = MagicMock(side_effect=[False]*5 + [True]*10 + [False]*30)
        
        frame_size = 960
        silence_frame = b"\x00" * frame_size
        speech_frame = (np.random.randn(480) * 10).astype(np.int16).tobytes()
        
        async def audio_gen():
            for _ in range(45):
                if _ < 5:
                    yield AudioChunk(silence_frame, 16000)
                elif _ < 15:
                    yield AudioChunk(speech_frame, 16000)
                else:
                    yield AudioChunk(silence_frame, 16000)
        
        transcripts = []
        async for text in stt.transcribe_stream(audio_gen()):
            print(f"Captured Transcript: {text}")
            transcripts.append(text)
        
        assert len(transcripts) > 0
        print("STT Refactored Logic Test Passed!")

async def test_tts_refactored_logic():
    print("Testing TTS Refactored Logic (Mocked Model)...")
    
    with patch("src.services.voice_interface.tts_service.SpeechT5Processor") as MockProc, \
         patch("src.services.voice_interface.tts_service.SpeechT5ForTextToSpeech") as MockModel, \
         patch("src.services.voice_interface.tts_service.SpeechT5HifiGan") as MockVocoder:
        
        tts = TTSService()
        
        # Mock processor output
        mock_tensor = MagicMock()
        mock_tensor.shape = (1, 10)
        mock_inputs = {"input_ids": mock_tensor}
        
        mock_proc_out = MagicMock()
        mock_proc_out.__getitem__.side_effect = mock_inputs.__getitem__
        mock_proc_out.to.return_value = mock_proc_out
        
        MockProc.from_pretrained.return_value = MagicMock(return_value=mock_proc_out)
        
        # Mock speech generation result
        mock_speech = MagicMock()
        mock_speech.cpu().numpy.return_value = np.zeros(16000, dtype=np.float32)
        tts.model.generate_speech.return_value = mock_speech
        
        async def text_gen():
            yield "Hello world."
            yield " How "
            yield "are you today?"
            
        chunks = []
        async for chunk in tts.synthesize_stream(text_gen()):
            chunks.append(chunk)
            
        print(f"Generated {len(chunks)} audio chunks.")
        # "Hello world." is one sentence. "How are you today?" is another.
        # Should have generated audio for both.
        assert len(chunks) > 0
        print("TTS Refactored Logic Test Passed!")

if __name__ == "__main__":
    asyncio.run(test_stt_refactored_logic())
    asyncio.run(test_tts_refactored_logic())
