"""Whisper STT integration."""

# src/voice/stt.py

import whisper
import sounddevice as sd
import numpy as np
import os
from scipy.io.wavfile import write as wav_write
import tempfile


class WhisperSTT:
    """Local speech-to-text using OpenAI Whisper (no API key, fully offline)."""

    def __init__(self, model_size: str | None = None):
        model_size = model_size or os.environ.get("WHISPER_MODEL", "base.en")
        self.model = whisper.load_model(model_size)

    def record_audio(self, duration_seconds: int = 10, sample_rate: int = 16000) -> np.ndarray:
        """Record from microphone. Returns audio as numpy array."""
        print(f"🎙️  Recording for {duration_seconds}s... (press Ctrl+C to stop early)")
        audio = sd.rec(
            int(duration_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy audio array to text."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_write(tmp.name, sample_rate, (audio * 32767).astype(np.int16))
            result = self.model.transcribe(tmp.name, fp16=False)
        return result["text"].strip()

    def listen_and_transcribe(self, duration_seconds: int = 10) -> str:
        """Convenience method: record then transcribe."""
        audio = self.record_audio(duration_seconds)
        return self.transcribe_audio(audio)