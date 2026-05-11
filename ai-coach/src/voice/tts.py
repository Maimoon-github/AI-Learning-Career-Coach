"""Kokoro TTS integration."""

# src/voice/tts.py

import os
import sounddevice as sd
from kokoro_onnx import Kokoro


class KokoroTTS:
    """
    Local TTS using Kokoro ONNX — CPU-friendly, zero API cost.
    Voice IDs: af_sky, af_bella, am_adam, bf_emma, bm_george
    """

    def __init__(self):
        self.voice = os.environ.get("TTS_VOICE", "af_sky")
        self.speed = float(os.environ.get("TTS_SPEED", "1.0"))
        self.kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

    def speak(self, text: str) -> None:
        """Generate speech and play it through the system speaker."""
        samples, sample_rate = self.kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang="en-us",
        )
        sd.play(samples, sample_rate)
        sd.wait()

    def synthesize_to_file(self, text: str, output_path: str) -> None:
        """Save speech to a WAV file instead of playing it."""
        from scipy.io.wavfile import write as wav_write
        samples, sample_rate = self.kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        wav_write(output_path, sample_rate, samples)