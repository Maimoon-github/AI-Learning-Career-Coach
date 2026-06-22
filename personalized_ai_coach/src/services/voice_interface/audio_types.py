# src/services/voice_interface/audio_types.py
from __future__ import annotations

class AudioChunk:
    """Simple offline data container passing PCM streams between WebRTC and STT."""
    def __init__(self, data: bytes, sample_rate: int, encoding: str):
        self.data = data
        self.sample_rate = sample_rate
        self.encoding = encoding
