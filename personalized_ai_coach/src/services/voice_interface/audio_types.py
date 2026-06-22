# src/services/voice_interface/audio_types.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AudioChunk:
    """
    Production-grade data container for PCM audio streams.
    Ensures immutable passing of audio blocks between WebRTC and ML services.
    """
    data: bytes
    sample_rate: int
    encoding: str = "pcm_s16le"
