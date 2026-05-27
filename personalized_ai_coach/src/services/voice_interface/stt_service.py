from __future__ import annotations

import asyncio
import io
import os
from typing import AsyncGenerator, Optional

import httpx
import numpy as np
import structlog
import webrtcvad

from .audio_stream_handler import AudioChunk

log = structlog.get_logger(__name__)

class STTService:
    """Streaming Speech‑to‑Text using OpenAI Whisper API (or local fallback)."""

    def __init__(self, use_api: bool = True):
        self.use_api = use_api and bool(os.getenv("OPENAI_API_KEY"))
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/audio/transcriptions"
        self.vad = webrtcvad.Vad(3)  # most aggressive
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # bytes
        self.silence_threshold_frames = 30  # 900ms silence

    async def transcribe_stream(
        self, audio_stream: AsyncGenerator[AudioChunk, None]
    ) -> AsyncGenerator[str, None]:
        """
        Accepts chunks of PCM audio (16kHz mono, int16), detects speech segments,
        and yields transcribed text segments.
        """
        buffer = b""
        silence_count = 0

        async for chunk in audio_stream:
            if chunk.encoding != "pcm_s16le" or chunk.sample_rate != self.sample_rate:
                # simple resampling/stub – real conversion omitted for brevity
                continue
            buffer += chunk.data

            while len(buffer) >= self.frame_size:
                frame = buffer[:self.frame_size]
                buffer = buffer[self.frame_size:]

                is_speech = self.vad.is_speech(frame, self.sample_rate)
                if not is_speech:
                    silence_count += 1
                else:
                    silence_count = 0

                if silence_count >= self.silence_threshold_frames and len(buffer) > 0:
                    # end of utterance
                    text = await self._transcribe(buffer)
                    if text.strip():
                        yield text
                    buffer = b""
                    silence_count = 0

        # flush remaining
        if len(buffer) > self.frame_size // 2:
            text = await self._transcribe(buffer)
            if text.strip():
                yield text

    async def _transcribe(self, audio_bytes: bytes) -> str:
        """Send audio bytes to Whisper API or local model."""
        if self.use_api:
            return await self._api_transcribe(audio_bytes)
        else:
            return await self._local_transcribe(audio_bytes)

    async def _api_transcribe(self, audio_bytes: bytes) -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
            data = {"model": "whisper-1", "language": "en"}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = await client.post(self.base_url, files=files, data=data, headers=headers)
            resp.raise_for_status()
            return resp.json().get("text", "")

    async def _local_transcribe(self, audio_bytes: bytes) -> str:
        # Fallback: use local whisper (load model once)
        import whisper
        model = getattr(self, "_local_model", None)
        if model is None:
            self._local_model = whisper.load_model("base")
            model = self._local_model
        # Convert bytes to numpy float32
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = await asyncio.to_thread(model.transcribe, audio_np, language="en", fp16=False)
        return result["text"].strip()