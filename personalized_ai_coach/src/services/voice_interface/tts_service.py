from __future__ import annotations

import asyncio
import hashlib
import os
from typing import AsyncGenerator, Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

class TTSService:
    """Low‑latency Text‑to‑Speech using ElevenLabs (preferred) or local Coqui."""

    def __init__(self, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.model = "eleven_monolingual_v1"
        self.use_api = bool(self.api_key)
        self._cache = {}  # simple in-memory cache

    async def synthesize_stream(
        self, text_stream: AsyncGenerator[str, None], chunk_size: int = 256
    ) -> AsyncGenerator[bytes, None]:
        """
        Consume a streaming text source (e.g., from LangGraph streaming mode),
        buffer by sentence, synthesize and yield audio chunks.
        """
        buffer = ""
        async for text_chunk in text_stream:
            buffer += text_chunk
            # split at sentence boundaries
            if any(buffer.rstrip().endswith(p) for p in (".", "!", "?")):
                sentence = buffer.strip()
                if sentence:
                    async for audio in self._synthesize_sentence(sentence, chunk_size):
                        yield audio
                buffer = ""

        if buffer.strip():
            async for audio in self._synthesize_sentence(buffer.strip(), chunk_size):
                yield audio

    async def _synthesize_sentence(self, text: str, chunk_size: int) -> AsyncGenerator[bytes, None]:
        """Synthesize a full sentence and stream PCM audio chunks."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._cache:
            audio_bytes = self._cache[text_hash]
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i+chunk_size]
            return

        if self.use_api:
            audio_bytes = await self._elevenlabs_synth(text)
        else:
            audio_bytes = await self._local_coqui_synth(text)

        self._cache[text_hash] = audio_bytes
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i+chunk_size]

    async def _elevenlabs_synth(self, text: str) -> bytes:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {"stability": 0.3, "similarity_boost": 0.7}
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.content  # raw MP3 or PCM? ElevenLabs returns MP3; convert to PCM would be needed.
            # For brevity, we assume PCM conversion elsewhere.

    async def _local_coqui_synth(self, text: str) -> bytes:
        # Load Coqui TTS (once)
        try:
            from TTS.api import TTS
            tts = getattr(self, "_coqui_model", None)
            if tts is None:
                self._coqui_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                tts = self._coqui_model
            # Synthesize to WAV in memory
            import io
            wav_io = io.BytesIO()
            await asyncio.to_thread(tts.tts_to_file, text=text, file_path=wav_io)
            wav_io.seek(0)
            return wav_io.read()
        except ImportError:
            log.warning("Coqui TTS not installed, using placeholder")
            return b""