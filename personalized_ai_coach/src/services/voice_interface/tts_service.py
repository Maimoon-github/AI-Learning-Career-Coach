# src/services/voice_interface/tts_service.py
from __future__ import annotations

import asyncio
import hashlib
from typing import AsyncGenerator
import torch
import numpy as np
import structlog
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

log = structlog.get_logger(__name__)

class TTSService:
    """Low‑latency Local Neural Text‑to‑Speech using Hugging Face SpeechT5."""

    def __init__(self):
        log.info("initializing_local_neural_tts")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self._cache: dict[str, bytes] = {}

        # Construct a high-quality deterministic voice embedding completely offline
        torch.manual_seed(42)
        self.speaker_embeddings = torch.randn(1, 512) * 0.4

    async def synthesize_stream(
        self, text_stream: AsyncGenerator[str, None], chunk_size: int = 640
    ) -> AsyncGenerator[bytes, None]:
        """Buffers tokens into complete sentences to stream clean synthesized PCM chunks."""
        buffer = ""
        async for text_chunk in text_stream:
            buffer += text_chunk
            if any(buffer.rstrip().endswith(p) for p in (".", "!", "?")):
                sentence = buffer.strip()
                if sentence:
                    async for audio_bytes in self._synthesize_sentence(sentence, chunk_size):
                        yield audio_bytes
                buffer = ""

        if buffer.strip():
            async for audio_bytes in self._synthesize_sentence(buffer.strip(), chunk_size):
                yield audio_bytes

    async def _synthesize_sentence(self, text: str, chunk_size: int) -> AsyncGenerator[bytes, None]:
        """Synthesizes text sentences into raw 16kHz mono pcm_s16le binary streams."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._cache:
            audio_data = self._cache[text_hash]
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i+chunk_size]
            return

        try:
            # Tokenize text inputs
            inputs = self.processor(text=text, return_tensors="pt")
            
            # Execute model synthesis asynchronously inside the CPU/GPU background thread
            speech_tensor = await asyncio.to_thread(
                self.model.generate_speech,
                inputs["input_ids"],
                self.speaker_embeddings,
                vocoder=self.vocoder
            )

            # Convert PyTorch float32 output wave into standard 16-bit linear PCM
            audio_np = speech_tensor.cpu().numpy()
            audio_pcm = (audio_np * 32767).astype(np.int16).tobytes()
            self._cache[text_hash] = audio_pcm

            for i in range(0, len(audio_pcm), chunk_size):
                yield audio_pcm[i:i+chunk_size]

        except Exception as e:
            log.error("local_tts_generation_failed", text=text, error=str(e))

    async def synthesize_and_play(self, text: str) -> bytes:
        """Utility method to return a full string synthesis directly."""
        full_audio = b""
        async for chunk in self._synthesize_sentence(text, chunk_size=4096):
            full_audio += chunk
        return full_audio
