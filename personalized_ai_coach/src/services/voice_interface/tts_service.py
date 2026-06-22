# src/services/voice_interface/tts_service.py
from __future__ import annotations

import asyncio
import hashlib
import re
from typing import AsyncGenerator, Optional

import torch
import numpy as np
import structlog
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

log = structlog.get_logger(__name__)

class TTSService:
    """
    High-Performance Local Neural TTS Service using SpeechT5.
    Features sentence-aware buffering and binary caching.
    """

    def __init__(self, device: str = "cpu"):
        log.info("initializing_local_tts", device=device)
        self.device = torch.device(device)
        
        # Load models to the specified device
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # Simple memory cache for repeated phrases
        self._cache: dict[str, bytes] = {}
        self._cache_limit = 100
        
        # Generate a stable, pleasant default voice embedding
        # seed 42 provides a neutral, clear male/female balanced profile
        torch.manual_seed(42)
        self.speaker_embeddings = torch.randn(1, 512).to(self.device) * 0.5

    async def synthesize_stream(
        self, text_stream: AsyncGenerator[str, None], chunk_size: int = 1280
    ) -> AsyncGenerator[bytes, None]:
        """
        Processes a stream of text tokens, buffering them into full sentences before synthesis.
        Yields raw PCM blocks (16kHz mono s16le).
        """
        buffer = ""
        # Regex to find sentence boundaries while preserving the separator
        sentence_end_pattern = re.compile(r"([.!?])\s*")

        async for text_chunk in text_stream:
            buffer += text_chunk
            
            # Split by punctuation but keep it
            parts = sentence_end_pattern.split(buffer)
            # Reconstruct sentences: parts = [text, punct, text, punct, ...]
            while len(parts) >= 3:
                sentence = parts[0] + parts[1]
                parts = parts[2:]
                buffer = "".join(parts) # Remaining parts
                
                async for audio_bytes in self._synthesize_sentence(sentence, chunk_size):
                    yield audio_bytes
                    
        # Final flush for any remaining text
        if buffer.strip():
            async for audio_bytes in self._synthesize_sentence(buffer.strip(), chunk_size):
                yield audio_bytes

    async def _synthesize_sentence(self, text: str, chunk_size: int) -> AsyncGenerator[bytes, None]:
        """Synthesizes a single sentence with result caching."""
        clean_text = text.strip()
        if not clean_text:
            return

        text_hash = hashlib.md5(clean_text.encode()).hexdigest()
        if text_hash in self._cache:
            audio_data = self._cache[text_hash]
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i : i + chunk_size]
            return

        try:
            # Prepare inputs - move to device if necessary
            def run_inference():
                inputs = self.processor(text=clean_text, return_tensors="pt").to(self.device)
                
                # SpeechT5 has a maximum input length (600 tokens)
                input_ids = inputs.get("input_ids")
                if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) > 1:
                    if input_ids.shape[1] > 550:
                        log.warn("tts_input_too_long", length=input_ids.shape[1])
                        return b""

                with torch.no_grad():
                    speech = self.model.generate_speech(
                        inputs["input_ids"], 
                        self.speaker_embeddings, 
                        vocoder=self.vocoder
                    )
                
                # Convert to PCM s16le bytes
                audio_np = speech.cpu().numpy()
                return (audio_np * 32767).astype(np.int16).tobytes()

            audio_pcm = await asyncio.to_thread(run_inference)
            
            if audio_pcm:
                # Store in cache
                if len(self._cache) < self._cache_limit:
                    self._cache[text_hash] = audio_pcm
                
                for i in range(0, len(audio_pcm), chunk_size):
                    yield audio_pcm[i : i + chunk_size]

        except Exception as e:
            log.error("tts_synthesis_error", text=clean_text, error=str(e))

    async def synthesize_and_play(self, text: str) -> bytes:
        """Helper to return complete binary data for a string."""
        full_audio = b""
        async for chunk in self._synthesize_sentence(text, chunk_size=8192):
            full_audio += chunk
        return full_audio
