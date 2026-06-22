# src/services/voice_interface/stt_service.py
from __future__ import annotations

import asyncio
from typing import AsyncGenerator
import numpy as np
import structlog
import webrtcvad
from faster_whisper import WhisperModel

from .audio_types import AudioChunk

log = structlog.get_logger(__name__)

class STTService:
    """Streaming Local Speech‑to‑Text powered by faster-whisper (CTranslate2)."""

    def __init__(self, model_size: str = "base"):
        # Runs entirely locally. Device will auto-fallback to CPU if CUDA is absent.
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.vad = webrtcvad.Vad(3)  # Most aggressive filtering mode
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # 960 bytes
        self.silence_threshold_frames = 25  # ~750ms of silence flags end of utterance

    async def transcribe_stream(
        self, audio_stream: AsyncGenerator[AudioChunk, None]
    ) -> AsyncGenerator[str, None]:
        """Consumes PCM audio chunks, runs VAD, and performs fast local transcription."""
        raw_buffer = b""
        speech_accumulator = b""
        silence_count = 0
        speech_detected = False

        async for chunk in audio_stream:
            if chunk.encoding != "pcm_s16le" or chunk.sample_rate != self.sample_rate:
                continue
            raw_buffer += chunk.data

            while len(raw_buffer) >= self.frame_size:
                frame = raw_buffer[:self.frame_size]
                raw_buffer = raw_buffer[self.frame_size:]

                is_speech = self.vad.is_speech(frame, self.sample_rate)
                
                if is_speech:
                    if not speech_detected:
                        log.debug("speech_started")
                    speech_detected = True
                    silence_count = 0
                    speech_accumulator += frame
                else:
                    silence_count += 1
                    if speech_detected:
                        speech_accumulator += frame

                # Trigger inference when user stops talking after an utterance
                if speech_detected and silence_count >= self.silence_threshold_frames:
                    log.debug("speech_ended", buffer_size=len(speech_accumulator))
                    text = await self._local_transcribe(speech_accumulator)
                    if text.strip():
                        yield text
                    
                    # Reset state for next utterance
                    speech_accumulator = b""
                    silence_count = 0
                    speech_detected = False

        # Flush any trailing voice buffer elements
        if speech_detected and speech_accumulator:
            text = await self._local_transcribe(speech_accumulator)
            if text.strip():
                yield text

    async def _local_transcribe(self, audio_bytes: bytes) -> str:
        """Processes raw byte blocks inside a thread pool via faster-whisper."""
        if not audio_bytes:
            return ""
        
        try:
            # Convert binary s16le PCM back into normalized float32 numpy arrays for Whisper
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            def run_inference():
                segments, _ = self.model.transcribe(audio_np, beam_size=3, language="en")
                return "".join([seg.text for seg in segments]).strip()

            return await asyncio.to_thread(run_inference)
        except Exception as e:
            log.error("local_stt_inference_failed", error=str(e))
            return ""
