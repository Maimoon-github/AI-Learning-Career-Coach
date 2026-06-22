# src/services/voice_interface/stt_service.py
from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncGenerator, Optional

import numpy as np
import structlog
import webrtcvad
from faster_whisper import WhisperModel

from .audio_types import AudioChunk

log = structlog.get_logger(__name__)

class STTService:
    """
    Robust, Local Speech‑to‑Text Service powered by faster-whisper.
    Implements advanced VAD with pre-roll buffering and utterance segmentation.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        log.info("initializing_local_stt", model_size=model_size, device=device)
        # Load model with quantization for speed/efficiency on local hardware
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.vad = webrtcvad.Vad(3)  # Aggressive filtering to minimize background noise
        
        # Audio configuration constants
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2
        
        # Utterance detection tuning
        self.silence_threshold_frames = 25  # ~750ms of silence flags end of speech
        self.pre_roll_frames = 10           # ~300ms pre-roll to avoid clipping starts
        self.max_utterance_duration_s = 30  # Safety cap for memory usage
        
        # Circular buffer for pre-roll
        self._pre_roll_buffer: deque[bytes] = deque(maxlen=self.pre_roll_frames)

    async def transcribe_stream(
        self, audio_stream: AsyncGenerator[AudioChunk, None]
    ) -> AsyncGenerator[str, None]:
        """
        Segments and transcribes an incoming stream of PCM chunks in real-time.
        Uses a sliding window for VAD and accumulates audio only when speech is active.
        """
        raw_buffer = b""
        speech_accumulator = b""
        silence_count = 0
        speech_detected = False
        bytes_per_sec = self.sample_rate * 2

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
                        log.debug("voice_activity_started")
                        speech_detected = True
                        # Include pre-roll to capture full onset
                        speech_accumulator = b"".join(self._pre_roll_buffer)
                    
                    silence_count = 0
                    speech_accumulator += frame
                else:
                    self._pre_roll_buffer.append(frame)
                    if speech_detected:
                        silence_count += 1
                        speech_accumulator += frame

                # Cap max utterance length
                if len(speech_accumulator) > self.max_utterance_duration_s * bytes_per_sec:
                    log.warn("max_utterance_limit_reached", duration=self.max_utterance_duration_s)
                    silence_count = self.silence_threshold_frames

                # Trigger inference when user stops talking or limit is hit
                if speech_detected and silence_count >= self.silence_threshold_frames:
                    text = await self._local_transcribe(speech_accumulator)
                    if text:
                        yield text
                    
                    # Reset state for next segment
                    speech_accumulator = b""
                    silence_count = 0
                    speech_detected = False
                    self._pre_roll_buffer.clear()

        # Final flush for trailing audio
        if speech_detected and speech_accumulator:
            text = await self._local_transcribe(speech_accumulator)
            if text:
                yield text

    async def _local_transcribe(self, audio_bytes: bytes) -> str:
        """Runs Whisper inference in a background thread to prevent event loop blocking."""
        if not audio_bytes or len(audio_bytes) < 4000: # Ignore clicks/tiny noise
            return ""
        
        try:
            # Convert PCM s16le to float32 normalized [-1, 1]
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            def perform_inference():
                # language="en" helps keep it fast and deterministic
                segments, _ = self.model.transcribe(
                    audio_np, 
                    beam_size=3, 
                    language="en", 
                    vad_filter=True # Secondary internal VAD check for precision
                )
                return " ".join([s.text for s in segments]).strip()

            result = await asyncio.to_thread(perform_inference)
            if result:
                log.info("stt_inference_complete", length=len(result))
            return result
            
        except Exception as e:
            log.error("local_stt_failure", error=str(e))
            return ""
