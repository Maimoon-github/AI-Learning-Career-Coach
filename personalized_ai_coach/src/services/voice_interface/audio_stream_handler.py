from __future__ import annotations

import asyncio
import io
import os
import wave
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_THRESHOLD_FRAMES = 30  # 900ms of silence = end of utterance


# ── Speech-to-Text ────────────────────────────────────────────────────────────

class STTService:
    """
    Real-time Speech-to-Text using OpenAI Whisper (local).
    Implements voice activity detection (VAD) to segment utterances.
    """

    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper
            log.info("stt.loading_model", size=self.model_size)
            self._model = whisper.load_model(self.model_size)
            log.info("stt.model_loaded")
        return self._model

    async def transcribe_audio(self, audio_bytes: bytes, language: str = "en") -> str:
        """Transcribe raw PCM audio bytes to text."""
        model = await asyncio.get_event_loop().run_in_executor(None, self._load_model)

        # Convert raw PCM to numpy float32
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.transcribe(audio_np, language=language, fp16=False),
        )
        text = result.get("text", "").strip()
        log.info("stt.transcribed", text_length=len(text))
        return text

    async def stream_transcribe(
        self, audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        """
        Consume a streaming audio source and yield partial transcriptions
        at utterance boundaries (VAD-detected silence).
        """
        try:
            import webrtcvad
            vad = webrtcvad.Vad(3)  # Aggressiveness 3 = most aggressive filtering
        except ImportError:
            log.warning("webrtcvad_not_available", fallback="simple_chunking")
            vad = None

        buffer = b""
        silence_frames = 0

        async for chunk in audio_stream:
            buffer += chunk

            if vad:
                # Process in 30ms frames for VAD
                while len(buffer) >= FRAME_SIZE * 2:
                    frame = buffer[: FRAME_SIZE * 2]
                    buffer = buffer[FRAME_SIZE * 2 :]
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    if not is_speech:
                        silence_frames += 1
                    else:
                        silence_frames = 0

                    if silence_frames >= SILENCE_THRESHOLD_FRAMES and len(buffer) > 0:
                        text = await self.transcribe_audio(buffer)
                        if text:
                            yield text
                        buffer = b""
                        silence_frames = 0
            else:
                # Fallback: yield every ~3 seconds of audio
                if len(buffer) >= SAMPLE_RATE * 2 * 3:
                    text = await self.transcribe_audio(buffer)
                    if text:
                        yield text
                    buffer = b""

        # Flush remaining buffer
        if len(buffer) > SAMPLE_RATE * 2 * 0.5:  # At least 0.5s of audio
            text = await self.transcribe_audio(buffer)
            if text:
                yield text


# ── Text-to-Speech ────────────────────────────────────────────────────────────

class TTSService:
    """
    Text-to-Speech synthesis. Uses TTS library (Coqui) for local synthesis.
    Streams audio to output device for real-time playback.
    """

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> None:
        self.model_name = model_name
        self._tts = None

    def _load_tts(self):
        if self._tts is None:
            try:
                from TTS.api import TTS
                log.info("tts.loading_model", model=self.model_name)
                self._tts = TTS(model_name=self.model_name, progress_bar=False)
                log.info("tts.model_loaded")
            except ImportError:
                log.warning("TTS_library_not_available")
        return self._tts

    async def synthesize(self, text: str) -> bytes:
        """Convert text to WAV audio bytes."""
        tts = await asyncio.get_event_loop().run_in_executor(None, self._load_tts)
        if tts is None:
            return b""

        def _synth():
            buf = io.BytesIO()
            tts.tts_to_file(text=text, file_path=buf)
            buf.seek(0)
            return buf.read()

        audio_bytes = await asyncio.get_event_loop().run_in_executor(None, _synth)
        log.info("tts.synthesized", text_length=len(text), audio_bytes=len(audio_bytes))
        return audio_bytes

    async def stream_speak(self, text_stream: AsyncGenerator[str, None]) -> None:
        """
        Consume a streaming text source and play audio in near-real-time.
        Buffers text by sentence boundary before synthesis to reduce latency.
        """
        import sounddevice as sd

        sentence_buffer = ""
        async for chunk in text_stream:
            sentence_buffer += chunk
            # Synthesize and play at sentence boundaries
            if any(sentence_buffer.rstrip().endswith(p) for p in (".", "!", "?")):
                if sentence_buffer.strip():
                    audio = await self.synthesize(sentence_buffer.strip())
                    if audio:
                        await self._play_audio(audio)
                sentence_buffer = ""

        # Flush remaining text
        if sentence_buffer.strip():
            audio = await self.synthesize(sentence_buffer.strip())
            if audio:
                await self._play_audio(audio)

    async def _play_audio(self, wav_bytes: bytes) -> None:
        """Play WAV bytes through the default audio output device."""
        try:
            import sounddevice as sd

            def _play():
                with wave.open(io.BytesIO(wav_bytes)) as wf:
                    audio_np = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    sd.play(audio_np, samplerate=wf.getframerate(), blocking=True)

            await asyncio.get_event_loop().run_in_executor(None, _play)
        except Exception as exc:
            log.error("audio_playback_error", error=str(exc))