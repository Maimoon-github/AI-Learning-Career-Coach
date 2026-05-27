from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Callable, Optional, Awaitable

import structlog
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame
from .audio_types import AudioChunk
from .stt_service import STTService
from .tts_service import TTSService

log = structlog.get_logger(__name__)

class AudioStreamHandler:
    """
    Full‑duplex WebRTC audio handler using aiortc.
    Manages a single peer connection, forwards incoming audio to STT,
    and sends synthesized audio from TTS back to the client.
    """

    def __init__(self, stt_service: STTService, tts_service: TTSService):
        self.stt = stt_service
        self.tts = tts_service
        self.pc: Optional[RTCPeerConnection] = None
        self.relay = MediaRelay()
        self._audio_track: Optional[AudioStreamTrack] = None
        self._tts_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    async def handle_offer(self, sdp: str, sdp_type: str = "offer") -> str:
        """Process a WebRTC SDP offer, set up tracks, return answer."""
        self.pc = RTCPeerConnection()

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                log.info("audio_track_received")
                self._audio_track = self.relay.subscribe(track)
                # Launch STT processing
                self._tasks.append(asyncio.create_task(self._process_incoming_audio()))

        # Create an outgoing audio track that reads from TTS queue
        outgoing_track = OutgoingAudioTrack(self._tts_audio_queue)
        self.pc.addTrack(outgoing_track)

        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription.sdp

    async def _process_incoming_audio(self):
        """Read PCM frames from the incoming audio track and feed to STT."""
        if not self._audio_track:
            return
        # Wrapping the track as an async generator of AudioChunk
        async def chunk_generator():
            while True:
                frame: AudioFrame = await self._audio_track.recv()
                # Convert to 16kHz mono PCM (simplified)
                yield AudioChunk(data=frame.to_ndarray().tobytes(), sample_rate=16000, encoding="pcm_s16le")

        async for transcript in self.stt.transcribe_stream(chunk_generator()):
            # Emit transcript event to LangGraph (via callback)
            if self.on_transcript:
                await self.on_transcript(transcript)

    async def send_tts_stream(self, text_stream: AsyncGenerator[str, None]):
        """Feed a stream of text (e.g., from LLM) to TTS and queue audio for outgoing track."""
        async for audio_chunk in self.tts.synthesize_stream(text_stream):
            await self._tts_audio_queue.put(audio_chunk)

    async def close(self):
        """Clean up peer connection and tasks."""
        if self.pc:
            await self.pc.close()
        for task in self._tasks:
            task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # Event callbacks (to be set by main.py)
    on_transcript: Optional[Callable[[str], Awaitable[None]]] = None
    on_error: Optional[Callable[[Exception], Awaitable[None]]] = None
    on_resume: Optional[Callable[[dict], Awaitable[None]]] = None

    async def prompt_hitl(self, presentation: dict):
        """Present HITL options via voice and collect user response."""
        # Synthesise the report summary
        report = presentation.get("weekly_report", {})
        summary = f"Week {presentation.get('current_week')} is complete. {report.get('headline_stat', '')}"
        await self.tts.synthesize_and_play(summary)
        # Ask for decision
        await self.tts.synthesize_and_play("Do you approve, revise, or end this session?")
        # Listen for response
        response = await self._listen_for_hitl_response()
        # Resume workflow with user's decision
        if self.on_resume:
            await self.on_resume(response)

    async def _listen_for_hitl_response(self) -> dict:
        """Wait for STT to recognise approve/revise/end."""
        # Simplified: would wait for a transcript from the STT stream
        # For production, use a queue and a timeout
        import asyncio
        # In real implementation, this would listen to the STT stream.
        # For demo, return a default.
        await asyncio.sleep(0.5)
        return {"hitl_action": "approve", "user_feedback": None}


class OutgoingAudioTrack(AudioStreamTrack):
    """Custom audio track that reads PCM chunks from a queue and frames them."""

    def __init__(self, queue: asyncio.Queue[bytes]):
        super().__init__()
        self.queue = queue
        self._timestamp = 0
        self._sample_rate = 16000
        self._channels = 1

    async def recv(self):
        data = await self.queue.get()
        # Create an AudioFrame
        frame = AudioFrame(format="s16", layout="mono", samples=len(data)//2)
        frame.planes[0].update(data)
        frame.sample_rate = self._sample_rate
        frame.time_base = 1 / self._sample_rate
        frame.pts = self._timestamp
        self._timestamp += frame.samples
        return frame



