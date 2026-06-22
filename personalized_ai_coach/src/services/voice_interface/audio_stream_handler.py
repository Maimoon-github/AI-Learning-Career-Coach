# src/services/voice_interface/audio_stream_handler.py
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Callable, Optional, Awaitable

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
    """Full‑duplex open-source WebRTC audio handler with zero cloud dependencies."""

    def __init__(self, stt_service: STTService, tts_service: TTSService):
        self.stt = stt_service
        self.tts = tts_service
        self.pc: Optional[RTCPeerConnection] = None
        self.relay = MediaRelay()
        self._audio_track: Optional[AudioStreamTrack] = None
        self._tts_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        
        # Callbacks hooked up via main orchestration runtime
        self.on_transcript: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_error: Optional[Callable[[Exception], Awaitable[None]]] = None
        self.on_resume: Optional[Callable[[dict], Awaitable[None]]] = None

    async def handle_offer(self, sdp: str, sdp_type: str = "offer") -> str:
        """Set up the remote/local description and maps full-duplex WebRTC tracks."""
        self.pc = RTCPeerConnection()

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                log.info("incoming_webrtc_audio_track_registered")
                self._audio_track = self.relay.subscribe(track)
                self._tasks.append(asyncio.create_task(self._process_incoming_audio()))

        # Build and configure the outgoing local neural audio generator track
        outgoing_track = OutgoingAudioTrack(self._tts_audio_queue)
        self.pc.addTrack(outgoing_track)

        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription.sdp

    async def _process_incoming_audio(self):
        """Streams received WebRTC PyAV matrix frames directly down into the local STT processor."""
        if not self._audio_track:
            return

        import av
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        async def chunk_generator():
            while True:
                try:
                    frame: AudioFrame = await self._audio_track.recv()
                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        yield AudioChunk(
                            data=r_frame.to_ndarray().tobytes(), 
                            sample_rate=16000, 
                            encoding="pcm_s16le"
                        )
                except Exception as e:
                    log.error("incoming_audio_processing_failed", error=str(e))
                    break

        try:
            async for transcript in self.stt.transcribe_stream(chunk_generator()):
                if self.on_transcript and transcript.strip():
                    await self.on_transcript(transcript)
        except Exception as e:
            if self.on_error:
                await self.on_error(e)

    async def send_tts_stream(self, text_stream: AsyncGenerator[str, None]):
        """Bridges a generative string token generator directly into the local TTS queue."""
        async for audio_chunk in self.tts.synthesize_stream(text_stream):
            await self._tts_audio_queue.put(audio_chunk)

    async def prompt_hitl(self, presentation: dict):
        """Voice prompts the Human-In-The-Loop layer directly using the local pipeline."""
        report = presentation.get("weekly_report", {})
        summary = f"Week {presentation.get('current_week')} processing complete. {report.get('headline_stat', '')}."
        
        # Generate raw binary blocks offline
        summary_pcm = await self.tts.synthesize_and_play(summary)
        prompt_pcm = await self.tts.synthesize_and_play("Do you approve, revise, or end this session?")
        
        # Send raw chunks out to WebRTC queue elements
        await self._tts_audio_queue.put(summary_pcm)
        await self._tts_audio_queue.put(prompt_pcm)
        
        response = await self._listen_for_hitl_response()
        if self.on_resume:
            await self.on_resume(response)

    async def _listen_for_hitl_response(self) -> dict:
        """Placeholder for human workflow interactive loop evaluation."""
        await asyncio.sleep(0.5)
        return {"hitl_action": "approve", "user_feedback": None}

    async def close(self):
        """Safely disposes loops, active tracks, and peer socket mappings."""
        if self.pc:
            await self.pc.close()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


class OutgoingAudioTrack(AudioStreamTrack):
    """Custom audio track slicing continuous raw local PCM queues into WebRTC packets."""

    def __init__(self, queue: asyncio.Queue[bytes]):
        super().__init__()
        self.queue = queue
        self._timestamp = 0
        self._sample_rate = 16000
        # 20ms frames at 16kHz = 320 samples per frame (640 bytes total per WebRTC cycle)
        self._frame_samples = 320 
        self._bytes_per_frame = self._frame_samples * 2
        self._remainder = b""

    async def recv(self) -> AudioFrame:
        """Retrieves and slices queued data into precise 20ms chunks expected by WebRTC."""
        while len(self._remainder) < self._bytes_per_frame:
            data = await self.queue.get()
            self._remainder += data
            self.queue.task_done()

        # Slice off an exact 20ms packet segment
        chunk = self._remainder[:self._bytes_per_frame]
        self._remainder = self._remainder[self._bytes_per_frame:]

        # Initialize PyAV internal media frames
        frame = AudioFrame(format="s16", layout="mono", samples=self._frame_samples)
        frame.planes[0].update(chunk)
        frame.sample_rate = self._sample_rate
        frame.time_base = 1 / self._sample_rate
        frame.pts = self._timestamp
        
        self._timestamp += self._frame_samples
        return frame
