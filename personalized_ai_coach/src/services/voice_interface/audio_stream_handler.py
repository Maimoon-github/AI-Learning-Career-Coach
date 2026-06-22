# src/services/voice_interface/audio_stream_handler.py
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Callable, Optional, Awaitable

import av
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
    Full‑duplex WebRTC Audio Manager.
    Bridges local open-source STT/TTS services with browser-based RTC tracks.
    """

    def __init__(self, stt_service: STTService, tts_service: TTSService):
        self.stt = stt_service
        self.tts = tts_service
        
        self.pc: Optional[RTCPeerConnection] = None
        self._audio_track: Optional[AudioStreamTrack] = None
        self._tts_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._tasks: set[asyncio.Task] = set()
        self.relay = MediaRelay()
        
        # High-level event hooks
        self.on_transcript: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_error: Optional[Callable[[Exception], Awaitable[None]]] = None
        self.on_resume: Optional[Callable[[dict], Awaitable[None]]] = None

    async def handle_offer(self, sdp: str, sdp_type: str = "offer") -> str:
        """
        Processes a WebRTC offer and returns a corresponding answer SDP.
        Orchestrates track registration for bidirectional voice.
        """
        self.pc = RTCPeerConnection()
        log.info("webrtc_offer_received", type=sdp_type)

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                log.info("incoming_audio_track_detected")
                self._audio_track = self.relay.subscribe(track)
                task = asyncio.create_task(self._process_incoming_audio())
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        # Register the outgoing synthesized audio track
        outgoing_track = OutgoingAudioTrack(self._tts_queue)
        self.pc.addTrack(outgoing_track)

        # Negotiate connection
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp, sdp_type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        return self.pc.localDescription.sdp

    async def _process_incoming_audio(self):
        """Resamples and streams RTC audio frames into the STT engine."""
        if not self._audio_track:
            return

        # Initialize resampler (WebRTC is typically 48kHz, we need 16kHz)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        async def generate_chunks():
            while True:
                try:
                    frame: AudioFrame = await self._audio_track.recv()
                    resampled = resampler.resample(frame)
                    for r_frame in resampled:
                        yield AudioChunk(
                            data=r_frame.to_ndarray().tobytes(),
                            sample_rate=16000
                        )
                except Exception as e:
                    log.debug("audio_track_stopped", error=str(e))
                    break

        try:
            async for transcript in self.stt.transcribe_stream(generate_chunks()):
                if self.on_transcript:
                    await self.on_transcript(transcript)
        except Exception as e:
            log.error("stt_stream_failure", error=str(e))
            if self.on_error:
                await self.on_error(e)

    async def send_tts_stream(self, text_stream: AsyncGenerator[str, None]):
        """Consumes a text token generator and queues the resulting audio bytes."""
        try:
            async for audio_chunk in self.tts.synthesize_stream(text_stream):
                await self._tts_queue.put(audio_chunk)
        except Exception as e:
            log.error("tts_stream_failure", error=str(e))

    async def prompt_hitl(self, presentation: dict):
        """Utility for Human-In-The-Loop voice probing."""
        msg = f"Week {presentation.get('current_week')} check. Approve or refine?"
        audio = await self.tts.synthesize_and_play(msg)
        await self._tts_queue.put(audio)
        
        # In a real impl, this would wait for a 'yes/no' transcript
        if self.on_resume:
            await self.on_resume({"hitl_action": "approve"})

    async def close(self):
        """Gracefully terminates WebRTC connection and background workers."""
        if self.pc:
            await self.pc.close()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("webrtc_handler_closed")


class OutgoingAudioTrack(AudioStreamTrack):
    """
    Slices raw PCM segments from a queue into standardized 20ms WebRTC packets.
    Ensures continuous, clock-synchronized audio playback in the browser.
    """

    def __init__(self, queue: asyncio.Queue[bytes]):
        super().__init__()
        self.queue = queue
        self._sample_rate = 16000
        self._frame_samples = 320 # 20ms at 16kHz
        self._bytes_per_frame = self._frame_samples * 2
        self._timestamp = 0
        self._leftover = b""

    async def recv(self) -> AudioFrame:
        """Blocks until a 20ms frame is ready to be sent."""
        while len(self._leftover) < self._bytes_per_frame:
            # Block until more audio is synthesized
            data = await self.queue.get()
            self._leftover += data
            self.queue.task_done()

        # Extract exactly 20ms
        packet = self._leftover[:self._bytes_per_frame]
        self._leftover = self._leftover[self._bytes_per_frame:]

        # Build PyAV frame
        frame = AudioFrame(format="s16", layout="mono", samples=self._frame_samples)
        frame.planes[0].update(packet)
        frame.sample_rate = self._sample_rate
        frame.time_base = 1 / self._sample_rate
        frame.pts = self._timestamp
        
        self._timestamp += self._frame_samples
        return frame
