"""The worker's microphone input, with the phone's pre-connect buffer heard once.

The phone records the first words into a buffer while it connects and sends it as one
byte stream; its live track then starts a moment before the end of that buffer, so the
live audio opens by repeating the buffer's last half second or so. LiveKit feeds both to
the speech-to-text and the first words arrive twice. This input drops the repeat by
timing: a live frame the worker received before the buffer finished arriving is the
repeat, so the live frames heard before that moment are dropped, at most the buffer's
length in total, counted from the first live frame. Every later live frame passes.

Frames carry no receive time, so each live frame is stamped as it comes off the track,
from the moment the track is created. The buffer's completion is the moment LiveKit hands
it to the input.

The input also runs one noise canceller over the buffer and then the kept live audio, in
order: LiveKit hands the canceller to its audio stream, which would run it on live frames
while the buffer is still being processed.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

import livekit.agents.voice.room_io.room_io as upstream
from livekit import rtc
from livekit.agents.voice import room_io
from livekit.agents.voice.room_io._input import _ParticipantAudioInputStream
from livekit.agents.voice.room_io.types import NoiseCancellationParams

logger = logging.getLogger("mentat.voice")


def _part(frame: Any, begin: int, end: int) -> Any:
    """The frame's samples per channel from begin to end, as a frame of their own."""
    if begin == 0 and end == frame.samples_per_channel:
        return frame
    width = frame.num_channels * 2
    data = bytes(frame.data)[begin * width:end * width]
    return rtc.AudioFrame(data, frame.sample_rate, frame.num_channels, end - begin)


class _LiveAudio:
    """The live track as the audio input reads it: the repeat dropped, noise cancelled in order.

    Stands in for rtc.AudioStream, which the input only iterates and closes. The stream is
    read from creation, so every frame is stamped with the time it arrived.
    """

    def __init__(
        self,
        stream: Any,
        *,
        sample_rate: int,
        cancel: Callable[[rtc.AudioFrame], rtc.AudioFrame],
    ) -> None:
        self._stream = stream
        self._sample_rate = sample_rate
        self._cancel = cancel
        self._stamped: deque[tuple[float, Any]] = deque()
        self._changed = asyncio.Event()
        self._ended = False
        self._first: float | None = None
        self._skip = 0
        self._stamping = asyncio.create_task(self._stamp())

    async def _stamp(self) -> None:
        try:
            while True:
                try:
                    event = await self._stream.__anext__()
                except StopAsyncIteration:
                    return
                now = time.monotonic()
                if self._first is None:
                    self._first = now
                self._stamped.append((now, event.frame))
                self._changed.set()
        finally:
            self._ended = True
            self._changed.set()

    def process_buffer(self, frames: Iterable[Any]) -> Iterable[Any]:
        """Drop the live audio heard before the buffer completed, and cancel noise from the buffer.

        The returned frames are the buffer through the canceller; they are consumed before the
        live audio is read.
        """
        buffer = list(frames)
        completed = time.monotonic()
        heard = 0
        for stamp, frame in self._stamped:
            if stamp >= completed:
                break
            heard += frame.samples_per_channel
        seconds = 0.0
        if buffer:
            seconds = sum(frame.samples_per_channel for frame in buffer) / buffer[0].sample_rate
        self._skip = min(round(seconds * self._sample_rate), heard)
        logger.info(
            "dropped %.2f s of live audio heard before the pre-connect buffer completed: "
            "first live frame at %s, buffer completed at %.3f",
            self._skip / self._sample_rate,
            "none" if self._first is None else f"{self._first:.3f}",
            completed,
        )
        return map(self._cancel, buffer)

    def __aiter__(self) -> _LiveAudio:
        return self

    async def __anext__(self) -> rtc.AudioFrameEvent:
        while True:
            kept = self._keep(await self._next_stamped())
            if kept is not None:
                return rtc.AudioFrameEvent(self._cancel(kept))

    async def aclose(self) -> None:
        self._stamping.cancel()
        await self._stream.aclose()

    async def _next_stamped(self) -> Any:
        while not self._stamped:
            if self._ended:
                self._raise_end()
            self._changed.clear()
            await self._changed.wait()
        return self._stamped.popleft()[1]

    def _raise_end(self) -> None:
        """End the iteration, or re-raise the error the stamping stopped with."""
        if not self._stamping.cancelled():
            error = self._stamping.exception()
            if error is not None:
                raise error
        raise StopAsyncIteration

    def _keep(self, frame: Any) -> Any | None:
        """The frame without the samples still to drop, or None when the whole frame is dropped."""
        count = frame.samples_per_channel
        dropped = min(self._skip, count)
        self._skip -= dropped
        if dropped and dropped == count:
            return None
        return _part(frame, dropped, count)


_LIVE: contextvars.ContextVar[_LiveAudio] = contextvars.ContextVar("preconnect_live_audio")


class _PreConnectAudioInput(_ParticipantAudioInputStream):
    """LiveKit's participant audio input, reading the track through _LiveAudio."""

    def _cancel_noise(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        processor = self._processor
        if processor is None or not processor.enabled:
            return frame
        try:
            return processor._process(frame)
        except Exception:
            logger.warning("noise cancellation failed, passing the frame through", exc_info=True)
            return frame

    def _create_stream(self, track: rtc.Track, participant: rtc.Participant) -> Any:
        noise_cancellation = self._noise_cancellation
        if callable(noise_cancellation):
            noise_cancellation = noise_cancellation(NoiseCancellationParams(participant, track))
            self._update_processor(
                noise_cancellation if isinstance(noise_cancellation, rtc.FrameProcessor) else None
            )
        stream = rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            frame_size_ms=self._frame_size_ms,
            noise_cancellation=(
                None if isinstance(noise_cancellation, rtc.FrameProcessor) else noise_cancellation
            ),
        )
        return _LiveAudio(stream, sample_rate=self._sample_rate, cancel=self._cancel_noise)

    async def _forward_task(self, old_task: Any, stream: Any, *rest: Any) -> None:
        _LIVE.set(stream)
        await super()._forward_task(old_task, stream, *rest)

    def _apply_audio_processor(self, frames: Iterable[rtc.AudioFrame]) -> Iterable[rtc.AudioFrame]:
        return _LIVE.get().process_buffer(frames)


class PreConnectRoomIO(room_io.RoomIO):
    """RoomIO whose microphone input hears the pre-connect buffer once."""

    async def start(self) -> None:
        # RoomIO.start builds its audio input from this module global and never yields to
        # the loop while it does, so the swap cannot be seen by another RoomIO.
        stock = upstream._ParticipantAudioInputStream
        upstream._ParticipantAudioInputStream = _PreConnectAudioInput
        try:
            await super().start()
        finally:
            upstream._ParticipantAudioInputStream = stock
