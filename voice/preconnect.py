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
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections import deque
from collections.abc import Iterable
from typing import Any

import numpy as np
import livekit.agents.voice.room_io.room_io as upstream
from livekit import rtc
from livekit.agents.voice import room_io
from livekit.agents.voice.room_io._input import _ParticipantAudioInputStream

logger = logging.getLogger("mentat.voice")

REPEAT_TAIL_SECONDS = (0.2, 0.4, 0.6)
REPEAT_SEARCH_EXTRA_SECONDS = 0.2
REPEAT_AGREE_SECONDS = 0.03
REPEAT_MIN_CORRELATION = 0.3


def _part(frame: Any, begin: int, end: int) -> Any:
    """The frame's samples per channel from begin to end, as a frame of their own."""
    if begin == 0 and end == frame.samples_per_channel:
        return frame
    width = frame.num_channels * 2
    data = bytes(frame.data)[begin * width:end * width]
    return rtc.AudioFrame(data, frame.sample_rate, frame.num_channels, end - begin)


def cut_repeat(buffer: np.ndarray, live: np.ndarray, rate: int, heard_before_open: int) -> tuple[int, str]:
    """How many live samples repeat the pre-connect buffer, and the path that decided it.

    The live audio opens by repeating the buffer's last half second or so, time-stretched and
    reshaped on the way through the jitter buffer. Each of the buffer's last 0.2, 0.4 and 0.6 s is
    matched within the first len(buffer) plus 0.2 s of live by zero-mean normalized
    cross-correlation. When all three correlate at least 0.3 and end within 30 ms of each other,
    the repeat ends at their median end ("aligned"). Otherwise the repeat is the live audio heard
    before the buffer opened, at most the buffer's length ("timing").
    """
    timing = min(heard_before_open, len(buffer)), "timing"
    window = live[:min(len(live), len(buffer) + round(REPEAT_SEARCH_EXTRA_SECONDS * rate))]
    ends = []
    for seconds in REPEAT_TAIL_SECONDS:
        length = round(seconds * rate)
        if length > len(buffer) or length > len(window):
            return timing
        end, correlation = _best_match(buffer[-length:], window)
        if correlation < REPEAT_MIN_CORRELATION:
            return timing
        ends.append(end)
    if max(ends) - min(ends) > round(REPEAT_AGREE_SECONDS * rate):
        return timing
    return sorted(ends)[1], "aligned"


def _best_match(tail: np.ndarray, window: np.ndarray) -> tuple[int, float]:
    """Where the tail ends in the window at its best zero-mean normalized correlation, and that correlation."""
    length = len(tail)
    centred = tail.astype(np.float64) - tail.mean()
    tail_energy = float(np.dot(centred, centred))
    if tail_energy == 0.0:
        return 0, 0.0
    samples = window.astype(np.float64)
    products = np.fft.irfft(np.fft.rfft(samples) * np.conj(np.fft.rfft(centred, len(samples))), len(samples))
    products = products[:len(samples) - length + 1]
    sums = np.concatenate([[0.0], np.cumsum(samples)])
    squares = np.concatenate([[0.0], np.cumsum(samples * samples)])
    window_sums = sums[length:] - sums[:-length]
    window_squares = squares[length:] - squares[:-length]
    spread = np.maximum(window_squares - window_sums * window_sums / length, 0.0)
    denominator = np.sqrt(spread * tail_energy)
    correlations = np.divide(products, denominator, out=np.zeros_like(products), where=denominator > 0.0)
    best = int(np.argmax(correlations))
    return best + length, float(correlations[best])


class _LiveAudio:
    """The live track as the audio input reads it, with the repeat dropped.

    Stands in for rtc.AudioStream, which the input only iterates and closes. The stream is
    read from creation, so every frame is stamped with the time it arrived.
    """

    def __init__(self, stream: Any, *, sample_rate: int) -> None:
        self._stream = stream
        self._sample_rate = sample_rate
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

    def process_buffer(self, frames: Iterable[Any]) -> list[Any]:
        """Drop the live audio heard before the buffer completed; return the buffer as it arrived.

        The buffer comes back in order and is consumed before the live audio is read.
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
        return buffer

    def __aiter__(self) -> _LiveAudio:
        return self

    async def __anext__(self) -> rtc.AudioFrameEvent:
        while True:
            kept = self._keep(await self._next_stamped())
            if kept is not None:
                return rtc.AudioFrameEvent(kept)

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

    def _create_stream(self, track: rtc.Track, participant: rtc.Participant) -> Any:
        stream = rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            frame_size_ms=self._frame_size_ms,
        )
        return _LiveAudio(stream, sample_rate=self._sample_rate)

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
