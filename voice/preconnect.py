"""The worker's microphone input, with the phone's pre-connect buffer heard once.

The phone records the first words into a buffer while it connects and sends it as one
byte stream; its live track then starts a moment before the end of that buffer, so the
live audio opens by repeating the buffer's last half second or so. LiveKit feeds both to
the speech-to-text and the first words arrive twice. This input drops the repeat by
aligning the buffer's tail with the live head (see cut_repeat). Once the buffer arrives, the
live track is held until its head covers the buffer's length plus 0.2 s, so the alignment
has the whole window, or the track ends, or that span has passed. When no tail aligns, the
repeat falls back to timing: a live frame the worker received before the buffer's stream
opened is the repeat, so the live frames heard before that moment are dropped, at most the
buffer's length in total. Every frame after the cut passes.

Frames carry no receive time, so each live frame is stamped as it comes off the track,
from the moment the track is created. The buffer's stream opening is recorded by the
handler as the phone's byte stream arrives, per track.
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
from livekit.agents.voice.room_io._pre_connect_audio import PreConnectAudioHandler

logger = logging.getLogger("mentat.voice")

REPEAT_TAIL_SECONDS = (0.2, 0.4, 0.6)
REPEAT_SEARCH_EXTRA_SECONDS = 0.2
REPEAT_AGREE_SECONDS = 0.05
REPEAT_MIN_CORRELATION = 0.3
REPEAT_FRAME_SECONDS = 0.01
REPEAT_VOICED_FRACTION = 0.05


def _part(frame: Any, begin: int, end: int) -> Any:
    """The frame's samples per channel from begin to end, as a frame of their own."""
    if begin == 0 and end == frame.samples_per_channel:
        return frame
    width = frame.num_channels * 2
    data = bytes(frame.data)[begin * width:end * width]
    return rtc.AudioFrame(data, frame.sample_rate, frame.num_channels, end - begin)


def _seconds(frames: list[Any]) -> float:
    return sum(frame.samples_per_channel for frame in frames) / frames[0].sample_rate


def _mono_at_rate(frames: list[Any], rate: int) -> np.ndarray:
    """The frames as one mono int16 array at the given rate, the way the live track is heard."""
    samples = np.concatenate([np.frombuffer(bytes(frame.data), dtype=np.int16) for frame in frames])
    samples = samples.astype(np.float64)
    channels = frames[0].num_channels
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    source = frames[0].sample_rate
    if source != rate:
        at = np.arange(round(len(samples) * rate / source)) * (source / rate)
        samples = np.interp(at, np.arange(len(samples)), samples)
    return np.round(samples).astype(np.int16)


def _voiced_end(buffer: np.ndarray, rate: int) -> int:
    """Where the buffer's last voiced 10 ms frame ends, in samples.

    A frame is voiced when its RMS is at least 5% of the buffer's 95th-percentile frame RMS.
    """
    step = round(REPEAT_FRAME_SECONDS * rate)
    levels = np.array([
        np.sqrt(np.mean(np.square(buffer[at:at + step].astype(np.float64))))
        for at in range(0, len(buffer), step)
    ])
    if levels.size == 0:
        return 0
    voiced = np.flatnonzero(levels >= REPEAT_VOICED_FRACTION * np.percentile(levels, 95))
    return min((int(voiced[-1]) + 1) * step, len(buffer))


def cut_repeat(buffer: np.ndarray, live: np.ndarray, rate: int, heard_before_open: int) -> tuple[int, str]:
    """How many live samples repeat the pre-connect buffer, and the path that decided it.

    The live audio opens by repeating the buffer's last half second or so, time-stretched and
    reshaped on the way through the jitter buffer. The buffer's voiced part ends at its last
    voiced 10 ms frame, and the quiet span after it is a pause the repeat may carry too. Each of
    the voiced part's last 0.2, 0.4 and 0.6 s is matched within the first len(buffer) plus 0.2 s of
    live by zero-mean normalized cross-correlation. When all three correlate at least 0.3 and end
    within 0.05 s of each other, the repeat ends at their median end plus the quiet span
    ("aligned"). Otherwise the repeat is the live audio heard before the buffer opened, at most
    the buffer's length ("timing"), and so is any buffer whose voiced part is shorter than 0.6 s.
    """
    timing = min(heard_before_open, len(buffer)), "timing"
    voiced_end = _voiced_end(buffer, rate)
    voiced = buffer[:voiced_end]
    window = live[:min(len(live), len(buffer) + round(REPEAT_SEARCH_EXTRA_SECONDS * rate))]
    ends = []
    for seconds in REPEAT_TAIL_SECONDS:
        length = round(seconds * rate)
        if length > len(voiced) or length > len(window):
            return timing
        end, correlation = _best_match(voiced[-length:], window)
        if correlation < REPEAT_MIN_CORRELATION:
            return timing
        ends.append(end)
    if max(ends) - min(ends) > round(REPEAT_AGREE_SECONDS * rate):
        return timing
    return sorted(ends)[1] + len(buffer) - voiced_end, "aligned"


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

    def __init__(self, stream: Any, *, sample_rate: int, track_id: str) -> None:
        self._stream = stream
        self._sample_rate = sample_rate
        self.track_id = track_id
        self._stamped: deque[tuple[float, Any]] = deque()
        self._changed = asyncio.Event()
        self._ended = False
        self._skip = 0
        self._stamping = asyncio.create_task(self._stamp())

    async def _stamp(self) -> None:
        try:
            while True:
                try:
                    event = await self._stream.__anext__()
                except StopAsyncIteration:
                    return
                self._stamped.append((time.monotonic(), event.frame))
                self._changed.set()
        finally:
            self._ended = True
            self._changed.set()

    async def hold(self, frames: list[Any]) -> None:
        """Wait until the live head covers the buffer's window, the stream ends, or the span passes.

        The window is the buffer's length plus 0.2 s. The span is timed on the event loop, so a
        stalled track releases the buffer on time whether or not any frame arrives.
        """
        span = _seconds(frames) + REPEAT_SEARCH_EXTRA_SECONDS
        needed = round(span * self._sample_rate)
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + span
        while not self._ended and self._stamped_samples() < needed:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except TimeoutError:
                break
        logger.info("pre-connect hold: %.2f s", loop.time() - started)

    def process_buffer(self, frames: Iterable[Any], opened: float) -> list[Any]:
        """Cut the buffer's repeat from the live head; return the buffer as it arrived.

        The buffer comes back in order and is consumed before the live audio is read. The
        live frames stamped so far are the head the repeat is matched in; the frames heard
        before the buffer's stream opened are the timing fallback.
        """
        buffer = list(frames)
        if buffer:
            live = self._head()
            heard = 0
            for stamp, frame in self._stamped:
                if stamp >= opened:
                    break
                heard += frame.samples_per_channel
            cut, path = cut_repeat(_mono_at_rate(buffer, self._sample_rate), live, self._sample_rate, heard)
        else:
            cut, path = 0, "none"
        self._skip = cut
        logger.info("pre-connect cut: path=%s dropped=%.2f s", path, cut / self._sample_rate)
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

    def _stamped_samples(self) -> int:
        return sum(frame.samples_per_channel for _, frame in self._stamped)

    def _head(self) -> np.ndarray:
        """The stamped live samples, oldest first, as mono int16 at the live rate."""
        pieces = [np.zeros(0, dtype=np.int16)]
        pieces += [np.frombuffer(bytes(frame.data), dtype=np.int16) for _, frame in self._stamped]
        return np.concatenate(pieces)

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
        return _LiveAudio(stream, sample_rate=self._sample_rate, track_id=track.sid)

    async def _forward_task(self, old_task: Any, stream: Any, *rest: Any) -> None:
        _LIVE.set(stream)
        await super()._forward_task(old_task, stream, *rest)

    def _apply_audio_processor(self, frames: Iterable[rtc.AudioFrame]) -> Iterable[rtc.AudioFrame]:
        live = _LIVE.get()
        return live.process_buffer(frames, self._pre_connect_audio_handler.opened_at(live.track_id))


class _OpenRecordingHandler(PreConnectAudioHandler):
    """LiveKit's pre-connect handler, recording when each track's buffer stream opened.

    An arrived buffer is held until the live head covers it (see _LiveAudio.hold). The wait runs
    in the forward task, where _LIVE is set. A buffer that never arrives, or fails, is cut as none.
    """

    def __init__(self, room: rtc.Room, *, timeout: float) -> None:
        super().__init__(room, timeout=timeout)
        self._opened: dict[str, float] = {}

    def opened_at(self, track_id: str) -> float:
        """When the track's buffer stream opened, on the monotonic clock."""
        return self._opened[track_id]

    async def wait_for_data(self, track_id: str) -> list[rtc.AudioFrame]:
        try:
            frames = await super().wait_for_data(track_id)
        except Exception:
            logger.info("pre-connect cut: path=none dropped=0.00 s")
            raise
        if frames:
            await _LIVE.get().hold(frames)
        return frames

    async def _read_audio_task(self, reader: rtc.ByteStreamReader, participant_id: str) -> None:
        # A buffer is only ever set by a stream that reaches this point, so every buffer has its open time.
        attributes = reader.info.attributes
        if attributes and (track_id := attributes.get("trackId")):
            self._opened[track_id] = time.monotonic()
        await super()._read_audio_task(reader, participant_id)


class PreConnectRoomIO(room_io.RoomIO):
    """RoomIO whose microphone input hears the pre-connect buffer once."""

    async def start(self) -> None:
        # RoomIO.start builds its audio input and pre-connect handler from these module globals
        # and never yields to the loop before it does, so the swaps cannot be seen by another RoomIO.
        stock_input = upstream._ParticipantAudioInputStream
        stock_handler = upstream.PreConnectAudioHandler
        upstream._ParticipantAudioInputStream = _PreConnectAudioInput
        upstream.PreConnectAudioHandler = _OpenRecordingHandler
        try:
            await super().start()
        finally:
            upstream._ParticipantAudioInputStream = stock_input
            upstream.PreConnectAudioHandler = stock_handler
