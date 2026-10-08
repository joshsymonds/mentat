"""The worker's microphone input, with the phone's pre-connect buffer heard once.

The phone records the first words into a buffer while it connects and sends it as one
byte stream; its live track then starts a moment before the end of that buffer, so the
live audio opens by repeating the buffer's last half second or so. LiveKit feeds both to
the speech-to-text and the first words arrive twice. This input drops the repeat, and
runs one noise canceller over the buffer and then the live audio, in order: LiveKit
hands the canceller to its audio stream, which runs it on live frames while the buffer
is still being processed.

The live copy went through Opus, so it is close to the buffer but not identical. The
repeat is found by correlating the raw (uncancelled) buffer against the raw live head
at one low common rate, only within the first seconds of live audio. Without a
confident match every live sample is kept.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from array import array
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from itertools import accumulate
from math import sqrt
from operator import mul
from typing import Any

import livekit.agents.voice.room_io.room_io as upstream
from livekit import rtc
from livekit.agents.voice import room_io
from livekit.agents.voice.room_io._input import _ParticipantAudioInputStream
from livekit.agents.voice.room_io.types import NoiseCancellationParams

logger = logging.getLogger("mentat.voice")

# Matching runs on block averages at this rate; the live and buffer rates differ.
COMMON_RATE = 2000
# Live audio is held for matching no longer than this, and never past the buffer's length.
HEAD_SECONDS = 1.5
FIRST_LOOK_SECONDS = 0.5
LOOK_STEP_SECONDS = 0.25
# A repeat shorter than this is not matched; there is too little audio to trust.
MIN_OVERLAP_SECONDS = 0.25
# Normalised correlation over the repeat. Opus copies score near 1, other speech far lower.
MATCH_SCORE = 0.85
# Windows quieter than this (int16 units, after averaging) carry nothing to match.
MIN_RMS = 30.0

_LOOKING, _SKIPPING, _PASSING = "looking", "skipping", "passing"


def _mono(frames: Iterable[Any]) -> array[int]:
    """Samples of the frames as one mono int16 array, channels averaged."""
    samples: array[int] = array("h")
    for frame in frames:
        raw: array[int] = array("h")
        raw.frombytes(bytes(frame.data))
        channels = frame.num_channels
        if channels == 1:
            samples.extend(raw)
        else:
            lanes = (raw[lane::channels] for lane in range(channels))
            samples.extend(sum(group) // channels for group in zip(*lanes))
    return samples


def _decimate(samples: Sequence[int], rate: int) -> list[float]:
    """Differenced block averages of the samples at COMMON_RATE.

    Each block is taken relative to the one before it. That weights down the low frequencies,
    where the Opus copy drifts in phase, and the first block is taken against silence.
    """
    size = rate / COMMON_RATE
    totals = list(accumulate(samples, initial=0))
    edges = [round(block * size) for block in range(int(len(samples) / size) + 1)]
    blocks = [
        (totals[end] - totals[start]) / (end - start) for start, end in zip(edges, edges[1:])
    ]
    return [value - before for before, value in zip([0.0, *blocks], blocks)]


def _find_repeat(buffer: Sequence[float], live: Sequence[float]) -> int | None:
    """How many blocks at the end of the buffer the start of live repeats, or None.

    The live track and the buffer record the same microphone, so the live head, when it
    repeats anything, repeats the buffer from some point to its very end. That leaves one
    unknown, where in the buffer the live head begins, and the best fit is the start
    whose normalised correlation over the overlap is highest.
    """
    least = int(MIN_OVERLAP_SECONDS * COMMON_RATE)
    if len(buffer) < least or len(live) < least:
        return None
    quiet = MIN_RMS * MIN_RMS
    buffer_energy = list(accumulate((value * value for value in buffer), initial=0.0))
    live_energy = list(accumulate((value * value for value in live), initial=0.0))
    best_score, best_start = 0.0, 0
    for start in range(len(buffer) - least + 1):
        width = min(len(live), len(buffer) - start)
        heard = buffer_energy[start + width] - buffer_energy[start]
        said = live_energy[width]
        if heard < quiet * width or said < quiet * width:
            continue
        score = sum(map(mul, buffer[start:start + width], live)) / sqrt(heard * said)
        if score > best_score:
            best_score, best_start = score, start
    if best_score < MATCH_SCORE:
        return None
    return len(buffer) - best_start


def _repeated_samples(buffer: Sequence[Any], live: Sequence[Any], live_rate: int) -> int:
    """Samples per channel at the start of the live frames that repeat the buffer's end."""
    buffer_rate = buffer[0].sample_rate
    buffer_blocks = _decimate(_mono(buffer), buffer_rate)
    blocks = _find_repeat(buffer_blocks, _decimate(_mono(live), live_rate))
    if blocks is None:
        return 0
    buffer_samples = sum(frame.samples_per_channel for frame in buffer)
    longest = round(buffer_samples * live_rate / buffer_rate)
    return min(round(blocks * live_rate / COMMON_RATE), longest)


def _drop(frame: Any, count: int) -> Any:
    """The frame without its first count samples per channel."""
    data = bytes(frame.data)[count * frame.num_channels * 2:]
    return rtc.AudioFrame(
        data, frame.sample_rate, frame.num_channels, frame.samples_per_channel - count
    )


class _LiveAudio:
    """The live track as the audio input reads it: repeats dropped, noise cancelled in order.

    Stands in for rtc.AudioStream, which the input only iterates and closes.
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
        self._buffer: list[Any] = []
        self._state: str | None = None
        self._held: list[Any] = []
        self._held_samples = 0
        self._next_look = 0
        self._limit = 0
        self._to_skip = 0
        self._ready: deque[Any] = deque()
        self._ended = False

    def process_buffer(self, frames: Iterable[Any]) -> Iterable[Any]:
        """Remember the raw buffer, and cancel noise from its frames as they are consumed."""
        self._buffer = list(frames)
        return map(self._cancel, self._buffer)

    def __aiter__(self) -> _LiveAudio:
        return self

    async def __anext__(self) -> rtc.AudioFrameEvent:
        while not self._ready:
            if self._ended:
                raise StopAsyncIteration
            try:
                event = await self._stream.__anext__()
            except StopAsyncIteration:
                self._ended = True
                await self._settle(final=True)
                continue
            await self._take(event.frame)
        return rtc.AudioFrameEvent(self._ready.popleft())

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def _take(self, frame: Any) -> None:
        if self._state is None:
            self._begin()
        if self._state == _LOOKING:
            self._held.append(frame)
            self._held_samples += frame.samples_per_channel
            if self._held_samples >= self._next_look:
                await self._settle(final=False)
        elif self._state == _SKIPPING:
            dropped = min(self._to_skip, frame.samples_per_channel)
            self._to_skip -= dropped
            if self._to_skip == 0:
                self._state = _PASSING
            if dropped < frame.samples_per_channel:
                self._ready.append(self._cancel(_drop(frame, dropped)))
        else:
            self._ready.append(self._cancel(frame))

    def _begin(self) -> None:
        seconds = 0.0
        if self._buffer:
            samples = sum(frame.samples_per_channel for frame in self._buffer)
            seconds = samples / self._buffer[0].sample_rate
        if seconds < MIN_OVERLAP_SECONDS:
            self._state = _PASSING
            return
        self._state = _LOOKING
        head = min(HEAD_SECONDS, seconds)
        self._limit = round(head * self._sample_rate)
        self._next_look = round(min(FIRST_LOOK_SECONDS, head) * self._sample_rate)

    async def _settle(self, *, final: bool) -> None:
        if self._state != _LOOKING or not self._held:
            return
        repeated = await asyncio.to_thread(
            _repeated_samples, self._buffer, self._held, self._sample_rate
        )
        if not repeated and not final and self._held_samples < self._limit:
            step = round(LOOK_STEP_SECONDS * self._sample_rate)
            self._next_look = min(self._held_samples + step, self._limit)
            return
        held, self._held = self._held, []
        self._to_skip = repeated
        self._state = _SKIPPING if repeated else _PASSING
        if repeated:
            logger.info("dropped %d live samples repeating the pre-connect buffer", repeated)
        for frame in held:
            await self._take(frame)


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
