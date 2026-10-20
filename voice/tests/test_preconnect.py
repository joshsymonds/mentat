"""The worker's microphone input hears the phone's pre-connect buffer once, on synthetic audio.

The system python has no livekit, so the module is imported against small stand-ins for the
few LiveKit pieces it touches; where livekit is installed the real ones are used instead.
"""

import asyncio
import contextlib
import importlib
import importlib.util
import os
import re
import stat
import sys
import tempfile
import types
import unittest
import wave
from array import array
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

VOICE_DIR = Path(__file__).resolve().parent.parent
LIVE_RATE = 24000
BUFFER_RATE = 48000
FRAME_SECONDS = 0.05


def _install_livekit_stand_ins():
    """Just enough of livekit for voice/preconnect.py to import and run."""

    class AudioFrame:
        def __init__(self, data, sample_rate, num_channels, samples_per_channel):
            self._data = bytes(data)
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.samples_per_channel = samples_per_channel

        @property
        def data(self):
            return memoryview(self._data).cast("h")

    @dataclass
    class AudioFrameEvent:
        frame: AudioFrame

    class FrameProcessor:
        pass

    class AudioStream:
        @classmethod
        def from_track(cls, **kwargs):
            raise AssertionError("tests patch from_track")

    class Chan:
        def __init__(self):
            self._items = asyncio.Queue()

        async def send(self, item):
            self._items.put_nowait(item)

        def close(self):
            self._items.put_nowait(None)

        def __aiter__(self):
            return self

        async def __anext__(self):
            item = await self._items.get()
            if item is None:
                raise StopAsyncIteration
            return item

    class ParticipantAudioInputStream:
        """The part of LiveKit's input stream preconnect.py builds on, as it behaves in 1.8.1."""

        def __init__(self, room, *, sample_rate, num_channels, noise_cancellation,
                     auto_gain_control=True, pre_connect_audio_handler, frame_size_ms=50):
            self._sample_rate = sample_rate
            self._num_channels = num_channels
            self._frame_size_ms = frame_size_ms
            self._pre_connect_audio_handler = pre_connect_audio_handler
            self._pre_connect_audio_publications = set()
            self._processor = noise_cancellation if isinstance(noise_cancellation, FrameProcessor) else None
            self._data_ch = Chan()
            self._attached = True

        def _process_frame(self, frame):
            pass

        def _apply_audio_processor(self, frames):
            for frame in frames:
                yield self._processor._process(frame) if self._processor else frame

        def _resample_frames(self, frames):
            yield from frames

        async def _forward_task(self, old_task, stream, track, publication, participant):
            key = (participant.identity, publication.sid)
            if (
                self._pre_connect_audio_handler
                and track_pb2.AudioTrackFeature.TF_PRECONNECT_BUFFER in publication.audio_features
                and key not in self._pre_connect_audio_publications
            ):
                try:
                    frames = await self._pre_connect_audio_handler.wait_for_data(track.sid)
                    self._pre_connect_audio_publications.add(key)
                    for frame in self._resample_frames(self._apply_audio_processor(frames)):
                        await self._data_ch.send(frame)
                except Exception:
                    self._pre_connect_audio_publications.add(key)
            async for event in stream:
                self._process_frame(event.frame)
                await self._data_ch.send(event.frame)
            await self._data_ch.send(AudioFrame(b"\0\0", self._sample_rate, 1, 1))

    PRE_CONNECT_AUDIO_BUFFER_STREAM = "lk.agent.pre-connect-audio-buffer"

    class PreConnectAudioHandler:
        """LiveKit 1.8.1's pre-connect handler, for the raw PCM the tests send."""

        def __init__(self, room, *, timeout, max_delta_s=1.0):
            self._room = room
            self._timeout = timeout
            self._max_delta_s = max_delta_s
            self._buffers = {}

        def register(self):
            def _handler(reader, participant_id):
                asyncio.create_task(self._read_audio_task(reader, participant_id))

            self._room.register_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM, _handler)

        async def wait_for_data(self, track_id):
            fut = self._buffers.setdefault(track_id, asyncio.get_running_loop().create_future())
            try:
                return await asyncio.wait_for(fut, self._timeout)
            finally:
                self._buffers.pop(track_id)

        async def _read_audio_task(self, reader, participant_id):
            attributes = reader.info.attributes
            if not attributes or not (track_id := attributes.get("trackId")):
                return
            fut = self._buffers.setdefault(track_id, asyncio.get_running_loop().create_future())
            rate, channels = int(attributes["sampleRate"]), int(attributes["channels"])
            frames = []
            async for chunk in reader:
                frames.append(rtc.AudioFrame(chunk, rate, channels, len(chunk) // (2 * channels)))
            if not fut.done():
                fut.set_result(frames)

    class RoomIO:
        def __init__(self, agent_session, room, *, participant=None, options=None):
            self._room = room
            self._options = options
            self._audio_input = None
            self._pre_connect_audio_handler = None

        async def start(self):
            audio = self._options.get_audio_input_options()
            if audio.pre_connect_audio:
                self._pre_connect_audio_handler = upstream.PreConnectAudioHandler(
                    room=self._room, timeout=audio.pre_connect_audio_timeout)
                self._pre_connect_audio_handler.register()
            self._audio_input = upstream._ParticipantAudioInputStream(
                self._room,
                sample_rate=audio.sample_rate,
                num_channels=1,
                noise_cancellation=audio.noise_cancellation,
                auto_gain_control=False,
                pre_connect_audio_handler=self._pre_connect_audio_handler,
            )

        @property
        def audio_input(self):
            return self._audio_input

    def module(name, **attributes):
        stand_in = types.ModuleType(name)
        vars(stand_in).update(attributes)
        return stand_in

    track_pb2 = module("livekit.rtc._proto.track_pb2",
                       AudioTrackFeature=SimpleNamespace(TF_PRECONNECT_BUFFER=1))
    upstream = module("livekit.agents.voice.room_io.room_io",
                      RoomIO=RoomIO, _ParticipantAudioInputStream=ParticipantAudioInputStream,
                      PreConnectAudioHandler=PreConnectAudioHandler)
    pre_connect_audio = module("livekit.agents.voice.room_io._pre_connect_audio",
                               PreConnectAudioHandler=PreConnectAudioHandler,
                               PRE_CONNECT_AUDIO_BUFFER_STREAM=PRE_CONNECT_AUDIO_BUFFER_STREAM)
    room_io = module("livekit.agents.voice.room_io", RoomIO=RoomIO, room_io=upstream,
                     _input=module("livekit.agents.voice.room_io._input",
                                   _ParticipantAudioInputStream=ParticipantAudioInputStream),
                     _pre_connect_audio=pre_connect_audio)
    rtc = module("livekit.rtc", AudioFrame=AudioFrame, AudioFrameEvent=AudioFrameEvent,
                 FrameProcessor=FrameProcessor, AudioStream=AudioStream, Track=object, Participant=object,
                 _proto=module("livekit.rtc._proto", track_pb2=track_pb2))
    voice = module("livekit.agents.voice", room_io=room_io)
    agents = module("livekit.agents", voice=voice)
    livekit = module("livekit", rtc=rtc, agents=agents)
    return {
        "livekit": livekit, "livekit.rtc": rtc, "livekit.rtc._proto": rtc._proto,
        "livekit.rtc._proto.track_pb2": track_pb2, "livekit.agents": agents,
        "livekit.agents.voice": voice, "livekit.agents.voice.room_io": room_io,
        "livekit.agents.voice.room_io.room_io": upstream,
        "livekit.agents.voice.room_io._input": room_io._input,
        "livekit.agents.voice.room_io._pre_connect_audio": pre_connect_audio,
    }


def _load_preconnect():
    sys.path.insert(0, str(VOICE_DIR))
    if importlib.util.find_spec("livekit") is not None:
        return importlib.import_module("preconnect"), False
    stand_ins = _install_livekit_stand_ins()
    sys.modules.update(stand_ins)
    try:
        return importlib.import_module("preconnect"), True
    finally:
        for name in stand_ins:
            del sys.modules[name]
        sys.modules.pop("preconnect", None)
        sys.path.remove(str(VOICE_DIR))


preconnect, STAND_INS = _load_preconnect()
rtc = preconnect.rtc
if STAND_INS:
    PRECONNECT_FEATURE = 1
else:
    from livekit.rtc._proto.track_pb2 import AudioTrackFeature

    PRECONNECT_FEATURE = AudioTrackFeature.TF_PRECONNECT_BUFFER


# -- synthetic audio -------------------------------------------------------------------------

def ramp(seconds, rate=LIVE_RATE):
    """Int16 samples that are each their own index, so any kept or dropped span is visible."""
    return array("h", ((index % 65536) - 32768 for index in range(round(seconds * rate))))


def silence(seconds, rate=BUFFER_RATE):
    """Frames of silence, as the phone's pre-connect buffer is sent."""
    return frames_of(array("h", bytes(2 * round(seconds * rate))), rate)


def frames_of(samples, rate, seconds=FRAME_SECONDS):
    step = round(rate * seconds)
    return [
        rtc.AudioFrame(samples[at:at + step].tobytes(), rate, 1, len(samples[at:at + step]))
        for at in range(0, len(samples), step)
    ]


def samples_of(frames):
    out = array("h")
    for frame in frames:
        out.extend(array("h", bytes(frame.data)))
    return out


def speechlike(seconds, seed, rate=LIVE_RATE):
    """Low-passed noise at speech-like levels as int16 samples, so a tail matches only its own repeat."""
    count = round(seconds * rate)
    smooth = np.convolve(np.random.default_rng(seed).standard_normal(count), np.ones(4) / 4, mode="same")
    return np.round(smooth / smooth.std() * 4000).astype(np.int16)


def quiet_noise(seconds, seed):
    """Near-silent noise at 1% of speechlike's level as int16 samples, as a pause or a quiet tail has it."""
    count = round(seconds * LIVE_RATE)
    return np.round(np.random.default_rng(seed).standard_normal(count) * 40).astype(np.int16)


def jittered_repeat(tail, *, noise_scale, seed, rate=LIVE_RATE):
    """The tail as a jittered live track replays it: 20 ms chunks, one dropped and two duplicated
    (about 2% longer), with noise at noise_scale times the tail's own level."""
    chunk = round(0.02 * rate)
    chunks = [tail[at:at + chunk] for at in range(0, len(tail), chunk)]
    played = []
    for index, piece in enumerate(chunks):
        if index == 20:
            continue
        played.append(piece)
        if index in (9, 31):
            played.append(piece)
    stretched = np.concatenate(played).astype(np.float64)
    noise = np.random.default_rng(seed).standard_normal(len(stretched)) * noise_scale * tail.std()
    return np.round(stretched + noise).astype(np.int16)


# -- harness ---------------------------------------------------------------------------------

class HalvingCanceller(rtc.FrameProcessor):
    """A noise canceller that halves every sample, so any frame it touches comes out different."""

    def __init__(self):
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    def _process(self, frame):
        halved = array("h", (sample // 2 for sample in array("h", bytes(frame.data))))
        return rtc.AudioFrame(halved.tobytes(), frame.sample_rate, frame.num_channels, frame.samples_per_channel)

    def _close(self):
        pass


class ScriptedStream:
    """Stands in for rtc.AudioStream: yields the frames, then ends."""

    def __init__(self, frames):
        self._frames = deque(frames)
        self.closed = False

    async def __anext__(self):
        if self._frames:
            return rtc.AudioFrameEvent(self._frames.popleft())
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class GatedStream:
    """Stands in for rtc.AudioStream: frames reach it when the test pushes them, and it ends when told."""

    def __init__(self):
        self._frames = deque()
        self._changed = asyncio.Event()
        self._ended = False
        self.closed = False

    def push(self, frames):
        self._frames.extend(frames)
        self._changed.set()

    def end(self):
        self._ended = True
        self._changed.set()

    async def __anext__(self):
        while not self._frames:
            if self._ended:
                raise StopAsyncIteration
            self._changed.clear()
            await self._changed.wait()
        return rtc.AudioFrameEvent(self._frames.popleft())

    async def aclose(self):
        self.closed = True
        self.end()


class FakeTime:
    """Stands in for the time module in preconnect, so tests choose when each frame is heard."""

    def __init__(self):
        self.now = 0.0
        self.reads = 0

    def monotonic(self):
        self.reads += 1
        return self.now


async def settle():
    """Lets the input's stamping task take whatever the stream has been given."""
    for _ in range(3):
        await asyncio.sleep(0)


async def drain(live):
    out = []
    async for event in live:
        out.append(event.frame)
    return out


class FakeHandler:
    def __init__(self, frames):
        self._frames = frames

    async def wait_for_data(self, track_id):
        return self._frames

    def opened_at(self, track_id):
        return 0.0


def build_input(handler):
    return preconnect._PreConnectAudioInput(
        mock.MagicMock(),
        sample_rate=LIVE_RATE,
        num_channels=1,
        noise_cancellation=None,
        auto_gain_control=False,
        pre_connect_audio_handler=handler,
        frame_size_ms=50,
    )


class LateBufferHandler:
    """A handler whose wait gives up, and whose buffer completes only after that.

    As in LiveKit's handler, a buffer that completes after a wait timed out is kept, and a
    later wait for the same track returns it at once.
    """

    def __init__(self):
        self.give_up = asyncio.Event()
        self.late = asyncio.get_running_loop().create_future()

    async def wait_for_data(self, track_id):
        if self.late.done():
            return self.late.result()
        await self.give_up.wait()
        raise TimeoutError


def track_streams(*streams):
    """Patches rtc.AudioStream.from_track to hand out the given streams, one per call."""
    pending = list(streams)
    return mock.patch.object(rtc.AudioStream, "from_track", staticmethod(lambda **kwargs: pending.pop(0)))


async def run_input(case_live, handler):
    """Drive the input's forward task over a track the way LiveKit does; returns what it forwarded."""
    track = SimpleNamespace(sid="TR_audio")
    publication = SimpleNamespace(sid="TR_audio", source=0, audio_features=[PRECONNECT_FEATURE])
    participant = SimpleNamespace(identity="phone")
    with mock.patch.object(rtc.AudioStream, "from_track", staticmethod(lambda **kwargs: ScriptedStream(case_live))):
        audio_input = build_input(handler)
        stream = audio_input._create_stream(track, participant)
        await audio_input._forward_task(None, stream, track, publication, participant)
    audio_input._data_ch.close()
    return [frame async for frame in audio_input._data_ch]


class AsyncTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Building seconds of synthetic audio inside a test would trip the loop's slow-callback log.
        asyncio.get_running_loop().set_debug(False)


# -- tests -----------------------------------------------------------------------------------

class LiveAudioTest(AsyncTest):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.clock = FakeTime()
        clock = mock.patch.object(preconnect, "time", self.clock)
        clock.start()
        self.addCleanup(clock.stop)
        self.stream = GatedStream()
        self.live = preconnect._LiveAudio(self.stream, sample_rate=LIVE_RATE, track_id="TR_audio")

    async def test_live_frames_heard_before_the_buffer_completed_are_dropped_and_the_rest_pass(self):
        live = ramp(2.0)
        frames = frames_of(live, LIVE_RATE)
        self.stream.push(frames[:10])  # 0.5 s heard from t=0
        await settle()
        self.clock.now = 0.7
        list(self.live.process_buffer(silence(1.0), opened=0.7))
        self.stream.push(frames[10:])
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out), live[round(0.5 * LIVE_RATE):])

    async def test_a_frame_straddling_the_drop_bound_is_trimmed_to_the_samples_past_it(self):
        live = ramp(2.0)
        frames = frames_of(live, LIVE_RATE)
        self.stream.push(frames[:15])  # 0.75 s heard from t=0; the buffer is 0.73 s, so frame 14 straddles
        await settle()
        self.clock.now = 0.8
        list(self.live.process_buffer(silence(0.73), opened=0.8))
        self.stream.push(frames[15:])
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out), live[round(0.73 * LIVE_RATE):])

    async def test_no_more_than_the_buffer_length_is_dropped(self):
        live = ramp(2.0)
        self.stream.push(frames_of(live, LIVE_RATE))  # all 2 s heard from t=0
        await settle()
        self.clock.now = 1.0
        list(self.live.process_buffer(silence(0.5), opened=1.0))
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out), live[round(0.5 * LIVE_RATE):])

    async def test_a_slow_send_drops_one_second_of_the_live_audio_heard_before_the_stream_opened(self):
        buffer = speechlike(2.3, seed=1, rate=BUFFER_RATE)
        live = speechlike(3.0, seed=2)
        frames = frames_of(live, LIVE_RATE)
        self.stream.push(frames[:46])  # 2.3 s heard from t=0, before the stream opened at 2.3
        await settle()
        self.clock.now = 2.3
        list(self.live.process_buffer(frames_of(buffer, BUFFER_RATE), opened=2.3))
        self.stream.push(frames[46:])
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out).tolist(), live[round(1.0 * LIVE_RATE):].tolist())

    async def test_live_frames_heard_after_the_buffer_completed_all_pass(self):
        live = ramp(1.0)
        self.clock.now = 0.0
        list(self.live.process_buffer(silence(1.0), opened=0.0))
        self.clock.now = 0.5
        self.stream.push(frames_of(live, LIVE_RATE))
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out), live)

    async def test_an_empty_buffer_drops_nothing(self):
        live = ramp(1.0)
        self.stream.push(frames_of(live, LIVE_RATE))
        await settle()
        self.clock.now = 0.5
        list(self.live.process_buffer([], opened=0.5))
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out), live)

    async def test_without_a_buffer_live_audio_flows_unchanged_and_unheld(self):
        frames = frames_of(ramp(1.0), LIVE_RATE)
        self.stream.push(frames)  # not ended: the track stays open
        await settle()

        first = await asyncio.wait_for(anext(self.live), timeout=1)

        self.assertIs(first.frame, frames[0])

    async def test_the_cut_logs_once_with_its_path_and_the_seconds_dropped(self):
        self.stream.push(frames_of(ramp(0.5), LIVE_RATE))  # first live frame heard at t=0
        await settle()
        self.clock.now = 0.7

        with self.assertLogs("mentat.voice", level="INFO") as logs:
            list(self.live.process_buffer(silence(1.0), opened=0.7))

        self.assertEqual({record.levelname for record in logs.records}, {"INFO"})
        self.assertEqual(pre_connect_messages(logs, "pre-connect cut:"), ["pre-connect cut: path=timing dropped=0.50 s"])
        self.assertEqual(pre_connect_messages(logs, "pre-connect tails:"), ["pre-connect tails: quiet=0.00 s none"])

    async def test_aclose_stops_the_stamping_and_closes_the_stream(self):
        self.stream.push(frames_of(ramp(0.5), LIVE_RATE))
        await settle()
        stamped = self.clock.reads
        self.assertGreater(stamped, 0)

        await self.live.aclose()
        self.stream.push(frames_of(ramp(0.5), LIVE_RATE))
        await settle()

        self.assertTrue(self.stream.closed)
        self.assertEqual(self.clock.reads, stamped)


class PreConnectDedupeTest(AsyncTest):
    async def test_a_configured_noise_canceller_never_touches_the_buffer_or_the_live_audio(self):
        buffer = frames_of(ramp(1.0), LIVE_RATE)
        live = frames_of(ramp(2.0), LIVE_RATE)
        track = SimpleNamespace(sid="TR_audio")
        publication = SimpleNamespace(sid="TR_audio", source=0, audio_features=[PRECONNECT_FEATURE])
        participant = SimpleNamespace(identity="phone")
        built = {}

        def from_track(**kwargs):
            built.update(kwargs)
            return ScriptedStream(live)

        with mock.patch.object(rtc.AudioStream, "from_track", staticmethod(from_track)):
            audio_input = preconnect._PreConnectAudioInput(
                mock.MagicMock(),
                sample_rate=LIVE_RATE,
                num_channels=1,
                noise_cancellation=HalvingCanceller(),
                auto_gain_control=False,
                pre_connect_audio_handler=FakeHandler(buffer),
                frame_size_ms=50,
            )
            stream = audio_input._create_stream(track, participant)
            await audio_input._forward_task(None, stream, track, publication, participant)
        audio_input._data_ch.close()
        forwarded = [frame async for frame in audio_input._data_ch]

        self.assertIsNone(built.get("noise_cancellation"))
        self.assertEqual([id(frame) for frame in forwarded[:len(buffer)]], [id(frame) for frame in buffer])
        self.assertEqual(samples_of(forwarded[len(buffer):-1]), samples_of(live))

    async def test_the_buffer_and_the_live_audio_reach_rtc_exactly_as_received(self):
        buffer = silence(1.0, LIVE_RATE)
        live = frames_of(ramp(2.0), LIVE_RATE)

        forwarded = await run_input(live, FakeHandler(buffer))

        self.assertEqual([id(frame) for frame in forwarded[:len(buffer)]], [id(frame) for frame in buffer])
        self.assertEqual(samples_of(forwarded[len(buffer):-1]), samples_of(live))

    async def test_a_timed_out_buffer_leaves_live_audio_untouched(self):
        live = frames_of(ramp(2.0), LIVE_RATE)
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=0.01)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            forwarded = await run_input(live, handler)

        self.assertEqual(samples_of(forwarded[:-1]), samples_of(live))
        self.assertEqual(pre_connect_messages(logs, "pre-connect cut:"), [NO_CUT])

    async def test_a_buffer_error_other_than_a_timeout_leaves_live_audio_untouched(self):
        live = frames_of(ramp(2.0), LIVE_RATE)
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)
        failing = mock.AsyncMock(side_effect=RuntimeError("byte stream failed"))

        with mock.patch.object(preconnect.PreConnectAudioHandler, "wait_for_data", failing):
            with self.assertLogs("mentat.voice", "INFO") as logs:
                forwarded = await run_input(live, handler)

        self.assertEqual(samples_of(forwarded[:-1]), samples_of(live))
        self.assertEqual(pre_connect_messages(logs, "pre-connect cut:"), [NO_CUT])

    async def test_a_buffer_completing_after_the_timeout_drops_no_live_audio_before_or_after(self):
        live = frames_of(ramp(3.0), LIVE_RATE)
        handler = LateBufferHandler()
        first, second = GatedStream(), GatedStream()
        track = SimpleNamespace(sid="TR_audio")
        publication = SimpleNamespace(sid="TR_audio", source=0, audio_features=[PRECONNECT_FEATURE])
        participant = SimpleNamespace(identity="phone")

        with track_streams(first, second):
            audio_input = build_input(handler)
            first_live = audio_input._create_stream(track, participant)
            first_task = asyncio.create_task(
                audio_input._forward_task(None, first_live, track, publication, participant)
            )
            first.push(live[:10])  # heard while the wait for the buffer is still open
            await settle()
            handler.give_up.set()
            await settle()
            first.push(live[10:20])  # heard after the timeout, before the buffer completes
            await settle()
            handler.late.set_result(silence(1.0))  # the buffer completes late
            await settle()
            first.push(live[20:30])  # heard after the late completion
            await settle()

            first_task.cancel()
            second_live = audio_input._create_stream(track, participant)
            second_task = asyncio.create_task(
                audio_input._forward_task(None, second_live, track, publication, participant)
            )
            second.push(live[30:])
            second.end()
            await second_task

        audio_input._data_ch.close()
        forwarded = [frame async for frame in audio_input._data_ch]

        self.assertEqual(samples_of(forwarded[:-1]), samples_of(live))


class OpenTimedHandler:
    """A handler whose buffer completes when the test releases it, and which reports when its stream opened."""

    def __init__(self, frames, opened):
        self.done = asyncio.get_running_loop().create_future()
        self._frames, self._opened = frames, opened

    async def wait_for_data(self, track_id):
        return await self.done

    def opened_at(self, track_id):
        return self._opened


class OpenTimeInputTest(AsyncTest):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.clock = FakeTime()
        clock = mock.patch.object(preconnect, "time", self.clock)
        clock.start()
        self.addCleanup(clock.stop)

    async def test_live_audio_heard_before_the_buffer_stream_opened_is_dropped_and_the_rest_pass(self):
        live = ramp(2.0)
        frames = frames_of(live, LIVE_RATE)
        stream = GatedStream()
        buffer = silence(1.0, LIVE_RATE)
        handler = OpenTimedHandler(buffer, opened=0.3)
        track = SimpleNamespace(sid="TR_audio")
        publication = SimpleNamespace(sid="TR_audio", source=0, audio_features=[PRECONNECT_FEATURE])
        participant = SimpleNamespace(identity="phone")

        with track_streams(stream):
            audio_input = build_input(handler)
            live_stream = audio_input._create_stream(track, participant)
            forward = asyncio.create_task(
                audio_input._forward_task(None, live_stream, track, publication, participant)
            )
            self.clock.now = 0.0
            stream.push(frames[:6])  # 0.3 s heard before the stream opened at 0.3
            await settle()
            self.clock.now = 0.3
            stream.push(frames[6:12])  # 0.3 s heard once the stream had opened
            await settle()
            self.clock.now = 0.6
            handler.done.set_result(buffer)  # the buffer completes at 0.6
            await settle()
            stream.push(frames[12:])
            stream.end()
            await forward

        audio_input._data_ch.close()
        forwarded = [frame async for frame in audio_input._data_ch]

        self.assertEqual(samples_of(forwarded[len(buffer):-1]), live[round(0.3 * LIVE_RATE):])


NO_CUT = "pre-connect cut: path=none dropped=0.00 s"


TAIL_ENTRY = r"(\d+\.\d{2}) s end=(\d+\.\d{2}) s r=(\d+\.\d{2})"


def pre_connect_messages(logs, prefix):
    return [record.getMessage() for record in logs.records if record.getMessage().startswith(prefix)]


def start_input(stream, handler):
    """Starts the input's forward task over one track's stream, as LiveKit does on subscription."""
    track = SimpleNamespace(sid="TR_audio")
    publication = SimpleNamespace(sid="TR_audio", source=0, audio_features=[PRECONNECT_FEATURE])
    participant = SimpleNamespace(identity="phone")
    with track_streams(stream):
        audio_input = build_input(handler)
        live_stream = audio_input._create_stream(track, participant)
    forward = asyncio.create_task(audio_input._forward_task(None, live_stream, track, publication, participant))
    return audio_input, forward


class CutTest(AsyncTest):
    async def test_a_jittered_repeat_of_the_buffer_tail_is_cut_by_alignment(self):
        buffer = speechlike(1.04, seed=1, rate=BUFFER_RATE)
        at_live_rate = np.interp(
            np.arange(round(len(buffer) * LIVE_RATE / BUFFER_RATE)) * BUFFER_RATE / LIVE_RATE,
            np.arange(len(buffer)), buffer,
        )
        tail = np.round(at_live_rate).astype(np.int16)[-round(0.9 * LIVE_RATE):]
        repeat = jittered_repeat(tail, noise_scale=1.2, seed=3)
        live = np.concatenate([repeat, speechlike(0.6, seed=4)])
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)
        await arrive(handler, pcm_chunks(buffer, BUFFER_RATE))

        with self.assertLogs("mentat.voice", "INFO") as logs:
            await run_input(frames_of(live, LIVE_RATE), handler)

        [cut] = pre_connect_messages(logs, "pre-connect cut:")
        match = re.fullmatch(r"pre-connect cut: path=aligned dropped=(\d+\.\d{2}) s", cut)
        self.assertIsNotNone(match, cut)
        self.assertLessEqual(abs(float(match[1]) - len(repeat) / LIVE_RATE), 0.05)

    async def test_no_repeat_is_cut_by_timing_at_the_live_heard_before_the_stream_opened(self):
        clock = FakeTime()
        patch = mock.patch.object(preconnect, "time", clock)
        patch.start()
        self.addCleanup(patch.stop)
        buffer = speechlike(1.0, seed=1, rate=BUFFER_RATE)
        live = speechlike(2.0, seed=2)
        frames = frames_of(live, LIVE_RATE)
        stream = GatedStream()
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            audio_input, forward = start_input(stream, handler)
            stream.push(frames[:6])  # 0.3 s heard before the stream opens
            await settle()
            clock.now = 0.3
            await arrive(handler, pcm_chunks(buffer, BUFFER_RATE))
            stream.push(frames[6:])
            stream.end()
            await forward
        audio_input._data_ch.close()
        forwarded = [frame async for frame in audio_input._data_ch]

        self.assertEqual(pre_connect_messages(logs, "pre-connect cut:"), ["pre-connect cut: path=timing dropped=0.30 s"])
        live_forwarded = [frame for frame in forwarded if any(frame is heard for heard in frames)]
        self.assertEqual([id(frame) for frame in live_forwarded], [id(frame) for frame in frames[6:]])

    async def test_an_empty_buffer_is_cut_as_none(self):
        live = frames_of(ramp(1.0), LIVE_RATE)
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)
        await arrive(handler, [])

        with self.assertLogs("mentat.voice", "INFO") as logs:
            forwarded = await run_input(live, handler)

        self.assertEqual(samples_of(forwarded[:-1]), samples_of(live))
        self.assertEqual(pre_connect_messages(logs, "pre-connect cut:"), [NO_CUT])
        self.assertEqual(pre_connect_messages(logs, "pre-connect hold:"), [])


class HoldTest(AsyncTest):
    async def test_the_buffer_is_held_until_the_live_head_covers_its_window(self):
        buffer = speechlike(1.0, seed=1, rate=BUFFER_RATE)
        live = frames_of(speechlike(2.0, seed=2), LIVE_RATE)
        stream = GatedStream()
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            audio_input, forward = start_input(stream, handler)
            await arrive(handler, pcm_chunks(buffer, BUFFER_RATE))
            stream.push(live[:20])  # 1.0 s of live, short of the 1.2 s window
            await settle()
            self.assertEqual(pre_connect_messages(logs, "pre-connect cut:"), [])

            stream.push(live[20:24])  # 1.2 s of live covers the window
            await settle()
            self.assertEqual(len(pre_connect_messages(logs, "pre-connect cut:")), 1)

            stream.end()
            await forward

        self.assertEqual(len(pre_connect_messages(logs, "pre-connect hold:")), 1)

    async def test_a_stream_that_ends_before_the_window_releases_the_held_buffer(self):
        buffer = speechlike(1.0, seed=1, rate=BUFFER_RATE)
        live = frames_of(speechlike(2.0, seed=2), LIVE_RATE)
        stream = GatedStream()
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            audio_input, forward = start_input(stream, handler)
            await arrive(handler, pcm_chunks(buffer, BUFFER_RATE))
            stream.push(live[:10])
            stream.end()
            await asyncio.wait_for(forward, 0.5)  # well inside the 1.2 s span

        self.assertEqual(
            pre_connect_messages(logs, "pre-connect cut:"), ["pre-connect cut: path=timing dropped=0.00 s"]
        )

    def test_a_stalled_live_track_releases_the_held_buffer_after_the_window_span(self):
        loop = ClockLoop()
        clock = SimpleNamespace(monotonic=loop.time)
        with mock.patch.object(preconnect, "time", clock), asyncio.Runner(loop_factory=lambda: loop) as runner:
            runner.run(self._stalls())

    async def _stalls(self):
        buffer = speechlike(1.0, seed=1, rate=BUFFER_RATE)
        live = frames_of(speechlike(2.0, seed=2), LIVE_RATE)
        stream = GatedStream()
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            audio_input, forward = start_input(stream, handler)
            await arrive(handler, pcm_chunks(buffer, BUFFER_RATE))
            stream.push(live[:10])  # 0.5 s of live, then the track stalls
            await asyncio.sleep(1.5)
            self.assertEqual(
                pre_connect_messages(logs, "pre-connect cut:"), ["pre-connect cut: path=timing dropped=0.00 s"]
            )
            stream.end()
            await forward

        [hold] = pre_connect_messages(logs, "pre-connect hold:")
        match = re.fullmatch(r"pre-connect hold: (\d+\.\d{2}) s", hold)
        self.assertIsNotNone(match, hold)
        self.assertAlmostEqual(float(match[1]), 1.2, delta=0.1)


def room_options(timeout=3.0):
    if STAND_INS:
        audio = SimpleNamespace(sample_rate=LIVE_RATE, noise_cancellation=None,
                                pre_connect_audio=True, pre_connect_audio_timeout=timeout)
        return SimpleNamespace(get_audio_input_options=lambda: audio)
    return preconnect.room_io.RoomOptions(
        audio_input=preconnect.room_io.AudioInputOptions(pre_connect_audio_timeout=timeout),
        audio_output=False,
        text_output=False,
    )


class GatedReader:
    """Stands in for rtc.ByteStreamReader: chunks reach it when the test pushes them, and it ends when told."""

    def __init__(self, track_id, rate=BUFFER_RATE):
        self.info = SimpleNamespace(
            attributes={"trackId": track_id, "sampleRate": str(rate), "channels": "1"},
            mime_type="",
        )
        self._chunks = deque()
        self._changed = asyncio.Event()
        self._ended = False

    def push(self, chunk):
        self._chunks.append(chunk)
        self._changed.set()

    def end(self):
        self._ended = True
        self._changed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        while not self._chunks:
            if self._ended:
                raise StopAsyncIteration
            self._changed.clear()
            await self._changed.wait()
        return self._chunks.popleft()


def pcm_chunks(samples, rate, seconds=FRAME_SECONDS):
    """The samples as the phone's buffer stream sends them, one chunk per frame's worth of audio."""
    step = round(rate * seconds)
    return [samples[at:at + step].tobytes() for at in range(0, len(samples), step)]


async def arrive(handler, chunks, track_id="TR_audio", rate=BUFFER_RATE):
    """Hands the handler a pre-connect buffer stream of the given chunks, which then ends."""
    reader = GatedReader(track_id, rate)
    reading = asyncio.create_task(handler._read_audio_task(reader, "phone"))
    await settle()
    for chunk in chunks:
        reader.push(chunk)
    reader.end()
    await reading


class ClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock moves only when it would otherwise sleep, so a 9 s wait takes no time."""

    def __init__(self):
        super().__init__()
        self._now = 0.0
        selector = self._selector
        select = selector.select

        def skip_to_next_timer(timeout=None):
            if timeout:
                self._now += timeout
            return select(0)

        selector.select = skip_to_next_timer

    def time(self):
        return self._now


class OpenRecordingHandlerTest(AsyncTest):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.clock = FakeTime()
        clock = mock.patch.object(preconnect, "time", self.clock)
        clock.start()
        self.addCleanup(clock.stop)

    async def test_the_open_time_of_each_buffer_stream_is_recorded_per_track(self):
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)
        reader = GatedReader("TR_audio")
        self.clock.now = 0.7

        reading = asyncio.create_task(handler._read_audio_task(reader, "phone"))
        await settle()
        reader.push(bytes(2 * 1000))
        reader.end()
        await reading

        self.assertEqual(handler.opened_at("TR_audio"), 0.7)

    async def test_an_arrived_buffer_is_held_by_the_handler_until_the_live_head_covers_its_window(self):
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)
        stream = GatedStream()
        live = frames_of(speechlike(2.0, seed=2), LIVE_RATE)
        preconnect._LIVE.set(preconnect._LiveAudio(stream, sample_rate=LIVE_RATE, track_id="TR_audio"))

        with self.assertLogs("mentat.voice", "INFO") as logs:
            wait = asyncio.create_task(handler.wait_for_data("TR_audio"))
            await arrive(handler, pcm_chunks(speechlike(1.0, seed=1, rate=BUFFER_RATE), BUFFER_RATE))
            stream.push(live[:20])  # 1.0 s of live, short of the 1.2 s window
            await settle()
            self.assertFalse(wait.done())

            stream.push(live[20:24])  # 1.2 s of live covers the window
            frames = await asyncio.wait_for(wait, timeout=1)
            stream.end()

        self.assertEqual(sum(frame.samples_per_channel for frame in frames), BUFFER_RATE)
        self.assertEqual(len(pre_connect_messages(logs, "pre-connect hold:")), 1)

    async def test_a_timeout_is_logged_as_none_and_raised_by_the_handler(self):
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=0.01)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            with self.assertRaises(TimeoutError):
                await handler.wait_for_data("TR_audio")

        self.assertEqual(pre_connect_messages(logs, "pre-connect cut:"), [NO_CUT])

    async def test_a_wait_error_is_logged_as_none_and_raised_by_the_handler(self):
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)
        failing = mock.AsyncMock(side_effect=RuntimeError("byte stream failed"))

        with mock.patch.object(preconnect.PreConnectAudioHandler, "wait_for_data", failing):
            with self.assertLogs("mentat.voice", "INFO") as logs:
                with self.assertRaises(RuntimeError):
                    await handler.wait_for_data("TR_audio")

        self.assertEqual(pre_connect_messages(logs, "pre-connect cut:"), [NO_CUT])

    def test_a_buffer_completing_nine_seconds_after_the_wait_begins_is_used(self):
        loop = ClockLoop()
        clock = SimpleNamespace(monotonic=loop.time)
        with mock.patch.object(preconnect, "time", clock), asyncio.Runner(loop_factory=lambda: loop) as runner:
            runner.run(self._nine_seconds())

    async def _nine_seconds(self):
        handler = preconnect._OpenRecordingHandler(mock.MagicMock(), timeout=10.0)
        reader = GatedReader("TR_audio")
        buffer = bytes(2 * BUFFER_RATE)
        preconnect._LIVE.set(preconnect._LiveAudio(GatedStream(), sample_rate=LIVE_RATE, track_id="TR_audio"))
        wait = asyncio.create_task(handler.wait_for_data("TR_audio"))  # the wait begins at subscription
        await asyncio.sleep(0.7)
        reading = asyncio.create_task(handler._read_audio_task(reader, "phone"))  # the stream opens
        await asyncio.sleep(8.3)
        reader.push(buffer)  # the buffer completes at 9 s
        reader.end()
        await reading

        frames = await wait

        self.assertEqual(sum(frame.samples_per_channel for frame in frames), BUFFER_RATE)


class PreConnectRoomIOTest(AsyncTest):
    async def test_start_installs_the_input_and_restores_livekits(self):
        upstream = preconnect.upstream
        stock = upstream._ParticipantAudioInputStream
        session = mock.MagicMock()
        room = mock.MagicMock()
        voice_io = preconnect.PreConnectRoomIO(agent_session=session, room=room, options=room_options(timeout=10.0))

        await voice_io.start()

        self.assertIsInstance(voice_io.audio_input, preconnect._PreConnectAudioInput)
        self.assertIs(upstream._ParticipantAudioInputStream, stock)

    async def test_start_builds_the_open_recording_handler_with_the_options_timeout(self):
        upstream = preconnect.upstream
        stock = upstream.PreConnectAudioHandler
        voice_io = preconnect.PreConnectRoomIO(agent_session=mock.MagicMock(), room=mock.MagicMock(),
                                               options=room_options(timeout=10.0))

        await voice_io.start()

        handler = voice_io._pre_connect_audio_handler
        self.assertIsInstance(handler, preconnect._OpenRecordingHandler)
        self.assertEqual(handler._timeout, 10.0)
        self.assertIs(upstream.PreConnectAudioHandler, stock)

    def test_it_is_a_room_io(self):
        self.assertTrue(issubclass(preconnect.PreConnectRoomIO, preconnect.room_io.RoomIO))


async def started_room_io(**kwargs):
    voice_io = preconnect.PreConnectRoomIO(
        agent_session=mock.MagicMock(), room=mock.MagicMock(), options=room_options(timeout=10.0), **kwargs
    )
    await voice_io.start()
    return voice_io


async def record_turn(voice_io, clock, track_id, buffer, live, *, base, rate=BUFFER_RATE):
    """One track's pre-connect turn through the room IO's input, on the clock.

    The first live frames are heard at base, the buffer's stream opens at base + 0.2 s, and the rest
    of the live head is heard at base + 0.3 s. The head is the 1.2 s the buffer's window needs.
    """
    handler = voice_io._pre_connect_audio_handler
    audio_input = voice_io.audio_input
    stream = GatedStream()
    track = SimpleNamespace(sid=track_id)
    publication = SimpleNamespace(sid=track_id, source=0, audio_features=[PRECONNECT_FEATURE])
    participant = SimpleNamespace(identity="phone")
    frames = frames_of(live, LIVE_RATE)
    with track_streams(stream):
        live_stream = audio_input._create_stream(track, participant)
    clock.now = base
    stream.push(frames[:4])
    await settle()
    forward = asyncio.create_task(audio_input._forward_task(None, live_stream, track, publication, participant))
    await settle()
    clock.now = base + 0.2
    await arrive(handler, pcm_chunks(buffer, rate), track_id=track_id, rate=rate)
    clock.now = base + 0.3
    stream.push(frames[4:24])
    await settle()
    stream.end()
    await forward


async def forwarded_frames(voice_io):
    audio_input = voice_io.audio_input
    audio_input._data_ch.close()
    return [frame async for frame in audio_input._data_ch]


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getnchannels(), wav.readframes(wav.getnframes())


def read_arrivals(path):
    return [(float(offset), int(count)) for offset, count in (line.split() for line in path.read_text().splitlines())]


class PreConnectRecordingTest(AsyncTest):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.clock = FakeTime()
        clock = mock.patch.object(preconnect, "time", self.clock)
        clock.start()
        self.addCleanup(clock.stop)

    async def test_a_processed_buffer_is_written_with_its_samples_and_the_live_head_it_was_cut_from(self):
        buffer = speechlike(1.0, seed=1, rate=BUFFER_RATE)
        live = speechlike(1.2, seed=2)
        with tempfile.TemporaryDirectory() as directory:
            voice_io = await started_room_io(record_dir=Path(directory))

            await record_turn(voice_io, self.clock, "TR_one", buffer, live, base=0.0)

            self.assertEqual(sorted(os.listdir(directory)), sorted([
                "TR_one-preconnect-buffer-turn-001.wav",
                "TR_one-preconnect-live-turn-001.wav",
                "TR_one-preconnect-live-turn-001.txt",
            ]))
            self.assertEqual(
                read_wav(Path(directory, "TR_one-preconnect-buffer-turn-001.wav")),
                (BUFFER_RATE, 1, buffer.tobytes()),
            )
            self.assertEqual(
                read_wav(Path(directory, "TR_one-preconnect-live-turn-001.wav")),
                (LIVE_RATE, 1, live.tobytes()),
            )
            self.assertEqual(
                read_arrivals(Path(directory, "TR_one-preconnect-live-turn-001.txt")),
                [(-0.2, 1200)] * 4 + [(0.1, 1200)] * 20,
            )
            for name in os.listdir(directory):
                self.assertEqual(stat.S_IMODE(os.stat(Path(directory, name)).st_mode), 0o600)

    async def test_each_processed_buffer_takes_the_next_number(self):
        buffer = speechlike(1.0, seed=1, rate=BUFFER_RATE)
        live = speechlike(1.2, seed=2)
        with tempfile.TemporaryDirectory() as directory:
            voice_io = await started_room_io(record_dir=Path(directory))

            await record_turn(voice_io, self.clock, "TR_one", buffer, live, base=0.0)
            await record_turn(voice_io, self.clock, "TR_two", buffer, live, base=1.0)

            self.assertEqual(sorted(os.listdir(directory)), sorted([
                "TR_one-preconnect-buffer-turn-001.wav",
                "TR_one-preconnect-live-turn-001.wav",
                "TR_one-preconnect-live-turn-001.txt",
                "TR_two-preconnect-buffer-turn-002.wav",
                "TR_two-preconnect-live-turn-002.wav",
                "TR_two-preconnect-live-turn-002.txt",
            ]))

    async def test_an_existing_turn_is_never_overwritten(self):
        buffer = speechlike(1.0, seed=1, rate=BUFFER_RATE)
        live = speechlike(1.2, seed=2)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "TR_one-preconnect-buffer-turn-001.wav").write_bytes(b"kept")
            voice_io = await started_room_io(record_dir=Path(directory))

            await record_turn(voice_io, self.clock, "TR_one", buffer, live, base=0.0)

            self.assertEqual(Path(directory, "TR_one-preconnect-buffer-turn-001.wav").read_bytes(), b"kept")
            self.assertTrue(Path(directory, "TR_one-preconnect-buffer-turn-002.wav").is_file())
            self.assertTrue(Path(directory, "TR_one-preconnect-live-turn-002.txt").is_file())

    async def test_without_a_record_dir_nothing_is_written_and_the_cut_is_the_same(self):
        # A buffer at the live rate passes LiveKit's resampler unchanged. LiveKit's gain control then
        # processes the live output, so the cut is compared by its log line and the whole output.
        buffer = speechlike(1.0, seed=1)
        live = speechlike(1.2, seed=2)
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            with self.assertLogs("mentat.voice", "INFO") as unrecorded_logs:
                unrecorded_io = await started_room_io()
                await record_turn(unrecorded_io, self.clock, "TR_one", buffer, live, base=0.0, rate=LIVE_RATE)
                unrecorded = await forwarded_frames(unrecorded_io)
            self.assertEqual(os.listdir(directory), [])

        with tempfile.TemporaryDirectory() as record_dir, self.assertLogs("mentat.voice", "INFO") as recorded_logs:
            recorded_io = await started_room_io(record_dir=Path(record_dir))
            await record_turn(recorded_io, self.clock, "TR_one", buffer, live, base=0.0, rate=LIVE_RATE)
            recorded = await forwarded_frames(recorded_io)

        cut = pre_connect_messages(unrecorded_logs, "pre-connect cut:")
        self.assertEqual(cut, ["pre-connect cut: path=timing dropped=0.20 s"])
        self.assertEqual(pre_connect_messages(recorded_logs, "pre-connect cut:"), cut)
        self.assertEqual(samples_of(recorded), samples_of(unrecorded))

    async def test_a_write_failure_is_logged_and_the_cut_still_applies(self):
        buffer = speechlike(1.0, seed=1)
        live = speechlike(1.2, seed=2)
        unrecorded_io = await started_room_io()
        await record_turn(unrecorded_io, self.clock, "TR_one", buffer, live, base=0.0, rate=LIVE_RATE)
        unrecorded = await forwarded_frames(unrecorded_io)

        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory, "recordings")
            blocked.write_bytes(b"")  # a file where the recording directory should be
            voice_io = await started_room_io(record_dir=blocked)
            with self.assertLogs("mentat.voice", "ERROR") as logs:
                await record_turn(voice_io, self.clock, "TR_one", buffer, live, base=0.0, rate=LIVE_RATE)
            cut = await forwarded_frames(voice_io)

        self.assertEqual([record.getMessage() for record in logs.records], ["failed to write pre-connect recording"])
        self.assertEqual(samples_of(cut), samples_of(unrecorded))


class CutRepeatTest(unittest.TestCase):
    def setUp(self):
        self.buffer = speechlike(1.04, seed=1)
        self.tail = self.buffer[-round(0.92 * LIVE_RATE):]

    def test_a_stretched_jittered_repeat_is_aligned_and_cut_at_its_end(self):
        repeat = jittered_repeat(self.tail, noise_scale=1.2, seed=3)
        live = np.concatenate([repeat, speechlike(0.6, seed=4)])

        cut, path = preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual(path, "aligned")
        self.assertLessEqual(abs(cut - len(repeat)), round(0.05 * LIVE_RATE))

    def test_a_voiced_repeat_stretched_by_a_gap_before_its_end_is_aligned_and_cut_at_its_end(self):
        repeat = self.buffer[-round(0.9 * LIVE_RATE):]
        split = len(repeat) - round(0.25 * LIVE_RATE)
        live = np.concatenate([repeat[:split], speechlike(0.04, seed=5), repeat[split:], speechlike(0.6, seed=4)])

        cut, path = preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual(path, "aligned")
        self.assertLessEqual(abs(cut - (len(repeat) + round(0.04 * LIVE_RATE))), round(0.05 * LIVE_RATE))

    def test_a_buffer_ending_in_a_quiet_pause_is_aligned_and_cut_after_the_pause(self):
        speech = speechlike(0.7, seed=1)
        buffer = np.concatenate([speech, quiet_noise(0.3, seed=7)])
        voiced_repeat = jittered_repeat(speech[-round(0.6 * LIVE_RATE):], noise_scale=1.2, seed=3)
        repeat = np.concatenate([voiced_repeat, quiet_noise(0.3, seed=8)])
        live = np.concatenate([repeat, speechlike(0.6, seed=4)])

        cut, path = preconnect.cut_repeat(buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual(path, "aligned")
        self.assertLessEqual(abs(cut - len(repeat)), round(0.05 * LIVE_RATE))

    def test_a_buffer_whose_voiced_part_is_shorter_than_the_longest_tail_is_timing(self):
        buffer = np.concatenate([speechlike(0.5, seed=1), quiet_noise(0.5, seed=7)])
        live = np.concatenate([buffer, speechlike(0.5, seed=4)])

        self.assertEqual(
            preconnect.cut_repeat(buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE)),
            (round(0.3 * LIVE_RATE), "timing"),
        )

    def test_no_repeat_is_timing_cut_at_the_samples_heard_before_open(self):
        live = speechlike(1.0, seed=2)

        self.assertEqual(
            preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.5 * LIVE_RATE)),
            (round(0.5 * LIVE_RATE), "timing"),
        )

    def test_timing_is_capped_at_the_buffer_length(self):
        buffer = self.buffer[:round(0.8 * LIVE_RATE)]
        live = speechlike(1.0, seed=2)

        self.assertEqual(
            preconnect.cut_repeat(buffer, live, LIVE_RATE, 2 * LIVE_RATE),
            (len(buffer), "timing"),
        )

    def test_timing_is_capped_at_one_second(self):
        live = speechlike(1.0, seed=2)

        self.assertEqual(
            preconnect.cut_repeat(self.buffer, live, LIVE_RATE, 2 * LIVE_RATE),
            (LIVE_RATE, "timing"),
        )

    def test_tails_that_end_apart_are_timing(self):
        # The buffer's last 0.2, 0.4 and 0.6 s, back to back: each tail matches its own copy
        # exactly, ending at 0.2, 0.6 and 1.2 s, so no two of the three agree within 0.05 s.
        tail = self.buffer[-round(0.6 * LIVE_RATE):]
        live = np.concatenate([
            tail[-round(0.2 * LIVE_RATE):], tail[-round(0.4 * LIVE_RATE):], tail, speechlike(0.5, seed=6),
        ])

        self.assertEqual(
            preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE)),
            (round(0.3 * LIVE_RATE), "timing"),
        )

    def test_three_tails_at_or_above_point_three_that_end_apart_are_timing(self):
        matches = [
            (0.2, round(0.40 * LIVE_RATE), 0.90),
            (0.4, round(0.60 * LIVE_RATE), 0.90),
            (0.6, round(0.80 * LIVE_RATE), 0.90),
        ]
        live = speechlike(1.0, seed=2)

        with mock.patch.object(preconnect, "_match_tails", return_value=matches):
            self.assertEqual(
                preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE)),
                (round(0.3 * LIVE_RATE), "timing"),
            )

    def test_a_pair_ending_six_hundredths_of_a_second_apart_is_timing(self):
        matches = [
            (0.2, round(0.90 * LIVE_RATE), 0.10),
            (0.4, round(0.50 * LIVE_RATE), 0.90),
            (0.6, round(0.56 * LIVE_RATE), 0.90),
        ]
        live = speechlike(1.0, seed=2)

        with mock.patch.object(preconnect, "_match_tails", return_value=matches):
            self.assertEqual(
                preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE)),
                (round(0.3 * LIVE_RATE), "timing"),
            )

    def test_two_agreeing_tails_align_when_the_third_is_a_weak_outlier(self):
        # lcdzk1ju's live tails: 0.20 s end=0.93 s r=0.60, 0.40 s end=1.15 s r=0.35, 0.60 s end=0.97 s r=0.46
        matches = [
            (0.2, round(0.93 * LIVE_RATE), 0.60),
            (0.4, round(1.15 * LIVE_RATE), 0.35),
            (0.6, round(0.97 * LIVE_RATE), 0.46),
        ]
        live = speechlike(1.0, seed=2)

        with mock.patch.object(preconnect, "_match_tails", return_value=matches):
            cut, path = preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual(path, "aligned")
        self.assertEqual(cut, (round(0.93 * LIVE_RATE) + round(0.97 * LIVE_RATE)) // 2)

    def test_of_two_agreeing_pairs_the_one_with_the_closer_ends_aligns(self):
        matches = [
            (0.2, round(0.40 * LIVE_RATE), 0.90),
            (0.4, round(0.44 * LIVE_RATE), 0.90),
            (0.6, round(0.46 * LIVE_RATE), 0.90),
        ]
        live = speechlike(1.0, seed=2)

        with mock.patch.object(preconnect, "_match_tails", return_value=matches):
            cut, path = preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual(path, "aligned")
        self.assertEqual(cut, (round(0.44 * LIVE_RATE) + round(0.46 * LIVE_RATE)) // 2)

    def test_a_single_tail_at_or_above_point_three_with_the_others_below_is_timing(self):
        matches = [
            (0.2, round(0.5 * LIVE_RATE), 0.90),
            (0.4, round(0.9 * LIVE_RATE), 0.10),
            (0.6, round(0.7 * LIVE_RATE), 0.10),
        ]
        live = speechlike(1.0, seed=2)

        with mock.patch.object(preconnect, "_match_tails", return_value=matches):
            self.assertEqual(
                preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE)),
                (round(0.3 * LIVE_RATE), "timing"),
            )

    def test_tails_that_correlate_below_point_three_are_timing(self):
        repeat = jittered_repeat(self.tail, noise_scale=4.0, seed=3)
        live = np.concatenate([repeat, speechlike(0.6, seed=4)])

        self.assertEqual(
            preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE)),
            (round(0.3 * LIVE_RATE), "timing"),
        )

    def test_a_buffer_too_short_for_the_longest_tail_is_timing(self):
        buffer = speechlike(0.5, seed=1)
        live = np.concatenate([buffer, speechlike(0.5, seed=4)])

        self.assertEqual(
            preconnect.cut_repeat(buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE)),
            (round(0.3 * LIVE_RATE), "timing"),
        )

    def test_a_live_head_too_short_for_the_longest_tail_is_timing(self):
        live = self.buffer[-round(0.3 * LIVE_RATE):]

        self.assertEqual(
            preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.4 * LIVE_RATE)),
            (round(0.4 * LIVE_RATE), "timing"),
        )

    def test_an_aligned_cut_logs_its_three_tails_with_the_ends_and_correlations_it_decided(self):
        repeat = jittered_repeat(self.tail, noise_scale=1.2, seed=3)
        live = np.concatenate([repeat, speechlike(0.6, seed=4)])

        with self.assertLogs("mentat.voice", "INFO") as logs:
            cut, path = preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual(path, "aligned")
        [line] = pre_connect_messages(logs, "pre-connect tails:")
        match = re.fullmatch(r"pre-connect tails: quiet=(\d+\.\d{2}) s (.+)", line)
        self.assertIsNotNone(match, line)
        entries = [re.fullmatch(TAIL_ENTRY, entry) for entry in match[2].split(", ")]
        self.assertTrue(all(entries), line)
        self.assertEqual([float(entry[1]) for entry in entries], [0.2, 0.4, 0.6])

        voiced_end = preconnect._voiced_end(self.buffer, LIVE_RATE)
        voiced = self.buffer[:voiced_end]
        window = live[:len(self.buffer) + round(0.2 * LIVE_RATE)]
        decided = [
            preconnect._best_match(voiced[-round(seconds * LIVE_RATE):], window)
            for seconds in (0.2, 0.4, 0.6)
        ]
        for entry, (end, correlation) in zip(entries, decided):
            self.assertAlmostEqual(float(entry[2]), end / LIVE_RATE, delta=0.0051)
            self.assertAlmostEqual(float(entry[3]), correlation, delta=0.0051)
        ends = [float(entry[2]) for entry in entries]
        self.assertAlmostEqual(sorted(ends)[1] + float(match[1]), cut / LIVE_RATE, delta=0.0101)

    def test_a_no_repeat_timing_cut_logs_all_three_tails_with_their_sub_point_three_correlations(self):
        live = speechlike(1.0, seed=2)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            cut, path = preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.5 * LIVE_RATE))

        self.assertEqual((cut, path), (round(0.5 * LIVE_RATE), "timing"))
        [line] = pre_connect_messages(logs, "pre-connect tails:")
        match = re.fullmatch(r"pre-connect tails: quiet=(\d+\.\d{2}) s (.+)", line)
        self.assertIsNotNone(match, line)
        self.assertEqual(match[1], "0.00")
        entries = [re.fullmatch(TAIL_ENTRY, entry) for entry in match[2].split(", ")]
        self.assertTrue(all(entries), line)
        self.assertEqual([float(entry[1]) for entry in entries], [0.2, 0.4, 0.6])
        voiced_end = preconnect._voiced_end(self.buffer, LIVE_RATE)
        voiced = self.buffer[:voiced_end]
        window = live[:len(self.buffer) + round(0.2 * LIVE_RATE)]
        decided = [
            preconnect._best_match(voiced[-round(seconds * LIVE_RATE):], window)
            for seconds in (0.2, 0.4, 0.6)
        ]
        for entry, (end, correlation) in zip(entries, decided):
            self.assertAlmostEqual(float(entry[2]), end / LIVE_RATE, delta=0.0051)
            self.assertAlmostEqual(float(entry[3]), correlation, delta=0.0051)
            self.assertLess(correlation, 0.3)

    def test_a_timing_cut_logs_every_tail_that_fits_including_those_below_point_three(self):
        loud = np.round(np.random.default_rng(9).standard_normal(round(0.25 * LIVE_RATE)) * 20000)
        loud = loud.clip(-32768, 32767).astype(np.int16)
        clean = self.tail[-round(0.2 * LIVE_RATE):]
        live = np.concatenate([loud, clean, speechlike(0.6, seed=4)])

        with self.assertLogs("mentat.voice", "INFO") as logs:
            cut, path = preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual((cut, path), (round(0.3 * LIVE_RATE), "timing"))
        [line] = pre_connect_messages(logs, "pre-connect tails:")
        match = re.fullmatch(r"pre-connect tails: quiet=(\d+\.\d{2}) s (.+)", line)
        self.assertIsNotNone(match, line)
        self.assertEqual(match[1], "0.00")
        entries = [re.fullmatch(TAIL_ENTRY, entry) for entry in match[2].split(", ")]
        self.assertTrue(all(entries), line)
        self.assertEqual([float(entry[1]) for entry in entries], [0.2, 0.4, 0.6])
        self.assertAlmostEqual(float(entries[0][2]), 0.45, delta=0.0051)
        self.assertGreater(float(entries[0][3]), 0.95)
        self.assertTrue(all(float(entry[3]) < 0.3 for entry in entries[1:]), line)

    def test_a_silent_buffer_logs_none_rather_than_matching_its_flat_tails(self):
        buffer = np.zeros(self.buffer.size, dtype=np.int16)
        live = speechlike(1.0, seed=2)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            cut, path = preconnect.cut_repeat(buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual((cut, path), (round(0.3 * LIVE_RATE), "timing"))
        [line] = pre_connect_messages(logs, "pre-connect tails:")
        self.assertRegex(line, r"^pre-connect tails: quiet=\d+\.\d{2} s none$")

    def test_a_buffer_too_short_for_any_tail_logs_none(self):
        buffer = speechlike(0.1, seed=1)
        live = speechlike(1.0, seed=2)

        with self.assertLogs("mentat.voice", "INFO") as logs:
            cut, path = preconnect.cut_repeat(buffer, live, LIVE_RATE, round(0.3 * LIVE_RATE))

        self.assertEqual((cut, path), (len(buffer), "timing"))
        [line] = pre_connect_messages(logs, "pre-connect tails:")
        self.assertRegex(line, r"^pre-connect tails: quiet=\d+\.\d{2} s none$")

    def test_an_empty_buffer_logs_no_tails_line(self):
        with self.assertNoLogs("mentat.voice", "INFO"):
            cut, path = preconnect.cut_repeat(np.zeros(0, dtype=np.int16), self.buffer, LIVE_RATE, 0)

        self.assertEqual((cut, path), (0, "timing"))


if __name__ == "__main__":
    unittest.main()
