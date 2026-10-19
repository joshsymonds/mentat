"""The worker's microphone input hears the phone's pre-connect buffer once, on synthetic audio.

The system python has no livekit, so the module is imported against small stand-ins for the
few LiveKit pieces it touches; where livekit is installed the real ones are used instead.
"""

import asyncio
import importlib
import importlib.util
import sys
import types
import unittest
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

    class RoomIO:
        def __init__(self, agent_session, room, *, participant=None, options=None):
            self._room = room
            self._options = options
            self._audio_input = None

        async def start(self):
            audio = self._options.audio_input
            self._audio_input = upstream._ParticipantAudioInputStream(
                self._room,
                sample_rate=audio.sample_rate,
                num_channels=1,
                noise_cancellation=audio.noise_cancellation,
                auto_gain_control=False,
                pre_connect_audio_handler=None,
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
                      RoomIO=RoomIO, _ParticipantAudioInputStream=ParticipantAudioInputStream)
    room_io = module("livekit.agents.voice.room_io", RoomIO=RoomIO, room_io=upstream,
                     _input=module("livekit.agents.voice.room_io._input",
                                   _ParticipantAudioInputStream=ParticipantAudioInputStream))
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
    def __init__(self, frames=None, error=None):
        self._frames, self._error = frames, error

    async def wait_for_data(self, track_id):
        if self._error:
            raise self._error
        return self._frames


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
        self.live = preconnect._LiveAudio(self.stream, sample_rate=LIVE_RATE)

    async def test_live_frames_heard_before_the_buffer_completed_are_dropped_and_the_rest_pass(self):
        live = ramp(2.0)
        frames = frames_of(live, LIVE_RATE)
        self.stream.push(frames[:10])  # 0.5 s heard from t=0
        await settle()
        self.clock.now = 0.7
        list(self.live.process_buffer(silence(1.0)))
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
        list(self.live.process_buffer(silence(0.73)))
        self.stream.push(frames[15:])
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out), live[round(0.73 * LIVE_RATE):])

    async def test_no_more_than_the_buffer_length_is_dropped(self):
        live = ramp(2.0)
        self.stream.push(frames_of(live, LIVE_RATE))  # all 2 s heard from t=0
        await settle()
        self.clock.now = 1.0
        list(self.live.process_buffer(silence(0.5)))
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out), live[round(0.5 * LIVE_RATE):])

    async def test_live_frames_heard_after_the_buffer_completed_all_pass(self):
        live = ramp(1.0)
        self.clock.now = 0.0
        list(self.live.process_buffer(silence(1.0)))
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
        list(self.live.process_buffer([]))
        self.stream.end()

        out = await drain(self.live)

        self.assertEqual(samples_of(out), live)

    async def test_without_a_buffer_live_audio_flows_unchanged_and_unheld(self):
        frames = frames_of(ramp(1.0), LIVE_RATE)
        self.stream.push(frames)  # not ended: the track stays open
        await settle()

        first = await asyncio.wait_for(anext(self.live), timeout=1)

        self.assertIs(first.frame, frames[0])

    async def test_the_drop_logs_once_with_the_seconds_dropped_and_the_arrival_times(self):
        self.stream.push(frames_of(ramp(0.5), LIVE_RATE))  # first live frame heard at t=0
        await settle()
        self.clock.now = 0.7

        with self.assertLogs("mentat.voice", level="INFO") as logs:
            list(self.live.process_buffer(silence(1.0)))

        [record] = logs.records
        self.assertEqual(record.levelname, "INFO")
        self.assertRegex(
            record.getMessage(),
            r"^dropped 0\.50 s of live audio .*first live frame at 0\.000, buffer completed at 0\.700$",
        )

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

        forwarded = await run_input(live, FakeHandler(error=TimeoutError()))

        self.assertEqual(samples_of(forwarded[:-1]), samples_of(live))

    async def test_a_buffer_error_other_than_a_timeout_leaves_live_audio_untouched(self):
        live = frames_of(ramp(2.0), LIVE_RATE)

        forwarded = await run_input(live, FakeHandler(error=RuntimeError("byte stream failed")))

        self.assertEqual(samples_of(forwarded[:-1]), samples_of(live))

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


class PreConnectRoomIOTest(AsyncTest):
    async def test_start_installs_the_input_and_restores_livekits(self):
        upstream = preconnect.upstream
        stock = upstream._ParticipantAudioInputStream
        if STAND_INS:
            options = SimpleNamespace(audio_input=SimpleNamespace(sample_rate=LIVE_RATE, noise_cancellation=None))
        else:
            options = preconnect.room_io.RoomOptions(
                audio_input=preconnect.room_io.AudioInputOptions(),
                audio_output=False,
                text_output=False,
            )
        session = mock.MagicMock()
        room = mock.MagicMock()
        voice_io = preconnect.PreConnectRoomIO(agent_session=session, room=room, options=options)

        await voice_io.start()

        self.assertIsInstance(voice_io.audio_input, preconnect._PreConnectAudioInput)
        self.assertIs(upstream._ParticipantAudioInputStream, stock)

    def test_it_is_a_room_io(self):
        self.assertTrue(issubclass(preconnect.PreConnectRoomIO, preconnect.room_io.RoomIO))


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

    def test_no_repeat_is_timing_cut_at_the_samples_heard_before_open(self):
        live = speechlike(1.0, seed=2)

        self.assertEqual(
            preconnect.cut_repeat(self.buffer, live, LIVE_RATE, round(0.5 * LIVE_RATE)),
            (round(0.5 * LIVE_RATE), "timing"),
        )

    def test_timing_is_capped_at_the_buffer_length(self):
        live = speechlike(1.0, seed=2)

        self.assertEqual(
            preconnect.cut_repeat(self.buffer, live, LIVE_RATE, 2 * LIVE_RATE),
            (len(self.buffer), "timing"),
        )

    def test_tails_that_end_apart_are_timing(self):
        tail = self.buffer[-round(0.6 * LIVE_RATE):]
        split = round(0.4 * LIVE_RATE)
        live = np.concatenate([tail[:split], speechlike(0.04, seed=5), tail[split:], speechlike(0.5, seed=6)])

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


if __name__ == "__main__":
    unittest.main()
