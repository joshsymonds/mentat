"""The worker's microphone input hears the phone's pre-connect buffer once, on synthetic audio.

The system python has no livekit, so the module is imported against small stand-ins for the
few LiveKit pieces it touches; where livekit is installed the real ones are used instead.
"""

import asyncio
import gzip
import importlib
import importlib.util
import math
import random
import re
import struct
import sys
import threading
import types
import unittest
from array import array
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

VOICE_DIR = Path(__file__).resolve().parent.parent
LIVE_RATE = 24000
BUFFER_RATE = 48000
FRAME_SECONDS = 0.05
BUFFER_END = 1.0
TOLERANCE = LIVE_RATE // 1000  # a millisecond of live samples


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
            self._noise_cancellation = noise_cancellation
            self._pre_connect_audio_handler = pre_connect_audio_handler
            self._pre_connect_audio_publications = set()
            self._processor = noise_cancellation if isinstance(noise_cancellation, FrameProcessor) else None
            self._data_ch = Chan()
            self._attached = True

        def _update_processor(self, processor):
            self._processor = processor

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
                except TimeoutError:
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
                                   _ParticipantAudioInputStream=ParticipantAudioInputStream),
                     types=module("livekit.agents.voice.room_io.types",
                                  NoiseCancellationParams=lambda participant, track: (participant, track)))
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
        "livekit.agents.voice.room_io.types": room_io.types,
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

def syllables(seed, start, end):
    """A run of voiced syllables as (start, duration, pitch, amplitude) between two times."""
    rng = random.Random(seed)
    out, now = [], start
    while True:
        now += rng.uniform(0.02, 0.05)
        length = rng.uniform(0.12, 0.2)
        if now + length > end:
            return out
        out.append((now, length, rng.uniform(95, 220), rng.uniform(3000, 6000)))
        now += length


def voice_at(plan, moment, pitch_scale=1.0, phase=0.0):
    for start, length, pitch, amplitude in plan:
        if start <= moment < start + length:
            into = moment - start
            envelope = math.sin(math.pi * into / length) ** 2
            glide = 1 + 0.03 * math.sin(2 * math.pi * 5 * into)
            return amplitude * envelope * sum(
                math.sin(2 * math.pi * harmonic * pitch * pitch_scale * into * glide + phase + harmonic) / harmonic
                for harmonic in range(1, 7)
            )
    return 0.0


def record(plan, begin, end, rate, *, seed, noise=0.0, gain=1.0, pitch_scale=1.0, phase=0.0):
    """Int16 samples of the plan between two times, with a quiet room and optional codec noise."""
    rng = random.Random(seed)
    count = round((end - begin) * rate)
    samples = array("h")
    for index in range(count):
        value = gain * voice_at(plan, begin + index / rate, pitch_scale, phase)
        value += rng.gauss(0, 40 + noise)
        samples.append(max(-32768, min(32767, round(value))))
    return samples


FIXTURE = VOICE_DIR / "tests" / "fixtures" / "preconnect-eval-line.pcm.gz"
EVAL_TOLERANCE = LIVE_RATE // 100  # ten milliseconds of live samples
# The fixture is four back-to-back sections of the eval's line at 24 kHz mono int16: the phone's
# buffer (the rendered line, 0.0 to 1.0 s), a live head that is an Opus copy of the line from
# 0.3 s, the same copy after 0.3 s of silence, and another TTS rendering of the same sentence.
EVAL_SECTIONS = {
    "buffer": (0, 24000),
    "copy": (24000, 48000),
    "copy_after_silence": (48000, 79200),
    "other_rendering": (79200, 103200),
}


def eval_line(section):
    """One section of the eval's recorded line, as int16 samples at 24 kHz."""
    with gzip.open(FIXTURE) as recording:
        raw = recording.read()
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    first, last = EVAL_SECTIONS[section]
    return array("h", samples[first:last])


def eval_case(live_section):
    """The eval's line as the phone sends it: the buffer repeated up to 48 kHz, the live head at 24 kHz."""
    doubled = array("h", (value for value in eval_line("buffer") for _ in (0, 1)))
    return SimpleNamespace(
        buffer=frames_of(doubled, BUFFER_RATE),
        live=frames_of(eval_line(live_section), LIVE_RATE),
    )


def frames_of(samples, rate):
    step = round(rate * FRAME_SECONDS)
    return [
        rtc.AudioFrame(samples[at:at + step].tobytes(), rate, 1, len(samples[at:at + step]))
        for at in range(0, len(samples), step)
    ]


def samples_of(frames):
    out = array("h")
    for frame in frames:
        out.extend(array("h", bytes(frame.data)))
    return out


class Overlap:
    """One speaker: the phone buffer ends at BUFFER_END, the live track starts overlap seconds before."""

    def __init__(self, plan, *, overlap, live_seconds=2.0, buffer_rate=BUFFER_RATE):
        self.overlap = overlap
        live_begin = BUFFER_END - overlap
        self.buffer = frames_of(record(plan, 0.0, BUFFER_END, buffer_rate, seed=1), buffer_rate)
        self.live = frames_of(
            record(plan, live_begin, live_begin + live_seconds, LIVE_RATE, seed=2, noise=150, gain=0.96),
            LIVE_RATE,
        )

    @property
    def repeated(self):
        return round(self.overlap * LIVE_RATE)


# -- harness ---------------------------------------------------------------------------------

def recording_canceller():
    """A noise canceller that changes nothing and notes the frames it was shown, in order."""

    class Canceller(rtc.FrameProcessor):
        def __init__(self):
            self.seen = []
            self._enabled = True

        @property
        def enabled(self):
            return self._enabled

        @enabled.setter
        def enabled(self, value):
            self._enabled = value

        def _process(self, frame):
            self.seen.append(frame)
            return frame

        def _close(self):
            pass

    return Canceller()


class ScriptedStream:
    """Stands in for rtc.AudioStream: yields the frames, then waits like an open track."""

    def __init__(self, frames, processor=None, ends=True):
        # rtc runs a processor handed to it on every frame as it arrives, before anyone reads it.
        self._frames = deque(processor._process(frame) if processor else frame for frame in frames)
        self._ends = ends
        self.closed = False

    async def __anext__(self):
        if self._frames:
            return rtc.AudioFrameEvent(self._frames.popleft())
        if self._ends:
            raise StopAsyncIteration
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


async def drain(live):
    out = []
    async for event in live:
        out.append(event.frame)
    return out


async def dedupe(case, canceller=None):
    """The live frames that come out for a buffer and a live track, with the buffer processed first."""
    canceller = canceller or recording_canceller()
    live = preconnect._LiveAudio(
        ScriptedStream(case.live), sample_rate=LIVE_RATE, cancel=canceller._process
    )
    list(live.process_buffer(case.buffer))
    return await drain(live)


def kept_after_skipping(case, out):
    """How many live samples were skipped, checking the rest is the live track unchanged."""
    live = samples_of(case.live)
    kept = samples_of(out)
    skipped = len(live) - len(kept)
    assert kept == live[skipped:], "the kept audio is not the tail of the live track"
    return skipped


MISS = re.compile(
    r"no repeat of the (?P<buffer>[\d.]+) s pre-connect buffer in the (?P<held>[\d.]+) s of live audio "
    r"held: best score (?P<score>-?[\d.]+), live onset (?P<onset>[\d.]+) s, "
    r"buffer position (?P<position>[\d.]+) s"
)


def removed_span(case, out):
    """The live samples the output lacks, as (begin, end), checking nothing else changed."""
    live = samples_of(case.live)
    kept = samples_of(out)
    removed = len(live) - len(kept)
    begin = next(
        (at for at, (heard, said) in enumerate(zip(live, kept)) if heard != said), len(kept)
    )
    assert kept[begin:] == live[begin + removed:], "the output is not the live track minus one span"
    return begin, begin + removed


class FakeHandler:
    def __init__(self, frames=None, error=None):
        self._frames, self._error = frames, error

    async def wait_for_data(self, track_id):
        if self._error:
            raise self._error
        return self._frames


def build_input(canceller, handler):
    return preconnect._PreConnectAudioInput(
        mock.MagicMock(),
        sample_rate=LIVE_RATE,
        num_channels=1,
        noise_cancellation=canceller,
        auto_gain_control=False,
        pre_connect_audio_handler=handler,
        frame_size_ms=50,
    )


async def run_input(case_live, case_buffer, canceller, *, buffered=True):
    """Drive the input's forward task over a track the way LiveKit does; returns what it forwarded."""
    handler = FakeHandler(case_buffer if buffered else None, error=None if buffered else TimeoutError())
    built = {}

    def from_track(**kwargs):
        processor = kwargs["noise_cancellation"]
        built["processor"] = processor
        return ScriptedStream(case_live, processor if isinstance(processor, rtc.FrameProcessor) else None)

    track = SimpleNamespace(sid="TR_audio")
    publication = SimpleNamespace(sid="TR_audio", source=0, audio_features=[PRECONNECT_FEATURE])
    participant = SimpleNamespace(identity="phone")
    with mock.patch.object(rtc.AudioStream, "from_track", staticmethod(from_track)):
        audio_input = build_input(canceller, handler)
        stream = audio_input._create_stream(track, participant)
        await audio_input._forward_task(None, stream, track, publication, participant)
    audio_input._data_ch.close()
    forwarded = [frame async for frame in audio_input._data_ch]
    return forwarded, built


class AsyncTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Building seconds of synthetic audio inside a test would trip the loop's slow-callback log.
        asyncio.get_running_loop().set_debug(False)


# -- tests -----------------------------------------------------------------------------------

class PreConnectDedupeTest(AsyncTest):
    async def test_buffer_and_overlapping_live_head_yield_one_copy_of_the_overlap(self):
        case = Overlap(syllables(7, 0.0, 3.0), overlap=0.7)

        skipped = kept_after_skipping(case, await dedupe(case))

        self.assertAlmostEqual(skipped, case.repeated, delta=TOLERANCE)

    async def test_a_short_overlap_found_only_once_the_head_is_long_enough_is_still_skipped(self):
        case = Overlap(syllables(11, 0.0, 3.0), overlap=0.3)

        skipped = kept_after_skipping(case, await dedupe(case))

        self.assertAlmostEqual(skipped, case.repeated, delta=TOLERANCE)

    async def test_no_overlap_keeps_every_sample(self):
        plan = syllables(7, 0.0, 3.0)
        case = Overlap(plan, overlap=-0.2)  # the live track starts after the buffer ended

        self.assertEqual(kept_after_skipping(case, await dedupe(case)), 0)

    async def test_no_overlap_logs_one_miss_with_the_held_length_and_the_best_fit(self):
        case = Overlap(syllables(7, 0.0, 3.0), overlap=-0.2)

        with self.assertLogs("mentat.voice", level="INFO") as logs:
            self.assertEqual(kept_after_skipping(case, await dedupe(case)), 0)

        self.assertMiss(logs, held=1.0, buffer=BUFFER_END)

    async def test_a_silent_head_logs_that_no_onset_was_found(self):
        case = Overlap(syllables(7, 0.0, 3.0), overlap=0.7)
        case.live = frames_of(array("h", bytes(2 * 2 * LIVE_RATE)), LIVE_RATE)

        with self.assertLogs("mentat.voice", level="INFO") as logs:
            self.assertEqual(kept_after_skipping(case, await dedupe(case)), 0)

        [record] = logs.records
        self.assertEqual(record.levelname, "INFO")
        self.assertRegex(record.getMessage(), r"^no onset found in the 1\.00 s of live audio held ")
        self.assertNotIn("score", record.getMessage())

    async def test_a_matched_copy_logs_only_the_dropped_samples(self):
        case = Overlap(syllables(7, 0.0, 3.0), overlap=0.7)

        with self.assertLogs("mentat.voice", level="INFO") as logs:
            skipped = kept_after_skipping(case, await dedupe(case))

        [record] = logs.records
        dropped = re.fullmatch(r"dropped (\d+) live samples repeating the pre-connect buffer", record.getMessage())
        self.assertIsNotNone(dropped, record.getMessage())
        self.assertEqual(int(dropped[1]), skipped)

    def assertMiss(self, logs, *, held, buffer):
        """Exactly one INFO record on mentat.voice, saying the head was held and matched to nothing."""
        [record] = logs.records
        self.assertEqual(record.levelname, "INFO")
        miss = MISS.fullmatch(record.getMessage())
        self.assertIsNotNone(miss, record.getMessage())
        self.assertAlmostEqual(float(miss["held"]), held, delta=0.01)
        self.assertAlmostEqual(float(miss["buffer"]), buffer, delta=0.01)
        self.assertLess(float(miss["score"]), preconnect.MATCH_SCORE)
        self.assertLess(float(miss["onset"]), float(miss["held"]))
        self.assertLessEqual(float(miss["position"]), float(miss["buffer"]))

    async def test_buffer_with_a_silent_tail_still_dedupes_the_earlier_speech(self):
        speech = syllables(5, 0.0, 0.62) + syllables(6, 1.1, 3.0)  # nothing is said from 0.62 to 1.1
        case = Overlap(speech, overlap=0.85)

        skipped = kept_after_skipping(case, await dedupe(case))

        self.assertAlmostEqual(skipped, case.repeated, delta=TOLERANCE)

    async def test_a_word_said_again_after_the_copy_is_kept(self):
        word = syllables(9, 0.3, 0.9)
        said_again = [(start + 0.9, length, pitch, amplitude) for start, length, pitch, amplitude in word]
        case = Overlap(word + said_again, overlap=0.65)
        live_begin = BUFFER_END - 0.65
        again = record(word, 0.3, 0.9, LIVE_RATE, seed=3, noise=150, gain=0.96, pitch_scale=1.05, phase=2.0)
        case.live = frames_of(
            record(word, live_begin, BUFFER_END, LIVE_RATE, seed=2, noise=150, gain=0.96)
            + record([], 0.0, 0.2, LIVE_RATE, seed=4)
            + again,
            LIVE_RATE,
        )

        skipped = kept_after_skipping(case, await dedupe(case))

        self.assertAlmostEqual(skipped, case.repeated, delta=TOLERANCE)

    async def test_similar_speech_that_is_not_a_copy_is_kept(self):
        word = syllables(9, 0.3, 0.9)
        case = Overlap(word, overlap=-0.1)
        case.live = frames_of(
            record(word, 0.3, 0.9, LIVE_RATE, seed=3, noise=150, gain=0.96, pitch_scale=1.05, phase=2.0)
            + record([], 0.0, 1.0, LIVE_RATE, seed=4),
            LIVE_RATE,
        )

        self.assertEqual(kept_after_skipping(case, await dedupe(case)), 0)

    async def test_the_opus_copy_of_the_eval_line_is_skipped_from_its_start(self):
        case = eval_case("copy")

        skipped = kept_after_skipping(case, await dedupe(case))

        self.assertAlmostEqual(skipped, round(0.7 * LIVE_RATE), delta=EVAL_TOLERANCE)

    async def test_the_opus_copy_after_a_quiet_lead_keeps_the_lead_and_skips_the_copy(self):
        case = eval_case("copy_after_silence")

        begin, end = removed_span(case, await dedupe(case))

        self.assertGreaterEqual(begin, 0.28 * LIVE_RATE)
        self.assertLessEqual(begin, 0.30 * LIVE_RATE)
        self.assertAlmostEqual(end, LIVE_RATE, delta=EVAL_TOLERANCE)

    async def test_another_rendering_of_the_eval_line_is_kept_whole(self):
        case = eval_case("other_rendering")

        with self.assertLogs("mentat.voice", level="INFO") as logs:
            self.assertEqual(kept_after_skipping(case, await dedupe(case)), 0)

        self.assertMiss(logs, held=1.0, buffer=1.0)

    async def test_another_rendering_after_a_quiet_lead_is_kept_whole(self):
        case = eval_case("other_rendering")
        silence = array("h", bytes(round(0.3 * LIVE_RATE) * 2))
        case.live = frames_of(silence + eval_line("other_rendering"), LIVE_RATE)

        self.assertEqual(kept_after_skipping(case, await dedupe(case)), 0)

    async def test_matching_runs_off_the_event_loop(self):
        case = Overlap(syllables(7, 0.0, 3.0), overlap=0.7)
        threads = []
        real = preconnect._repeated_samples

        def noting(*arguments):
            threads.append(threading.current_thread())
            return real(*arguments)

        with mock.patch.object(preconnect, "_repeated_samples", noting):
            await dedupe(case)

        self.assertTrue(threads)
        self.assertNotIn(threading.current_thread(), threads)

    async def test_the_canceller_sees_every_buffer_frame_then_every_live_frame(self):
        case = Overlap(syllables(7, 0.0, 3.0), overlap=0.7, buffer_rate=LIVE_RATE)
        canceller = recording_canceller()

        _, built = await run_input(case.live, case.buffer, canceller)

        # rtc must not be handed the canceller: it would run it on live frames while the buffer waits.
        self.assertIsNone(built["processor"])
        buffer_ids = {id(frame) for frame in case.buffer}
        shown = ["buffer" if id(frame) in buffer_ids else "live" for frame in canceller.seen]
        live_shown = shown.count("live")
        self.assertEqual(shown, ["buffer"] * len(case.buffer) + ["live"] * live_shown)
        self.assertGreater(live_shown, 0)
        self.assertLess(live_shown, len(case.live), "the repeated head should never reach the canceller")

    async def test_without_a_preconnect_buffer_live_audio_flows_unchanged_and_unheld(self):
        case = Overlap(syllables(7, 0.0, 3.0), overlap=0.7)
        canceller = recording_canceller()
        live = preconnect._LiveAudio(
            ScriptedStream(case.live, ends=False), sample_rate=LIVE_RATE, cancel=canceller._process
        )

        first = await asyncio.wait_for(anext(live), timeout=1)

        self.assertIs(first.frame, case.live[0])
        self.assertEqual(len(canceller.seen), 1)

    async def test_a_timed_out_buffer_leaves_live_audio_untouched(self):
        case = Overlap(syllables(7, 0.0, 3.0), overlap=0.7)
        canceller = recording_canceller()

        forwarded, _ = await run_input(case.live, case.buffer, canceller, buffered=False)

        self.assertEqual(samples_of(forwarded[:-1]), samples_of(case.live))


class PreConnectRoomIOTest(AsyncTest):
    async def test_start_installs_the_input_and_restores_livekits(self):
        upstream = preconnect.upstream
        stock = upstream._ParticipantAudioInputStream
        canceller = recording_canceller()
        if STAND_INS:
            options = SimpleNamespace(audio_input=SimpleNamespace(sample_rate=LIVE_RATE, noise_cancellation=canceller))
        else:
            options = preconnect.room_io.RoomOptions(
                audio_input=preconnect.room_io.AudioInputOptions(noise_cancellation=canceller),
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


if __name__ == "__main__":
    unittest.main()
