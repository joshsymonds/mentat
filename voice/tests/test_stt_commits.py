"""Worker-side Scribe commit cadence, driven through fake streams and a fake clock."""

import ast
import asyncio
import logging
import unittest
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

AGENT_PATH = Path(__file__).resolve().parents[1] / "agent.py"
SOURCE = AGENT_PATH.read_text()
TREE = ast.parse(SOURCE)
INTERVAL = 5.0
AUDIO_TICK = 0.25


def _top_level(name):
    return next(
        node for node in TREE.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )


def _front_agent_method(name):
    front_agent = _top_level("FrontAgent")
    return next(
        node for node in front_agent.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _exec(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(AGENT_PATH), "exec"), namespace)
    return namespace


class FakeTime:
    """A fake clock that also stands in for asyncio.wait_for, so the controller's
    real background loop sleeps on fake time and fires when the clock reaches it.
    As time passes it reports the seconds to `hear`, the way Josh's audio streams in."""

    def __init__(self):
        self.now = 0.0
        self.hear = None
        self.waits = 0
        self._timer = None
        self._deadline = None

    def clock(self):
        return self.now

    async def wait_for(self, awaitable, timeout):
        self.waits += 1
        wake = asyncio.ensure_future(awaitable)
        if timeout is None:
            await wake
            return
        self._timer = asyncio.get_running_loop().create_future()
        self._deadline = self.now + timeout
        try:
            await asyncio.wait({wake, self._timer}, return_when=asyncio.FIRST_COMPLETED)
            if not wake.done():
                raise TimeoutError
        finally:
            wake.cancel()
            self._timer.cancel()
            self._timer = None

    async def settle(self):
        for _ in range(20):
            await asyncio.sleep(0)

    async def advance(self, seconds):
        end = self.now + seconds
        while True:
            # Audio streams in tick by tick, so the worker can act on it as it arrives.
            tick_end = min(end, self.now + AUDIO_TICK)
            if self._timer is not None and not self._timer.done() and self._deadline <= tick_end:
                self._move_to(max(self.now, self._deadline))
                self._timer.set_result(None)
            elif self.now < end:
                self._move_to(tick_end)
            else:
                await self.settle()
                return
            await self.settle()

    def _move_to(self, when):
        # Audio reaches the stream before the timer at `when` fires, so a commit due then sees it.
        if self.hear is not None and when > self.now:
            self.hear(when - self.now)
        self.now = when


class FakeStream:
    """Stands in for the ElevenLabs SpeechStream: flush raises once it is closed."""

    def __init__(self, time):
        self._time = time
        self.flushes = []
        self.closed = False

    def flush(self):
        if self.closed:
            raise RuntimeError("SpeechStream is closed")
        self.flushes.append(self._time.now)


class FakeScribe:
    """Stands in for elevenlabs.STT; update_options reconnects the same stream objects."""

    def __init__(self, time=None, **options):
        self.options = options
        self.opened = []
        self.reconnects = 0
        self._time = time

    def stream(self, **kwargs):
        stream = FakeStream(self._time)
        self.opened.append(stream)
        return stream

    def update_options(self, **options):
        self.reconnects += 1


class ScribeCommitTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.time = FakeTime()
        self.time.hear = self._josh_audio
        fake_asyncio = SimpleNamespace(**vars(asyncio))
        fake_asyncio.wait_for = self.time.wait_for
        self.logger = Mock(spec=logging.Logger)
        self.namespace = _exec(
            [_top_level("ScribeCommitController")],
            {"asyncio": fake_asyncio, "time": SimpleNamespace(monotonic=self.time.clock),
             "Any": Any, "Callable": Callable, "logger": self.logger},
        )
        self.controller = self.namespace["ScribeCommitController"](clock=self.time.clock)
        self.controller.start()
        self.stream = FakeStream(self.time)
        self.controller.set_stream(self.stream)
        await self.time.settle()

    async def asyncTearDown(self):
        await self.controller.aclose()

    def _josh_audio(self, seconds):
        self.controller.audio_sent(seconds)

    @property
    def flushes(self):
        return self.stream.flushes

    async def speech_starts_and_vad_ends(self):
        self.controller.user_state("speaking")
        await self.time.advance(1)
        self.controller.user_state("listening")
        await self.time.settle()


class CommitCadenceTest(ScribeCommitTestCase):
    async def test_quiet_commits_every_five_seconds(self):
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [5.0])
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [5.0])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [5.0, 10.0])
        await self.time.advance(15)
        self.assertEqual(self.flushes, [5.0, 10.0, 15.0, 20.0, 25.0])

    async def test_commits_continue_every_five_seconds_while_the_agent_speaks(self):
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [5.0])
        await self.time.advance(15)
        self.assertEqual(self.flushes, [5.0, 10.0, 15.0, 20.0])

    async def test_commits_at_agent_speech_end_once_five_seconds_have_passed(self):
        await self.time.advance(5)
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(3)
        self.speech_ended_by_josh_final()
        await self.time.advance(3)
        self.assertEqual(self.flushes, [5.0])
        self.controller.agent_state("speaking", "listening")
        await self.time.settle()
        self.assertEqual(self.flushes, [5.0, 11.0])

    async def test_agent_speech_end_after_a_long_reply_waits_for_the_next_spacing(self):
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(22)
        self.assertEqual(self.flushes, [5.0, 10.0, 15.0, 20.0])
        self.controller.agent_state("speaking", "listening")
        await self.time.advance(2.9)
        self.assertEqual(self.flushes, [5.0, 10.0, 15.0, 20.0])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [5.0, 10.0, 15.0, 20.0, 25.0])

    async def test_a_barge_in_holds_commits_through_a_five_second_boundary(self):
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(3)
        self.controller.user_state("speaking")
        await self.time.advance(5)
        self.assertEqual(self.flushes, [])
        self.controller.agent_state("speaking", "listening")
        await self.time.advance(30)
        self.assertEqual(self.flushes, [])

    async def test_a_barge_in_that_ends_locally_waits_for_its_final_transcript(self):
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(3)
        self.controller.user_state("speaking")
        await self.time.advance(4)
        self.controller.user_state("listening")
        await self.time.advance(30)
        self.assertEqual(self.flushes, [])
        self.controller.close_line()
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [42.0])

    async def test_commits_resume_while_the_agent_speaks_after_the_barge_in_final(self):
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(3)
        self.controller.user_state("speaking")
        await self.time.advance(4)
        self.controller.user_state("listening")
        self.controller.close_line()
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [12.0])

    def speech_ended_by_josh_final(self):
        self.controller.user_state("speaking")
        self.controller.user_state("listening")
        self.controller.close_line()

    async def test_agent_speech_end_does_not_commit_within_five_seconds_of_the_last_commit(self):
        await self.time.advance(5)
        self.assertEqual(self.flushes, [5.0])
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(1)
        self.controller.agent_state("speaking", "listening")
        await self.time.advance(3.9)
        self.assertEqual(self.flushes, [5.0])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [5.0, 10.0])

    async def test_workers_commits_are_never_closer_than_five_seconds(self):
        for _ in range(4):
            await self.time.advance(2)
            self.controller.agent_state("listening", "speaking")
            await self.time.advance(0.5)
            self.controller.agent_state("speaking", "listening")
            await self.time.advance(0.5)
        await self.time.advance(20)
        gaps = [later - earlier for earlier, later in zip(self.flushes, self.flushes[1:])]
        self.assertGreaterEqual(len(self.flushes), 3)
        self.assertTrue(all(gap >= INTERVAL for gap in gaps), self.flushes)

    async def test_no_commit_while_josh_speaks(self):
        await self.time.advance(3)
        self.controller.user_state("speaking")
        await self.time.advance(60)
        self.assertEqual(self.flushes, [])

    async def test_no_commit_after_speech_ends_locally_with_no_partial_yet(self):
        await self.speech_starts_and_vad_ends()
        await self.time.advance(60)
        self.assertEqual(self.flushes, [])
        self.controller.agent_state("listening", "speaking")
        self.controller.agent_state("speaking", "listening")
        await self.time.advance(60)
        self.assertEqual(self.flushes, [])

    async def test_final_transcript_resumes_cadence_five_seconds_later(self):
        await self.speech_starts_and_vad_ends()
        await self.time.advance(20)
        self.controller.close_line()
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [26.0])

    async def test_final_transcript_while_vad_hears_josh_does_not_start_the_quiet_clock(self):
        self.controller.user_state("speaking")
        await self.time.advance(1)
        self.controller.close_line()
        await self.time.advance(30)
        self.assertEqual(self.flushes, [])
        self.controller.user_state("listening")
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [36.0])

    async def test_speech_resuming_after_a_final_heard_mid_speech_is_pending_again(self):
        self.controller.user_state("speaking")
        self.controller.close_line()
        self.controller.user_state("listening")
        await self.time.advance(2)
        self.controller.user_state("speaking")
        await self.time.advance(1)
        self.controller.user_state("listening")
        await self.time.advance(60)
        self.assertEqual(self.flushes, [])

    async def test_turn_completion_resumes_cadence_five_seconds_later(self):
        await self.speech_starts_and_vad_ends()
        await self.time.advance(20)
        agent = SimpleNamespace(_scribe_commits=self.controller, _closed=True)
        namespace = _exec([_front_agent_method("on_user_turn_completed")], {"logger": self.logger})
        await namespace["on_user_turn_completed"](agent, None, SimpleNamespace(text_content=""))
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [26.0])

    async def test_recovery_resumes_cadence_five_seconds_after_the_reconnect(self):
        scribe = FakeScribe()
        namespace = _exec(
            [_front_agent_method("recover_stalled_stt"), _top_level("stt_secondary_languages")],
            {"logger": self.logger, "STT_STALL_REPLY": "Say it again?"},
        )
        agent = SimpleNamespace(
            _scribe_commits=self.controller, _closed=False, _stt_provider=scribe,
            _tts_provider=Mock(), _voice_language="en", _voice_mode="normal",
            _default_voice="voice", session=SimpleNamespace(say=Mock()),
        )
        await self.speech_starts_and_vad_ends()
        await self.time.advance(20)
        self.assertEqual(self.flushes, [])
        namespace["recover_stalled_stt"](agent, 4.0)
        self.assertEqual(scribe.reconnects, 1)
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [26.0])

    async def test_recovery_reconnect_then_agent_speech_end_waits_five_seconds(self):
        scribe = FakeScribe()
        namespace = _exec(
            [_front_agent_method("recover_stalled_stt"), _top_level("stt_secondary_languages")],
            {"logger": self.logger, "STT_STALL_REPLY": "Say it again?"},
        )
        agent = SimpleNamespace(
            _scribe_commits=self.controller, _closed=False, _stt_provider=scribe,
            _tts_provider=Mock(), _voice_language="en", _voice_mode="normal",
            _default_voice="voice", session=SimpleNamespace(say=Mock()),
        )
        await self.speech_starts_and_vad_ends()
        await self.time.advance(20)
        namespace["recover_stalled_stt"](agent, 4.0)
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(0.3)
        self.controller.agent_state("speaking", "listening")
        await self.time.advance(4.6)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [26.0])

    async def test_close_stops_the_background_loop(self):
        await self.controller.aclose()
        await self.time.advance(60)
        self.assertEqual(self.flushes, [])
        self.assertIsNone(self.controller._task)


class SessionStartGapTest(ScribeCommitTestCase):
    """A Scribe session start counts as the worker's last commit."""

    async def hold_agent_speech(self, seconds):
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(seconds)

    async def test_new_stream_then_agent_speech_end_waits_five_seconds(self):
        await self.hold_agent_speech(20)
        fresh = FakeStream(self.time)
        self.controller.set_stream(fresh)
        self.controller.agent_state("speaking", "listening")
        await self.time.advance(4.9)
        self.assertEqual(fresh.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(fresh.flushes, [25.0])

    async def test_reconnect_then_agent_speech_end_waits_five_seconds(self):
        await self.hold_agent_speech(20)
        self.controller.session_started()
        self.controller.agent_state("speaking", "listening")
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [5.0, 10.0, 15.0, 20.0])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [5.0, 10.0, 15.0, 20.0, 25.0])

    async def test_reconnect_pushes_the_quiet_cadence_five_seconds_out(self):
        await self.time.advance(3)
        self.controller.session_started()
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [8.0])

    async def test_session_start_after_a_commit_still_waits_five_seconds(self):
        await self.time.advance(5)
        self.assertEqual(self.flushes, [5.0])
        await self.time.advance(2)
        self.controller.session_started()
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [5.0])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [5.0, 12.0])


class CommitFloorTest(ScribeCommitTestCase):
    """Scribe rejects a commit under 0.3 s of uncommitted audio, so a commit waits for that much."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        # Audio reaches the stream only where a test reports it.
        self.time.hear = None

    async def test_no_audio_means_no_commit_at_five_or_ten_seconds(self):
        await self.time.advance(5)
        self.assertEqual(self.flushes, [])
        await self.time.advance(5)
        self.assertEqual(self.flushes, [])

    async def test_point_two_nine_seconds_is_under_the_floor(self):
        for seconds in (0.1, 0.1, 0.09):
            self.controller.audio_sent(seconds)
        await self.time.advance(10)
        self.assertEqual(self.flushes, [])

    async def test_three_tenths_of_a_second_commits_on_the_cadence(self):
        self.controller.audio_sent(0.3)
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [5.0])

    async def test_a_commit_due_early_waits_for_the_floor_then_fires_on_arrival(self):
        self.controller.audio_sent(0.1)
        await self.time.advance(7)
        self.assertEqual(self.flushes, [])
        self.controller.audio_sent(0.2)
        await self.time.settle()
        self.assertEqual(self.flushes, [7.0])
        self.controller.audio_sent(1.0)
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [7.0])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [7.0, 12.0])

    async def test_waiting_for_the_floor_does_not_spin(self):
        self.controller.audio_sent(0.1)
        await self.time.advance(1)
        waits = self.time.waits
        await self.time.advance(60)
        self.assertEqual(self.time.waits, waits)
        self.assertEqual(self.flushes, [])

    async def test_each_commit_restarts_the_count(self):
        self.controller.audio_sent(5.0)
        await self.time.advance(5)
        self.assertEqual(self.flushes, [5.0])
        await self.time.advance(5)
        self.assertEqual(self.flushes, [5.0])
        self.controller.audio_sent(0.3)
        await self.time.settle()
        self.assertEqual(self.flushes, [5.0, 10.0])

    async def test_a_session_start_restarts_the_count(self):
        self.controller.audio_sent(0.5)
        await self.time.advance(3)
        self.controller.session_started()
        await self.time.advance(10)
        self.assertEqual(self.flushes, [])
        self.controller.audio_sent(0.3)
        await self.time.settle()
        self.assertEqual(self.flushes, [13.0])


class PassThroughSTT:
    """Stands in for the livekit default stt_node, which hands the frames on to STT."""

    def stt_node(self, audio, model_settings):
        return audio


async def _frames(frames):
    for frame in frames:
        yield frame


def _frame(samples):
    return SimpleNamespace(
        samples_per_channel=samples, sample_rate=16000, num_channels=1, data=bytes(2 * samples),
    )


class SttNodeAudioTest(unittest.IsolatedAsyncioTestCase):
    """FrontAgent.stt_node reports each frame's seconds to the commit controller."""

    def setUp(self):
        class_node = ast.ClassDef(
            name="FrontAgent",
            bases=[ast.Name(id="PassThroughSTT", ctx=ast.Load())],
            keywords=[],
            body=[_front_agent_method("stt_node")],
            decorator_list=[],
            type_params=[],
        )
        namespace = _exec(
            [ast.fix_missing_locations(class_node)],
            {"PassThroughSTT": PassThroughSTT, "Any": Any,
             "AsyncGenerator": AsyncGenerator, "AsyncIterable": AsyncIterable},
        )
        self.front_agent = namespace["FrontAgent"]
        self.frames = [_frame(800), _frame(1600), _frame(400)]
        self.seconds = [0.05, 0.1, 0.025]

    def agent(self, recorder):
        agent = self.front_agent.__new__(self.front_agent)
        agent._input_audio = recorder
        agent._scribe_commits = Mock()
        return agent

    async def handed_on(self, agent):
        return [frame async for frame in agent.stt_node(_frames(self.frames), None)]

    def reported(self, agent):
        return [args.args[0] for args in agent._scribe_commits.audio_sent.call_args_list]

    async def test_frames_are_reported_to_the_controller_without_a_recorder(self):
        agent = self.agent(recorder=None)
        self.assertEqual(await self.handed_on(agent), self.frames)
        self.assertEqual(self.reported(agent), self.seconds)

    async def test_frames_are_recorded_and_reported_with_a_recorder(self):
        recorder = Mock()
        agent = self.agent(recorder=recorder)
        self.assertEqual(await self.handed_on(agent), self.frames)
        recorder.begin.assert_called_once_with()
        self.assertEqual(recorder.capture.call_args_list, [call(frame) for frame in self.frames])
        self.assertEqual(self.reported(agent), self.seconds)


class StreamHandoffTest(ScribeCommitTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        fake_elevenlabs = SimpleNamespace(STT=FakeScribe)
        namespace = _exec(
            [_top_level("CommitSTT")],
            {"elevenlabs": fake_elevenlabs, "ScribeCommitController": object, "Any": Any},
        )
        self.scribe = namespace["CommitSTT"](commits=self.controller, time=self.time, model="m")

    async def test_the_stream_the_stt_opens_is_the_stream_the_worker_commits(self):
        opened = self.scribe.stream(conn_options="conn")
        self.assertIs(self.scribe.opened[-1], opened)
        await self.time.advance(5)
        self.assertEqual(opened.flushes, [5.0])
        self.assertEqual(self.stream.flushes, [])

    async def test_commits_survive_update_options_reconnects(self):
        opened = self.scribe.stream()
        await self.time.advance(5)
        self.scribe.update_options(secondary_languages=["es"])
        await self.time.advance(5)
        self.scribe.update_options(secondary_languages=[])
        await self.time.advance(5)
        self.assertEqual(self.scribe.reconnects, 2)
        self.assertEqual(opened.flushes, [5.0, 10.0, 15.0])

    async def test_a_stream_the_stt_opens_counts_as_a_session_start(self):
        self.controller.agent_state("listening", "speaking")
        await self.time.advance(20)
        opened = self.scribe.stream()
        self.controller.agent_state("speaking", "listening")
        await self.time.advance(4.9)
        self.assertEqual(opened.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(opened.flushes, [25.0])

    async def test_a_closed_stream_is_dropped_without_killing_the_cadence(self):
        first = self.scribe.stream()
        await self.time.advance(5)
        first.closed = True
        await self.time.advance(20)
        self.assertEqual(first.flushes, [5.0])
        self.logger.warning.assert_called_once()
        second = self.scribe.stream()
        await self.time.advance(5)
        self.assertEqual(second.flushes, [30.0])
        await self.time.advance(5)
        self.assertEqual(second.flushes, [30.0, 35.0])

    async def test_build_stt_without_a_cadence_still_opens_streams(self):
        namespace = _exec(
            [_top_level("CommitSTT"), _top_level("build_stt")],
            {"elevenlabs": SimpleNamespace(STT=FakeScribe), "ScribeCommitController": object,
             "Any": Any, "STT_MODEL": "scribe", "STT_SERVER_VAD": {}},
        )
        scribe = namespace["build_stt"]("key", ("Mentat",))
        opened = scribe.stream()
        self.assertIs(scribe.opened[-1], opened)
        self.assertEqual(scribe.options["keyterms"], ["Mentat"])

    async def test_close_drops_the_stream(self):
        opened = self.scribe.stream()
        await self.controller.aclose()
        await self.time.advance(60)
        self.assertEqual(opened.flushes, [])


def _entrypoint_handler(event_name):
    entry = next(
        node for node in TREE.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "entrypoint"
    )
    for node in ast.walk(entry):
        if isinstance(node, ast.FunctionDef) and any(
            ast.unparse(decorator) == f"session.on('{event_name}')"
            for decorator in node.decorator_list
        ):
            return ast.FunctionDef(
                name=node.name, args=node.args, body=node.body, decorator_list=[],
                returns=node.returns, type_comment=None, type_params=[],
                lineno=node.lineno, col_offset=node.col_offset,
            )
    raise AssertionError(f"entrypoint has no {event_name} handler")


class SessionEventsTest(ScribeCommitTestCase):
    """The entrypoint's own session handlers drive the real controller."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        namespace = _exec(
            [
                _entrypoint_handler("agent_state_changed"),
                _entrypoint_handler("user_state_changed"),
                _entrypoint_handler("user_input_transcribed"),
            ],
            {
                "Any": Any, "scribe_commits": self.controller, "ending_policy": Mock(),
                "time": SimpleNamespace(monotonic=self.time.clock), "logger": self.logger,
                "_rearm_timer": Mock(),
            },
        )
        self.agent_state = namespace["_on_agent_state"]
        self.user_state = namespace["_on_user_state"]
        self.transcribed = namespace["_on_user_input_transcribed"]

    async def test_agent_speech_end_event_commits(self):
        await self.time.advance(5)
        self.agent_state(SimpleNamespace(old_state="listening", new_state="speaking"))
        await self.time.advance(3)
        self.user_state(SimpleNamespace(old_state="listening", new_state="speaking"))
        self.user_state(SimpleNamespace(old_state="speaking", new_state="listening"))
        self.transcribed(SimpleNamespace(is_final=True, transcript="wait"))
        await self.time.advance(3)
        self.assertEqual(self.flushes, [5.0])
        self.agent_state(SimpleNamespace(old_state="speaking", new_state="listening"))
        await self.time.settle()
        self.assertEqual(self.flushes, [5.0, 11.0])

    async def test_speech_start_event_holds_commits_until_a_final_transcript(self):
        self.user_state(SimpleNamespace(old_state="listening", new_state="speaking"))
        await self.time.advance(1)
        self.user_state(SimpleNamespace(old_state="speaking", new_state="listening"))
        self.transcribed(SimpleNamespace(is_final=False, transcript="turn the"))
        await self.time.advance(30)
        self.assertEqual(self.flushes, [])
        self.transcribed(SimpleNamespace(is_final=True, transcript="turn the lights off"))
        await self.time.advance(4.9)
        self.assertEqual(self.flushes, [])
        await self.time.advance(0.1)
        self.assertEqual(self.flushes, [36.0])


class SessionWiringTest(unittest.TestCase):
    """Building the controller inside entrypoint cannot run offline, so its lines are pinned."""

    def test_one_controller_is_shared_by_stt_agent_and_cleanup(self):
        entrypoint = ast.get_source_segment(SOURCE, next(
            node for node in TREE.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "entrypoint"
        ))
        self.assertIn("scribe_commits = ScribeCommitController()", entrypoint)
        self.assertIn("scribe_commits.start()", entrypoint)
        self.assertIn("scribe_commits=scribe_commits", entrypoint)
        self.assertIn('ctx.proc.userdata["private"].keyterms, scribe_commits', entrypoint)
        self.assertIn("await self._scribe_commits.aclose()", ast.unparse(_front_agent_method("aclose")))


if __name__ == "__main__":
    unittest.main()
