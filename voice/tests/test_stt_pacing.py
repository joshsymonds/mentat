"""Backlog pacing of the audio FrontAgent.stt_node hands to Scribe, driven through a fake clock."""

import ast
import asyncio
import logging
import time
import unittest
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

AGENT_PATH = Path(__file__).resolve().parents[1] / "agent.py"
TREE = ast.parse(AGENT_PATH.read_text())
SAMPLE_RATE = 16000
FRAME_SAMPLES = 800
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
FREE_SECONDS = 5.0
HELD_GAP = FRAME_SECONDS / 4
EPSILON = 1e-9


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


class PassThroughSTT:
    """Stands in for the livekit default stt_node, which hands the frames on to STT."""

    def stt_node(self, audio, model_settings):
        return audio


class FakeClock:
    """Wall time that only moves when the pacer sleeps or the audio source reaches a new frame."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _frame(samples=FRAME_SAMPLES):
    return SimpleNamespace(
        samples_per_channel=samples, sample_rate=SAMPLE_RATE, num_channels=1,
        data=bytes(2 * samples),
    )


def _burst(seconds):
    return [(0.0, FRAME_SAMPLES)] * round(seconds / FRAME_SECONDS)


async def _arriving(clock, arrivals):
    """Hand on frames as the audio source has them: a frame waits until the clock reaches it."""
    for when, samples in arrivals:
        clock.now = max(clock.now, when)
        yield _frame(samples)


class PacerTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = FakeClock()
        self.logger = Mock(spec=logging.Logger)
        namespace = _exec(
            [_top_level("AudioPacer")],
            {
                "asyncio": asyncio, "time": time, "logger": self.logger,
                "Any": Any, "AsyncGenerator": AsyncGenerator, "AsyncIterable": AsyncIterable,
                "Awaitable": Awaitable, "Callable": Callable,
            },
        )
        self.pacer_class = namespace["AudioPacer"]

    async def released(self, arrivals):
        """Times at which each frame is handed on, read at the moment it is received."""
        pacer = self.pacer_class(clock=self.clock.clock, sleep=self.clock.sleep)
        times = []
        async for _ in pacer.paced(_arriving(self.clock, arrivals)):
            times.append(self.clock.now)
        return times


class BacklogPacingTest(PacerTestCase):
    async def test_a_burst_under_five_seconds_is_never_delayed(self):
        times = await self.released(_burst(4.9))
        self.assertEqual(times, [0.0] * 98)
        self.assertEqual(self.clock.slept, [])
        self.logger.info.assert_not_called()

    async def test_a_fifteen_second_burst_passes_five_seconds_at_once_then_at_most_four_times(self):
        times = await self.released(_burst(15))
        free, held = times[:100], times[100:]
        self.assertEqual(free, [0.0] * 100)
        self.assertEqual(len(held), 200)
        self.assertAlmostEqual(held[0], HELD_GAP, places=9)
        gaps = [later - earlier for earlier, later in zip(times[99:], times[100:])]
        for gap in gaps:
            self.assertGreaterEqual(gap, HELD_GAP - EPSILON)
        self.assertAlmostEqual(times[-1], 2.5, delta=FRAME_SECONDS)

    async def test_real_time_frames_after_a_burst_are_never_delayed(self):
        live = [(2.5 + FRAME_SECONDS * (index + 1), FRAME_SAMPLES) for index in range(50)]
        times = await self.released(_burst(15) + live)
        self.assertEqual(len(times), 350)
        for released_at, (arrived_at, _) in zip(times[300:], live):
            self.assertAlmostEqual(released_at, arrived_at, places=9)

    async def test_one_info_line_per_engagement_with_the_backlog_seconds(self):
        after_a_pause = [(40.0, FRAME_SAMPLES)] * 300
        await self.released(_burst(15) + after_a_pause)
        self.assertEqual(self.logger.info.call_count, 2)
        for engagement in self.logger.info.call_args_list:
            self.assertAlmostEqual(engagement.args[1], FREE_SECONDS + FRAME_SECONDS, places=9)

    async def test_the_backlog_drains_to_zero_so_a_pause_restores_free_audio(self):
        times = await self.released(_burst(15) + [(40.0, FRAME_SAMPLES)] * 100)
        self.assertEqual(times[300:], [40.0] * 100)
        self.assertEqual(self.logger.info.call_count, 1)


class SttNodePacingTest(PacerTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        pacer = partial(self.pacer_class, clock=self.clock.clock, sleep=self.clock.sleep)
        front_agent = _exec(
            [ast.fix_missing_locations(ast.ClassDef(
                name="FrontAgent",
                bases=[ast.Name(id="PassThroughSTT", ctx=ast.Load())],
                keywords=[],
                body=[_front_agent_method("stt_node")],
                decorator_list=[],
                type_params=[],
            ))],
            {"PassThroughSTT": PassThroughSTT, "Any": Any, "AsyncGenerator": AsyncGenerator,
             "AsyncIterable": AsyncIterable, "AudioPacer": pacer},
        )["FrontAgent"]
        self.agent = front_agent.__new__(front_agent)
        self.captured = []
        self.reported = []
        self.recorder = Mock()
        self.recorder.capture.side_effect = lambda frame: self.captured.append(self.clock.now)
        self.commits = Mock()
        self.commits.audio_sent.side_effect = lambda seconds: self.reported.append(
            (self.clock.now, seconds)
        )
        self.agent._input_audio = self.recorder
        self.agent._scribe_commits = self.commits

    async def test_stt_node_hands_on_every_frame_in_order_paced_then_recorded_and_reported(self):
        frames = [_frame(FRAME_SAMPLES) for _ in range(300)]

        async def source():
            for frame in frames:
                yield frame

        handed = [frame async for frame in self.agent.stt_node(source(), None)]
        self.assertEqual(len(handed), 300)
        self.assertTrue(all(out is frame for out, frame in zip(handed, frames)))
        self.assertEqual(self.captured[:100], [0.0] * 100)
        self.assertAlmostEqual(self.captured[100], HELD_GAP, places=9)
        self.assertEqual(len(self.reported), 300)
        self.assertEqual(
            [when for when, _ in self.reported], self.captured,
        )
        for _, seconds in self.reported:
            self.assertAlmostEqual(seconds, FRAME_SECONDS, places=9)


if __name__ == "__main__":
    unittest.main()
