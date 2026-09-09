"""Measure the front's one real judgment: answer it myself, or ask Mentat.

Every scenario in scenarios.jsonl is put to the same model the room hears,
wearing the same persona, holding the same ask_mentat tool, and the only thing
recorded is which way it went. That makes an edit to persona.md checkable:
change the wording, run this, compare the table.

The fidelity is the point, so nothing here is a copy of production. The
instructions come out of persona.md through request.split_persona, the tool
schema is lifted off agent.FrontAgent.ask_mentat itself — the real docstring,
the real arguments — and the arguments come back through the same parser the
framework uses to execute a call. A persona or a tool description edited in
agent.py changes this eval without anyone remembering to update it.

Online, and therefore not part of any gate: run it as `just eval-voice` with
LIVEKIT_INFERENCE_API_KEY and LIVEKIT_INFERENCE_API_SECRET in the environment,
under the flake's voice-env interpreter (it needs livekit-agents). One turn per
scenario against a non-reasoning model costs a rounding error — the whole file
is well under a cent — so run it freely.

    python run.py [--min-accuracy 0.9] [--concurrency 4]

Exits non-zero below --min-accuracy, which defaults to 0: the bar is the user's
to set, and a recipe that fails a build on a prompt's mood would be worse than
no recipe.
"""

from __future__ import annotations

import os
import argparse
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# agent.py and request.py sit one level up and are imported flat, the way
# voice/tests does it: the voice tree is scripts next to each other, not a
# package, because agent.py is run by path from a systemd unit.
sys.path.insert(0, str(HERE.parent))

from livekit.agents import APIError, llm
from livekit.agents.inference import LLM
from livekit.agents.llm.utils import parse_function_arguments
from livekit.agents.voice.generation import update_instructions

import agent
from request import PRIVATE_CONTEXT_ENV, load_private_context, with_private_context
from scoring import (
    Response,
    Result,
    Scenario,
    ToolCall,
    exit_code,
    judge,
    load_scenarios,
    report,
)

#: Enough to keep the gateway busy without turning a 30-line file into a burst
#: that trips rate limiting.
CONCURRENCY = 4

#: One retry, and only for errors the SDK itself calls retryable. Its own
#: conn_options already retry inside a single call; this covers the turn that
#: exhausts them, so a flaky minute does not read as a persona regression.
RETRY_DELAY_S = 2.0


def build_context(instructions: str, scenario: Scenario) -> llm.ChatContext:
    """The chat context one scenario puts to the model.

    Instructions go in through the framework's own helper rather than as a
    hand-rolled system message: it is what AgentSession does with an Agent's
    instructions, down to the position in the context.
    """
    chat_ctx = llm.ChatContext.empty()
    for role, text in scenario.history:
        chat_ctx.add_message(role=role, content=text)
    chat_ctx.add_message(role="user", content=scenario.utterance)
    update_instructions(chat_ctx, instructions=instructions, add_if_missing=True)
    return chat_ctx


def to_response(collected: llm.CollectedResponse) -> Response:
    """One collected turn, in the shape the scoring understands.

    Arguments that cannot be recovered even by the framework's repairing parser
    make the whole turn an error rather than a consult with empty arguments:
    the raw string is what a reader needs to see, and an argumentless consult
    would quietly satisfy any scenario that pins nothing.
    """
    calls: list[ToolCall] = []
    for call in collected.tool_calls:
        try:
            arguments = parse_function_arguments(call.arguments)
        except ValueError as err:
            return Response(text=collected.text, error=f"bad tool arguments: {err}")
        calls.append(ToolCall(name=call.name, arguments=arguments))
    return Response(text=collected.text, tool_calls=tuple(calls))


async def run_scenario(
    model: LLM,
    tools: list[llm.Tool | llm.Toolset],
    instructions: str,
    scenario: Scenario,
) -> Result:
    """One scenario, judged. A failed turn is recorded, never raised.

    A single unreachable gateway must not take the other twenty-nine scenarios
    with it: the run is worth reading with one line marked as an error.
    """
    chat_ctx = build_context(instructions, scenario)
    for attempt in (1, 2):
        try:
            collected = await model.chat(chat_ctx=chat_ctx, tools=tools).collect()
        except APIError as err:
            if attempt == 2 or not err.retryable:
                return judge(scenario, Response(error=f"{type(err).__name__}: {err}"))
            await asyncio.sleep(RETRY_DELAY_S)
        except asyncio.TimeoutError as err:
            if attempt == 2:
                return judge(scenario, Response(error=f"timeout: {err}"))
            await asyncio.sleep(RETRY_DELAY_S)
        else:
            return judge(scenario, to_response(collected))
    raise AssertionError("unreachable: every attempt returns or retries")


async def run_all(
    scenarios: list[Scenario], model_name: str, concurrency: int
) -> list[Result]:
    """Every scenario, in file order, a few at a time.

    The front is built here, not described here: a real FrontAgent hands over
    both halves of what the model sees — the persona it was constructed with,
    and ask_mentat bound to it. Bound matters. The tool read off the class
    still carries `self`, and the schema built from it does not describe the
    tool Luna is actually offered.

    One LLM instance for the whole run, as the session holds one for a call:
    it owns the connection pool, and thirty of them would spend the run opening
    sockets.
    """
    instructions, voice_card = agent.load_persona()
    # The same fold production does, so an eval with MENTAT_VOICE_PRIVATE set
    # scores the persona the room actually hears; unset scores the public one.
    private = load_private_context(os.environ.get(PRIVATE_CONTEXT_ENV))
    instructions = with_private_context(instructions, private)
    # Constructed exactly as entrypoint constructs it. Nothing here consults —
    # the tool is declared to the model and never executed — but a stand-in
    # front would be a second definition of the thing under measurement.
    front = agent.FrontAgent(
        pronunciations=private.pronunciations,
        instructions=instructions,
        voice_card=voice_card,
        room_name="eval",
        mentat_url=agent.DEFAULT_MENTAT_URL,
    )
    model = LLM(model_name)
    limit = asyncio.Semaphore(concurrency)

    async def one(scenario: Scenario) -> Result:
        async with limit:
            return await run_scenario(model, front.tools, front.instructions, scenario)

    try:
        return await asyncio.gather(*(one(scenario) for scenario in scenarios))
    finally:
        await model.aclose()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=HERE / "scenarios.jsonl",
        help="labelled scenario file (default: the one next to this script)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="exit non-zero below this fraction (default: 0, never fails)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY, help="turns in flight at once"
    )
    # The room's own model by default: measuring anything else would score a
    # voice Josh never hears. --model is for deliberately comparing the two.
    parser.add_argument(
        "--model",
        default=agent.FRONT_MODEL,
        help=f"inference model id (default: {agent.FRONT_MODEL})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios = load_scenarios(args.scenarios)
    results = asyncio.run(run_all(scenarios, args.model, args.concurrency))
    print(report(results))
    return exit_code(results, args.min_accuracy)


if __name__ == "__main__":
    raise SystemExit(main())
