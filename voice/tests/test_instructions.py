"""Offline policy contracts for backend web search and voice delegation."""

import sys
import unittest
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VOICE_DIR))

from request import VOICE_CARD_MARKER, split_persona

ROOT = VOICE_DIR.parent

PERSONAL_TOOLS = (
    "For questions about Josh's own data or systems, use his personal tools; "
    "do not use `web_search`."
)
STABLE_FACTS = "Answer stable common knowledge directly without searching."
SEARCH_POLICY = (
    "For current, time-sensitive, local, niche, or otherwise uncertain facts, "
    "use ToolSearch to discover `web_search` and search before answering; "
    "never guess confidently."
)
SOURCE_POLICY = "Name the source you checked in your answer."
SOURCE_ATTRIBUTION_POLICY = (
    "Attribute each claim only to the source that actually supports it; name the "
    "site you checked, and never attribute a result to a site that did not provide it."
)
RESULT_SITE_POLICY = (
    "Name a source site only if it appears among the search results you actually "
    "used to answer."
)
CHECKED_PAGE_POLICY = (
    "Never claim you checked a page unless that page appeared among the search "
    "results you actually used to answer."
)
STALE_RESULT_POLICY = (
    "If a search result's date or source version is older than the date asked about, "
    "qualify it as potentially stale and do not present it as current fact without "
    "checking the current source."
)
VERIFY_POLICY = (
    "Treat search-result snippets and summaries as unverified leads. Check the "
    "result's source and context to judge what it supports; when that cannot be "
    "established from search results, say that and state uncertainty instead of "
    "treating a summary as fact."
)
FRESHNESS_POLICY = (
    "Check the source's publication or update date and whether it applies to the "
    "date asked about. If sources conflict, are undated, or may be stale, state "
    "uncertainty rather than present an unverified summary as fact."
)
VOICE_DELEGATION = (
    "Delegate current, local, niche, or otherwise uncertain factual questions "
    "for backend lookup instead of guessing, including when a result may be stale "
    "or sources disagree."
)
VOICE_STABLE = (
    "For factual questions with stable common knowledge, respond directly rather "
    "than delegating."
)

OPENING_POLICY = (
    "When the call opens, speak first. Greet Josh in a few words and ask what's up, "
    "then listen. Let the call context shape the greeting: the time of day, the day "
    "of the week, where he is, or that he's driving or traveling. Lead with the most "
    "specific thing you know: driving, traveling, or a named place beats the time of "
    "day. Vary the wording from call to call."
)
COMPLETION_POLICY = (
    'When Mentat confirms an action or answers Josh\'s request, relay it briefly and '
    'stop. Do not add "anything else?" or offer more help;'
)


class InstructionPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = (ROOT / "prompt.md").read_text()
        cls.persona = (VOICE_DIR / "persona.md").read_text()
        cls.instructions, marker, cls.voice_card = cls.persona.partition(VOICE_CARD_MARKER)
        if not marker:
            raise AssertionError("persona must retain the voice-card split marker")

    def assert_policy_present(self, text, clause):
        normalized = " ".join(text.lower().split())
        self.assertTrue(clause.lower() in normalized, f"Missing affirmative policy: {clause}")

    def test_backend_search_policy_excludes_personal_data_and_keeps_stable_facts_direct(self):
        self.assert_policy_present(self.prompt, PERSONAL_TOOLS)
        self.assert_policy_present(self.prompt, STABLE_FACTS)

    def test_backend_searches_uncertain_facts_and_verifies_sources_and_dates(self):
        for clause in (
            SEARCH_POLICY,
            SOURCE_POLICY,
            SOURCE_ATTRIBUTION_POLICY,
            RESULT_SITE_POLICY,
            CHECKED_PAGE_POLICY,
            VERIFY_POLICY,
            FRESHNESS_POLICY,
            STALE_RESULT_POLICY,
        ):
            with self.subTest(clause=clause):
                self.assert_policy_present(self.prompt, clause)

    def test_voice_delegates_uncertain_facts_but_answers_stable_knowledge_directly(self):
        self.assert_policy_present(self.instructions, VOICE_DELEGATION)
        self.assert_policy_present(self.instructions, VOICE_STABLE)

    def test_voice_greets_first_and_does_not_prolong_a_finished_request(self):
        self.assert_policy_present(self.instructions, OPENING_POLICY)
        self.assert_policy_present(self.instructions, COMPLETION_POLICY)

    def test_policy_assertions_reject_negation_or_removal(self):
        mutations = (
            (PERSONAL_TOOLS, PERSONAL_TOOLS.replace("do not use", "use")),
            (STABLE_FACTS, STABLE_FACTS.replace("directly without searching", "only after searching")),
            (VERIFY_POLICY, VERIFY_POLICY.replace("unverified leads", "verified facts")),
            (
                SOURCE_ATTRIBUTION_POLICY,
                SOURCE_ATTRIBUTION_POLICY.replace("never attribute", "attribute"),
            ),
            (RESULT_SITE_POLICY, RESULT_SITE_POLICY.replace("only if", "even if")),
            (
                CHECKED_PAGE_POLICY,
                CHECKED_PAGE_POLICY.replace("Never claim", "Claim"),
            ),
            (
                STALE_RESULT_POLICY,
                STALE_RESULT_POLICY.replace("qualify it as potentially stale", "present it as current"),
            ),
            (FRESHNESS_POLICY, ""),
        )
        for clause, mutated in mutations:
            with self.subTest(clause=clause):
                with self.assertRaises(AssertionError):
                    self.assert_policy_present(mutated, clause)

    def test_voice_card_split_and_spoken_style_are_preserved(self):
        instructions, voice_card = split_persona(self.persona)
        self.assertEqual(instructions, self.instructions.strip())
        self.assertEqual(voice_card, self.voice_card.strip())
        self.assertIn("spoken prose short", voice_card)
        self.assertIn("Never use lists", voice_card)


if __name__ == "__main__":
    unittest.main()
