import unittest

from muse_spark.prompt_compiler import build_stateful_turn_plan


class StatefulPromptTests(unittest.TestCase):
    def test_build_stateful_turn_plan_emits_system_preamble_and_latest_user_turn(self):
        plan = build_stateful_turn_plan(
            [
                {"role": "system", "content": "You are a sharp frontend engineer."},
                {"role": "developer", "content": "Prefer concise answers."},
                {"role": "user", "content": "Explain the architecture."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "Now refactor the navbar."},
            ],
            max_chars=4000,
        )

        self.assertIn("<conversation_setup>", plan.system_preamble)
        self.assertIn("<system_instructions>", plan.system_preamble)
        self.assertIn("Prefer concise answers.", plan.system_preamble)
        # No more fixed bootstrap scaffolding (READY handshake, format
        # instructions) — only the caller's own system messages are emitted.
        self.assertNotIn("Reply with exactly READY", plan.system_preamble)
        self.assertIn("<user_message>", plan.user_prompt)
        self.assertIn("Now refactor the navbar.", plan.user_prompt)
        self.assertNotIn("Explain the architecture.", plan.user_prompt)
        self.assertNotIn("[assistant]", plan.user_prompt)

    def test_build_stateful_turn_plan_skips_preamble_when_no_system_messages(self):
        plan = build_stateful_turn_plan(
            [{"role": "user", "content": "Write me a landing page hero."}],
            max_chars=4000,
        )

        # With bootstrap removed, requests without system messages produce
        # no preamble — Meta just sees the user message and replies. Turn 1
        # is intentionally "leaner" in exchange for one less round-trip.
        self.assertEqual(plan.system_preamble, "")
        self.assertIn("Write me a landing page hero.", plan.user_prompt)
