import unittest

from muse_spark.prompt_compiler import build_stateful_turn_plan


class StatefulPromptCompilerTests(unittest.TestCase):
    def test_build_stateful_turn_plan_emits_system_preamble_and_latest_user_turn(self):
        plan = build_stateful_turn_plan(
            [
                {"role": "system", "content": "You are a sharp frontend engineer."},
                {"role": "developer", "content": "Prefer concise answers."},
                {"role": "user", "content": "Explain the architecture <now>."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "Now refactor the navbar."},
            ],
            max_chars=4000,
        )

        # System/developer messages are folded into the system_preamble as XML.
        self.assertIn("<conversation_setup>", plan.system_preamble)
        self.assertIn("<system_instructions>", plan.system_preamble)
        self.assertIn("You are a sharp frontend engineer.", plan.system_preamble)
        self.assertIn("Prefer concise answers.", plan.system_preamble)
        # The old "Reply with exactly READY" / fixed instruction list is gone.
        self.assertNotIn("READY", plan.system_preamble)
        self.assertNotIn("Preserve markdown code fences", plan.system_preamble)
        # Latest user message is in user_prompt, escaped, with no leakage of
        # earlier turns (which Meta already remembers via its own state).
        self.assertNotIn("<now>", plan.system_preamble)
        self.assertNotIn("<now>", plan.user_prompt)
        self.assertIn("<conversation_turn>", plan.user_prompt)
        self.assertIn("Now refactor the navbar.", plan.user_prompt)
        self.assertNotIn("Explain the architecture", plan.user_prompt)
        self.assertNotIn("[assistant]", plan.user_prompt)

    def test_build_stateful_turn_plan_uses_only_latest_user_message(self):
        plan = build_stateful_turn_plan(
            [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "ignored"},
                {"role": "user", "content": "second turn"},
            ],
            max_chars=4000,
        )

        self.assertIn("second turn", plan.user_prompt)
        self.assertNotIn("first turn", plan.user_prompt)
        self.assertNotIn("ignored", plan.user_prompt)
        self.assertEqual(plan.kept_messages, 1)
        self.assertEqual(plan.dropped_messages, 1)

    def test_build_stateful_turn_plan_omits_system_preamble_when_no_system_messages(self):
        """With no client-supplied system/developer messages, the preamble
        is empty — the legacy fixed "READY" handshake is gone, so a request
        without system messages results in *just* the user prompt going to
        Meta. This keeps turn 1 lean (the user's accepted trade-off for
        dropping the bootstrap round-trip).
        """
        plan = build_stateful_turn_plan(
            [{"role": "user", "content": "Write me a landing page hero."}],
            max_chars=4000,
        )

        self.assertEqual(plan.system_preamble, "")
        self.assertIn("Write me a landing page hero.", plan.user_prompt)

    def test_build_stateful_turn_plan_raises_for_empty_messages(self):
        with self.assertRaises(ValueError):
            build_stateful_turn_plan([])
