import unittest

from muse_spark.prompt_compiler import build_stateful_turn_plan


class StatefulPromptCompilerTests(unittest.TestCase):
    def test_build_stateful_turn_plan_emits_xml_bootstrap_and_latest_user_turn(self):
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

        self.assertIn("<conversation_setup>", plan.bootstrap_prompt)
        self.assertIn("<system_instructions>", plan.bootstrap_prompt)
        self.assertIn("<acknowledgement>READY</acknowledgement>", plan.bootstrap_prompt)
        self.assertIn("You are a sharp frontend engineer.", plan.bootstrap_prompt)
        self.assertIn("Prefer concise answers.", plan.bootstrap_prompt)
        self.assertNotIn("<now>", plan.bootstrap_prompt)
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

    def test_build_stateful_turn_plan_raises_for_empty_messages(self):
        with self.assertRaises(ValueError):
            build_stateful_turn_plan([])
