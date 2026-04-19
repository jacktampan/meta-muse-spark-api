import unittest

from muse_spark.prompt_compiler import build_stateful_turn_plan


class StatefulPromptTests(unittest.TestCase):
    def test_build_stateful_turn_plan_emits_xml_bootstrap_and_latest_user_turn(self):
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

        self.assertIn("<conversation_setup>", plan.bootstrap_prompt)
        self.assertIn("<system_instructions>", plan.bootstrap_prompt)
        self.assertIn("Reply with exactly READY.", plan.bootstrap_prompt)
        self.assertIn("Preserve markdown code fences when returning code.", plan.bootstrap_prompt)
        self.assertIn("<user_message>", plan.user_prompt)
        self.assertIn("Now refactor the navbar.", plan.user_prompt)
        self.assertNotIn("Explain the architecture.", plan.user_prompt)
        self.assertNotIn("[assistant]", plan.user_prompt)

    def test_build_stateful_turn_plan_always_includes_bootstrap_for_first_turn(self):
        plan = build_stateful_turn_plan(
            [{"role": "user", "content": "Write me a landing page hero."}],
            max_chars=4000,
        )

        self.assertTrue(plan.bootstrap_prompt)
        self.assertIn("<conversation_setup>", plan.bootstrap_prompt)
        self.assertIn("Write me a landing page hero.", plan.user_prompt)
