import unittest

from muse_spark.prompt_compiler import compile_chat_messages


class PromptCompilerTests(unittest.TestCase):
    def test_compile_messages_merges_system_messages_and_preserves_roles(self):
        compiled = compile_chat_messages(
            [
                {"role": "system", "content": "You are a senior frontend engineer."},
                {"role": "system", "content": "Prefer clean React patterns."},
                {"role": "user", "content": "Here is a component."},
                {"role": "assistant", "content": "Show me the code."},
                {"role": "user", "content": "Refactor it."},
            ],
            max_chars=4000,
        )

        self.assertIn("System instructions:", compiled.prompt)
        self.assertIn("- You are a senior frontend engineer.", compiled.prompt)
        self.assertIn("- Prefer clean React patterns.", compiled.prompt)
        self.assertIn("[user]\nHere is a component.", compiled.prompt)
        self.assertIn("[assistant]\nShow me the code.", compiled.prompt)
        self.assertIn("[user]\nRefactor it.", compiled.prompt)
        self.assertFalse(compiled.truncated)

    def test_compile_messages_truncates_oldest_turns_and_keeps_latest(self):
        compiled = compile_chat_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "old user " * 40},
                {"role": "assistant", "content": "old assistant " * 40},
                {"role": "user", "content": "latest request: refactor this navbar"},
            ],
            max_chars=220,
        )

        self.assertTrue(compiled.truncated)
        self.assertIn("latest request: refactor this navbar", compiled.prompt)
        self.assertNotIn("old assistant", compiled.prompt)
        self.assertGreaterEqual(compiled.dropped_messages, 1)

    def test_compile_messages_keeps_a_contiguous_newest_suffix(self):
        compiled = compile_chat_messages(
            [
                {"role": "user", "content": "old tiny"},
                {"role": "assistant", "content": "middle very long " * 40},
                {"role": "user", "content": "latest small"},
            ],
            max_chars=220,
        )

        self.assertIn("latest small", compiled.prompt)
        self.assertNotIn("old tiny", compiled.prompt)
        self.assertNotIn("middle very long", compiled.prompt)

    def test_compile_messages_adds_json_mode_instruction(self):
        compiled = compile_chat_messages(
            [{"role": "user", "content": "Return a JSON object with title and description."}],
            response_format={"type": "json_object"},
            max_chars=4000,
        )

        self.assertIn("Return only valid JSON.", compiled.prompt)
        self.assertIn("Do not include markdown fences or explanatory text.", compiled.prompt)

    def test_compile_messages_adds_max_tokens_and_stop_guidance(self):
        compiled = compile_chat_messages(
            [{"role": "user", "content": "Write a React hero component."}],
            max_tokens=250,
            stop=["</html>", "DONE"],
            max_chars=4000,
        )

        self.assertIn("Keep the answer within about 250 tokens if possible.", compiled.prompt)
        self.assertIn("Stop when you reach one of these sequences if possible: </html>, DONE", compiled.prompt)

    def test_compile_messages_treats_developer_role_as_instruction_preamble(self):
        compiled = compile_chat_messages(
            [
                {"role": "developer", "content": "Prefer terse answers."},
                {"role": "user", "content": "Refactor this component."},
            ],
            max_chars=4000,
        )

        self.assertIn("System instructions:", compiled.prompt)
        self.assertIn("- Prefer terse answers.", compiled.prompt)
        self.assertNotIn("[developer]", compiled.prompt)

    def test_compile_messages_preserves_non_text_content_repr(self):
        compiled = compile_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this asset."},
                        {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                    ],
                }
            ],
            max_chars=4000,
        )

        self.assertIn("Describe this asset.", compiled.prompt)
        self.assertIn("image_url", compiled.prompt)
        self.assertIn("demo.png", compiled.prompt)

    def test_compile_messages_system_only_truncation_metadata_is_correct(self):
        messages = [{"role": "system", "content": "abc"}]
        full = compile_chat_messages(messages, max_chars=4000)
        exact = compile_chat_messages(messages, max_chars=len(full.prompt))
        shorter = compile_chat_messages(messages, max_chars=max(1, len(full.prompt) - 2))

        self.assertFalse(exact.truncated)
        self.assertTrue(shorter.truncated)
        self.assertLessEqual(len(shorter.prompt), max(1, len(full.prompt) - 2))

    def test_compile_messages_truncates_latest_turn_without_chopping_requirements(self):
        compiled = compile_chat_messages(
            [{"role": "user", "content": "x" * 600}],
            max_chars=210,
        )

        self.assertTrue(compiled.truncated)
        self.assertIn("Response requirements:", compiled.prompt)
        self.assertTrue(compiled.prompt.endswith("Preserve markdown code fences when returning code."))

    def test_compile_messages_raises_for_empty_messages(self):
        with self.assertRaises(ValueError):
            compile_chat_messages([], max_chars=4000)
