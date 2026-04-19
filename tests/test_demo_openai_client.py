import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from muse_spark import demo_openai_client


class FakeChunk:
    def __init__(self, content=None, role=None, finish_reason=None):
        self.choices = [
            type(
                "Choice",
                (),
                {
                    "delta": type("Delta", (), {"content": content, "role": role})(),
                    "finish_reason": finish_reason,
                },
            )()
        ]


class DemoOpenAIClientTests(unittest.TestCase):
    def test_build_parser_has_visual_demo_defaults(self):
        parser = demo_openai_client.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(args.model, "meta/muse-spark")
        self.assertEqual(args.prompt, "Reply with exactly: muse spark is live")
        self.assertFalse(args.stream)

    def test_run_completion_nonstream_returns_text(self):
        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return type(
                            "Response",
                            (),
                            {
                                "choices": [
                                    type(
                                        "Choice",
                                        (),
                                        {
                                            "message": type(
                                                "Message", (), {"content": "muse spark is live"}
                                            )()
                                        },
                                    )()
                                ]
                            },
                        )()

        text = demo_openai_client.run_completion(
            client=FakeClient(),
            model="meta/muse-spark",
            prompt="Reply with exactly: muse spark is live",
            stream=False,
        )

        self.assertEqual(text, "muse spark is live")

    def test_run_completion_stream_concatenates_chunk_text(self):
        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return iter(
                            [
                                FakeChunk(role="assistant"),
                                FakeChunk(content="muse "),
                                FakeChunk(content="spark "),
                                FakeChunk(content="is live"),
                                FakeChunk(finish_reason="stop"),
                            ]
                        )

        text = demo_openai_client.run_completion(
            client=FakeClient(),
            model="meta/muse-spark",
            prompt="Reply with exactly: muse spark is live",
            stream=True,
        )

        self.assertEqual(text, "muse spark is live")

    def test_main_prints_visual_friendly_output(self):
        stdout = io.StringIO()
        with patch.object(
            demo_openai_client,
            "create_client",
            return_value=object(),
        ), patch.object(
            demo_openai_client,
            "run_completion",
            return_value="muse spark is live",
        ), redirect_stdout(stdout):
            exit_code = demo_openai_client.main([])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("base_url: http://127.0.0.1:8000/v1", output)
        self.assertIn("model: meta/muse-spark", output)
        self.assertIn("prompt: Reply with exactly: muse spark is live", output)
        self.assertIn("response:\nmuse spark is live", output)
