from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
OPENAI_DEMO_PATH = REPO_ROOT / "examples" / "x-teaser-demo" / "openai_demo.py"


class XTeaserDemoFilesTests(unittest.TestCase):
    def test_openai_demo_is_camera_friendly_minimal_sdk_example(self):
        content = OPENAI_DEMO_PATH.read_text()

        self.assertIn("from openai import OpenAI", content)
        self.assertIn('base_url="http://127.0.0.1:8000/v1"', content)
        self.assertIn('api_key="x"', content)
        self.assertIn('model="meta/muse-spark"', content)
        self.assertIn('Reply with exactly: muse spark is live', content)
        self.assertIn("print(resp.choices[0].message.content)", content)
