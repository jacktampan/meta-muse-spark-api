"""Unit tests for streaming-side helpers added to fix output stability issues.

Covers:
- ``muse_spark.client._resync_text`` (Fix #4: resync on full-event mismatch)
- ``muse_spark.client._is_completion_op`` and ``_event_is_complete``
  (Fix #2: explicit completion-signal detection so we exit promptly instead
  of relying on the recv() idle timeout, which previously caused truncation)
- ``muse_spark.api._clean_assistant_text`` and ``_ScaffoldingStripper``
  (Fix #4: strip stateful-turn scaffolding tags that occasionally leak from
  Meta's stateful conversation context into user-visible output)
"""
from __future__ import annotations

import unittest

from muse_spark.api import _ScaffoldingStripper, _clean_assistant_text
from muse_spark.client import _event_is_complete, _is_completion_op, _resync_text


class ResyncTextTests(unittest.TestCase):
    def test_no_change_returns_empty(self):
        self.assertEqual(_resync_text("hello world", "hello world"), [])

    def test_extension_returns_only_suffix(self):
        self.assertEqual(_resync_text("hello", "hello world"), [" world"])

    def test_empty_current_returns_full_new(self):
        self.assertEqual(_resync_text("", "hello world"), ["hello world"])

    def test_late_divergence_returns_tail_after_common_prefix(self):
        # Meta corrected the last word — small backtrack, accept duplication
        # to converge on Meta's final text.
        result = _resync_text("Hello, world!", "Hello, World!")
        self.assertEqual(result, ["World!"])

    def test_large_backtrack_drops_correction(self):
        # If Meta rewrites a long span we already streamed, emitting the
        # divergent tail would just stack garbled duplication on top of
        # the original (SSE can't retract bytes). Drop the correction.
        current = "x" + "a" * 200  # 201 chars, divergence at index 1
        new_full = "x" + "b" * 200
        self.assertEqual(_resync_text(current, new_full), [])

    def test_backtrack_threshold_is_inclusive(self):
        # backtrack==max_backtrack is still applied; backtrack>max_backtrack drops.
        current = "abcdefghij"  # 10 chars
        new_full = "abXXXXXXXX"  # diverges at index 2 → backtrack=8
        self.assertEqual(
            _resync_text(current, new_full, max_backtrack=8),
            ["XXXXXXXX"],
        )
        self.assertEqual(
            _resync_text(current, new_full, max_backtrack=7),
            [],
        )

    def test_shorter_new_returns_empty(self):
        # We never roll back already-streamed characters.
        self.assertEqual(_resync_text("hello world", "hello"), [])


class CompletionSignalTests(unittest.TestCase):
    def test_replace_state_complete_is_completion(self):
        self.assertTrue(_is_completion_op("replace", "/sections/0/state", "COMPLETE"))
        self.assertTrue(_is_completion_op("replace", "/state", "DONE"))
        self.assertTrue(_is_completion_op("replace", "/state", "finished"))

    def test_replace_is_complete_true_is_completion(self):
        self.assertTrue(
            _is_completion_op("replace", "/sections/0/is_complete", True)
        )

    def test_unrelated_replace_is_not_completion(self):
        self.assertFalse(_is_completion_op("replace", "/state", "STREAMING"))
        self.assertFalse(_is_completion_op("delta", "/state", "COMPLETE"))
        self.assertFalse(_is_completion_op(None, None, None))

    def test_event_with_response_state_complete(self):
        self.assertTrue(_event_is_complete({"response": {"state": "complete"}}))
        self.assertTrue(_event_is_complete({"response": {"is_complete": True}}))

    def test_event_without_completion_signal(self):
        self.assertFalse(_event_is_complete({"response": {"state": "STREAMING"}}))
        self.assertFalse(_event_is_complete({}))
        self.assertFalse(_event_is_complete({"response": "not-a-dict"}))


class CleanAssistantTextTests(unittest.TestCase):
    def test_strips_user_message_wrapper(self):
        text = "<conversation_turn><user_message>Hi</user_message></conversation_turn>"
        self.assertEqual(_clean_assistant_text(text), "Hi")

    def test_collapses_bare_ready_signal(self):
        # Meta occasionally echoes the bootstrap acknowledgement; that's
        # internal scaffolding and must not surface to API consumers.
        self.assertEqual(_clean_assistant_text("READY"), "")
        self.assertEqual(_clean_assistant_text("READY."), "")

    def test_preserves_normal_text(self):
        self.assertEqual(_clean_assistant_text("Hello, world!"), "Hello, world!")

    def test_returns_empty_string_for_none_or_empty_input(self):
        # Type annotation says ``str`` so callers (e.g. JSON serialisation)
        # don't have to defensively handle None.
        self.assertEqual(_clean_assistant_text(None), "")
        self.assertEqual(_clean_assistant_text(""), "")


class ScaffoldingStripperTests(unittest.TestCase):
    def test_emits_chunks_without_tags_immediately(self):
        stripper = _ScaffoldingStripper()
        self.assertEqual(stripper.feed("Hello"), "Hello")
        self.assertEqual(stripper.feed(" world"), " world")
        self.assertEqual(stripper.flush(), "")

    def test_holds_partial_tag_until_completion(self):
        stripper = _ScaffoldingStripper()
        emitted = []
        emitted.append(stripper.feed("Hello "))
        emitted.append(stripper.feed("<user_mes"))  # partial tag — buffered
        emitted.append(stripper.feed("sage>world</user_message> done"))
        emitted.append(stripper.flush())
        joined = "".join(emitted)
        self.assertNotIn("<user_message", joined)
        self.assertNotIn("</user_message", joined)
        self.assertIn("Hello", joined)
        self.assertIn("world", joined)
        self.assertIn("done", joined)

    def test_strips_complete_tag_in_single_chunk(self):
        stripper = _ScaffoldingStripper()
        out = stripper.feed("Hello <conversation_turn>x</conversation_turn> end")
        self.assertEqual(out, "Hello x end")

    def test_flush_drops_bare_ready_acknowledgement(self):
        stripper = _ScaffoldingStripper()
        stripper.feed("READY")
        # Bare "READY" stuck in buffer (no tag, no later content) must be
        # dropped at flush time so it never surfaces to clients.
        self.assertEqual(stripper.flush(), "")


if __name__ == "__main__":
    unittest.main()
