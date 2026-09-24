import unittest
import json
from router import prune_messages


class TestPruneMessages(unittest.TestCase):
    def test_no_prune_small(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        res, pruned, orig, final, stats = prune_messages(msgs, max_chars=1000)
        self.assertFalse(pruned)
        self.assertEqual(len(res), 2)
        self.assertEqual(stats["dedupes"], 0)

    def test_tool_output_truncation(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        for i in range(5):
            call_id = f"call_{i}"
            msgs.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "test"}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": "x" * 2000})

        res, pruned, orig, final, stats = prune_messages(msgs, max_chars=100000, keep_recent_tool_turns=2)
        self.assertTrue(pruned)
        self.assertIn("[Output evicted:", res[3]["content"])
        self.assertIn("[Output evicted:", res[5]["content"])
        self.assertIn("[Output evicted:", res[7]["content"])
        self.assertEqual(len(res[9]["content"]), 2000)
        self.assertEqual(len(res[11]["content"]), 2000)
        self.assertEqual(stats["evicted_tools"], 3)

    def test_donut_window_preserves_tool_pairing(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "root task"}]
        for i in range(15):
            call_id = f"call_{i}"
            msgs.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "test"}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": f"output_{i} " * 50})

        res, pruned, orig, final, stats = prune_messages(msgs, max_chars=500, keep_recent_tool_turns=2, min_tail_messages=4)
        self.assertTrue(pruned)
        self.assertEqual(res[0]["role"], "system")
        self.assertEqual(res[1]["role"], "user")
        self.assertIn("[System Note:", res[2]["content"])
        tail = res[3:]
        for idx, m in enumerate(tail):
            if m.get("role") == "tool":
                self.assertGreater(idx, 0, "Tool message cannot be the first message in tail")
                prev = tail[idx - 1]
                self.assertTrue(
                    (prev.get("role") == "assistant" and any(tc["id"] == m.get("tool_call_id") for tc in prev.get("tool_calls", []))) or
                    prev.get("role") == "tool"
                )

    def test_multi_tool_call_tail_boundary(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        for i in range(5):
            call_1 = f"c_{i}_1"
            call_2 = f"c_{i}_2"
            msgs.append({
                "role": "assistant",
                "tool_calls": [
                    {"id": call_1, "type": "function", "function": {"name": "test1"}},
                    {"id": call_2, "type": "function", "function": {"name": "test2"}}
                ]
            })
            msgs.append({"role": "tool", "tool_call_id": call_1, "content": f"res_{i}_1 " * 200})
            msgs.append({"role": "tool", "tool_call_id": call_2, "content": f"res_{i}_2 " * 200})

        res, pruned, orig, final, stats = prune_messages(msgs, max_chars=400, keep_recent_tool_turns=1, min_tail_messages=2)
        self.assertTrue(pruned)
        marker_idx = 2
        tail = res[marker_idx + 1:]
        self.assertEqual(tail[0]["role"], "assistant")
        self.assertEqual(len(tail[0]["tool_calls"]), 2)
        self.assertEqual(tail[1]["role"], "tool")
        self.assertEqual(tail[2]["role"], "tool")

    def test_option_b_head_preserves_clarification_turns(self):
        msgs = [
            {"role": "system", "content": "Hermes sys instructions"},
            {"role": "user", "content": "Initial inquiry"},
            {"role": "assistant", "content": "Clarifying question: which file?"},
            {"role": "user", "content": "Spec: edit player.gd with CSG nodes"},
        ]
        for i in range(10):
            call_id = f"c_{i}"
            msgs.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "test"}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": f"data_{i} " * 200})

        res, pruned, orig, final, stats = prune_messages(msgs, max_chars=400, keep_recent_tool_turns=1, min_tail_messages=2)
        self.assertTrue(pruned)
        self.assertEqual(res[0]["content"], "Hermes sys instructions")
        self.assertEqual(res[1]["content"], "Initial inquiry")
        self.assertEqual(res[2]["content"], "Clarifying question: which file?")
        self.assertEqual(res[3]["content"], "Spec: edit player.gd with CSG nodes")
        self.assertIn("[System Note:", res[4]["content"])

    def test_dcp_tool_deduplication(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        # Turn 1: read_file("player.gd")
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "player.gd"})}}]
        })
        msgs.append({"role": "tool", "tool_call_id": "c1", "content": "old_code " * 100})

        # Turn 2: unrelated tool
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
        })
        msgs.append({"role": "tool", "tool_call_id": "c2", "content": "result"})

        # Turn 3: read_file("player.gd") again!
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "player.gd"})}}]
        })
        msgs.append({"role": "tool", "tool_call_id": "c3", "content": "new_code " * 100})

        res, pruned, orig, final, stats = prune_messages(msgs, max_chars=100000, keep_recent_tool_turns=3)
        self.assertTrue(pruned)
        self.assertEqual(stats["dedupes"], 1)
        # c1 should be pruned with duplicate notice
        self.assertIn("[Duplicate read_file pruned:", res[3]["content"])
        self.assertIn("Superseded by newer call", res[3]["content"])
        # c3 (latest) should remain intact
        self.assertEqual(res[7]["content"], "new_code " * 100)

    def test_dcp_resolved_error_tombstoning(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        # Old turn with a Python traceback
        traceback_output = (
            "Traceback (most recent call last):\n"
            "  File \"player.gd\", line 42, in _ready\n"
            "    self.initialize_nodes()\n"
            "SyntaxError: invalid syntax in player.gd\n" + ("extra trace info\n" * 50)
        )
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": "err_call", "type": "function", "function": {"name": "terminal", "arguments": "gdscript player.gd"}}]
        })
        msgs.append({"role": "tool", "tool_call_id": "err_call", "content": traceback_output})

        # 4 subsequent successful tool turns
        for i in range(4):
            call_id = f"c_{i}"
            msgs.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "ok"}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": "success"})

        res, pruned, orig, final, stats = prune_messages(msgs, max_chars=100000, keep_recent_tool_turns=2)
        self.assertTrue(pruned)
        self.assertEqual(stats["errors_purged"], 1)
        self.assertIn("[Historical error output pruned:", res[3]["content"])
        self.assertIn("SyntaxError:", res[3]["content"])


if __name__ == "__main__":
    unittest.main()
