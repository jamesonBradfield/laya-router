import unittest
from router import prune_messages


class TestPruneMessages(unittest.TestCase):
    def test_no_prune_small(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        res, pruned, orig, final = prune_messages(msgs, max_chars=1000)
        self.assertFalse(pruned)
        self.assertEqual(len(res), 2)

    def test_tool_output_truncation(self):
        # Create 5 tool turns, each with 2000 chars of output
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        for i in range(5):
            call_id = f"call_{i}"
            msgs.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "test"}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": "x" * 2000})

        # Keep recent 2 turns
        res, pruned, orig, final = prune_messages(msgs, max_chars=100000, keep_recent_tool_turns=2)
        self.assertTrue(pruned)
        # First 3 tool outputs should be evicted
        self.assertIn("[Output evicted:", res[3]["content"])
        self.assertIn("[Output evicted:", res[5]["content"])
        self.assertIn("[Output evicted:", res[7]["content"])
        # Last 2 tool outputs should be intact
        self.assertEqual(len(res[9]["content"]), 2000)
        self.assertEqual(len(res[11]["content"]), 2000)

    def test_donut_window_preserves_tool_pairing(self):
        # Create 15 turns
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "root task"}]
        for i in range(15):
            call_id = f"call_{i}"
            msgs.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "test"}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": f"output_{i} " * 50})

        # Force donut windowing with very small max_chars
        res, pruned, orig, final = prune_messages(msgs, max_chars=500, keep_recent_tool_turns=2, min_tail_messages=4)
        self.assertTrue(pruned)
        # Check that res[0] is system, res[1] is user
        self.assertEqual(res[0]["role"], "system")
        self.assertEqual(res[1]["role"], "user")
        # Check marker is present
        self.assertIn("[System Note:", res[2]["content"])
        # Check that in tail, every tool message has a valid tool_call_id and follows an assistant message
        tail = res[3:]
        for idx, m in enumerate(tail):
            if m.get("role") == "tool":
                self.assertGreater(idx, 0, "Tool message cannot be the first message in tail")
                prev = tail[idx - 1]
                # Either previous is an assistant with tool_calls, or previous is another tool from a multi-call
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

        res, pruned, orig, final = prune_messages(msgs, max_chars=400, keep_recent_tool_turns=1, min_tail_messages=2)
        self.assertTrue(pruned)
        # Tail must never start with a tool message
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
        # Then add 10 tool turns
        for i in range(10):
            call_id = f"c_{i}"
            msgs.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "test"}}]})
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": f"data_{i} " * 200})

        res, pruned, orig, final = prune_messages(msgs, max_chars=400, keep_recent_tool_turns=1, min_tail_messages=2)
        self.assertTrue(pruned)
        # Option B pins everything up to first tool call (res[0], res[1], res[2], res[3])
        self.assertEqual(res[0]["content"], "Hermes sys instructions")
        self.assertEqual(res[1]["content"], "Initial inquiry")
        self.assertEqual(res[2]["content"], "Clarifying question: which file?")
        self.assertEqual(res[3]["content"], "Spec: edit player.gd with CSG nodes")
        self.assertIn("[System Note:", res[4]["content"])

if __name__ == "__main__":
    unittest.main()
