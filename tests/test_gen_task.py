"""Unit tests for gen_task.py's pieces(): splitting a batched LLM reply (compute_reward_1..4) into separate
candidates, used by every batching experiment (exp_i_generalize, exp_k_llm_robustness, exp_l_batch_quality)."""
import gen_task as g


def test_pieces_indep_arm_returns_the_whole_text_unsplit():
    text = "```python\ndef compute_reward(x):\n    return x\n```"
    assert g.pieces("indep", text) == [(None, text)]


def test_pieces_batch4_splits_four_functions_and_renames_each_to_compute_reward():
    text = (
        "```python\ndef compute_reward_1(x):\n    return x\n```\n"
        "```python\ndef compute_reward_2(x):\n    return 2 * x\n```\n"
        "```python\ndef compute_reward_3(x):\n    return 3 * x\n```\n"
        "```python\ndef compute_reward_4(x):\n    return 4 * x\n```"
    )
    out = g.pieces("batch4", text)
    assert [k for k, _ in out] == [1, 2, 3, 4]
    for k, piece in out:
        assert f"compute_reward_{k}(" not in piece  # renamed away
        assert "def compute_reward(" in piece       # ...to the plain, trainable name
        assert piece.startswith("```python\n") and piece.rstrip().endswith("```")


def test_pieces_batch4_keeps_each_functions_own_body():
    text = (
        "```python\ndef compute_reward_1(x):\n    return 111\n```\n"
        "```python\ndef compute_reward_2(x):\n    return 222\n```"
    )
    out = g.pieces("batch4", text)
    bodies = dict(out)
    assert "return 111" in bodies[1]
    assert "return 222" in bodies[2]
    assert "return 111" not in bodies[2]  # no cross-contamination between pieces


def test_pieces_batch4_ignores_blocks_without_a_numbered_compute_reward():
    text = "```python\nimport torch\n```\n```python\ndef compute_reward_1(x):\n    return x\n```"
    out = g.pieces("batch4", text)
    assert [k for k, _ in out] == [1]  # the helper-only block is skipped, not mistaken for a candidate


def test_pieces_batch4_with_no_matching_blocks_returns_empty():
    assert g.pieces("batch4", "no code here at all") == []
