"""Unit tests for the pure text-processing helpers in eureka_ab.py: code extraction from an LLM reply,
response parsing, and candidate-directory naming. None of this needs torch, Isaac Gym or the network --
eureka_ab imports cleanly without any of those (verified: it only needs the Eureka repo's text-only
utils.extract_task_code / utils.misc at import time)."""
import json

import eureka_ab as e


# ---------------------------------------------------------------- extract_code

def test_extract_code_pulls_a_single_function_out_of_a_python_fence():
    reply = "Here is the reward function:\n```python\ndef compute_reward(x):\n    return x\n```\nLet me know if you need changes."
    assert e.extract_code(reply) == "def compute_reward(x):\n    return x"


def test_extract_code_keeps_only_the_last_def_when_a_fence_has_two_functions():
    # documented, upstream-verbatim quirk: the trim-to-"def" loop has no break, so it keeps re-slicing to the
    # LAST "def" line it sees -- an earlier helper function in the same fence is silently dropped.
    reply = "```python\ndef helper():\n    pass\ndef compute_reward(x):\n    return x\n```"
    assert e.extract_code(reply) == "def compute_reward(x):\n    return x"


def test_extract_code_falls_back_to_the_raw_reply_when_nothing_is_delimited():
    # no code fence, no triple/double quotes anywhere -- none of the five extraction patterns can match,
    # so extract_code must fall back to the whole reply and still trim to the function definition.
    reply = "Explanation: here is the function.\ndef compute_reward(x):\n    return x"
    assert '"' not in reply and "```" not in reply
    assert e.extract_code(reply) == "def compute_reward(x):\n    return x"


def test_extract_code_prefers_the_python_fence_over_plain_triple_quotes():
    reply = 'Notes """not the function""" more notes\n```python\ndef compute_reward(x):\n    return x\n```'
    assert e.extract_code(reply) == "def compute_reward(x):\n    return x"


# ---------------------------------------------------------------- parse_response

def test_parse_response_extracts_content_finish_reason_usage_and_model():
    body = {"choices": [{"message": {"content": "the code"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}, "model": "minimax/minimax-m3.1-flash"}
    text, finish, usage, model = e.parse_response(json.dumps(body))
    assert text == "the code"
    assert finish == "stop"
    assert usage == {"prompt_tokens": 10, "completion_tokens": 20}
    assert model == "minimax/minimax-m3.1-flash"


def test_parse_response_handles_a_body_with_no_choices_without_crashing():
    body = {"choices": [], "usage": None, "model": "m"}
    text, finish, usage, model = e.parse_response(json.dumps(body))
    assert text is None
    assert finish is None


# ---------------------------------------------------------------- suffix_for

def test_suffix_for_encodes_round_and_candidate_with_the_arm_tag():
    suffix = e.suffix_for("round1/cand0")
    assert suffix == "GPT" + e.ARM["tag"] + "R1C0"


def test_suffix_for_is_stable_for_round_two():
    suffix = e.suffix_for("round2/cand3")
    assert suffix == "GPT" + e.ARM["tag"] + "R2C3"
