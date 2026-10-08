#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Writes random conversations in corpus-conversations.json's form (README.md).

    random_conversations.py SEED COUNT > conversations.json

Each conversation has 1-9 messages of every role in any order, content drawn
from a pool of tags, control-token texts, Jinja syntax, Unicode and
whitespace, assistant reasoning and tool calls, tools on some and the
thinking, reasoning-effort and preserve-thinking options on some. `collect.py
render --conversations` renders the corpus on them; chat_corpus_test checks
the interpreter against those renderings when LLMP_TEST_DATA names a
directory whose chat/corpus-conversations.json is this file and
LLMP_TEST_MODELS one whose chat-templates/ holds the corpus and the new
references.json.
"""

import sys

sys.dont_write_bytecode = True

import json
import random

CONTENTS = [None, "", "hi", " spaced \n", "a <think>x</think> b", "東京 😀\t", "<tool_call>\nz\n</tool_call>",
            "\n\n", "{{ x }} {% if %} \\n \\\\ 'q' \"dq\"", "<|im_end|><|eot_id|><｜end▁of▁sentence｜>",
            "[INST] x [/INST]", "line1\nline2\r\nline3", "What is 2+2?", "<tool_response>r</tool_response>",
            "   ", "x" * 300]
REASONINGS = [None, None, None, "", " ", "think hard", "\nr\n", "<think>nested</think>"]
ARGUMENTS = [{"city": "Paris", "days": 3}, {}, {"a": {"b": [1, None, "x\ny"]}, "c": 2.5, "d": True},
             {"q": "東京 \"quoted\" <tag>"}]
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the weather.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string", "description": "City"},
                                         "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search <web> & \"more\".", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}, "n": {"type": "integer", "default": 3}},
        "required": ["query"]}}},
    {"type": "function", "function": {"name": "noargs", "description": "",
                                      "parameters": {"type": "object", "properties": {}}}},
]
ROLES = ["system"] * 2 + ["user"] * 3 + ["assistant"] * 3 + ["tool"] * 2


def main() -> int:
    rng = random.Random(int(sys.argv[1]))
    out = []
    for i in range(int(sys.argv[2])):
        messages = []
        for _ in range(rng.randint(1, 9)):
            role = rng.choice(ROLES)
            m = {"role": role, "content": rng.choice(CONTENTS)}
            if role == "assistant":
                reasoning = rng.choice(REASONINGS)
                if reasoning is not None:
                    m["reasoning_content"] = reasoning
                if rng.random() < 0.3:
                    m["tool_calls"] = [{"name": rng.choice(["get_weather", "search", "noargs"]),
                                        "arguments": rng.choice(ARGUMENTS)} for _ in range(rng.randint(1, 2))]
            messages.append(m)
        conv = {"name": f"r{i:04d}", "messages": messages}
        if rng.random() < 0.4:
            conv["tools"] = rng.sample(TOOLS, rng.randint(1, 3))
        if rng.random() < 0.25:
            conv["add_generation_prompt"] = False
        options = {}
        t = rng.random()
        if t < 0.33:
            options["enable_thinking"] = True
        elif t < 0.66:
            options["enable_thinking"] = False
        if rng.random() < 0.4:
            options["reasoning_effort"] = rng.choice(["low", "medium", "high", "xhigh", "max"])
        if rng.random() < 0.3:
            options["preserve_thinking"] = rng.random() < 0.5
        if options:
            conv["options"] = options
        out.append(conv)
    print(json.dumps({"about": "random conversations (random_conversations.py)", "conversations": out},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
