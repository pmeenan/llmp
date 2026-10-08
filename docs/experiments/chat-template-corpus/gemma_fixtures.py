#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Writes the Gemma chat fixtures (README.md, docs/tokenizer.md).

    gemma_fixtures.py --gemma4 FILE --gemma4-e FILE --gemma4-2604 FILE --gemma3 FILE --out DIR

Renders the cases below with transformers' render_jinja_template (as
collect.py render does: Jinja2's sandbox, trim_blocks, lstrip_blocks,
transformers' tojson and raise_exception; BOS `<bos>` and EOS `<eos>`;
messages as llmpalooza passes them, a tool call's string arguments parsed as
the client's JSON) on Gemma 4's template (google/gemma-4-31B-it), the
E2B/E4B checkpoints', Google's of 2026-04-28 (NVIDIA's NVFP4 checkpoints')
and Gemma 3's, and writes DIR/gemma-4.json, DIR/gemma-4-e.json,
DIR/gemma-4-2604.json and DIR/gemma-3.json: per case the text, or "invalid"
with the template's exception. chat_test compares the native renderers with
them; chat_template_test the interpreter on the Gemma 4 templates,
which are in tests/unit/data/chat/templates (Apache-2.0). Gemma 3's template
(Gemma terms) stays in the corpus directory, outside the repository.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import pathlib


def call(name, arguments):
    return {"name": name, "arguments": arguments}


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer", "description": 7}},
        "required": ["city"]}}}
# Every branch of the declaration notation: nested objects, arrays of
# objects with required keys, type lists, nullable, enums, an object
# without properties (its own keys, filtered), a response, keys out of
# case order and a description that is not a string.
SCHEMA = {"type": "function", "function": {
    "name": "plan_trip", "description": "Plan a trip <fast> & \"exact\".",
    "parameters": {"type": "object", "properties": {
        "Zeta": {"type": "string"},
        "alpha": {"type": "object", "properties": {"inner": {"type": "number", "nullable": True}},
                  "required": ["inner"]},
        "Beta": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string", "description": "who"}}, "required": ["name"], "maxItems": 3}},
        "gamma": {"type": ["string", "null"], "description": ["a", 1]},
        "delta": {"type": "array", "items": {"type": ["integer", "null"], "enum": [1, 2, None]}},
        "eps": {"type": "object", "additionalProperties": False, "description": "free form"},
        "mode": {"type": "string", "enum": ["fast", "slow"], "nullable": True},
        "plain": "not a schema",
        "count": {"type": "integer", "minimum": 0.5}},
        "required": ["Zeta", "alpha"]},
    "response": {"type": "object", "description": "The plan."}}}
NO_PARAMS = {"type": "function", "function": {"name": "noargs", "description": ""}}
ODD_PARAMS = {"type": "function", "function": {
    "name": "odd", "parameters": {"type": "object", "required": "ab", "properties": {}},
    "response": {"description": None, "type": "string"}}}

TOOL_ROUND = [
    {"role": "system", "content": "You can call tools."},
    {"role": "user", "content": "Weather in Paris and Tokyo?"},
    {"role": "assistant", "content": "", "reasoning_content": "Two calls.",
     "tool_calls": [call("get_weather", {"city": "Paris", "days": 3, "unit": "celsius"}),
                    call("get_weather", {"city": "東京", "Days": 3,
                                         "extra": {"nested": [1, 2.5, None, True, False, "x\ny"],
                                                   "B": {"z": -0.0, "a": 1e300, "big": 123456789012345678901234,
                                                         "f": 1e-07, "one": 1.0}}})]},
    {"role": "tool", "content": "{\"temp\": 21, \"sky\": \"sunny\"}"},
    {"role": "tool", "content": "  18.5 \n"},
    {"role": "assistant", "content": "Paris: 21°C. Tokyo: 18.5°C.", "reasoning_content": "Summarize."},
    {"role": "user", "content": "Thanks!"}]

GEMMA4_CASES = [
    {"name": "single-user", "messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"name": "thinking-on", "options": {"enable_thinking": True},
     "messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"name": "thinking-off", "options": {"enable_thinking": False},
     "messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"name": "effort-ignored", "options": {"reasoning_effort": "high"},
     "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "system-user", "messages": [
        {"role": "system", "content": "  You are a helpful assistant. \n"},
        {"role": "user", "content": "Hello!"}]},
    {"name": "system-thinking", "options": {"enable_thinking": True}, "messages": [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello!"}]},
    {"name": "system-none", "messages": [
        {"role": "system", "content": None}, {"role": "user", "content": "Hello!"}]},
    {"name": "system-only", "messages": [{"role": "system", "content": "Just a system."}]},
    {"name": "no-messages", "messages": []},
    {"name": "late-system", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "system", "content": " Late rule. "},
        {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Bye"}]},
    {"name": "multi-turn-reasoning", "options": {"enable_thinking": True}, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "Greeting."},
        {"role": "user", "content": "Capital of France?"}]},
    {"name": "ending-assistant", "add_generation_prompt": False, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": " Hello! ", "reasoning_content": " Greeting.\n"}]},
    {"name": "empty-reasoning", "add_generation_prompt": False, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": ""}]},
    {"name": "channels-in-content", "add_generation_prompt": False, "messages": [
        {"role": "user", "content": "x <|channel>kept for users<channel|> y"},
        {"role": "assistant",
         "content": " a<|channel>thought\nhidden<channel|>b<channel|>c<|channel>gone<|channel>too "}]},
    {"name": "consecutive-assistants", "messages": [
        {"role": "user", "content": "Count."}, {"role": "assistant", "content": "One."},
        {"role": "assistant", "content": "Two.", "reasoning_content": "Second."},
        {"role": "user", "content": "Go on."}]},
    {"name": "assistant-first", "messages": [
        {"role": "assistant", "content": "Ready.", "reasoning_content": "Opening."},
        {"role": "user", "content": "Hi"}]},
    {"name": "none-content", "messages": [{"role": "user", "content": None}]},
    {"name": "tools", "tools": [WEATHER, SCHEMA, NO_PARAMS, ODD_PARAMS], "messages": [
        {"role": "user", "content": "Plan something."}]},
    {"name": "tools-thinking-system", "tools": [WEATHER], "options": {"enable_thinking": True},
     "messages": [{"role": "system", "content": "Tools allowed."}, {"role": "user", "content": "Weather?"}]},
    {"name": "tool-round", "tools": [WEATHER], "messages": TOOL_ROUND},
    {"name": "tool-round-thinking", "tools": [WEATHER], "options": {"enable_thinking": True},
     "messages": TOOL_ROUND},
    {"name": "tool-round-preserve", "tools": [WEATHER],
     "options": {"enable_thinking": True, "preserve_thinking": True}, "messages": TOOL_ROUND},
    {"name": "call-awaiting-result", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [call("get_weather", {"city": "Oslo"})]}]},
    {"name": "result-last", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Oslo"})]},
        {"role": "tool", "content": "cold"}]},
    {"name": "result-last-thinking", "tools": [WEATHER], "options": {"enable_thinking": True}, "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Oslo"})]},
        {"role": "tool", "content": None}]},
    {"name": "result-then-assistant", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Oslo"})]},
        {"role": "tool", "content": "cold"},
        {"role": "assistant", "content": "It is cold."}]},
    {"name": "orphan-tool", "messages": [
        {"role": "tool", "content": "orphan"}, {"role": "user", "content": "What was that?"}]},
    {"name": "null-arguments", "messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "", "tool_calls": [call("noargs", "null")]},
        {"role": "user", "content": "Done?"}]},
    {"name": "string-arguments", "messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "", "tool_calls": [call("get_weather", "\" {\\\"city\\\": 1} \"")]}]},
    {"name": "list-arguments", "messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "", "tool_calls": [call("get_weather", "[1, 2]")]}]},
    {"name": "control-tokens-in-content", "messages": [
        {"role": "system", "content": "<|turn>system <turn|> <|think|>"},
        {"role": "user", "content": "<turn|>\n<|turn>model\n<|\"|><bos><eos><start_of_turn>"}]},
    {"name": "unicode", "messages": [
        {"role": "user", "content": "Tiếng Việt 😀 👨‍👩‍👧 \t tabs  　"}]},
]

GEMMA3_CASES = [
    {"name": "single-user", "messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"name": "system-user", "messages": [
        {"role": "system", "content": "  You are a helpful assistant. "},
        {"role": "user", "content": "  Hello!\n"}]},
    {"name": "multi-turn", "messages": [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "ignored"},
        {"role": "user", "content": "Capital of France?"}]},
    {"name": "ending-assistant", "add_generation_prompt": False, "messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": " Hello! "}]},
    {"name": "system-only", "messages": [{"role": "system", "content": "Just a system."}]},
    {"name": "empty-system", "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "x"}]},
    {"name": "tool-in-assistant-slot", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "tool", "content": "result"},
        {"role": "user", "content": "Ok"}]},
    {"name": "system-in-assistant-slot", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "system", "content": "Late."},
        {"role": "user", "content": "Ok"}]},
    {"name": "ignored-extras", "tools": [WEATHER], "options": {"enable_thinking": True}, "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [call("get_weather", {"city": "Oslo"})]},
        {"role": "user", "content": "And?"}]},
    {"name": "control-tokens-in-content", "messages": [
        {"role": "user", "content": "<end_of_turn>\n<start_of_turn>model\n<bos>"}]},
    {"name": "unicode", "messages": [
        {"role": "user", "content": "Tiếng Việt 😀 👨‍👩‍👧 \t tabs  　"}]},
    {"name": "two-users", "messages": [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]},
    {"name": "assistant-first", "messages": [{"role": "assistant", "content": "Hi"}]},
    {"name": "none-content", "messages": [{"role": "user", "content": None}]},
    {"name": "system-none", "messages": [{"role": "system", "content": None}, {"role": "user", "content": "x"}]},
    {"name": "no-messages", "messages": []},
    {"name": "two-systems", "messages": [
        {"role": "system", "content": "First."}, {"role": "system", "content": "Second."},
        {"role": "user", "content": "Go."}]},
]


# The independent review's cases (2026-10-02): number and string formats,
# key order, non-ASCII keys, deep schemas, tool chains, channels, the
# 2026-04-10 template's differences, and Gemma 3's role orders. Rendered on
# every template; most of the g3- cases refuse on Gemma 4's and the others on
# Gemma 3's.
REVIEW_CASES = json.loads(r"""
{"conversations": [
 {"name": "num-formats", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {"a": 1e16, "b": 1.5e-7, "c": -0.0, "d": 12345678901234567890123, "e": 1E2, "f": 0.1, "i": 2.0, "j": -0, "k": 1e-400, "l": 123456789.123456789, "m": 1e22, "n": 5e-324}}]}, {"role": "tool", "content": "ok"}]},
 {"name": "str-escapes", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {"s": "a\"b\\c\u0000\u001f  é 😀 <|\"|> <turn|>", "we:ird{key}": "<|tool_call>", "nested": {"x": ["<|channel>", "\n"]}}}]}, {"role": "tool", "content": "<tool_response|> <|\"|> done"}]},
 {"name": "bools-nulls", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": null, "tool_calls": [{"name": "f", "arguments": {"a": [true, false, null, [], {}], "b": {"c": null}, "c": null}}]}, {"role": "tool", "content": null}, {"role": "tool", "content": ""}]},
 {"name": "key-case-order", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x", "tool_calls": [{"name": "f", "arguments": {"b": 1, "B": 2, "a": 3, "A": 4, "_x": 5, "Z": 6, "[": 7, "{": 8, "~": 9}}]}]},
 {"name": "keys-nonascii-lower", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {"città": 1, "zeta": 2, "город": 3, "东京": 4}}]}, {"role": "tool", "content": "r"}]},
 {"name": "tool-props-nonascii-lower", "tools": [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {"città": {"type": "string"}, "zeta": {"type": "string"}}}}}], "messages": [{"role": "user", "content": "q"}]},
 {"name": "schema-deep", "tools": [{"type": "function", "function": {"name": "deep", "description": "D <|\"|> <turn|>", "parameters": {"type": "object", "properties": {
    "n": {"type": "integer", "default": 3, "description": "with default"},
    "tl": {"type": ["string", "null"]},
    "up": {"type": "STRING", "enum": ["a", 1, null, true, 2.5]},
    "notype": {"description": "x"},
    "obj": {"type": "object", "additionalProperties": {"type": "string"}, "x-extra": {"type": "integer", "description": "inner"}, "nullable": true, "required": ["x-extra"]},
    "arr": {"type": "array", "items": {"type": "array", "items": {"type": "string", "enum": ["p", "q"]}, "description": "inner array", "minItems": 1}},
    "arrobj": {"type": "array", "nullable": true, "description": "objs", "items": {"type": "object", "properties": {"k": {"type": "object", "properties": {"z": {"type": "boolean"}}}}, "required": ["k"], "additionalProperties": false}},
    "anyof": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    "desc_obj": {"type": "string", "description": {"a": "b", "c": [1, 2.0]}},
    "desc_num": {"type": "number", "description": 1.0},
    "enum_obj": {"type": "string", "enum": {"Z": 1, "a": [null]}},
    "req_str": {"type": "object", "properties": {"a": {"type": "string"}}, "required": "ab"}
  }, "required": ["n", "tl"]}, "response": {"type": "object", "description": "resp"}}},
   {"type": "function", "function": {"name": "typeonly", "description": "", "parameters": {"type": "object"}}},
   {"type": "function", "function": {"name": "notype", "description": "nt", "parameters": {"properties": {"a": {"type": "string"}}}}},
   {"type": "function", "function": {"name": "uni_名前", "description": "ünï", "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "€"}}, "required": []}, "response": {"type": "string"}}}],
  "messages": [{"role": "system", "content": "  sys  "}, {"role": "user", "content": "q"}]},
 {"name": "chain-tools", "options": {"enable_thinking": true, "preserve_thinking": true}, "tools": [{"type": "function", "function": {"name": "f", "description": "d"}}], "messages": [
   {"role": "user", "content": "a"},
   {"role": "assistant", "content": "pre", "reasoning_content": "r1", "tool_calls": [{"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": {}}]},
   {"role": "tool", "content": "t1"}, {"role": "tool", "content": "t2"},
   {"role": "assistant", "content": "mid", "reasoning_content": "r2"},
   {"role": "assistant", "content": "", "reasoning_content": "r3", "tool_calls": [{"name": "f", "arguments": {"y": null}}]},
   {"role": "tool", "content": "t3"},
   {"role": "user", "content": "b"},
   {"role": "assistant", "content": "", "reasoning_content": "r4", "tool_calls": [{"name": "f", "arguments": {}}]},
   {"role": "tool", "content": "t4"}]},
 {"name": "chain-tools-nothink", "tools": [{"type": "function", "function": {"name": "f", "description": "d"}}], "messages": [
   {"role": "user", "content": "a"},
   {"role": "assistant", "content": "pre", "reasoning_content": "r1", "tool_calls": [{"name": "f", "arguments": {"x": 1}}]},
   {"role": "tool", "content": "t1"},
   {"role": "assistant", "content": "  ", "reasoning_content": "r2", "tool_calls": [{"name": "f", "arguments": {}}]}]},
 {"name": "call-no-result-content", "add_generation_prompt": false, "messages": [
   {"role": "user", "content": "a"},
   {"role": "assistant", "content": "visible <|channel>hidden<channel|>", "tool_calls": [{"name": "f", "arguments": {}}]}]},
 {"name": "result-with-content-last", "options": {"enable_thinking": true}, "messages": [
   {"role": "user", "content": "a"},
   {"role": "assistant", "content": "done", "tool_calls": [{"name": "f", "arguments": {}}]},
   {"role": "tool", "content": "x"}]},
 {"name": "channels", "add_generation_prompt": false, "messages": [
   {"role": "user", "content": " <|channel>u<channel|> "},
   {"role": "assistant", "content": "<channel|>a"},
   {"role": "user", "content": "x"},
   {"role": "assistant", "content": "<|channel>never closed"},
   {"role": "user", "content": "y"},
   {"role": "assistant", "content": "x<channel|>y<channel|>z<|channel>w"},
   {"role": "user", "content": "z"},
   {"role": "assistant", "content": " 　 <|channel>t<channel|> ans  ", "reasoning_content": "\n r \n"}]},
 {"name": "whitespace-unicode", "messages": [{"role": "system", "content": "　sys \u0085"}, {"role": "user", "content": "\u001chi\u001f "}]},
 {"name": "late-system-tools", "tools": [{"type": "function", "function": {"name": "f", "description": "d"}}], "messages": [{"role": "user", "content": "a"}, {"role": "system", "content": " s "}, {"role": "assistant", "content": "b", "reasoning_content": "rr"}, {"role": "system", "content": null}]},
 {"name": "only-tools-gen", "tools": [{"type": "function", "function": {"name": "f", "description": "d"}}], "messages": []},
 {"name": "only-system-thinking", "options": {"enable_thinking": true}, "messages": [{"role": "system", "content": null}]},
 {"name": "tool-first", "messages": [{"role": "tool", "content": "orphan"}, {"role": "assistant", "content": "a", "tool_calls": [{"name": "f", "arguments": {}}]}, {"role": "user", "content": "u"}]},
 {"name": "assistant-calls-then-user", "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {}}]}, {"role": "user", "content": "b"}]},
 {"name": "consecutive-asst-end", "add_generation_prompt": false, "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"}, {"role": "assistant", "content": "y"}]},
 {"name": "consecutive-asst-gen", "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x", "tool_calls": [{"name": "f", "arguments": {}}]}, {"role": "tool", "content": "t"}, {"role": "assistant", "content": "y"}]},
 {"name": "g3-system-ws", "messages": [{"role": "system", "content": "  keep  \n"}, {"role": "user", "content": "  u  "}, {"role": "assistant", "content": " a "}, {"role": "tool", "content": " t "}, {"role": "system", "content": "s2"}, {"role": "user", "content": "u2"}]},
 {"name": "g3-empty-system", "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "u"}]},
 {"name": "g3-system-assistant", "messages": [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]},
 {"name": "g3-asst-calls", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {"a": 1}}]}, {"role": "user", "content": "v"}]},
 {"name": "param-named-type", "tools": [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {"type": {"type": "string"}, "description": {"type": "string", "description": "a description"}, "city": {"type": "string"}}, "required": ["type"]}}}], "messages": [{"role": "user", "content": "q"}]},
 {"name": "ws-content-after-results", "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "  ", "tool_calls": [{"name": "f", "arguments": {}}]}, {"role": "tool", "content": "t"}, {"role": "user", "content": "b"}]},
 {"name": "g3-asst-none", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": null, "tool_calls": [{"name": "f", "arguments": {"a": 1}}]}, {"role": "user", "content": "v"}]}
]}
""")["conversations"]
# The second review's: case mapping beyond ASCII (Cyrillic upper-cased, keys
# that sort differently once lower-cased in full, `ß` upper-cased to `SS`),
# reasoning kept or not.
REVIEW_CASES += [
    {"name": "fb-upper-type", "tools": [{"type": "function", "function": {
        "name": "t", "description": "d", "parameters": {"type": "object", "properties": {"x": {"type": "строка"}}}}}],
     "messages": [{"role": "user", "content": "q"}]},
    {"name": "fb-dictsort-order", "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [call("f", {"Äb": 1, "ä": 2})]},
        {"role": "tool", "content": "r"}]},
    {"name": "fb-items-type", "tools": [{"type": "function", "function": {
        "name": "t", "description": "d", "parameters": {"type": "object", "properties": {
            "x": {"type": "array", "items": {"type": ["ß", "null"]}}}}}}],
     "messages": [{"role": "user", "content": "q"}]},
    {"name": "fb-preserve", "options": {"enable_thinking": True, "preserve_thinking": True}, "messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r1"},
        {"role": "user", "content": "c"}]},
    {"name": "fb-user-reasoning", "messages": [
        {"role": "user", "content": "a", "reasoning_content": "ur"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}]},
    {"name": "case-mapping", "tools": [{"type": "function", "function": {
        "name": "t", "description": "d", "parameters": {"type": "object", "properties": {
            "Ωmega": {"type": "ǆ"}, "ωmega": {"type": "straße"}, "İd": {"type": "ﬃ"}}}}}],
     "messages": [{"role": "user", "content": "q"},
                  {"role": "assistant", "content": "", "tool_calls": [call("f", {"ΟΔΟΣ": 1, "οδος": 2, "Σ": 3})]},
                  {"role": "tool", "content": "r"}]},
]
GEMMA4_CASES += REVIEW_CASES
GEMMA3_CASES += REVIEW_CASES


def template_messages(case):
    """The messages as llmpalooza hands a template (chat/template.cc); string
    arguments are the client's JSON, which llmpalooza parses."""
    out = []
    for m in case["messages"]:
        msg = {"role": m["role"], "content": m.get("content")}
        if "reasoning_content" in m:
            msg["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            msg["tool_calls"] = [{"type": "function", "function": {
                "name": c["name"],
                "arguments": json.loads(c["arguments"]) if isinstance(c["arguments"], str) else c["arguments"]}}
                for c in m["tool_calls"]]
        out.append(msg)
    return out


def write_json(path: pathlib.Path, value) -> None:
    """Each case on its own line (tokenizer-reference/generate.py's form)."""
    lines = ["{"]
    keys = list(value)
    for i, key in enumerate(keys):
        comma = "," if i + 1 < len(keys) else ""
        v = value[key]
        if isinstance(v, list):
            lines.append(f"  {json.dumps(key)}: [")
            for j, element in enumerate(v):
                lines.append("    " + json.dumps(element, ensure_ascii=False) + ("," if j + 1 < len(v) else ""))
            lines.append("  ]" + comma)
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(v, ensure_ascii=False)}{comma}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_name(path.name + ".license").write_text(
        "SPDX-FileCopyrightText: 2026 llmpalooza contributors\nSPDX-License-Identifier: Apache-2.0\n")


def fixture(label: str, template_path: pathlib.Path, cases, out: pathlib.Path) -> None:
    import transformers
    from transformers.utils import chat_template_utils

    data = template_path.read_bytes()
    template = data.decode("utf-8")
    entries = []
    for case in cases:
        entry = {k: case[k] for k in ("name", "messages", "tools", "options", "add_generation_prompt") if k in case}
        kwargs = dict(case.get("options", {}))
        if "enable_thinking" in kwargs:
            kwargs["thinking"] = kwargs["enable_thinking"]  # as llmpalooza passes it
        try:
            rendered, _ = chat_template_utils.render_jinja_template(
                conversations=[template_messages(case)], tools=case.get("tools"), chat_template=template,
                add_generation_prompt=case.get("add_generation_prompt", True), bos_token="<bos>",
                eos_token="<eos>", **kwargs)
            entry["text"] = rendered[0]
        except Exception as e:  # noqa: BLE001 (the template's own exceptions are results)
            entry["error"] = "invalid"
            entry["reference_error"] = f"{type(e).__name__}: {e}"[:300]
        entries.append(entry)
    ok = sum("text" in e for e in entries)
    print(f"{label}: {ok}/{len(entries)} rendered")
    write_json(out / f"{label}.json", {
        "template": label,
        "template_sha256": hashlib.sha256(data).hexdigest(),
        "renderer": f"transformers {transformers.__version__} render_jinja_template",
        "cases": entries,
    })


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gemma4", required=True)
    p.add_argument("--gemma4-e", required=True)
    p.add_argument("--gemma4-2604", required=True)
    p.add_argument("--gemma3", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    fixture("gemma-4", pathlib.Path(args.gemma4), GEMMA4_CASES, out)
    fixture("gemma-4-e", pathlib.Path(args.gemma4_e), GEMMA4_CASES, out)
    fixture("gemma-4-2604", pathlib.Path(args.gemma4_2604), GEMMA4_CASES, out)
    fixture("gemma-3", pathlib.Path(args.gemma3), GEMMA3_CASES, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
