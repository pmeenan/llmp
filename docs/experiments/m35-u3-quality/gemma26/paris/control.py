#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen Paris control data and validation; no model implementation."""
import sys
sys.dont_write_bytecode = True

import hashlib
import importlib
import json
import math
from pathlib import Path
import re

INPUT_SHA = '3d0e98600de96f8a96f7bab3655d85b9d2e974c6e39b768d092e6fa1961069ec'
RAW_REFERENCE_SHA = '847191af80ee1cead7979e06d825fc6460e1e73a408e6a415c48c9bc2c28d7ef'
RAW_LAUNCH_SHA = '1b30eec5096cf3383bf3f3b410b1ea75a6279594c7abcfb83760975803a66325'
SERVICE_SHA = '0f1135a2283fe63797e15c8dffe48c9df39b0baae91a57f925b5ba21c3e3252e'
RUNTIME_SHA = 'd18c7b7a9c8c6cf383a6f607a8606d2ce2026c4b2a8cb47632ec424601704dff'
RUNTIME_SOURCE_SHA = 'd0c8f0023776b0994de9e97beff04cb7bcd3e77ddad78bf87390bb312794cff4'
ARTIFACT = '1d20ce879b2fe72c947699357da5e20d3f6388e9cc7858b9eb6fcc2516cd3875'
ARTIFACT_METADATA = {
    'manifest.json': 'aeedabdf082a32883e9c197a7a6347135ac0cdfb190aacca833003bc26fababa',
    'index.json': '470041519492eae2e8c084b8b1ed6182a8bd72d60da42d326c4c102f542ffc2a',
    'spec.json': '7e163fda1b1e528b38f1c1f65d905de8a08acc44dea5ca452304e80760525d91',
}
PORT = 18976
STOP_IDS = [1, 106, 50]
TEMPLATE_SHA = '94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413'
LOADER_SHA = '61732d66793a85ca140cb090a2c8aecad23da141eb4980f3cb8767953f3127a1'
STAGE_SHA = '9fb846ebf68691cd63465c999d0a2e4ca9dd64037b09c2998bae230370a52593'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(path, digest):
    got = sha(path)
    require(got == digest, f'identity changed: {path}: {got}')
    return got


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def inputs(path=None):
    path = Path(path) if path else Path(__file__).with_name('input.json')
    checked(path, INPUT_SHA)
    value = json.loads(path.read_bytes())
    ids = value['prompt_token_ids']
    require(len(ids) == 35 and ids[0] == 2 and ids.count(2) == 1 and
            all(type(i) is int and i >= 0 for i in ids), 'invalid frozen input')
    return value


def registration(value):
    require(re.fullmatch(r'[0-9a-f]{40}', value) is not None,
            'supply the full reviewed GPU amendment commit, not the input-freeze commit')
    require(not value.startswith('03655eaf'), 'input-freeze commit does not register GPU collection')
    return value


def raw_reference():
    parent = Path(__file__).resolve().parent.parent
    checked(parent / 'reference.py', RAW_REFERENCE_SHA)
    sys.path.insert(0, str(parent))
    raw = importlib.import_module('reference')
    require(Path(raw.__file__).resolve() == parent / 'reference.py', 'wrong reference module')
    return raw


def raw_launcher():
    raw_reference()
    parent = Path(__file__).resolve().parent.parent
    checked(parent / 'launch.py', RAW_LAUNCH_SHA)
    launch = importlib.import_module('launch')
    require(Path(launch.__file__).resolve() == parent / 'launch.py', 'wrong lifecycle module')
    return launch


def body():
    return dict(model='gemma26', prompt=inputs()['prompt_token_ids'], max_tokens=64,
                temperature=0, top_p=1, add_special_tokens=False, stream=False,
                logprobs=1, return_tokens_as_token_ids=True)


def verdict(text, finish):
    return 'pass' if text.strip() == 'Paris' and finish == 'stop' else 'fail'


def native_output(obj):
    require(type(obj.get('choices')) is list and len(obj['choices']) == 1, 'expected one choice')
    choice = obj['choices'][0]
    require(type(choice.get('text')) is str, 'missing reply text')
    finish = choice.get('finish_reason')
    require(finish in ('stop', 'length'), 'unexpected finish reason')
    usage = obj.get('usage', {})
    require(all(type(usage.get(k)) is int for k in
                ('prompt_tokens', 'completion_tokens', 'total_tokens')), 'missing token counts')
    n = usage['completion_tokens']
    require(usage['prompt_tokens'] == 35 and 0 < n <= 64 and
            usage['total_tokens'] == 35 + n, 'token accounting mismatch')
    require(finish != 'length' or n == 64, 'early length stop')
    require(usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0,
            'fresh request unexpectedly reused prompt tokens')
    scores = choice.get('logprobs')
    require(type(scores) is dict, 'missing generated scores')
    tokens, losses = scores.get('tokens'), scores.get('token_logprobs')
    require(type(tokens) is list and type(losses) is list and len(tokens) == len(losses) == n,
            'selected-token score count mismatch')
    require(all(type(v) in (float, int) and math.isfinite(v) and v <= 0 for v in losses),
            'nonfinite, positive or missing selected-token score')
    require(all(type(t) is str and re.fullmatch(r'token_id:[0-9]+', t) for t in tokens),
            'completion did not return exact generated token IDs')
    ids = [int(t.split(':')[1]) for t in tokens]
    require(not any(i in STOP_IDS for i in ids[:-1]), 'continued past a registered stop')
    require((finish == 'stop') == (ids[-1] in STOP_IDS), 'unexpected native stop contract')
    return dict(prompt_tokens=35, output_token_ids=ids, completion_tokens=n,
                text=choice['text'], finish_reason=finish, selected_token_logprobs=losses,
                semantic_verdict=verdict(choice['text'], finish))
