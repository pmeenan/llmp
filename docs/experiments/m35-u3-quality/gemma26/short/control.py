#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen short-score contracts; reuse authenticated identity and target validators."""
import sys
sys.dont_write_bytecode = True

import hashlib
import importlib.util
import json
from pathlib import Path

INPUT_SHA = '389216b176c694cd5f43e3068b20d6fad97e09cbd956bcd6c2837d6b400858b2'
SOURCE_HASHES_SHA = '80ad88e3b7f01de67e86f72c2ec59e3fdac6ec108ee0961bf434b1b6efe5a9b4'
PARIS_CONTROL_SHA = 'e01815d98d1395d631847211275e771ac208be7593689edba790564f00294ce9'
QWEN_CONTRACT_SHA = '697485e5f71da2d2bafd77f564d799cba410a10252f53bdf6de6cc13bf121e17'
PAIR_TOOL_SHA = '9bf1d1597a8d6e97d65a648a9db68d8a51c694cc3772a6f01ba158fc420137bd'
PAIR_PREREG_SHA = 'f9a06071f0c7f75cf4472f6ad80a0125db4fae418cc460c9a1f9d42f4b27dbd9'
HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(path, digest):
    got = sha(path)
    require(got == digest, f'identity changed: {path}')
    return got


def module(path, digest, name):
    checked(path, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


paris = module(HERE.parent / 'paris/control.py', PARIS_CONTROL_SHA, 'paris_control')
targets = module(HERE.parent.parent / 'qwen27/contract.py', QWEN_CONTRACT_SHA, 'qwen_contract')
encoded = paris.encoded
raw_reference = paris.raw_reference
raw_launcher = paris.raw_launcher
RUNTIME_SHA, RUNTIME_SOURCE_SHA = paris.RUNTIME_SHA, paris.RUNTIME_SOURCE_SHA
SERVICE_SHA = paris.SERVICE_SHA
ARTIFACT, ARTIFACT_METADATA = paris.ARTIFACT, paris.ARTIFACT_METADATA
TEMPLATE_SHA, LOADER_SHA, STAGE_SHA = paris.TEMPLATE_SHA, paris.LOADER_SHA, paris.STAGE_SHA
PORT = 18977


def inputs():
    path = HERE / 'ids.json'
    checked(path, INPUT_SHA)
    ids = json.loads(path.read_bytes())
    targets.valid_ids(ids, 262144, count=1024)
    require(ids[0] == 2 and ids.count(2) == 1, 'BOS must appear exactly once')
    return ids


def registration(value):
    paris.registration(value)
    require(not value.startswith(('3202f46f', '24846f55')),
            'Paris/input-freeze registration does not register likelihood')
    checked(HERE / 'pair-prereg.json', PAIR_PREREG_SHA)
    return value


def source_check():
    path = HERE / 'source-hashes.json'
    checked(path, SOURCE_HASHES_SHA)
    expected = json.loads(path.read_bytes())
    spec = importlib.util.find_spec('vllm')
    require(spec is not None and spec.origin is not None, 'no installed vLLM source')
    root = Path(spec.origin).parent
    actual = {name: checked(root / name, digest) for name, digest in expected.items()}
    require(len(actual) == 37, 'source inventory changed')
    return actual


def body():
    return dict(model='gemma26', prompt=inputs(), max_tokens=0, temperature=0,
                top_p=1, add_special_tokens=False, stream=False, echo=True,
                prompt_logprobs=1, logprobs=1, return_tokens_as_token_ids=True)


def native_output(obj):
    ids = inputs()
    choices = obj.get('choices')
    require(not obj.get('error') and isinstance(choices, list) and len(choices) == 1,
            'invalid native completion')
    choice = choices[0]
    scores = choice.get('logprobs')
    require(isinstance(scores, dict) and
            scores.get('tokens') == [f'token_id:{i}' for i in ids],
            'complete native echoed prompt IDs differ or are missing')
    usage = obj.get('usage', {})
    require(all(type(usage.get(k)) is int for k in
                ('prompt_tokens', 'completion_tokens', 'total_tokens')) and
            usage['prompt_tokens'] == usage['total_tokens'] == 1024 and
            usage['completion_tokens'] == 0, 'native score-only usage differs')
    require(usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0,
            'fresh score request reused prefix state')
    require(choice.get('finish_reason') in ('stop', 'length'), 'invalid score finish')
    # This validator's optional prompt_token_ids fallback is not our ID proof:
    # complete echoed literal token labels were independently required above.
    return targets.native_records(obj, ids, 1, 0)


def source_output(outputs):
    ids = inputs()
    require(len(outputs) == 1, 'expected one source request')
    request = outputs[0]
    require(list(request.prompt_token_ids) == ids and
            all(type(i) is int for i in request.prompt_token_ids) and request.finished and
            len(request.outputs) == 1, 'source prompt IDs/count or completion differs')
    rows = request.prompt_logprobs
    require(isinstance(rows, list) and len(rows) == 1024 and rows[0] is None,
            'source teacher row geometry differs')
    capture = targets.ActualTargets(ids, 1, 0)
    for pos in range(1, len(ids)):
        row = rows[pos]
        entry = row.get(ids[pos]) if isinstance(row, dict) else None
        value = getattr(entry, 'logprob', None)
        capture.add([pos], [ids[pos]], [value], [[]], [[]])
    unused = request.outputs[0]
    require(len(unused.token_ids) == 1 and type(unused.token_ids[0]) is int and
            0 <= unused.token_ids[0] < 262144 and unused.finish_reason in ('stop', 'length'),
            'source required one-token decode differs')
    return capture.finish(), dict(token_ids=list(unused.token_ids), text=unused.text,
                                  finish_reason=unused.finish_reason,
                                  stop_reason=unused.stop_reason, used_as_teacher=False)


def write_losses(out, rows):
    require(len(rows) == 1023, 'wrong scored target count')
    # Compact JSONL is required: encoded() intentionally pretty-prints other receipts.
    data = b''.join(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                    + b'\n' for row in rows)
    (out / 'losses.jsonl').write_bytes(data)
    return dict(targets=len(rows), losses_sha256=sha(out / 'losses.jsonl'))
