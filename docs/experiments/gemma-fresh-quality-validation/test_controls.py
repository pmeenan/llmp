#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiny admission/metric controls. No tokenizer, contexts or logit payloads."""
import hashlib
import importlib.util
import json
import pathlib
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result

judge = module('schedule_judge', HERE / 'analyze.py')
inputs = module('validation_inputs', HERE.parent / 'gemma-input-preparation/prepare.py')

def refuses(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError('invalid admission accepted')

assert judge.recipe(128)['max_rows'] == judge.recipe(1024)['max_rows'] == 1024
refuses(lambda: judge.recipe(256))
assert judge.nearest_p99(list(range(1024))) == 1013
assert judge.heapq.nlargest(2, range(3), key=[1, 1, 0].__getitem__) == [0, 1]
refuses(lambda: judge.nearest_p99([0] * 1023))
refuses(lambda: judge.nearest_p99([float('nan')] * 1024))
assert judge.quality.current_choices([0, 1], [1, 1]) == (1, 0, 0)
assert judge.quality.current_choices([0, 1], [2, 1]) == (1, 0, 1)
with tempfile.TemporaryDirectory() as temporary:
    root = pathlib.Path(temporary)
    path = root / 'receipt.json'; path.write_text('{"gate":1}')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert judge.authenticated(path, sha) == {'gate': 1}
    path.write_text('{"gate":2}')
    refuses(lambda: judge.authenticated(path, sha))
    path.write_text('not JSON')
    refuses(lambda: judge.authenticated(path, sha))  # SHA refuses before parse.
    manifest = dict(schema=1, rows=1024, scored_targets=1023, bos=2, bos_count=1,
                    input_bytes=4096, input_sha256='0'*64, vocab=262144, profile='26',
                    mode='literal', history_index=14)
    path.write_text(json.dumps(manifest)); sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert judge.manifest(path, sha)['history_index'] == 14
    for change in (dict(history_index=12), dict(profile='31'), dict(scored_targets=1024)):
        path.write_text(json.dumps(dict(manifest, **change)))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        refuses(lambda: judge.manifest(path, sha))
    corpus = b'a\n\n' * 400000
    slices = inputs.paragraphs(corpus, (13, 14))
    assert [x[0] for x in slices] == ['paragraph-13', 'paragraph-14']
    for _, text, offset, start, end in slices:
        assert offset < start < end and 32768 <= len(text) <= 65536
    refuses(lambda: inputs.paragraphs(b'no complete paragraphs', (13, 14)))
print('fixed capacity / nearest-rank metric / receipt replacement / heldout manifest / fixed text boundary: PASS')
