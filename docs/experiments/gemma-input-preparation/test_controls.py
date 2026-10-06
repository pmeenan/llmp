#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiny input/manifest controls only; no tokenizer, model, or quality outputs."""
import hashlib
import importlib.util
import json
import pathlib
import struct
import tempfile
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

quality = module('quality_inputs', HERE.parent / 'gemma-quality/analyze.py')
prepare = module('fresh_inputs', HERE / 'prepare.py')

def refuses(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError('invalid input accepted')

with tempfile.TemporaryDirectory() as temporary:
    root = pathlib.Path(temporary)
    (root / 'input1').mkdir()
    raw = struct.pack('<1024i', 2, *([100] * 1023))
    digest = hashlib.sha256(raw).hexdigest()
    (root / 'input1/ids.i32').write_bytes(raw)
    manifest = dict(schema=1, rows=1024, scored_targets=1023, bos=2, bos_count=1,
                    input_bytes=4096, input_sha256=digest, vocab=262144, profile='31')
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert quality.current_input() == (quality.ID_SHA, None)
    refuses(lambda: quality.current_ids(root))  # Default old b2d7 pin is mandatory.
    assert quality.current_input(path, sha) == (digest, sha)
    assert len(quality.current_ids(root, digest, True)) == 1024
    refuses(lambda: quality.current_input(path, sha, '26'))
    path.write_text(json.dumps(dict(manifest, input_sha256='0'*64)))
    refuses(lambda: quality.current_input(path, sha))  # Replaced self-consistent manifest.
    path.write_text(json.dumps(dict(manifest, scored_targets=1024)))
    wrong_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    refuses(lambda: quality.current_input(path, wrong_sha))  # No target after final row.
    repeated = struct.pack('<1024i', 2, 2, *([100] * 1022))
    (root / 'input1/ids.i32').write_bytes(repeated)
    refuses(lambda: quality.current_ids(root, hashlib.sha256(repeated).hexdigest(), True))
    refuses(lambda: prepare.ids(repeated))
    (root / 'input1/ids.i32').write_bytes(raw + struct.pack('<i', 2))
    refuses(lambda: quality.current_ids(root, hashlib.sha256(raw).hexdigest(), True))
    if len(sys.argv) == 2:
        binary = pathlib.Path(sys.argv[1]).resolve()
        invalid = root / 'invalid.gguf'
        invalid.write_bytes(b'not GGUF')
        text = root / 'text.txt'
        text.write_text('synthetic input')
        oversized = root / 'oversized.gguf'
        with oversized.open('wb') as file:
            file.truncate((32 << 20) + 1)
        for arguments in ([], [str(invalid), str(text), str(root/'out'), 'bad-mode'],
                          [str(root/'missing'), str(text), str(root/'out'), 'literal'],
                          [str(oversized), str(text), str(root/'out'), 'literal'],
                          [str(invalid), str(text), str(root/'out'), 'literal']):
            result = subprocess.run([str(binary), *arguments], capture_output=True, timeout=5)
            assert result.returncode == 1 and not (root/'out').exists()
    else:
        assert len(sys.argv) == 1
    assert len(prepare.ids(raw)) == 1024
    refuses(lambda: prepare.ids(struct.pack('<i', -1)))
print('fresh manifest pin/default refusal/ONE BOS/final-row boundary controls: PASS')
