#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""The two original Gemma launchers remain pinned and compiled with GGML flags."""
import hashlib
import json
import pathlib
import shlex
import sys

build = pathlib.Path(sys.argv[1])
receipt = json.loads((build / 'jitllm-receipt.json').read_text())
ggml = next(c for c in receipt['components'] if c['id'] == 'ggml')
root = pathlib.Path(ggml['source']) / 'ggml/src/ggml-cuda'
expected = {
    'topk-moe.cu': 'fdf688136cbf728b825c6a62534855a53e64cb73acb7f3dd3a3ab1b161456199',
    'topk-moe.cuh': 'cc1cec0c9b865624be058697b9a21048cfdaac0aca547bc6f6b18d54d820c88b',
    'moe-weighted-reduction.cu': 'c05a68615007e34ffbbd8a5f381a8f915c2cc0d38cd859351c8535ddb513c1d0',
    'moe-weighted-reduction.cuh': '89b9a6138bbc20ca12a82cc082a3b38da94cd83c2f41367ef6f8a41abd044638',
}
for name, digest in expected.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
assert ggml['version'] == 'b10964' and ggml['license'] == 'MIT' and not ggml['modified']
commands = json.loads((build / 'compile_commands.json').read_text())
for name in ('topk-moe.cu', 'moe-weighted-reduction.cu'):
    actual = [c for c in commands if pathlib.Path(c['file']).resolve() == (root / name).resolve()]
    assert len(actual) == int(receipt['cuda']), (name, len(actual))
    for command in actual:
        flags = shlex.split(command['command'])
        assert '-use_fast_math' in flags and '-extended-lambda' in flags, name
        assert 'GGML_JITLLM' in command['command'], name
print('Gemma MoE pinned original source and compile inventory: PASS')
