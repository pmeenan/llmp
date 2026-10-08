#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Check closed CLI refusal before provider work and reviewed comparison controls."""
import os
import pathlib
import subprocess
import sys
import tempfile

import prepare

if len(sys.argv) != 2:
    raise SystemExit('test_controls.py NATIVE_BINARY')
binary = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory(prefix='gemma26-compound-cli-') as directory:
    root = pathlib.Path(directory)
    ids = root / 'ids.i32'
    ids.write_bytes((2).to_bytes(4, 'little') * 1024)
    base = ['nonexistent-artifact', str(root / 'out'), '26', '4', 'joined', 'norm', str(ids)]
    cases = [([], {'LLMP_GEMMA26_PACKED_C4': '1'})]
    for index, bad in [(2, '31'), (3, '2'), (4, 'scalar'), (5, 'ordinary'), (5, 'rows-norm')]:
        args = base.copy()
        args[index] = bad
        cases.append((args, {'LLMP_GEMMA26_PACKED_C4': '1'}))
    for flag in ('0', 'bad'):
        cases.append((base, {'LLMP_GEMMA26_PACKED_C4': flag}))
    cases.append((base, {}))
    truncated = root / 'truncated.i32'
    truncated.write_bytes(ids.read_bytes()[:-1])
    cases.append((base[:-1] + [str(truncated)], {'LLMP_GEMMA26_PACKED_C4': '1'}))
    for args, override in cases:
        env = os.environ.copy()
        env.pop('LLMP_GEMMA26_PACKED_C4', None)
        env.update(override)
        result = subprocess.run([str(binary), *args], env=env, capture_output=True, timeout=10)
        if result.returncode != 2 or result.stdout or result.stderr or (root / 'out').exists():
            raise ValueError('closed CLI did not refuse before output/provider work')
    print('COMPOUND_CLI_CONTROLS 10 refusals PASS')
    large = root / 'sparse-native-identity'
    with large.open('wb') as file:
        file.truncate((128 << 20) + 1)
    try:
        prepare.record(large)
        raise AssertionError('source identity limit accepted oversized file')
    except ValueError:
        pass
    expected = subprocess.check_output(['sha256sum', str(large)], text=True).split()[0]
    identity = prepare.record(large, 256 << 20)
    if identity['bytes'] != (128 << 20) + 1 or identity['sha256'] != expected:
        raise ValueError('large native binary streaming identity mismatch')
    with large.open('wb') as file:
        file.truncate((256 << 20) + 1)
    try:
        prepare.record(large, 256 << 20)
        raise AssertionError('binary identity cap accepted oversized file')
    except ValueError:
        pass
    print('COMPOUND_BINARY_BOUNDARY sparse >128MiB / <=256MiB / >256MiB PASS')
subprocess.run([sys.executable, '-B', str(pathlib.Path(__file__).with_name('compare.py')),
                '--selftest'], check=True)
