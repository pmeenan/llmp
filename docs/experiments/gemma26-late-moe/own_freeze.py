#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze native operators without reading any counterpart output bytes."""
import json
import math
import os
import pathlib
import runpy
import shlex
import struct
import sys

snapshot = runpy.run_path(str(pathlib.Path(__file__).with_name('source_snapshot.py')))
identity, admitted_receipts = snapshot['record'], snapshot['admitted_receipts']
success = runpy.run_path(str(pathlib.Path(__file__).with_name('official.py')))['success']


def valid(data, route):
    size = 4096 if route else 720896
    if len(data) != size:
        raise ValueError('incomplete whole operator output')
    if route:
        if any(not 0 <= value < 128 for (value,) in struct.iter_unpack('<i', data[:2048])):
            raise ValueError('noncanonical selected IDs')
        data = data[2048:]
    if any(not math.isfinite(value) for (value,) in struct.iter_unpack('<f', data)):
        raise ValueError('nonfinite output')


def main():
    if len(sys.argv) != 13:
        raise ValueError('PRE CAPTURE_RECEIPT INPUTS kind PRIMITIVE_FIRST PRIMITIVE_REPEAT FUSED_FIRST FUSED_REPEAT JOB_NAME NEW_FREEZE EXPECTED_PRE_SHA EXPECTED_CAPTURE_SHA')
    pre_path, capture_path, inputs = map(pathlib.Path, sys.argv[1:4])
    kind = sys.argv[4]
    if kind not in ('route', 'reduce'):
        raise ValueError('closed kind')
    roots = list(map(pathlib.Path, sys.argv[5:9]))
    job, output = sys.argv[9], pathlib.Path(sys.argv[10])
    receipts, admissions = admitted_receipts({'pre': (pre_path, sys.argv[11]),
                                              'capture': (capture_path, sys.argv[12])})
    pre = receipts['pre']
    if receipts['capture']['pre_sha256'] != admissions['pre']['sha256']:
        raise ValueError('capture/source chronology differs')
    release = json.loads((inputs / 'release.json').read_text())
    if release['kind'] != kind or release['capture_receipt'] != identity(capture_path) or \
            identity(inputs / 'inputs.tsv') != release['manifest']:
        raise ValueError('wrong authenticated input release')
    official, steps = success(job, 4)
    manifest_sha = release['manifest']['sha256']
    for step, root, policy in zip(steps, roots, ('primitive', 'primitive', 'fused', 'fused')):
        command = shlex.split(step['cmd'])
        expected = [str(inputs), manifest_sha, str(root), kind, policy]
        if len(command) != 6 or command[1:] != expected or \
                not command[0].endswith('/benchmarks/jitllm_gemma26_late_moe'):
            raise ValueError('actual native acquisition command/input/policy differs')
    log_path = pathlib.Path.home() / '.local/share/jitllm/jobs' / job / 'log'
    log = log_path.read_text()
    if log.splitlines().count('LATE_NATIVE_RETIRED complete=1') != 4:
        raise ValueError('missing native owner retirement markers')
    source = pathlib.Path.home() / 'src/jitLLM-wt/m3fixb'
    for name, expected in pre['source_files'].items():
        if identity(source / name, 32 << 20) != expected:
            raise ValueError('source changed before calibration freeze')
    if identity(source / 'build/spark-native/benchmarks/jitllm_gemma26_late_moe') != pre['native_operator_client']:
        raise ValueError('native binary changed')
    records = {}
    for policy, first, repeat in (('primitive', roots[0], roots[1]), ('fused', roots[2], roots[3])):
        for layer in (28, 29):
            paths = [root / f'layer-{layer}-{which}.bin'
                     for root in (first, repeat) for which in ('first', 'repeat')]
            if any(identity(path)['bytes'] != (4096 if kind == 'route' else 720896) for path in paths):
                raise ValueError('outside whole output size')
            data = paths[0].read_bytes()
            valid(data, kind == 'route')
            if any(path.read_bytes() != data for path in paths[1:]):
                raise ValueError('native eager/capture/fresh-process movement')
            records[f'{policy}/{layer}'] = [identity(path) for path in paths]
    os.umask(0o077)
    result = {'schema': 'gemma26-late-moe-native-own-v1', 'kind': kind,
              'prior_gate_receipt_identities': admissions,
              'pre_sha256': identity(pre_path)['sha256'],
              'capture_receipt_sha256': identity(capture_path)['sha256'],
              'input_manifest': release['manifest'], 'native_binary': pre['native_operator_client'],
              'own_output_proofs': records, 'official_native_acquisition': official,
              'counterpart_outputs_read': False,
              'claim': 'finite eager/captured/fresh-process own equality; no cross-engine allowance'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    output.chmod(0o400)


if __name__ == '__main__':
    main()
