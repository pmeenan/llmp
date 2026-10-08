#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate counterpart acquisition only after the native blind boundary."""
import json
import os
import pathlib
import runpy
import shlex
import sys

tool = pathlib.Path(__file__).resolve().parent
snapshot = runpy.run_path(str(tool / 'source_snapshot.py'))
identity, admitted_receipts = snapshot['record'], snapshot['admitted_receipts']
success = runpy.run_path(str(tool / 'official.py'))['success']
valid = runpy.run_path(str(tool / 'own_freeze.py'))['valid']


def main():
    if len(sys.argv) != 12:
        raise ValueError('PRE CAPTURE_RECEIPT INPUTS OWN_FREEZE PRIMITIVE_ROOT FUSED_ROOT JOB NEW_PROOF EXPECTED_PRE_SHA EXPECTED_CAPTURE_SHA EXPECTED_OWN_SHA')
    pre_path, capture_path, inputs, own_path, primitive, fused = map(pathlib.Path, sys.argv[1:7])
    job, output = sys.argv[7], pathlib.Path(sys.argv[8])
    receipts, admissions = admitted_receipts({'pre': (pre_path, sys.argv[9]),
        'capture': (capture_path, sys.argv[10]), 'own': (own_path, sys.argv[11])})
    pre, own = receipts['pre'], receipts['own']
    kind = own['kind']
    if own['schema'] != 'gemma26-late-moe-native-own-v1' or kind not in ('route', 'reduce') or \
            own['pre_sha256'] != identity(pre_path)['sha256'] or \
            own['capture_receipt_sha256'] != identity(capture_path)['sha256']:
        raise ValueError('unmatched native pre-exposure calibration')
    root = pathlib.Path.home() / '.local/share/llmp/gemma26-late-moe'
    source = pathlib.Path.home() / 'src/llmp-wt/m3fixb'
    if identity(root / 'oracle') != pre['original_operator_client']:
        raise ValueError('original client binary changed')
    for name, expected in pre['source_files'].items():
        if identity(source / name, 32 << 20) != expected:
            raise ValueError('client/source changed after native calibration')
    manifest = identity(inputs / 'inputs.tsv')
    if manifest != own['input_manifest']:
        raise ValueError('counterpart incoming inputs changed')
    official, steps = success(job, 2)
    for step, policy in zip(steps, ('primitive', 'fused')):
        command = shlex.split(step['cmd'])
        if len(command) != 5 or not command[0].endswith('/gemma26-late-moe/reference.sh') or \
                command[1:] != ['oracle', kind, policy, manifest['sha256']]:
            raise ValueError('actual original operator command/input/policy differs')
    log_path = pathlib.Path.home() / '.local/share/llmp/jobs' / job / 'log'
    if log_path.read_text().splitlines().count('LATE_ORIGINAL_RETIRED complete=1') != 2:
        raise ValueError('missing original context/device retirement')
    containers = {}
    files = {}
    for policy, directory in (('primitive', primitive), ('fused', fused)):
        if directory != root / f'original-{kind}-{policy}':
            raise ValueError('wrong actual counterpart output directory')
        name = f'{kind}-{policy}'
        cid = (root / (name + '.cid')).read_text().strip()
        container = json.loads((root / (name + '-container-retired.json')).read_text())
        if len(cid) != 64 or any(c not in '0123456789abcdef' for c in cid) or container != {
                'cid': cid, 'name': 'llmp-gemma26-late-moe-' + name,
                'owner_label': 'gemma26-late-moe', 'container_absent_after_checked_docker_query': True}:
            raise ValueError('missing owned original container absence proof')
        containers[policy] = container
        for layer in (28, 29):
            first, repeat = [directory / f'layer-{layer}-{which}.bin' for which in ('first', 'repeat')]
            expected_size = 4096 if kind == 'route' else 720896
            if identity(first)['bytes'] != expected_size or identity(repeat)['bytes'] != expected_size:
                raise ValueError('incomplete original whole output')
            data = first.read_bytes()
            valid(data, kind == 'route')
            if repeat.read_bytes() != data:
                raise ValueError('original own output movement')
            files[f'{policy}/{layer}'] = {'first': identity(first), 'repeat': identity(repeat)}
    proof = {'schema': 'gemma26-late-moe-original-v1', 'kind': kind,
             'prior_gate_receipt_identities': admissions,
             'pre_sha256': identity(pre_path)['sha256'], 'native_own_sha256': identity(own_path)['sha256'],
             'capture_receipt_sha256': identity(capture_path)['sha256'],
             'original_binary': pre['original_operator_client'], 'input_manifest': manifest,
             'official_original': official, 'container_retirements': containers, 'files': files}
    os.umask(0o077)
    with output.open('x') as file:
        json.dump(proof, file, indent=2); file.write('\n'); file.flush(); os.fsync(file.fileno())
    output.chmod(0o400)


if __name__ == '__main__':
    main()
