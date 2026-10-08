#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Admit immutable operator inputs only after unchanged-stock fidelity proof."""
import hashlib
import json
import math
import pathlib
import re
import runpy
import struct
import sys

observer = runpy.run_path(str(pathlib.Path(__file__).resolve().parent.parent /
                               'gemma26-dispatch-observation/analyze.py'))
snapshot = runpy.run_path(str(pathlib.Path(__file__).with_name('source_snapshot.py')))
identity, admitted_receipts = snapshot['record'], snapshot['admitted_receipts']
success = runpy.run_path(str(pathlib.Path(__file__).with_name('official.py')))['success']
ROLES = {
    'logits': (32768, 0, (128, 64, 1, 1), (4, 512, 32768, 32768)),
    'ids': (2048, 26, (8, 64, 1, 1), (4, 512, 32768, 32768)),
    'route-weights': (2048, 0, (1, 8, 64, 1), (4, 4, 32, 2048)),
    'experts': (5767168, 0, (2816, 8, 64, 1), (4, 11264, 90112, 5767168)),
    'scales': (2048, 0, (1, 8, 64, 1), (4, 4, 32, 2048)),
    'weights': (2048, 0, (1, 8, 64, 1), (4, 4, 32, 2048)),
    'sum': (720896, 0, (2816, 64, 1, 1), (4, 11264, 720896, 720896)),
}


def read(path, size):
    if not 0 < size <= 5767168 or not path.is_file() or path.is_symlink() or path.stat().st_size != size:
        raise ValueError('outside fixed regular file: ' + str(path))
    data = path.read_bytes()
    if len(data) != size:
        raise ValueError('truncated fixed file')
    return data


def descriptors(data):
    if len(data) > 16384:
        raise ValueError('unbounded descriptor metadata')
    result = {}
    for line in data.decode('ascii').splitlines():
        fields = line.split('\t')
        if len(fields) != 7:
            raise ValueError('malformed descriptor record')
        layer, role, size, dtype, address, ne, nb = fields
        key = (int(layer), role)
        if key in result or key[0] not in (28, 29) or role not in ROLES:
            raise ValueError('duplicate/outside fixed role')
        expected = ROLES[role]
        if (int(size), int(dtype), tuple(map(int, ne.split(':'))),
                tuple(map(int, nb.split(':')))) != expected:
            raise ValueError('outside actual captured descriptor')
        if not re.fullmatch(r'0x[0-9a-f]+', address) or int(address, 16) == 0:
            raise ValueError('null/malformed actual descriptor address')
        result[key] = {'bytes': int(size), 'type': int(dtype), 'ne': expected[2],
                       'nb': expected[3]}
    if set(result) != {(layer, role) for layer in (28, 29) for role in ROLES}:
        raise ValueError('incomplete actual descriptor groups')
    return result


def decisions(lines):
    result = observer['observations'](lines, require_complete=True)
    # Compare actual selected families/shapes and every gate, not diagnostic
    # host addresses or appended memcpy nodes. No CUDA replay-count claim.
    return {key: result[key] for key in (
        'host_graph_calls', 'host_graph_phase_counts', 'host_evaluated_or_captured_calls',
        'host_graph_policy_calls', 'fusion_decisions', 'product_decisions')}


def main():
    if len(sys.argv) != 10:
        raise ValueError('PRE LOG CAPTURE_DIRECTORY OLD_OBSERVER_LOG OLD_STOCK_A OLD_STOCK_B JOB_NAME NEW_RECEIPT EXPECTED_PRE_SHA')
    pre_path, log, root, old_log, old_a, old_b = map(pathlib.Path, sys.argv[1:7])
    job, output = sys.argv[7], pathlib.Path(sys.argv[8])
    receipts, admissions = admitted_receipts({'pre': (pre_path, sys.argv[9])})
    pre = receipts['pre']
    for key, name in (('capture_client', 'capture-client'), ('controller_library', 'controller.so')):
        if identity(root.parent / name) != pre[key]:
            raise ValueError('compiled capture identity changed')
    official, _ = success(job)
    if identity(log, 64 << 20) != official['log']:
        raise ValueError('not the officially retired capture log')
    container = json.loads((root.parent / 'capture-container-retired.json').read_text())
    cid = (root.parent / 'capture.cid').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{64}', cid) or container != {
            'cid': cid, 'name': 'llmp-gemma26-late-moe-capture',
            'owner_label': 'gemma26-late-moe', 'container_absent_after_checked_docker_query': True}:
        raise ValueError('missing owned capture-container absence proof')
    for source in (log, old_log):
        if source.stat().st_size > 64 << 20:
            raise ValueError('unbounded observation log')
    lines = log.read_text().splitlines()
    if lines.count('LATE_CLIENT_EXIT0 explicit_public_backend_teardown=source_verified') != 1 or \
            lines.count('LLMP_LATE retired_complete_roles=14 actual64calls=2') != 1:
        raise ValueError('missing successful public/backend/device retirement')
    chronology = [line for line in lines if line.startswith('LLMP_LATE graph64=1 ')]
    if len(chronology) != 2 or 'occurrence64=1 selected=0' not in chronology[0] or \
            'occurrence64=2 selected=1' not in chronology[1]:
        raise ValueError('wrong selected post-reset occurrence')
    actual = decisions(lines)
    if actual != decisions(old_log.read_text().splitlines()):
        raise ValueError('capture altered actual all-phase dispatch')
    head_proof = {}
    capture = root.parent / 'capture'
    for label, directory in (('capture', capture), ('old_a', old_a), ('old_b', old_b)):
        sha = observer['digest'](directory / 'heads.f32', observer['HEAD_BYTES'])
        if sha != observer['HEAD_SHA'] or observer['digest'](directory / 'inputs.i32', 4096) != observer['ID_SHA']:
            raise ValueError('capture changed complete heads or canonical IDs')
        head_proof[label] = sha
    metadata_size = (root / 'descriptors.tsv').stat().st_size
    if metadata_size > 16384:
        raise ValueError('unbounded descriptor metadata before allocation')
    records = descriptors(read(root / 'descriptors.tsv', metadata_size))
    files = {}
    for (layer, role), record in records.items():
        name = f'layer-{layer}-{role}.bin'
        data = read(root / name, record['bytes'])
        if role == 'ids':
            if any(not 0 <= value < 128 for (value,) in struct.iter_unpack('<i', data)):
                raise ValueError('noncanonical selected expert ID')
        elif any(not math.isfinite(value) for (value,) in struct.iter_unpack('<f', data)):
            raise ValueError('nonfinite/incomplete role')
        files[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    # This receipt authenticates all captured bytes for the parent. Only logits
    # are released to the native routing client before its endogenous freeze.
    receipt = {'schema': 'gemma26-late-moe-capture-v1', 'scope': 'owner0 second prefill64 layers28/29',
               'prior_gate_receipt_identities': admissions,
               'pre_sha256': identity(pre_path)['sha256'], 'official_capture': official,
               'capture_container_retirement': container,
               'head_fidelity': head_proof, 'all_phase_dispatch_unchanged': True,
               'descriptors': {f'{l}/{r}': v for (l, r), v in records.items()}, 'files': files,
               'routing_input_only': [f'layer-{l}-logits.bin' for l in (28, 29)],
               'claim': 'operator inputs admitted; counterpart outputs withheld; no cause or speed claim'}
    with output.open('x') as file:
        json.dump(receipt, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
