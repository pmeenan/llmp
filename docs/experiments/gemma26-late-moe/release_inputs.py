#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded input-only release; reduction weights follow native routing freeze."""
import hashlib
import json
import os
import pathlib
import runpy
import sys

snapshot = runpy.run_path(str(pathlib.Path(__file__).with_name('source_snapshot.py')))
identity, admitted_receipts = snapshot['record'], snapshot['admitted_receipts']


def names(kind):
    if kind not in ('route', 'reduce'):
        raise ValueError('closed operator kind')
    roles = ('logits',) if kind == 'route' else ('experts', 'scales', 'weights')
    return [f'layer-{layer}-{role}.bin' for layer in (28, 29) for role in roles]


def main():
    if len(sys.argv) not in (6, 8):
        raise ValueError('CAPTURE_RECEIPT CAPTURE_ROOT NEW_INPUT_DIRECTORY route|reduce EXPECTED_CAPTURE_SHA [ROUTING_OWN_FREEZE EXPECTED_ROUTING_OWN_SHA]')
    receipt_path, root, output = map(pathlib.Path, sys.argv[1:4])
    kind = sys.argv[4]
    pairs = {'capture': (receipt_path, sys.argv[5])}
    if kind == 'reduce':
        if len(sys.argv) != 8:
            raise ValueError('original weights remain withheld before routing own freeze')
        pairs['routing_own'] = (pathlib.Path(sys.argv[6]), sys.argv[7])
    elif kind != 'route' or len(sys.argv) != 6:
        raise ValueError('unexpected route release extension')
    receipts, admissions = admitted_receipts(pairs)
    receipt = receipts['capture']
    if receipt['schema'] != 'gemma26-late-moe-capture-v1' or not receipt['all_phase_dispatch_unchanged']:
        raise ValueError('unqualified captured inputs')
    if kind == 'reduce':
        routing = receipts['routing_own']
        if routing['schema'] != 'gemma26-late-moe-native-own-v1' or routing['kind'] != 'route' or \
                routing['capture_receipt_sha256'] != identity(receipt_path)['sha256']:
            raise ValueError('unmatched native routing blind boundary')
    os.umask(0o077)
    output.mkdir(mode=0o700)
    lines = []
    for name in names(kind):
        expected = receipt['files'][name]
        if identity(root / name) != expected:
            raise ValueError('captured operand changed')
        # Hard links preserve exact immutable captured inputs and add no copy
        # of withheld output roles. Caller owns this owner-only diagnostic tree.
        os.link(root / name, output / name, follow_symlinks=False)
        lines.append(f"{name}\t{expected['bytes']}\t{expected['sha256']}\n")
    with (output / 'inputs.tsv').open('x') as stream:
        stream.writelines(lines); stream.flush(); os.fsync(stream.fileno())
    proof = {'capture_receipt': admissions['capture'], 'kind': kind,
             'prior_gate_receipt_identities': admissions,
             'inputs': names(kind), 'manifest': identity(output / 'inputs.tsv')}
    with (output / 'release.json').open('x') as stream:
        json.dump(proof, stream, indent=2); stream.write('\n')
    print('input_manifest_sha256=' + proof['manifest']['sha256'])


if __name__ == '__main__':
    main()
