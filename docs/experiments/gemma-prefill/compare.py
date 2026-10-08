#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Read retained 8K heads; no quality allowance or gate is introduced."""
import array
import hashlib
import json
import math
import pathlib
import sys

VOCAB = 262144
CASES = ['native128-b2', 'native128-b3'] + [
    f'reference-ubatch{x}-a1' for x in (512, 1024, 2048, 4096, 8192)
] + ['reference-ubatch1024-a2', 'reference-ubatch1024-a3']


def head(path):
    raw = path.read_bytes()
    assert len(raw) == VOCAB * 4
    values = array.array('f')
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    assert all(map(math.isfinite, values))
    return values, hashlib.sha256(raw).hexdigest()


def compare(a, b):
    def distribution(v):
        maximum = max(v)
        weights = [math.exp(x - maximum) for x in v]
        total = math.fsum(weights)
        return weights, total, maximum + math.log(total)
    aw, at, al = distribution(a)
    bw, bt, bl = distribution(b)
    ai = max(range(VOCAB), key=a.__getitem__)
    bi = max(range(VOCAB), key=b.__getitem__)
    return dict(max_raw_delta=max(abs(x-y) for x, y in zip(a, b)),
                full_softmax_tv=.5 * math.fsum(abs(x/at-y/bt) for x, y in zip(aw, bw)),
                chosen_nll_delta=abs((al-a[bi])-(bl-b[bi])), argmax=ai, reference_argmax=bi)


def main():
    assert len(sys.argv) == 2
    root = pathlib.Path(sys.argv[1])
    raw = (root / 'input1' / 'ids.i32').read_bytes()
    assert len(raw) == 8227 * 4 and hashlib.sha256(raw).hexdigest() == (
        '6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b')
    result = dict(scope='two retained heads only; no 8K quality/noise/memory gate', heads={})
    for name in ('prefill.f32', 'final.f32'):
        values, hashes = {}, {}
        for case in CASES:
            values[case], hashes[case] = head(root / case / name)
        baseline = values['reference-ubatch1024-a1']
        result['heads'][name] = dict(
            sha256=hashes,
            relative_to_reference1024={case: compare(v, baseline) for case, v in values.items()})
        assert hashes['native128-b2'] == hashes['native128-b3'], 'native retained repeat differs'
        assert all(hashes[f'reference-ubatch1024-a{x}'] == hashes['reference-ubatch1024-a1']
                   for x in (2, 3)), 'fastest reference retained repeat differs'
    with (root / 'retained-heads.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
