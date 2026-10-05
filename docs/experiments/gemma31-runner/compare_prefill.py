#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded retained-head and paid-count proof; ROOT OFFICIAL_BOOKEND_LOG.

Run through installed spark-job --gpu or hostlock shared. Two retained
heads do not qualify 8K quality, generated histories or physical peak.
"""
import hashlib
import importlib.util
import json
import pathlib
import re
import struct
import sys

ID_SHA = '6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b'
CASES = ('reference-a1', 'native-b1', 'reference-a2', 'native-b2', 'reference-a3')


def main():
    assert len(sys.argv) == 3
    root, log = map(pathlib.Path, sys.argv[1:])
    module = pathlib.Path(__file__).resolve().parent.parent / 'gemma-prefill' / 'compare.py'
    spec = importlib.util.spec_from_file_location('retained_head_helpers', module)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    raw = (root / 'input1' / 'ids.i32').read_bytes()
    assert len(raw) == 8227 * 4 and hashlib.sha256(raw).hexdigest() == ID_SHA
    ids = struct.unpack('<8227i', raw)
    assert ids[0] == 2 and all(0 <= x < common.VOCAB for x in ids)
    result = dict(input_sha256=ID_SHA, scope='paid fixed-prefix work and two retained heads only',
                  processes={}, heads={})
    current = None
    for line in log.read_text().splitlines():
        if '] step ' in line and ' start: ' in line:
            match = re.search(r'(?:reference\.sh screen |gemma31-prefill/)(reference-a[123]|native-b[12])', line)
            current = None
            if match:
                arm = match[1]
                assert arm not in result['processes']
                current = dict(prefix=[], tokens=[], timer=None)
                result['processes'][arm] = current
        if current is None:
            continue
        if line.startswith('PREFILL_TIMED_PREFIX '):
            current['prefix'].append({k: int(v) for k, v in re.findall(r'(\w+)=([0-9]+)', line)})
        elif line.startswith('PREFILL_TOKEN '):
            current['tokens'].append({k: int(v) for k, v in re.findall(r'(\w+)=([0-9]+)', line)})
        elif line.startswith(('PREFILL_NATIVE ', 'PREFILL_REFERENCE ')):
            assert current['timer'] is None
            current['timer'] = dict(re.findall(r'(\w+)=([^ ]+)', line))
    assert tuple(result['processes']) == CASES, 'missing or reordered bookend processes'
    for arm, process in result['processes'].items():
        timer = process['timer']
        assert timer is not None
        assert process['prefix'] == [dict(appended=ids[8192+i], past=8193+i) for i in range(3)]
        assert len(process['tokens']) == 32
        assert [x['step'] for x in process['tokens']] == list(range(32))
        assert [x['forced'] for x in process['tokens']] == list(ids[8195:])
        assert all(0 <= x['argmax'] < common.VOCAB for x in process['tokens'])
        assert all(int(timer[k]) == v for k, v in dict(prefill_rows=8192, decode_chunks=32,
                                                     timed_start=8195, completed=8227,
                                                     context=16384).items())
        assert float(timer['prefill_seconds']) > 0 and float(timer['decode_seconds']) > 0
        if arm.startswith('native'):
            assert timer['masks'] == 'device' and int(timer['prefill_chunks']) == 64
            assert int(timer['intermediate_heads']) == 63
            assert all(int(timer[k]) == 0 for k in ('shared_vecq','row_products','norm_fused','rope_store'))
        else:
            assert int(timer['ubatch']) == 256 and int(timer['batch']) == 8192
            assert timer['fusion'] == 'enabled' and int(timer['intermediate_heads']) == 0
    for name in ('prefill.f32', 'final.f32'):
        values, hashes = {}, {}
        for arm in CASES:
            values[arm], hashes[arm] = common.head(root / arm / name)
        assert hashes['native-b1'] == hashes['native-b2'], 'native retained repeat changed'
        assert hashes['reference-a1'] == hashes['reference-a2'] == hashes['reference-a3'], 'reference retained repeat changed'
        result['heads'][name] = dict(sha256=hashes, relative_to_reference_a1={
            arm: common.compare(values[arm], values['reference-a1']) for arm in CASES})
    for other in ('reference-a2', 'reference-a3'):
        assert result['processes'][other]['tokens'] == result['processes']['reference-a1']['tokens']
    assert result['processes']['native-b1']['tokens'] == result['processes']['native-b2']['tokens']
    with (root / 'retained-heads.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
