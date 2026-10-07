#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""One native padded/bounded C12 short bookend; raw outputs remain external."""
import array
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, log):
    text = log.read_text()
    records, hashes = {}, {}
    for arm in ('B1', 'N1', 'N2', 'B2'):
        pieces = re.split(r'^\[spark-job [^\n]+\] step \d+/\d+ start: ', text, flags=re.M)
        matches = [s for s in pieces[1:] if s.split('\n', 1)[0].endswith(f' arm {arm}')]
        assert len(matches) == 1, (arm, len(matches))
        prefix = 'GEMMA3_WIDE_CYCLE'
        assert prefix + '_RETIRED' in matches[0]
        line = re.search(prefix + r' (.*)', matches[0])
        assert line
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line.group(1)))
        for key, value in dict(slots=12, input_identities=2, context=4096, per_owner_rows=128,
            wave_rows=256, paid_prefix_rows=6144, paid_teacher_rows=36, paid_generated_tokens=384,
            decode_steps=32, final_heads_paid=12).items():
            assert int(fields[key]) == value, (arm, key)
        for field in ('prefill_seconds', 'decode_seconds'):
            value = float(fields[field])
            assert math.isfinite(value) and value > 0
        assert fields['gpu_tokens'] == '528' and fields['host_overcharges'] == '0'
        assert 0 < int(fields['actual_plans']) <= int(fields['cache_event_bound']) == 160
        assert 0 < int(fields['plan_graph_bytes']) <= int(fields['cache_budget']) <= 64 * 2**30
        assert int(fields['selected_owner']) > 0 and int(fields['selected_packed']) > 0
        assert int(fields['bounded_whole12']) == int(arm.startswith('N'))
        assert (int(fields['selected_bounded']) > 0) == arm.startswith('N')
        for phase, expected in (('prefill', 60), ('decode', 32)):
            assert sum(int(fields[phase + '_' + item]) for item in ('eager', 'captured', 'replayed')) == expected
        assert int(fields['decode_replayed']) == 32 and int(fields['decode_captured']) == int(fields['decode_eager']) == 0
        folder = root / arm
        raw = (folder / 'chosen.i32').read_bytes()
        assert len(raw) == 384 * 4
        assert all(0 <= t < 262208 for t in struct.unpack('<384i', raw))
        past = struct.unpack('<12I', (folder / 'past.u32').read_bytes())
        assert past == tuple((768 if s in (2, 3, 8, 9, 10, 11) else 256) + 35 for s in range(12))
        final = (folder / 'final.f32').read_bytes()
        assert len(final) == 12 * 262208 * 4
        values = array.array('f'); values.frombytes(final)
        if sys.byteorder != 'little': values.byteswap()
        assert all(map(math.isfinite, values))
        hashes[arm] = {n: digest(folder / n) for n in ('chosen.i32', 'past.u32', 'final.f32')}
        records[arm] = fields
    # Both policies must reproduce their own output and each other exactly.
    assert hashes['N1'] == hashes['N2'], 'bounded own cycle outputs differ'
    assert hashes['B1'] == hashes['B2'], 'padded own cycle outputs differ'
    assert hashes['N1'] == hashes['B1'], 'padded/bounded natural choices or final heads differ'
    expected = dict(chosen_i32='2d606c0fd4cc15469bd3af0ce10c348931288a29e21bb4f5b724a2d1d629b8b3',
                    past_u32='a81d9f2ee71ac651407a2239dd7d7b85d5cf68f7bff14ce67a7fa9ef2ae73de6',
                    final_f32='b0b49b18725623e7fec3869f0dfeffd06485f4f9676c76747bc8f18eaa396d1d')
    assert hashes['B1'] == {n.replace('_', '.'):v for n,v in expected.items()}, 'historical padded outputs changed'
    assert all(records[a][key] == records['B1'][key] for a in records for key in
               ('prefill_captured','prefill_replayed','prefill_eager','decode_captured','decode_replayed','decode_eager'))
    means = {}
    for engine, arms in (('bounded', ('N1', 'N2')), ('padded', ('B1', 'B2'))):
        p = sum(float(records[a]['prefill_seconds']) for a in arms) / 2
        d = sum(float(records[a]['decode_seconds']) for a in arms) / 2
        means[engine] = dict(prefill_seconds=p, decode_seconds=d, cycle_seconds=p+d,
                             aggregate_decode_tokens_per_second=384/d)
    ratio = means['bounded']['cycle_seconds'] / means['padded']['cycle_seconds'] - 1
    result = dict(passed=True, scope='one native same-binary C12 short n=2 factor screen; concurrency, public admission, '
        'peak memory, 8K batching and sustained performance remain open', order=['B1', 'N1', 'N2', 'B2'],
        paid_prefix_rows=6144, paid_teacher_rows=36, paid_generated_tokens=384, final_heads_paid=12,
        output_hashes=hashes, arms=records, means=means, bounded_cycle_relative_to_padded=ratio)
    with (root / 'performance.json').open('x') as out:
        json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 3
    main(Path(sys.argv[1]), Path(sys.argv[2]))
