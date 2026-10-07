#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the bounded R/N/N/R GPU-token cycle after the quality gate."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def run(root):
    quality = json.loads((root / 'quality.json').read_text())
    assert quality['positive_margin_differences'] == 0
    assert math.isfinite(quality['mean_target_nll_delta'])
    assert math.expm1(quality['mean_target_nll_delta']) <= .03
    records = []
    for name, native in [('stock-cycle1', False), ('native-cycle1', True),
                         ('native-cycle2', True), ('stock-cycle2', False)]:
        text = (root / (name + '.log')).read_text()
        prefix = 'GEMMA2_PROBE ' if native else 'GEMMA2_STOCK '
        lines = [line for line in text.splitlines() if line.startswith(prefix)]
        assert len(lines) == 1
        fields = dict(item.split('=', 1) for item in lines[0].split()[1:])
        for key, expected in [('mode', 'greedy-cycle'), ('context', '4096'),
                              ('slots', '1'), ('chunk', '128'), ('prompt_rows', '256'),
                              ('untimed_rows', '3'), ('decode_rows', '32'), ('past', '291')]:
            assert fields[key] == expected, (name, key)
        if native:
            assert 'GEMMA2_RETIRED' in text
            assert fields['policy'] == 'optimized' and fields['coverage_violations'] == '0'
            assert fields['gpu_tokens'] == '44'
            assert int(fields['captured']) > 0 and int(fields['replayed']) > 0
            assert all(int(fields[key]) > 0 for key in
                       ['selected_norm_mul', 'selected_quant_geglu', 'selected_norm_add'])
            assert fields['selected_norm_rope'] == '0'
        else:
            assert 'GEMMA2_STOCK_RETIRED' in text
            assert fields['tokenized_equal'] == '1' and fields['fusion'] == 'stock'
            assert fields['graphs'] == 'allowed' and fields['final_head_paid'] == '1'
            assert fields['backend_tokens'] == '48'
            assert fields['sampled_logits_max'] == '256000'
            assert fields['sampled_logits_min'] == '256000'
            retired = json.loads((root / (name + '-container-retired.json')).read_text())
            assert retired['container_absent_after_checked_docker_query'] is True
        chosen = (root / name / 'chosen.i32').read_bytes()
        assert len(chosen) == 32 * 4
        assert all(0 <= token < 256000 for token in struct.unpack('<32i', chosen))
        head = (root / name / 'final.f32').read_bytes()
        assert len(head) == 256000 * 4
        assert all(math.isfinite(value) for value in struct.unpack('<256000f', head))
        prefill, decode = [float(fields[key]) for key in ['prefill_seconds', 'decode_seconds']]
        assert all(math.isfinite(value) and value > 0 for value in [prefill, decode])
        records.append(dict(name=name, native=native, prefill_seconds=prefill,
                            decode_seconds=decode, cycle_seconds=prefill + decode,
                            chosen_sha256=hashlib.sha256(chosen).hexdigest(),
                            final_sha256=hashlib.sha256(head).hexdigest()))
    for indices in [(0, 3), (1, 2)]:
        a, b = (records[index] for index in indices)
        assert a['chosen_sha256'] == b['chosen_sha256'], 'same-policy token repeat changed'
        assert a['final_sha256'] == b['final_sha256'], 'same-policy final head repeat changed'
    averages = {}
    for label, native in [('stock', False), ('native', True)]:
        selected = [row for row in records if row['native'] == native]
        averages[label] = {key: math.fsum(row[key] for row in selected) / 2 for key in
                           ['prefill_seconds', 'decode_seconds', 'cycle_seconds']}
    result = dict(scope='representative C1 GPU-token cycle; three supplied rows excluded',
                  order='RNNR', records=records, averages=averages,
                  cross_engine_tokens_equal=records[0]['chosen_sha256'] == records[1]['chosen_sha256'],
                  cross_engine_final_head_equal=records[0]['final_sha256'] == records[1]['final_sha256'],
                  native_over_stock={key: averages['native'][key] / averages['stock'][key]
                                     for key in averages['stock']})
    with (root / 'performance.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result), flush=True)
    assert result['cross_engine_tokens_equal'], 'matched greedy histories diverged'
    assert result['cross_engine_final_head_equal'], 'matched greedy final heads differ'


if __name__ == '__main__':
    run(Path(sys.argv[1]))
