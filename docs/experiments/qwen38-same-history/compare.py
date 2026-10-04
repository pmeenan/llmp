# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare literal-history first verifies; raw files stay outside the repository."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

VOCAB = 248320


def require(value, why):
    if not value:
        raise ValueError(why)


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON ' + value)
    return json.loads(path.read_text(), parse_constant=invalid)


def digest(ids):
    return hashlib.sha256(struct.pack('<' + str(len(ids)) + 'i', *ids)).hexdigest()


def rows(path, expected):
    raw = path.read_bytes()
    require(len(raw) == 4 * VOCAB * 4 and hashlib.sha256(raw).hexdigest() == expected,
            'complete first-verify F32 rows and recorded digest required')
    values = array('f')
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    require(values.itemsize == 4 and all(math.isfinite(x) for x in values), 'finite F32 rows required')
    return values


def accepted(first):
    require(first['rows'] == 4 and len(first['drafts']) == 3 and len(first['verdicts']) == 4,
            'four target rows for three actually offered drafts required')
    count = 0
    for proposal, verdict in zip(first['drafts'], first['verdicts']):
        if proposal != verdict:
            break
        count += 1
    require(first['kept'] == count + 1, 'kept must follow the actual contiguous agreement')
    return count


def check_verdicts(first, values):
    require(len(first['verdicts']) == 4, 'all four recorded verdicts required')
    for index in range(4):
        row = values[index * VOCAB:(index + 1) * VOCAB]
        require(max(range(VOCAB), key=row.__getitem__) == first['verdicts'][index],
                'recorded verdict differs from its own raw argmax')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='append', type=Path, required=True)
    parser.add_argument('--mia', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    mia = read(args.mia / 'receipt.json')
    require(mia['complete'] and mia['container_absent'] and mia['launch_reaped']
            and mia['timing_qualified'] is False and len(mia['controls']) == 5,
            'all reference controls and owned retirement required')
    require([(x['position'], x['repeat']) for x in mia['controls']]
            == [(0, False), (0, True), (1, False), (2, False), (3, False)],
            'exact cold/repeat/adjacent reference schedule required')
    report = dict(complete=False, timing_qualified=False, scope='four adjacent anchors from one frozen 32K history',
                  native_retirement_qualification='requires external supervisor success; spec.json alone does not prove retirement',
                  reference_receipt_sha256=hashlib.sha256((args.mia / 'receipt.json').read_bytes()).hexdigest(),
                  pairs=[])
    for directory in args.native:
        native = read(directory / 'spec.json')
        require(native['problems'] == [] and native['prompt_token_ids'] and native['draft_rows'] == 3
                and not native['adaptive_depth'] and native['window'] == 0,
                'passing native literal greedy controls required')
        greedy = [x for x in native['results'] if x['check'] == 'greedy']
        require(len(greedy) == len({x['prompt'] for x in greedy}),
                'unique native greedy cells required')
        cells = {x['prompt']: x for x in greedy}
        require(set(cells) == {'p0', 'p1', 'p2', 'p3'}, 'all four native anchors required')
        for control in mia['controls']:
            cell = cells['p' + str(control['position'])]
            first, other = cell['first_verify'], control['first_verify']
            require(digest(cell['prompt_ids']) == cell['prompt_ids_sha256'] == control['prompt_i32_sha256']
                    and first['position'] == other['position'] == cell['prompt_tokens'], 'identical initial history required')
            left = rows(directory / (cell['prompt'] + '-first-verify.f32'), first['logits_sha256'])
            require(Path(other['logits_file']).name == other['logits_file'], 'reference row basename required')
            right = rows(args.mia / 'traces' / other['logits_file'], other['logits_sha256'])
            check_verdicts(first, left)
            check_verdicts(other, right)
            n, m = accepted(first), accepted(other)
            require(m == other['accepted'], 'reference actual sampler count differs')
            anchor_equal = first['anchor_token'] == other['anchor_token']
            common = 1 if anchor_equal else 0
            if anchor_equal:
                for a, b in zip(first['drafts'], other['drafts']):
                    if a != b:
                        break
                    common += 1
            comparisons = []
            for index in range(common):
                a = left[index * VOCAB:(index + 1) * VOCAB]
                b = right[index * VOCAB:(index + 1) * VOCAB]
                differences = [abs(x - y) for x, y in zip(a, b)]
                comparisons.append(dict(row=index, native_verdict=first['verdicts'][index],
                                        reference_verdict=other['verdicts'][index],
                                        max_abs=max(differences), mean_abs=sum(differences) / VOCAB))
            report['pairs'].append(dict(head_rows=native['draft_vocab'], position=control['position'],
                reference_repeat=control['repeat'], cached_tokens=control['cached_tokens'],
                prompt_i32_sha256=control['prompt_i32_sha256'], anchor_equal=anchor_equal,
                native_offered=3, reference_offered=3, native_accepted=n, reference_accepted=m,
                native_drafts=first['drafts'], reference_drafts=other['drafts'],
                same_conditioned_rows=comparisons))
    require(all(p['anchor_equal'] for p in report['pairs']), 'unmatched verify anchors cannot qualify the pair')
    report['complete'] = True
    with args.out.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
