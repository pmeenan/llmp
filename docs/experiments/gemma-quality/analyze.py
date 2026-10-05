#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""External full-row analysis: ROOT calibrate|oracle; fixed teacher-forced IDs.

Run under installed spark-job on Spark, or hostlock shared locally.
Native calibration is frozen exclusively before oracle data are inspected.
"""
import array
import hashlib
import heapq
import json
import math
import pathlib
import struct
import sys

ROWS, VOCAB = 1024, 262144
ID_SHA = 'b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610'


def read_row(file):
    raw = file.read(VOCAB * 4)
    assert len(raw) == VOCAB * 4, 'likelihood row is truncated'
    values = array.array('f')
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    assert all(map(math.isfinite, values)), 'nonfinite likelihood row'
    return values, hashlib.sha256(raw).hexdigest()


def distribution(values):
    maximum = max(values)
    weights = [math.exp(x - maximum) for x in values]
    total = math.fsum(weights)
    return weights, total, maximum + math.log(total)


def main():
    assert len(sys.argv) == 3
    root, action = pathlib.Path(sys.argv[1]), sys.argv[2]
    raw_ids = (root / 'input1' / 'ids.i32').read_bytes()
    assert hashlib.sha256(raw_ids).hexdigest() == ID_SHA and len(raw_ids) == ROWS * 4
    ids = struct.unpack('<1024i', raw_ids)
    assert ids[0] == 2 and all(0 <= x < VOCAB for x in ids)
    frozen = root / 'native-noise-128-frozen.json'
    native = root / 'native-ordinary128' / 'logits.f32'
    assert native.stat().st_size == ROWS * VOCAB * 4
    if action == 'calibrate':
        assert not frozen.exists(), 'existing calibration cannot be replaced'
        norm = root / 'native-norm128' / 'logits.f32'
        assert norm.stat().st_size == ROWS * VOCAB * 4
        hashes, moves, argmax_ids, raw_deltas, own_mismatches = [], [], [], [], []
        with native.open('rb') as a, norm.open('rb') as b:
            for index in range(ROWS):
                av, ah = read_row(a)
                bv, bh = read_row(b)
                first, second = heapq.nlargest(2, range(VOCAB), key=av.__getitem__)
                other = max(range(VOCAB), key=bv.__getitem__)
                if first != other:
                    own_mismatches.append(dict(row=index, ordinary=first, norm=other))
                hashes.append([ah, bh])
                moves.append(abs((av[first]-av[second])-(bv[first]-bv[second])))
                argmax_ids.append(first)
                raw_deltas.append(max(abs(x-y) for x, y in zip(av, bv)))
        result = dict(input_sha256=ID_SHA, rows=ROWS, teacher_chunk=128,
                      native_policy='ordinary versus independently norm-fused; no shared Q8/store/rows',
                      p99_top2_margin_movement=sorted(moves)[math.ceil(.99*ROWS)-1],
                      byte_exact_rows=sum(a == b for a,b in hashes),
                      max_raw_delta=max(raw_deltas), native_own_argmax_mismatches=own_mismatches,
                      hashes=hashes, argmax_ids=argmax_ids)
        with frozen.open('x') as file:
            json.dump(result, file, indent=2)
            file.write('\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('hashes','argmax_ids')}))
        print('freeze_sha256='+hashlib.sha256(frozen.read_bytes()).hexdigest())
        return 0
    if action == 'verify-bytes':
        arms = ('native-ordinary128', 'reference128-unfused-first', 'reference128-unfused-repeat')
        hashes = {}
        for arm in arms:
            path = root / arm / 'logits.f32'
            assert path.stat().st_size == ROWS * VOCAB * 4
            digest = hashlib.sha256()
            with path.open('rb') as file:
                for data in iter(lambda: file.read(1048576), b''):
                    digest.update(data)
            hashes[arm] = digest.hexdigest()
        assert len(set(hashes.values())) == 1, 'retained full files differ bytewise'
        result = dict(input_sha256=ID_SHA, rows=ROWS, whole_file_sha256=hashes,
                      scope='direct retained byte verification; original aggregates and freeze unchanged')
        with (root / 'unfused-byte-verification.json').open('x') as file:
            json.dump(result, file, indent=2)
            file.write('\n')
        print(json.dumps(result, indent=2))
        return 0
    assert action in ('oracle', 'oracle-unfused')
    unfused = action == 'oracle-unfused'
    calibration = json.loads(frozen.read_text())
    assert calibration['input_sha256'] == ID_SHA and calibration['teacher_chunk'] == 128
    reference = root / ('reference128-unfused-first' if unfused else 'reference128-first') / 'logits.f32'
    repeat = root / ('reference128-unfused-repeat' if unfused else 'reference128-repeat') / 'logits.f32'
    assert reference.stat().st_size == repeat.stat().st_size == ROWS * VOCAB * 4
    rows, native_nll, reference_nll, mismatches = [], [], [], []
    repeat_exact = 0
    repeat_max_raw = 0.0
    with native.open('rb') as a, reference.open('rb') as b, repeat.open('rb') as c:
        for index in range(ROWS):
            av, ah = read_row(a)
            bv, bh = read_row(b)
            cv, ch = read_row(c)
            assert ah == calibration['hashes'][index][0]
            ni = calibration['argmax_ids'][index]
            ri = max(range(VOCAB), key=bv.__getitem__)
            repeat_exact += bh == ch
            repeat_max_raw = max(repeat_max_raw, max(abs(x-y) for x,y in zip(bv,cv)))
            aw, at, al = distribution(av)
            bw, bt, bl = distribution(bv)
            item = dict(row=index, native_argmax=ni, reference_argmax=ri,
                        max_raw_delta=max(abs(x-y) for x,y in zip(av,bv)),
                        full_softmax_tv=.5*math.fsum(abs(x/at-y/bt) for x,y in zip(aw,bw)),
                        chosen_nll_delta=abs((al-av[ri])-(bl-bv[ri])),
                        reference_repeat_byte_exact=bh==ch, native_reference_byte_exact=ah==bh)
            if index+1 < ROWS:
                target = ids[index+1]
                nn, rn = al-av[target], bl-bv[target]
                native_nll.append(nn)
                reference_nll.append(rn)
                item.update(target=target, native_target_nll=nn, reference_target_nll=rn)
            if ni != ri:
                margin = bv[ri]-bv[ni]
                mismatches.append(dict(row=index, native=ni, reference=ri, reference_margin=margin,
                                       within_frozen_noise=margin<=calibration['p99_top2_margin_movement']))
            rows.append(item)
            if index%128==0: print(f'QUALITY_ANALYSIS completed_rows={index+1}', flush=True)
    nm, rm = math.fsum(native_nll)/len(native_nll), math.fsum(reference_nll)/len(reference_nll)
    result = dict(input_sha256=ID_SHA, rows=ROWS, scored_targets=ROWS-1, teacher_chunk=128,
                  frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                  frozen_p99=calibration['p99_top2_margin_movement'],
                  reference_fusion='disabled diagnostic' if unfused else 'enabled production',
                  byte_exact_rows=sum(x['native_reference_byte_exact'] for x in rows),
                  native_mean_nll=nm, reference_mean_nll=rm,
                  native_ppl=math.exp(nm), reference_ppl=math.exp(rm),
                  ppl_relative_percent=100*math.expm1(nm-rm),
                  reference_repeat_byte_exact_rows=repeat_exact,
                  reference_repeat_max_raw_delta=repeat_max_raw,
                  strict_argmax_mismatches=mismatches, comparisons=rows,
                  scope='fixed shared teacher-forced prefixes; no competitive timing or model qualification')
    with (root/('comparison128-unfused.json' if unfused else 'comparison128.json')).open('x') as file:
        json.dump(result,file,indent=2)
        file.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','strict_argmax_mismatches')},indent=2))
    print('strict_argmax_mismatches='+str(len(mismatches)))
    print('outside_frozen_noise='+str(sum(not x['within_frozen_noise'] for x in mismatches)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
