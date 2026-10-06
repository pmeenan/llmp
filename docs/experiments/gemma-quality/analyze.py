#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""External full-row analysis: ROOT calibrate|oracle; fixed teacher-forced IDs.

Run under installed spark-job on Spark, or hostlock shared locally.
Native calibration is frozen exclusively before oracle data are inspected.
Historical ROOT modes retain their 128-row ordinary-versus-norm semantics.
current-freeze ROOT CHUNK VARIANT POLICY NORMMUL SOURCE_JSON SOURCE_SHA;
current-oracle ROOT FREEZE_SHA uses native-first/native-repeat and
reference-first/reference-repeat. SOURCE_JSON has a matching native_recipe.
Both current modes optionally append INPUT_MANIFEST MANIFEST_SHA for fresh inputs.
The legacy/default input remains pinned to b2d7; the final row has no target.
current-selftest runs only tiny metadata/tie controls, without logit files.
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


def current_recipe(chunk, variant, policy, normmul):
    assert 1 <= chunk <= 1024 and variant in ('26', '31')
    assert policy in ('ordinary', 'norm', 'both', 'norm_rope', 'norm_add', 'moe', 'all')
    assert normmul in ('normmul-off', 'normmul-on')
    return dict(context=4096, teacher_chunk=chunk, max_rows=chunk, variant=variant,
                policy=policy, normmul=normmul, all_outputs=True)


def current_input(manifest_path=None, expected_manifest_sha=None, expected_profile=None):
    if manifest_path is None:
        assert expected_manifest_sha is None
        return ID_SHA, None
    assert manifest_path.stat().st_size <= 65536
    raw = manifest_path.read_bytes()
    assert len(raw) <= 65536
    assert hashlib.sha256(raw).hexdigest() == expected_manifest_sha, 'input manifest changed'
    manifest = json.loads(raw)
    assert manifest['schema'] == 1 and manifest['rows'] == ROWS
    assert manifest['profile'] in ('26', '31')
    if expected_profile is not None:
        assert manifest['profile'] == expected_profile, 'input checkpoint differs from recipe'
    assert manifest['input_bytes'] == ROWS * 4 and manifest['vocab'] == VOCAB
    assert manifest['bos'] == 2 and manifest['bos_count'] == 1
    assert manifest['scored_targets'] == ROWS - 1, 'final row must remain unscored'
    digest = manifest['input_sha256']
    assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
    return digest, expected_manifest_sha


def current_ids(root, expected_input_sha=ID_SHA, fresh=False):
    path = root / 'input1/ids.i32'
    assert path.stat().st_size == ROWS * 4
    raw = path.read_bytes()
    assert len(raw) == ROWS * 4 and hashlib.sha256(raw).hexdigest() == expected_input_sha
    ids = struct.unpack('<1024i', raw)
    assert ids[0] == 2 and all(0 <= value < VOCAB for value in ids)
    if fresh:
        assert ids.count(2) == 1, 'fresh history must contain exactly one BOS'
    return ids


def current_choices(native, reference):
    ni = max(range(len(native)), key=native.__getitem__)
    ri = max(range(len(reference)), key=reference.__getitem__)
    return ni, ri, reference[ri] - reference[ni]


def current_selftest():
    assert current_choices([0, 1], [1, 1]) == (1, 0, 0), 'exact reference tie'
    assert current_choices([0, 1], [2, 1]) == (1, 0, 1), 'positive reference margin'
    assert current_choices([-0.0, 1], [0.0, 1]) == (1, 1, 0), 'signed zero winner'
    assert current_recipe(256, '31', 'both', 'normmul-on')['max_rows'] == 256
    assert current_recipe(1024, '26', 'all', 'normmul-on')['teacher_chunk'] == 1024
    for args in ((0, '31', 'both', 'normmul-on'), (1025, '26', 'all', 'normmul-on'),
                 (256, '32', 'both', 'normmul-on'), (256, '31', 'invalid', 'normmul-on'),
                 (256, '31', 'both', 'invalid')):
        try:
            current_recipe(*args)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid current recipe accepted')
    print('current recipe bounds, exact tie, positive margin and signed zero: PASS')


def current_freeze(root, recipe, source_path, expected_source_sha,
                   manifest_path=None, expected_manifest_sha=None):
    input_sha, manifest_sha = current_input(manifest_path, expected_manifest_sha, recipe['variant'])
    current_ids(root, input_sha, manifest_sha is not None)
    source_path = source_path.resolve()
    source_bytes = source_path.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == expected_source_sha
    source = json.loads(source_bytes)
    assert source['native_recipe'] == recipe, 'source recipe differs from requested calibration'
    paths = [root / arm / 'logits.f32' for arm in ('native-first', 'native-repeat')]
    assert all(path.stat().st_size == ROWS * VOCAB * 4 for path in paths)
    hashes, winners, whole = [], [], [hashlib.sha256(), hashlib.sha256()]
    with paths[0].open('rb') as first, paths[1].open('rb') as repeat:
        for index in range(ROWS):
            a, b = first.read(VOCAB * 4), repeat.read(VOCAB * 4)
            assert len(a) == len(b) == VOCAB * 4 and a == b, 'native own repeat changed'
            values = array.array('f')
            values.frombytes(a)
            if sys.byteorder != 'little':
                values.byteswap()
            assert all(map(math.isfinite, values)), 'nonfinite native likelihood row'
            hashes.append(hashlib.sha256(a).hexdigest())
            winners.append(max(range(VOCAB), key=values.__getitem__))
            whole[0].update(a)
            whole[1].update(b)
            if index % 128 == 0:
                print(f'CURRENT_SELF_REPEAT completed_rows={index+1}', flush=True)
        assert not first.read(1) and not repeat.read(1)
    result = dict(input_sha256=input_sha, rows=ROWS, vocab=VOCAB, native_recipe=recipe,
                  calibration='same-policy full-byte own repeat only; no inherited allowance',
                  byte_exact_rows=ROWS, max_raw_delta=0, p99_top2_margin_movement=0,
                  source_identities_path=str(source_path), source_identities_sha256=expected_source_sha,
                  source=source, whole_file_sha256=[item.hexdigest() for item in whole],
                  row_sha256=hashes, argmax_ids=winners)
    if manifest_sha is not None:
        result['input_manifest_sha256'] = manifest_sha
    frozen = root / 'current-native-repeat-frozen.json'
    with frozen.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print('freeze_sha256=' + hashlib.sha256(frozen.read_bytes()).hexdigest())


def current_oracle(root, expected_freeze_sha, manifest_path=None, expected_manifest_sha=None,
                   operational_calibration=None):
    frozen_bytes = (root / 'current-native-repeat-frozen.json').read_bytes()
    assert hashlib.sha256(frozen_bytes).hexdigest() == expected_freeze_sha
    frozen = json.loads(frozen_bytes)
    input_sha, manifest_sha = current_input(manifest_path, expected_manifest_sha,
                                            frozen['native_recipe']['variant'])
    assert frozen['rows'] == ROWS and frozen['vocab'] == VOCAB and frozen['input_sha256'] == input_sha
    assert frozen.get('input_manifest_sha256') == manifest_sha
    assert frozen['byte_exact_rows'] == ROWS and frozen['p99_top2_margin_movement'] == 0
    assert hashlib.sha256(pathlib.Path(frozen['source_identities_path']).read_bytes()).hexdigest() == frozen['source_identities_sha256']
    ids = current_ids(root, input_sha, manifest_sha is not None)
    paths = [root / arm / 'logits.f32' for arm in
             ('native-first', 'reference-first', 'reference-repeat')]
    assert all(path.stat().st_size == ROWS * VOCAB * 4 for path in paths)
    native_nll, reference_nll, tvs = [], [], []
    whole = [hashlib.sha256() for _ in paths]
    exact = strict = positive = tied = 0
    bounded_misses = []
    margin_bound = 0.0 if operational_calibration is None else operational_calibration[
        'p99_top2_margin_movement']
    assert math.isfinite(margin_bound) and margin_bound >= 0
    max_raw = max_chosen = 0.0
    with paths[0].open('rb') as a, paths[1].open('rb') as b, paths[2].open('rb') as c:
        for index in range(ROWS):
            av, ah = read_row(a)
            br, cr = b.read(VOCAB * 4), c.read(VOCAB * 4)
            assert len(br) == len(cr) == VOCAB * 4 and br == cr, 'reference own repeat changed'
            bv = array.array('f')
            bv.frombytes(br)
            if sys.byteorder != 'little':
                bv.byteswap()
            assert all(map(math.isfinite, bv)), 'nonfinite reference likelihood row'
            assert ah == frozen['row_sha256'][index], 'native changed after own freeze'
            # Hash the original little-endian bytes, including signed zero.
            native_raw = av.tobytes() if sys.byteorder == 'little' else struct.pack('<%df' % VOCAB, *av)
            for digest, raw in zip(whole, (native_raw, br, cr)):
                digest.update(raw)
            ni, ri, margin = current_choices(av, bv)
            assert ni == frozen['argmax_ids'][index]
            exact += ah == hashlib.sha256(br).hexdigest()
            if ni != ri:
                strict += 1
                positive += margin > 0
                tied += margin == 0
                if operational_calibration is not None:
                    bounded_misses.append(dict(row=index, native=ni, reference=ri,
                                              reference_margin=margin,
                                              within_frozen_bound=margin <= margin_bound))
            aw, at, al = distribution(av)
            bw, bt, bl = distribution(bv)
            max_raw = max(max_raw, max(abs(x-y) for x, y in zip(av, bv)))
            tvs.append(.5 * math.fsum(abs(x/at-y/bt) for x, y in zip(aw, bw)))
            max_chosen = max(max_chosen, abs((al-av[ri])-(bl-bv[ri])))
            if index + 1 < ROWS:
                native_nll.append(al-av[ids[index+1]])
                reference_nll.append(bl-bv[ids[index+1]])
            if index % 128 == 0:
                print(f'CURRENT_ORACLE completed_rows={index+1}', flush=True)
        assert not a.read(1) and not b.read(1) and not c.read(1)
    digests = [item.hexdigest() for item in whole]
    assert digests[0] == frozen['whole_file_sha256'][0] and digests[1] == digests[2]
    nm, rm = math.fsum(native_nll)/(ROWS-1), math.fsum(reference_nll)/(ROWS-1)
    result = dict(input_sha256=input_sha, rows=ROWS, scored_targets=ROWS-1,
                  native_recipe=frozen['native_recipe'], frozen_sha256=expected_freeze_sha,
                  source_identities_sha256=frozen['source_identities_sha256'],
                  expected_reference_recipe='score-ring: C1/context4096/F16/full heads, batch=ubatch=teacher_chunk, swa_full=false/kv_unified=false, fusion+graphs enabled; caller authenticates actual invocation',
                  whole_file_sha256=digests, byte_exact_rows=exact,
                  reference_repeat_byte_exact_rows=ROWS, strict_argmax_mismatches=strict,
                  positive_reference_margin_mismatches=positive, exact_reference_tie_mismatches=tied,
                  outside_frozen_zero_margin_movement=positive, max_raw_delta=max_raw,
                  max_full_softmax_tv=max(tvs), mean_full_softmax_tv=math.fsum(tvs)/ROWS,
                  max_reference_chosen_nll_delta=max_chosen,
                  native_mean_nll=nm, reference_mean_nll=rm,
                  native_ppl=math.exp(nm), reference_ppl=math.exp(rm),
                  ppl_relative_percent=100*math.expm1(nm-rm),
                  scope='current fixed teacher-forced recipe only; no inherited128 allowance, performance or model qualification')
    if operational_calibration is not None:
        outside = sum(not item['within_frozen_bound'] for item in bounded_misses)
        result.update(operational_calibration=operational_calibration,
                      strict_miss_rows=bounded_misses, outside_frozen_bound=outside,
                      operational_margin_gate_pass=outside == 0,
                      ppl_gate_pass=result['ppl_relative_percent'] <= 3,
                      strict_zero_margin_gate_pass=positive == 0,
                      scope='fresh fixed-capacity heldout history; unchanged history13 native schedule bound; old zero-bound failures remain failures')
    if manifest_sha is not None:
        result['input_manifest_sha256'] = manifest_sha
    with (root / 'current-comparison.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result, indent=2))


def main():
    if len(sys.argv) == 2 and sys.argv[1] == 'current-selftest':
        current_selftest()
        return 0
    if len(sys.argv) in (9, 11) and sys.argv[1] == 'current-freeze':
        recipe = current_recipe(int(sys.argv[3]), *sys.argv[4:7])
        current_freeze(pathlib.Path(sys.argv[2]), recipe, pathlib.Path(sys.argv[7]), sys.argv[8],
                       *((pathlib.Path(sys.argv[9]), sys.argv[10]) if len(sys.argv) == 11 else ()))
        return 0
    if len(sys.argv) in (4, 6) and sys.argv[1] == 'current-oracle':
        current_oracle(pathlib.Path(sys.argv[2]), sys.argv[3],
                       *((pathlib.Path(sys.argv[4]), sys.argv[5]) if len(sys.argv) == 6 else ()))
        return 0
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
