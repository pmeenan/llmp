#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Candidate-only repeat freeze, then authenticated complete-head comparison.

freeze ROOT IDS SOURCE_IDENTITIES; compare ROOT QUALITY FREEZE_SHA.
All scans run through the installed GPU supervisor. No ordinary noise update.
"""
import array
import hashlib
import json
import math
import pathlib
import sys
import struct
import tempfile

ROWS, VOCAB = 1024, 262144
ID_SHA = 'b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610'
STOCK_SHA = 'c07c711bfc732ce498182917c0cc3cfd74f21374975e9fa69c90ec88461cc256'
OLD_FREEZE = '2c97b2a4d65277f00b242180d7f54128a395284490d84db0db24e634adcfc901'


def decode(raw):
    values = array.array('f')
    assert values.itemsize == 4 and sys.byteorder == 'little'
    values.frombytes(raw)
    assert all(map(math.isfinite, values)), 'nonfinite logits'
    return values


def metrics(a, b):
    av, bv = decode(a), decode(b)
    assert len(av) == len(bv) and av
    return dict(byte_exact=a == b,
                strict_argmax_mismatch=max(range(len(av)), key=av.__getitem__) !=
                max(range(len(bv)), key=bv.__getitem__),
                max_raw_delta=max(abs(x-y) for x, y in zip(av, bv)))


def streams(paths):
    for path in paths:
        assert path.stat().st_size == ROWS * VOCAB * 4, path
    return [path.open('rb') for path in paths]


def write(path, result):
    with path.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write('\n')
    print(path.name, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


def freeze(root, ids, identities):
    raw_ids = ids.read_bytes()
    assert len(raw_ids) == ROWS * 4 and hashlib.sha256(raw_ids).hexdigest() == ID_SHA
    source_bytes = identities.read_bytes()
    source = json.loads(source_bytes)
    first, repeat = streams([root / 'all-first/logits.f32', root / 'all-repeat/logits.f32'])
    hashes, winners, whole = [], [], [hashlib.sha256(), hashlib.sha256()]
    try:
        for index in range(ROWS):
            a, b = first.read(VOCAB * 4), repeat.read(VOCAB * 4)
            assert len(a) == len(b) == VOCAB * 4 and a == b, 'candidate own repeat changed'
            values = decode(a)
            hashes.append(hashlib.sha256(a).hexdigest())
            winners.append(max(range(VOCAB), key=values.__getitem__))
            whole[0].update(a)
            whole[1].update(b)
            if index % 128 == 0:
                print('SELF_REPEAT completed_rows=' + str(index + 1), flush=True)
        assert not first.read(1) and not repeat.read(1)
    finally:
        first.close()
        repeat.close()
    write(root / 'candidate-self-repeat-frozen.json', dict(
        phase='candidate self-repeat only; no oracle input', rows=ROWS, vocab=VOCAB,
        context=4096, teacher_chunk=128, input_sha256=ID_SHA, policy='all (norm/RoPE, norm/residual, routing and scaled reduction)',
        byte_exact_rows=ROWS, max_raw_delta=0, p99_top2_margin_movement=0,
        source_identities_sha256=hashlib.sha256(source_bytes).hexdigest(), source=source,
        whole_file_sha256=[digest.hexdigest() for digest in whole],
        row_sha256=hashes, argmax_ids=winners, original_ordinary_freeze_sha256=OLD_FREEZE,
        limitation='Repeat stability only. Ordinary-to-candidate movement is not noise.'))
    (root / 'candidate-self-repeat-frozen.json').chmod(0o444)


def compare(root, quality, frozen_sha):
    frozen_bytes = (root / 'candidate-self-repeat-frozen.json').read_bytes()
    assert hashlib.sha256(frozen_bytes).hexdigest() == frozen_sha
    frozen = json.loads(frozen_bytes)
    assert frozen['input_sha256'] == ID_SHA and frozen['rows'] == ROWS
    assert hashlib.sha256((quality / 'input1/ids.i32').read_bytes()).hexdigest() == ID_SHA
    assert hashlib.sha256((quality / 'native-noise-128-frozen.json').read_bytes()).hexdigest() == OLD_FREEZE
    paths = [root / 'all-first/logits.f32', quality / 'reference128-first/logits.f32']
    files = streams(paths)
    whole = [hashlib.sha256(), hashlib.sha256()]
    ids = struct.unpack(f'<{ROWS}i', (quality / 'input1/ids.i32').read_bytes())
    original_noise = json.loads((quality / 'native-noise-128-frozen.json').read_text())
    exact, mismatches, outside, maximum, maximum_tv = 0, 0, 0, 0.0, 0.0
    candidate_nll, stock_nll, comparisons = [], [], []
    try:
        for index in range(ROWS):
            a, b = [file.read(VOCAB * 4) for file in files]
            assert len(a) == len(b) == VOCAB * 4
            assert hashlib.sha256(a).hexdigest() == frozen['row_sha256'][index]
            for digest, raw in zip(whole, (a, b)):
                digest.update(raw)
            av, bv = decode(a), decode(b)
            ai = max(range(VOCAB), key=av.__getitem__)
            bi = max(range(VOCAB), key=bv.__getitem__)
            exact += a == b
            mismatch = ai != bi
            mismatches += mismatch
            margin = bv[bi] - bv[ai]
            outside += mismatch and margin > original_noise['p99_top2_margin_movement']
            raw_delta = max(abs(x-y) for x, y in zip(av, bv))
            maximum = max(maximum, raw_delta)
            am, bm = max(av), max(bv)
            aw = [math.exp(x-am) for x in av]
            bw = [math.exp(x-bm) for x in bv]
            at, bt = math.fsum(aw), math.fsum(bw)
            tv = .5 * math.fsum(abs(x/at-y/bt) for x, y in zip(aw, bw))
            maximum_tv = max(maximum_tv, tv)
            row = dict(row=index, byte_exact=a == b, candidate_argmax=ai, stock_argmax=bi,
                       max_raw_delta=raw_delta, full_softmax_tv=tv,
                       reference_winner_margin=margin)
            if index + 1 < ROWS:
                target = ids[index+1]
                an, bn = am+math.log(at)-av[target], bm+math.log(bt)-bv[target]
                candidate_nll.append(an)
                stock_nll.append(bn)
                row.update(target=target, candidate_target_nll=an, stock_target_nll=bn)
            comparisons.append(row)
            if index % 128 == 0:
                print('COMPARISON completed_rows=' + str(index+1), flush=True)
        assert all(not file.read(1) for file in files)
    finally:
        for file in files:
            file.close()
    digests = [digest.hexdigest() for digest in whole]
    assert digests[0] == frozen['whole_file_sha256'][0] and digests[1] == STOCK_SHA
    candidate_mean = math.fsum(candidate_nll)/len(candidate_nll)
    stock_mean = math.fsum(stock_nll)/len(stock_nll)
    result = dict(candidate_freeze_sha256=frozen_sha, original_ordinary_freeze_sha256=OLD_FREEZE,
                  rows=ROWS, vocab=VOCAB, input_sha256=ID_SHA, teacher_chunk=128,
                  whole_file_sha256=digests, byte_exact_rows=exact,
                  strict_argmax_mismatches=mismatches, outside_original_noise=outside,
                  max_raw_delta=maximum, max_full_softmax_tv=maximum_tv,
                  scored_targets=ROWS-1, candidate_mean_nll=candidate_mean,
                  stock_mean_nll=stock_mean, candidate_ppl=math.exp(candidate_mean),
                  stock_ppl=math.exp(stock_mean),
                  ppl_relative_percent=100*math.expm1(candidate_mean-stock_mean),
                  comparisons=comparisons,
                  scores_inherited_by_complete_byte_identity=exact == ROWS)
    write(root / 'comparison-stock.json', result)


def selftest():
    def raw(values):
        return array.array('f', values).tobytes()
    assert metrics(raw([0.0, 1]), raw([-0.0, 1])) == dict(
        byte_exact=False, strict_argmax_mismatch=False, max_raw_delta=0.0)
    assert metrics(raw([1, 1]), raw([1, 2]))['strict_argmax_mismatch']
    try:
        decode(raw([math.nan]))
    except AssertionError:
        pass
    else:
        raise AssertionError('nonfinite input accepted')
    # Exercise the actual freeze/compare entry points on bounded synthetic
    # heads, rather than only a numerical-equality helper.
    global ROWS, VOCAB, ID_SHA, STOCK_SHA, OLD_FREEZE
    saved = ROWS, VOCAB, ID_SHA, STOCK_SHA, OLD_FREEZE
    try:
        ROWS, VOCAB = 2, 4
        with tempfile.TemporaryDirectory() as tmp:
            root, quality = pathlib.Path(tmp)/'candidate', pathlib.Path(tmp)/'quality'
            for path in (root/'all-first', root/'all-repeat', quality/'input1',
                         quality/'reference128-first'):
                path.mkdir(parents=True)
            ids = struct.pack('<2i', 2, 3)
            (quality/'input1/ids.i32').write_bytes(ids)
            ID_SHA = hashlib.sha256(ids).hexdigest()
            noise = b'{"p99_top2_margin_movement":0}\n'
            (quality/'native-noise-128-frozen.json').write_bytes(noise)
            OLD_FREEZE = hashlib.sha256(noise).hexdigest()
            ordinary = raw([0.0,1,2,3, 0.0,1,2,3])
            reference = raw([-0.0,1,2,3, 0.0,1,2,3])
            for arm in ('all-first','all-repeat'):
                (root/arm/'logits.f32').write_bytes(ordinary)
            (quality/'reference128-first/logits.f32').write_bytes(reference)
            STOCK_SHA = hashlib.sha256(reference).hexdigest()
            identities = root/'source-identities.json'
            identities.write_text('{}\n')
            freeze(root, quality/'input1/ids.i32', identities)
            digest = hashlib.sha256((root/'candidate-self-repeat-frozen.json').read_bytes()).hexdigest()
            compare(root, quality, digest)
            measured = json.loads((root/'comparison-stock.json').read_text())
            assert measured['byte_exact_rows'] == 1 and measured['max_raw_delta'] == 0
            assert measured['strict_argmax_mismatches'] == 0
            assert measured['ppl_relative_percent'] == 0
            assert (quality/'native-noise-128-frozen.json').read_bytes() == noise
            try:
                freeze(root, quality/'input1/ids.i32', identities)
            except FileExistsError:
                pass
            else:
                raise AssertionError('immutable freeze could be overwritten')
    finally:
        ROWS, VOCAB, ID_SHA, STOCK_SHA, OLD_FREEZE = saved
    print('signed-zero, tie/argmax, nonfinite, actual freeze/compare and overwrite controls: PASS')


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    if sys.argv[1] == 'selftest':
        assert len(sys.argv) == 2
        selftest()
    elif sys.argv[1] == 'freeze':
        assert len(sys.argv) == 5
        freeze(*map(pathlib.Path, sys.argv[2:]))
    elif sys.argv[1] == 'compare':
        assert len(sys.argv) == 5
        compare(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4])
    else:
        raise AssertionError('unknown mode')
