#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed Gemma31 C2 admission/own/strict comparison; no empirical allowance."""
import array
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

VOCAB, OWNERS, ROWS = 262144, 2, 33
INPUT_SHA = 'd584450079145f3d2c93f46ef24a0aabe2b3971279a1cbddbbb29f0a506ac5e3'
IMAGE = 'c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        while block := file.read(1048576):
            h.update(block)
    return h.hexdigest()


def small(path, limit=1048576):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size <= limit
    return path.read_bytes()


def load(path, expected):
    data = small(path)
    assert hashlib.sha256(data).hexdigest() == expected
    return json.loads(data)


def exclusive(path, record):
    with path.open('x') as file:
        json.dump(record, file, indent=2)
        file.write('\n')


def source(root, expected):
    s = load(root / 'source.json', expected)
    assert s['schema'] == 'gemma-release-c2-v1'
    assert s['recipe'] == {'profile': 31, 'owners': 2, 'context': 4096, 'max_rows': 256,
                           'headcap': 2, 'prefill': 992, 'steps': 32,
                           'margin_allowance': 0, 'conditional_limit': 0.03}
    assert s['input_sha256'] == INPUT_SHA and s['image_manifest'] == IMAGE
    assert s['native_owner_state_bytes'] == 922746880
    raw = s['raw_model']
    assert Path(raw['path']).name == 'gemma-4-31B-it-UD-Q4_K_XL.gguf'
    assert Path(raw['path']).is_file() and Path(raw['path']).stat().st_size == raw['bytes'] == 18822970304
    assert raw['sha256'] == '9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575'
    # Retained approved whole-checkpoint SHA; current name/length only, no repeated18GB read.
    measured = load(Path(s['native_receipts']['measured']['path']), s['native_receipts']['measured']['sha256'])
    canonical = load(Path(s['native_receipts']['canonical']['path']), s['native_receipts']['canonical']['sha256'])
    assert measured['source_lock']['sha256'] == s['native_receipts']['measured_source_lock_sha256']
    assert canonical['source_lock']['sha256'] == s['native_receipts']['canonical_source_lock_sha256']
    measured.pop('source_lock'); canonical.pop('source_lock')
    assert measured == canonical
    # This list is fixed in the preregistered source, not inferred from outputs.
    for item in s['protected_files']:
        path = Path(item['path'])
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
    return s


def inputs(path):
    b = small(path, 49152)
    assert len(b) == 49152 and hashlib.sha256(b).hexdigest() == INPUT_SHA
    ids = struct.unpack('<12288i', b)
    for owner in range(12):
        history = ids[owner * 1024:(owner + 1) * 1024]
        assert history[0] == 2 and history.count(2) == 1
        assert all(0 <= token < VOCAB for token in history)
    return ids


def head(file):
    b = file.read(VOCAB * 4)
    assert len(b) == VOCAB * 4 and sys.byteorder == 'little'
    a = array.array('f'); assert a.itemsize == 4; a.frombytes(b)
    assert all(math.isfinite(x) for x in a)
    return b, a


def fields(lines, prefix):
    matches = [line for line in lines if line.startswith(prefix + ' ')]
    assert len(matches) == 1
    return dict(piece.split('=', 1) for piece in matches[0].split()[1:])



def warm_policy(lines):
    # Immutable helper emits this once on the first warm decode, before reset/paid work.
    selected = [line for line in lines if line.startswith('JOINED_SELECTED phase=decode-first-build ')]
    assert len(selected) == 1
    f = fields(selected, 'JOINED_SELECTED')
    assert f == {'phase': 'decode-first-build', 'owner': '0', 'rows': '2', 'segments': '2',
                 'requested_rows': '2', 'requested_segments': '2', 'policy_basis': 'last-built',
                 'row_products': '0', 'norm_rope': '120', 'norm_add': '120', 'norm_fused': '121',
                 'rope_store': '0', 'owner_attention_steps': '60', 'requested_cohort8_steps': '0',
                 'requested_cohort12_steps': '0', 'requested_partial_cohort_steps': '0',
                 'shared_vecq': '0', 'gemma_route': '0', 'gemma_reduce': '0', 'lane_steps': '0'}


def admit(root, engine, arm, s, admission):
    assert engine in ('native', 'reference')
    directory = root / engine / arm
    ids = inputs(directory / 'inputs.i32')
    assert (directory / 'cohort.logits.f32').stat().st_size == ROWS * OWNERS * VOCAB * 4
    log = small(root / engine / (arm + '.log')).decode()
    lines = log.splitlines()
    prefix = 'JOINED_NATIVE' if engine == 'native' else 'JOINED_REFERENCE'
    assert lines.count(prefix + '_RETIRED production=1') == 1
    f = fields(lines, prefix)
    common = {'owners': '2', 'mode': 'joined', 'recipe': 'production', 'completed_waves': '32',
              'completed_units': '64', 'first_past': '992', 'retained_rows': '66',
              'input_mode': 'supplied', 'unequal_past': '0', 'ubatch': '256'}
    if engine == 'native':
        warm_policy(lines)
        del common['ubatch']
        common.update(variant='31', context='4096', max_rows='256', max_head_rows='2',
                      paid_gpu_groups='32', rows='2', segments='2', owner_requested='1',
                      owner_attention_steps='60', row_products='0', norm_rope='120',
                      norm_add='120', norm_fused='121', rope_store='0', shared_vecq='0',
                      gemma_route='0', gemma_reduce='0', heap_funded='73515008')
        assert int(f['captured_delta']) + int(f['replayed_delta']) == 32
    else:
        common.update(context='8192', per_sequence_context='4096', batch='256', seq_max='2',
                      physical_decode_groups='1', fusion='enabled', graphs='allowed')
        caches = re.findall(r'llama_kv_cache: size = .*?\(\s*(\d+) cells,\s*(\d+) layers,\s*(\d+)/(\d+) seqs\), K \(f16\): .*?V \(f16\):', log)
        assert sorted(tuple(map(int, x)) for x in caches) == [(1280, 50, 2, 2), (4096, 10, 2, 2)]
        profile = fields(lines, 'JOINED_REFERENCE_PROFILE')
        assert profile == {'recipe': 'production', 'variant': '31', 'mode': 'joined',
                           'context': '8192', 'batch': '256', 'ubatch': '256',
                           'swa_full': '0', 'kv_unified': '0', 'type_k': 'F16', 'type_v': 'F16'}
        driver_lines = [line for line in lines if line.startswith('C2_DRIVER ')]
        assert len(driver_lines) == 1
        meta, name = driver_lines[0].split(' name=', 1)
        driver = dict(x.split('=', 1) for x in meta.split()[1:])
        assert name == 'NVIDIA GB10' and driver['initialized'] == driver['current_context'] == '1'
        assert driver['ordinal'] == '0' and driver['devices'] == '1'
        assert driver['major'] == '12' and driver['minor'] == '1' and int(driver['sms']) > 0
        assert int(driver['driver_api']) > 0 and re.fullmatch('[0-9a-f]{32}', driver['uuid'])
        assert driver['libcuda'] == s['reference_driver']['resolved']
        # Model-process path is bound to the captured exact image closure identity.
        closure = load(Path(s['closure']['path']), s['closure']['sha256'])
        assert any(item == s['reference_driver'] for item in closure['runtime_dependencies'].values())
        retired = load(root / engine / (arm + '-container-retired.json'),
                       admission['retirement_sha256'])
        assert retired['container_absent_after_checked_docker_query'] is True
        assert retired['owner_label'] == 'gemma26-late-moe'
        assert re.fullmatch('[0-9a-f]{64}', retired['cid'])
        assert retired['name'] == 'llmp-gemma26-late-moe-release-c2-' + arm
    assert all(f[k] == v for k, v in common.items()) and float(f['seconds']) > 0
    records = {}
    for line in lines:
        if line.startswith('JOINED_FRONTIER ') or line.startswith('JOINED_TOKEN '):
            v = dict(x.split('=', 1) for x in line.split()[1:])
            owner = int(v['owner'])
            row = 0 if line.startswith('JOINED_FRONTIER ') else int(v['step']) + 1
            assert 0 <= owner < 2 and 0 <= row < 33 and (row, owner) not in records
            if row == 0:
                assert v['completed'] == '992'
            else:
                assert int(v['forced']) == ids[owner * 1024 + 992 + row - 1]
            records[row, owner] = int(v['argmax'])
    assert len(records) == 66
    wall = float(f['seconds'])
    assert math.isfinite(wall) and wall > 0
    return records, wall


def identities(directory, engine, s):
    names = ['inputs.i32', 'cohort.logits.f32']
    sizes = {'inputs.i32': 49152, 'cohort.logits.f32': 69206016}
    if engine == 'native':
        for phase in ('frontier', 'final'):
            names.append(phase + '.layout.txt')
            for owner in range(2):
                name = f'{phase}.owner{owner}.state.bin'
                names.append(name); sizes[name] = s['native_owner_state_bytes']
    result = {}
    for name in names:
        p = directory / name
        assert p.is_file() and not p.is_symlink()
        length = p.stat().st_size
        assert length == sizes[name] if name in sizes else 0 < length <= 65536
        result[name] = {'bytes': length, 'sha256': digest(p)}
    return result


def producer(admissions, engine, s, source_sha):
    a = admissions[engine]
    for arm in ('first', 'repeat'):
        j = a[arm]
        for key in ('job', 'final', 'steps', 'log'):
            assert digest(Path(j[key]['path'])) == j[key]['sha256']
        final = json.loads(small(Path(j['final']['path'])))
        steps = json.loads(small(Path(j['steps']['path'])))['steps']
        assert final['state'] == 'done' and final['rc'] == 0
        assert len(steps) == j['step_count'] == 1 and all(x['state'] == 'done' and x['rc'] == 0 for x in steps)
        expected = s['producer_commands'][engine][arm].replace('__SOURCE_SHA256__', source_sha)
        if engine == 'reference':
            expected = expected.replace('__NATIVE_OWN_SHA256__', admissions['native_own_sha256'])
        assert steps[0]['cmd'] == expected


def native_ready(root, source_sha, native_sha):
    s = source(root, source_sha)
    own = load(root / 'native-own.json', native_sha)
    assert own['engine'] == 'native' and own['source_sha256'] == source_sha
    assert own['analysis_sha256'] == digest(Path(__file__)) and own['complete_finite_repeat_exact'] is True
    assert own['files']['first'] == own['files']['repeat']
    assert own['files']['first']['inputs.i32'] == {'bytes': 49152, 'sha256': INPUT_SHA}
    assert own['files']['first']['cohort.logits.f32']['bytes'] == 69206016
    assert len(own['files']['first']) == 8


def own(root, engine, source_sha, admissions_path, admissions_sha):
    s = source(root, source_sha)
    admissions = load(admissions_path, admissions_sha)
    producer(admissions, engine, s, source_sha)
    if engine == 'reference':
        # The caller pins the independently completed native freeze before any stock own read.
        native_ready(root, source_sha, admissions['native_own_sha256'])
    out = {}
    for arm in ('first', 'repeat'):
        assert digest(root / engine / (arm + '.log')) == admissions[engine][arm]['output_log_sha256']
        records, wall = admit(root, engine, arm, s, admissions[engine][arm])
        directory = root / engine / arm
        with (directory / 'cohort.logits.f32').open('rb') as file:
            for index in range(66):
                _, row = head(file)
                assert max(range(VOCAB), key=row.__getitem__) == records[divmod(index, 2)]
            assert not file.read(1)
        out[arm] = identities(directory, engine, s)
    assert out['first'] == out['repeat']
    exclusive(root / (engine + '-own.json'), {'engine': engine, 'source_sha256': source_sha,
        'analysis_sha256': digest(Path(__file__)), 'admissions_sha256': admissions_sha,
        'complete_finite_repeat_exact': True, 'files': out})


def nll(row, target):
    peak = max(row)
    return math.log(math.fsum(math.exp(float(x) - peak) for x in row)) + (peak - row[target])


def compare(root, source_sha, native_sha, reference_sha):
    s = source(root, source_sha)
    for engine, sha in (('native', native_sha), ('reference', reference_sha)):
        own = load(root / (engine + '-own.json'), sha)
        assert own['source_sha256'] == source_sha and own['analysis_sha256'] == digest(Path(__file__))
        assert own['complete_finite_repeat_exact'] is True
        for arm in ('first', 'repeat'):
            assert identities(root / engine / arm, engine, s) == own['files'][arm]
    ids = inputs(root / 'native/first/inputs.i32')
    native_loss, reference_loss, differences = [], [], []
    exact = 0; maximum = 0.0
    with (root / 'native/first/cohort.logits.f32').open('rb') as af, (root / 'reference/first/cohort.logits.f32').open('rb') as bf:
        for index in range(66):
            ab, a = head(af); bb, b = head(bf)
            exact += ab == bb
            ai, bi = max(range(VOCAB), key=a.__getitem__), max(range(VOCAB), key=b.__getitem__)
            maximum = max(maximum, max(abs(float(x) - y) for x, y in zip(a, b)))
            if ai != bi:
                margin = float(b[bi]) - b[ai]
                assert margin >= 0
                differences.append({'row': index, 'reference_margin': margin, 'tie': margin == 0})
            row, owner = divmod(index, 2)
            if row < 32:
                target = ids[owner * 1024 + 992 + row]
                native_loss.append(nll(a, target)); reference_loss.append(nll(b, target))
    assert len(native_loss) == len(reference_loss) == 64
    native_mean, reference_mean = math.fsum(native_loss) / 64, math.fsum(reference_loss) / 64
    log_ratio = native_mean - reference_mean
    try:
        relative = math.expm1(log_ratio)
    except OverflowError:
        relative = None
    conditional_pass = relative is not None and relative <= 0.03
    strict = sum(not x['tie'] for x in differences)
    record = {'native_own_sha256': native_sha, 'reference_own_sha256': reference_sha,
              'source_sha256': source_sha, 'complete_rows': 66, 'byte_exact_rows': exact,
              'max_raw_delta': maximum, 'strict_positive_margin_differences': strict,
              'tie_differences': sum(x['tie'] for x in differences), 'differences': differences,
              'conditional_target_count': 64, 'native_mean_conditional_nll': native_mean,
              'reference_mean_conditional_nll': reference_mean, 'relative_conditional_loss': relative,
              'conditional_log_ratio': log_ratio, 'conditional_ratio_overflow': relative is None,
              'strict_gate_passed': strict == 0, 'conditional_gate_passed': conditional_pass,
              'quality_gate_passed': strict == 0 and conditional_pass, 'full_corpus_claim': False}
    exclusive(root / 'comparison.json', record)
    return 0 if record['quality_gate_passed'] else 1


def timing_ready(root, source_sha, native_sha, reference_sha, comparison_sha):
    source(root, source_sha)
    record = load(root / 'comparison.json', comparison_sha)
    assert record['source_sha256'] == source_sha and record['quality_gate_passed'] is True
    assert record['native_own_sha256'] == native_sha and record['reference_own_sha256'] == reference_sha
    for engine, sha in (('native', native_sha), ('reference', reference_sha)):
        own = load(root / (engine + '-own.json'), sha)
        assert own['source_sha256'] == source_sha and own['complete_finite_repeat_exact'] is True


def timing(root, source_sha, native_sha, reference_sha, comparison_sha, admissions_path, admissions_sha):
    timing_ready(root, source_sha, native_sha, reference_sha, comparison_sha)
    s = source(root, source_sha)
    admissions = load(admissions_path, admissions_sha)
    final = load(Path(admissions['final']['path']), admissions['final']['sha256'])
    steps = load(Path(admissions['steps']['path']), admissions['steps']['sha256'])['steps']
    for key in ('job', 'log'):
        assert digest(Path(admissions[key]['path'])) == admissions[key]['sha256']
    assert final['state'] == 'done' and final['rc'] == 0 and len(steps) == 4
    order = [('reference', 'timing-first'), ('native', 'timing-first'),
             ('native', 'timing-repeat'), ('reference', 'timing-repeat')]
    walls = {'native': [], 'reference': []}
    for step, (engine, arm) in zip(steps, order):
        assert step['state'] == 'done' and step['rc'] == 0
        expected = s['producer_commands'][engine][arm]
        for marker, value in (('__SOURCE_SHA256__', source_sha), ('__NATIVE_OWN_SHA256__', native_sha),
                              ('__REFERENCE_OWN_SHA256__', reference_sha), ('__COMPARISON_SHA256__', comparison_sha)):
            expected = expected.replace(marker, value)
        assert step['cmd'] == expected
        a = admissions['arms'][engine][arm]
        assert digest(root / engine / (arm + '.log')) == a['output_log_sha256']
        _, wall = admit(root, engine, arm, s, a)
        frozen = load(root / (engine + '-own.json'), native_sha if engine == 'native' else reference_sha)
        assert identities(root / engine / arm, engine, s) == frozen['files']['first']
        walls[engine].append(wall)
    means = {engine: sum(values) / 2 for engine, values in walls.items()}
    exclusive(root / 'timing.json', {'source_sha256': source_sha, 'native_own_sha256': native_sha,
        'reference_own_sha256': reference_sha, 'comparison_sha256': comparison_sha,
        'full_output_identities_match_own': True, 'walls': walls, 'means': means,
        'spreads': {engine: abs(values[0] - values[1]) for engine, values in walls.items()},
        'native_relative_elapsed': means['native'] / means['reference'] - 1,
        'scope': 'Short matched32 forced physicalC2 waves, no sustained/default qualification'})


def selftest():
    # Fixed row-to-target chronology includes frontier and excludes completed1024 row.
    pairs = [(row, owner, 992 + row) for row in range(33) for owner in range(2) if row < 32]
    assert len(pairs) == 64 and pairs[-1] == (31, 1, 1023)
    assert abs(nll([0.0, 0.0], 0) - math.log(2)) < 1e-15
    assert nll([1000.0, 999.0], 0) == nll([0.0, -1.0], 0)
    assert math.expm1(math.log1p(0.03)) <= 0.03 + 1e-16
    try:
        math.expm1(1e30)
        raise AssertionError('expected overflow control')
    except OverflowError:
        pass
    valid = ('JOINED_SELECTED phase=decode-first-build owner=0 rows=2 segments=2 '
             'requested_rows=2 requested_segments=2 policy_basis=last-built row_products=0 '
             'norm_rope=120 norm_add=120 norm_fused=121 rope_store=0 owner_attention_steps=60 '
             'requested_cohort8_steps=0 requested_cohort12_steps=0 requested_partial_cohort_steps=0 '
             'shared_vecq=0 gemma_route=0 gemma_reduce=0 lane_steps=0')
    warm_policy([valid])
    for bad in ([], [valid, valid], [valid.replace('rows=2', 'rows=1')],
                [valid.replace('owner_attention_steps=60', 'owner_attention_steps=0')]):
        try:
            warm_policy(bad)
        except AssertionError:
            continue
        raise AssertionError('invalid warm witness admitted')
    print('C2_ANALYSIS_SELFTEST_PASS')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['selftest']:
        selftest()
    elif len(args) == 3 and args[0] == 'guard':
        source(Path(args[1]), args[2])
    elif len(args) == 4 and args[0] == 'native-ready':
        native_ready(Path(args[1]), args[2], args[3])
    elif len(args) == 6 and args[0] == 'timing-ready':
        timing_ready(Path(args[1]), *args[2:])
    elif len(args) == 8 and args[0] == 'timing':
        timing(Path(args[1]), *args[2:6], Path(args[6]), args[7])
    elif len(args) == 6 and args[0] == 'own':
        own(Path(args[1]), args[2], args[3], Path(args[4]), args[5])
    elif len(args) == 5 and args[0] == 'compare':
        sys.exit(compare(Path(args[1]), args[2], args[3], args[4]))
    else:
        raise SystemExit('selftest | guard ROOT SOURCE_SHA | own ROOT ENGINE SOURCE_SHA ADMISSIONS SHA | compare ROOT SOURCE_SHA NATIVE_OWN_SHA REFERENCE_OWN_SHA')
