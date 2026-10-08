#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-serving admission, independent own freeze, strict lineage/loss and bookends."""
import array
import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from pathlib import Path

V, PREFIX, HISTORY, OUTPUTS = 262144, 8063, 8192, 129
IMAGE = 'c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while b := f.read(1 << 20): h.update(b)
    return h.hexdigest()


def small(p, cap=1 << 20):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size <= cap
    b = p.read_bytes(); assert len(b) <= cap
    return b


def load(p, expected):
    b = small(p); assert hashlib.sha256(b).hexdigest() == expected
    return json.loads(b)


def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def source(root, expected):
    s = load(root / 'source.json', expected)
    assert s['schema'] == 'gemma31-serving-bridge-v1'
    assert s['recipe'] == dict(profile=31, context=8192, max_rows=256, owners=[1, 4],
        prefix=8063, completed_waves=128, emitted_rows=129, ignore_eog=True,
        margin_allowance=0, conditional_limit=0.03, exact_argmax_lineage=True,
        conditional_targets='frozen-native-emitted-0-through-127')
    assert s['image_manifest'] == IMAGE
    assert s['native_state_bytes'] == {'generation': 1719664640, 'corpus': 922746880}
    raw = s['raw_model']; p = Path(raw['path'])
    assert p.is_file() and not p.is_symlink() and p.stat().st_size == raw['bytes'] == 18822970304
    assert p.name == 'gemma-4-31B-it-UD-Q4_K_XL.gguf'
    assert raw['sha256'] == '9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575'
    for item in s['protected_files']:
        p = Path(item['path']); assert p.stat().st_size == item['bytes']
        assert digest(p) == item['sha256']
    b = small(root / 'inputs.i32', 131072)
    assert len(b) == 131072 and hashlib.sha256(b).hexdigest() == s['inputs_sha256']
    manifest = load(root / 'input-manifest.json', s['input_manifest_sha256'])
    assert manifest['input_sha256'] == s['inputs_sha256'] and manifest['owners'] == 4
    assert manifest['rows'] == 8192 and manifest['profile'] == 31
    ids = struct.unpack('<32768i', b)
    for owner in range(4):
        seq = ids[owner*HISTORY:(owner+1)*HISTORY]
        assert seq[0] == 2 and seq.count(2) == 1 and all(0 <= x < V for x in seq)
    return s, ids


def fields(lines, prefix, phase):
    match = [x for x in lines if x.startswith(prefix+' ') and f'phase={phase} ' in x]
    assert len(match) == 1
    return dict(x.split('=', 1) for x in match[0].split()[1:])


def official(s, source_sha, key, admission):
    for field in ('job', 'final', 'steps', 'log'):
        p = Path(admission[field]['path']); assert digest(p) == admission[field]['sha256']
    final = json.loads(small(Path(admission['final']['path'])))
    steps = json.loads(small(Path(admission['steps']['path'])))['steps']
    assert final['state'] == 'done' and final['rc'] == 0
    assert len(steps) == 1 and steps[0]['state'] == 'done' and steps[0]['rc'] == 0
    expected = s['producer_commands'][key].replace('__SOURCE_SHA256__', source_sha)
    if '__NATIVE_OWN_SHA256__' in expected:
        expected = expected.replace('__NATIVE_OWN_SHA256__', admission['native_own_sha256'])
    if '__ANCHOR_SHA256__' in expected:
        expected = expected.replace('__ANCHOR_SHA256__', admission['anchor_proof_sha256'])
    for marker,field in (('__REFERENCE_OWN_SHA256__','reference_own_sha256'),('__COMPARISON_SHA256__','comparison_sha256')):
        if marker in expected: expected=expected.replace(marker,admission[field])
    assert steps[0]['cmd'] == expected


def admit(root, s, key, admission, phases):
    engine, cohort, mode = key.split('-')[:3]; count = int(cohort[1:])
    assert engine in ('native', 'reference') and count in (1, 4)
    assert mode in ('quality', 'cycle', 'corpus') and (mode != 'corpus' or count == 1)
    log = small(root / (key+'.log')).decode(); lines = log.splitlines()
    assert digest(root / (key+'.log')) == admission['output_log_sha256']
    assert lines.count('SERVING_PROOF_RETIRED' if engine == 'native' else 'SERVING_REFERENCE_RETIRED') == 1
    if engine == 'reference':
        meta = [x for x in lines if x.startswith('SERVING_DRIVER ')]
        assert len(meta) == 1
        info, name = meta[0].split(' name=', 1)
        d = dict(x.split('=', 1) for x in info.split()[1:])
        assert name == 'NVIDIA GB10' and d['initialized'] == d['current_context'] == '1'
        assert d['ordinal'] == '0' and d['devices'] == '1' and d['major'] == '12' and d['minor'] == '1'
        assert int(d['sms']) > 0 and int(d['driver_api']) > 0 and re.fullmatch('[0-9a-f]{32}', d['uuid'])
        assert d['libcuda'] == s['reference_driver']['resolved']
        closure = load(Path(s['closure']['path']), s['closure']['sha256'])
        assert s['reference_driver'] in closure['runtime_dependencies'].values()
        retired = load(root / (key+'-container-retired.json'), admission['retirement_sha256'])
        assert retired['container_absent_after_checked_docker_query'] is True
        assert retired['owner_label'] == 'gemma26-late-moe'
        assert retired['name'] == 'llmp-gemma26-late-moe-serving-'+key
        assert re.fullmatch('[0-9a-f]{64}', retired['cid'])
        caches = re.findall(r'llama_kv_cache: size = .*?\(\s*(\d+) cells,\s*(\d+) layers,\s*(\d+)/(\d+) seqs\), K \(f16\): .*?V \(f16\):', log)
        assert sorted(tuple(map(int,x)) for x in caches) == [(1280,50,count,count),(8192,10,count,count)]
    walls = {}
    for phase in phases:
        if mode == 'corpus':
            f = fields(lines, 'SERVING_CORPUS' if engine == 'native' else 'SERVING_REFERENCE_CORPUS', phase)
            assert f['rows'] == '1024' and f['targets'] == '1023' and f['query_rows'] == '1' and f['cursor'] == '1024'
            if engine == 'native':
                extra = json.loads(f['extra'])
                assert extra['gemma31_candidate'] is True and extra['max_rows'] == 256
                assert extra['norm_rope_requested'] is True and extra['norm_add_requested'] is True
                assert extra['owner_attention_requested'] is True and extra['row_invariant'] is False
        else:
            f = fields(lines, 'SERVING_CYCLE' if engine == 'native' else 'SERVING_REFERENCE', phase)
            assert all(f[k] == v for k,v in dict(owners=str(count),emitted=str(count*129),waves='128',cursor='8191',pending_position='8191',timed=str(int(phase=='paid'))).items())
            walls[phase] = {k:float(f[k+'_seconds']) for k in ('cycle','prefill','decode')}
            assert all(math.isfinite(x) and x > 0 for x in walls[phase].values())
            if engine == 'native':
                assert f['policy_basis'] == 'last-built' and f['rows'] == f['segments'] == str(count)
                assert int(f['norm_rope']) > 0 and int(f['norm_add']) > 0
                assert int(f['owner']) == (60 if count == 4 else 0)
                extra = json.loads(f['extra'])
                assert extra['gemma31_candidate'] is True and extra['max_rows'] == 256
                assert extra['norm_rope_requested'] is True and extra['norm_add_requested'] is True
                assert extra['owner_attention_requested'] is True and extra['row_invariant'] is False
            else:
                assert f['cache_max'] == '8190' and f['context'] == str(8192*count)
                assert f['per_sequence_context'] == '8192' and f['seq_max'] == str(count)
                assert f['batch'] == f['ubatch'] == '256' and f['F16'] == f['normal_ring'] == '1'
    if engine == 'native' and mode != 'corpus':
        # Only this fresh-process first/warm prefill is a new plan. Later cache hits do not
        # update last_built_policy; no claim that these counts are executed-wave counters.
        p = fields(lines, 'SERVING_PREFILL', phases[0])
        assert p['policy_basis'] == 'last-built' and p['rows'] == '127' and p['segments'] == '1'
        assert int(p['norm_rope']) > 0 and int(p['norm_add']) > 0 and p['owner'] == '0'
    return walls


def row(f):
    b = f.read(V*4); assert len(b) == V*4 and sys.byteorder == 'little'
    a = array.array('f'); assert a.itemsize == 4; a.frombytes(b)
    assert all(math.isfinite(x) for x in a)
    return b, a


def tokens(p, count):
    b = small(p, count*4); assert len(b) == count*4
    t = struct.unpack('<'+str(count)+'i', b)
    assert all(0 <= x < V for x in t)
    return t


def names(engine, count, mode):
    if engine == 'reference':
        return {'heads.f32': (1024 if mode == 'corpus' else count*129)*V*4} | (
            {} if mode == 'corpus' else {'tokens.i32':count*129*4,'final-head.f32':count*V*4})
    result = {'layout.txt': None}
    for owner in range(count):
        result[f'heads-{owner}.f32'] = (1024 if mode == 'corpus' else 129)*V*4
        result[f'state-{owner}.bin'] = 922746880 if mode == 'corpus' else 1719664640
        if mode != 'corpus':
            result[f'tokens-{owner}.i32'] = 129*4
            result[f'history-{owner}.i32'] = 8191*4
            result[f'final-head-{owner}.f32'] = V*4
    return result


def identities(directory, sizes):
    assert {p.name for p in directory.iterdir()} == set(sizes)
    records = {}
    for name, size in sizes.items():
        p = directory/name; assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size == size if size is not None else 0 < p.stat().st_size <= 65536
        records[name] = dict(bytes=p.stat().st_size, sha256=digest(p))
    return records


def frozen(root, s, source_sha, engine, count, mode, expected, reauthenticate=False):
    p = root/f'{engine}-c{count}-{mode}-own.json'; proof = load(p, expected)
    assert proof['source_sha256'] == source_sha and proof['engine'] == engine and proof['count'] == count and proof['mode'] == mode
    assert proof['analysis_sha256'] == digest(Path(__file__)) and proof['complete_finite_repeat_exact'] is True
    assert proof['inputs_sha256'] == s['inputs_sha256'] and proof['files']['first'] == proof['files']['repeat']
    assert set(proof['files']['first']) == set(names(engine,count,mode))
    for name,size in names(engine,count,mode).items():
        assert proof['files']['first'][name]['bytes'] == size if size is not None else 0 < proof['files']['first'][name]['bytes'] <= 65536
    for name,item in proof['files']['first'].items():
        p = root/f'{engine}-c{count}-{mode}'/'first'/name
        assert p.stat().st_size == item['bytes']
        if reauthenticate: assert digest(p) == item['sha256']
    return proof


def own(root, source_sha, key, admissions_path, admissions_sha):
    s, supplied = source(root, source_sha)
    a = load(admissions_path, admissions_sha); official(s,source_sha,key,a)
    engine,cohort,mode = key.split('-'); count=int(cohort[1:])
    if engine == 'reference': frozen(root,s,source_sha,'native',count,mode,a['native_own_sha256'])
    assert mode in ('quality','corpus')
    admit(root,s,key,a,('first','repeat'))
    records = {}
    for phase in ('first','repeat'):
        directory=root/key/phase
        records[phase]=identities(directory,names(engine,count,mode))
        if mode == 'corpus':
            with (directory/('heads-0.f32' if engine=='native' else 'heads.f32')).open('rb') as f:
                for _ in range(1024): row(f)
                assert not f.read(1)
            continue
        expected = [tokens(directory/f'tokens-{o}.i32',129) for o in range(count)] if engine=='native' else [tokens(directory/'tokens.i32',129*count)[o::count] for o in range(count)]
        files=[(directory/f'heads-{o}.f32').open('rb') for o in range(count)] if engine=='native' else [(directory/'heads.f32').open('rb')]
        try:
            for r in range(129):
                for o in range(count):
                    b,values=row(files[o if engine=='native' else 0])
                    assert max(range(V),key=values.__getitem__) == expected[o][r]
                    if r==128:
                        final=small(directory/f'final-head-{o}.f32') if engine=='native' else small(directory/'final-head.f32',count*V*4)[o*V*4:(o+1)*V*4]
                        assert b == final
            assert all(not f.read(1) for f in files)
        finally:
            for f in files: f.close()
        if engine=='native':
            for o in range(count):
                history=tokens(directory/f'history-{o}.i32',8191)
                assert history[:8063] == supplied[o*8192:o*8192+8063] and history[8063:] == expected[o][:128]
    assert records['first']==records['repeat']
    save(root/(key+'-own.json'),dict(engine=engine,count=count,mode=mode,source_sha256=source_sha,
        analysis_sha256=digest(Path(__file__)),admissions_sha256=admissions_sha,
        inputs_sha256=s['inputs_sha256'],complete_finite_repeat_exact=True,files=records))


def carrier(root,source_sha,count,native_sha):
    s,ids=source(root,source_sha); frozen(root,s,source_sha,'native',count,'quality',native_sha)
    values=list(ids)
    for o in range(count):
        path=root/f'native-c{count}-quality'/'first'/f'tokens-{o}.i32'
        proof=load(root/f'native-c{count}-quality-own.json',native_sha)
        assert digest(path) == proof['files']['first'][path.name]['sha256']
        emitted=tokens(path,129)
        values[o*8192+8063:o*8192+8191]=emitted[:128]
    b=struct.pack('<32768i',*values)
    with (root/f'anchors-c{count}.i32').open('xb') as f:f.write(b)
    save(root/f'anchors-c{count}.json',dict(source_sha256=source_sha,native_own_sha256=native_sha,
         inputs_sha256=s['inputs_sha256'],anchor_carrier_sha256=hashlib.sha256(b).hexdigest()))


# Preserve the already-qualified current-pin full-vocabulary FP64 calculation.
NUMERICAL_SHA = '841b67633e6960ddbc81c1895963f25db24a52bc23b291049e65fb46864edc89'
NUMERICAL_PATH = Path(__file__).resolve().parent.parent / 'gemma-release-c2/analyze.py'
assert digest(NUMERICAL_PATH) == NUMERICAL_SHA
spec = importlib.util.spec_from_file_location('gemma_current_c2_numerics', NUMERICAL_PATH)
numerics = importlib.util.module_from_spec(spec); spec.loader.exec_module(numerics)
nll = numerics.nll


def compare(root,source_sha,count,mode,native_sha,reference_sha):
    s,ids=source(root,source_sha)
    for engine,expected in (('native',native_sha),('reference',reference_sha)):
        frozen(root,s,source_sha,engine,count,mode,expected,True)
    nr=root/f'native-c{count}-{mode}'/'first'; rr=root/f'reference-c{count}-{mode}'/'first'
    generated=[]
    if mode=='quality':
        own=load(root/f'native-c{count}-quality-own.json',native_sha)
        for o in range(count):
            path=nr/f'tokens-{o}.i32'
            assert digest(path)==own['files']['first'][path.name]['sha256']
            generated.append(tokens(path,129))
    native=[(nr/f'heads-{o}.f32').open('rb') for o in range(count)]
    public=(rr/'heads.f32').open('rb')
    strict=ties=lineage=exact=0; losses=[[],[]]; rows=1024 if mode=='corpus' else 129
    try:
        for r in range(rows):
            for o in range(count):
                nb,n=row(native[o]); rb,ref=row(public)
                ni=max(range(V),key=n.__getitem__); ri=max(range(V),key=ref.__getitem__)
                margin=float(ref[ri])-float(ref[ni])
                strict+=int(margin>0); ties+=int(ni!=ri and margin==0); lineage+=int(ni!=ri); exact+=int(nb==rb)
                if r < rows-1:
                    target=ids[r+1] if mode=='corpus' else generated[o][r]
                    losses[0].append(nll(n,target)); losses[1].append(nll(ref,target))
        assert all(not f.read(1) for f in native) and not public.read(1)
    finally:
        for f in native:f.close()
        public.close()
    means=[math.fsum(x)/len(x) for x in losses]
    overflow=False
    try: relative=math.expm1(means[0]-means[1])
    except OverflowError: relative=None; overflow=True
    conditional=not overflow and relative<=.03
    lineage_ok=lineage==0 if mode=='quality' else True
    passed=strict==0 and conditional and lineage_ok
    absolute_ppl = []
    for mean in means:
        try: absolute_ppl.append(math.exp(mean))
        except OverflowError: absolute_ppl.append(None)
    result=dict(source_sha256=source_sha,native_own_sha256=native_sha,reference_own_sha256=reference_sha,
        count=count,mode=mode,rows=rows*count,targets=len(losses[0]),strict_positive_margin_differences=strict,
        tied_argmax_differences=ties,argmax_id_differences=lineage,byte_exact_heads=exact,
        mean_native_nll=means[0],mean_reference_nll=means[1],relative_loss=relative,
        absolute_perplexity=dict(native=absolute_ppl[0],reference=absolute_ppl[1]),
        loss_scope='authentic scalar corpus' if mode=='corpus' else 'generated-history conditional',
        relative_loss_overflow=overflow,conditional_gate=conditional,
        exact_greedy_lineage_gate=lineage_ok if mode=='quality' else None,pass_all=passed)
    save(root/f'comparison-c{count}-{mode}.json',result)
    print(json.dumps(result,allow_nan=False)); return 0 if passed else 1


def carrier_ready(root,source_sha,count,native_sha,carrier_sha):
    s,ids=source(root,source_sha)
    proof=frozen(root,s,source_sha,'native',count,'quality',native_sha)
    record=load(root/f'anchors-c{count}.json',carrier_sha)
    assert record['source_sha256']==source_sha and record['native_own_sha256']==native_sha
    assert record['inputs_sha256']==s['inputs_sha256']
    b=small(root/f'anchors-c{count}.i32',131072)
    assert len(b)==131072 and hashlib.sha256(b).hexdigest()==record['anchor_carrier_sha256']
    supplied=list(struct.unpack('<32768i',b)); expected=list(ids)
    for o in range(count):
        path=root/f'native-c{count}-quality'/'first'/f'tokens-{o}.i32'
        assert digest(path)==proof['files']['first'][path.name]['sha256']
        expected[o*8192+8063:o*8192+8191]=tokens(path,129)[:128]
    assert supplied==expected


def timing_ready(root,source_sha,count,native_sha,reference_sha,comparison_sha):
    s,_=source(root,source_sha)
    for engine,sha in (('native',native_sha),('reference',reference_sha)):
        frozen(root,s,source_sha,engine,count,'quality',sha)
    comparison=load(root/f'comparison-c{count}-quality.json',comparison_sha)
    assert comparison['source_sha256']==source_sha and comparison['pass_all'] is True
    assert comparison['native_own_sha256']==native_sha and comparison['reference_own_sha256']==reference_sha
    assert comparison['exact_greedy_lineage_gate'] is True


def timing(root,source_sha,count,native_sha,reference_sha,comparison_sha,admissions_path,admissions_sha):
    timing_ready(root,source_sha,count,native_sha,reference_sha,comparison_sha)
    s,_=source(root,source_sha); admissions=load(admissions_path,admissions_sha)
    order=['reference-first','native-first','native-repeat','reference-repeat']
    assert admissions['order']==order and len(admissions['arms'])==4
    elapsed={'native':[],'reference':[]}; components={}; records={}
    for label in order:
        engine,arm=label.split('-'); key=f'{engine}-c{count}-cycle-{arm}'
        admission=admissions['arms'][label]
        official(s,source_sha,key,admission)
        walls=admit(root,s,key,admission,('warm','paid'))
        own=frozen(root,s,source_sha,engine,count,'quality',native_sha if engine=='native' else reference_sha)
        directory=root/key/'paid'
        sizes={name:item['bytes'] for name,item in own['files']['first'].items() if not name.startswith('heads')}
        actual=identities(directory,sizes)
        expected={name:item for name,item in own['files']['first'].items() if not name.startswith('heads')}
        assert actual==expected
        # Quality trace already proves all rows. Paid mode proves complete IDs, final head,
        # and native initialized state/layout; no claim of archived paid intermediate heads.
        elapsed[engine].append(walls['paid']['cycle']); components[label]=walls['paid']; records[label]=actual
    means={engine:math.fsum(values)/2 for engine,values in elapsed.items()}
    save(root/f'timing-c{count}.json',dict(source_sha256=source_sha,native_own_sha256=native_sha,
        reference_own_sha256=reference_sha,comparison_sha256=comparison_sha,
        admissions_sha256=admissions_sha,order=order,complete_paid_outputs_match_own=True,
        cycle_seconds=elapsed,mean_cycle_seconds=means,
        native_relative_elapsed=means['native']/means['reference']-1,
        spreads={k:max(v)-min(v) for k,v in elapsed.items()},components=components,
        scope='short whole-serving cycles including prefill/frontend/publication/retirement; no sustained claim',
        files=records))


if __name__=='__main__':
    action=sys.argv[1]; root=Path(sys.argv[2]); source_sha=sys.argv[3]
    if action=='guard':source(root,source_sha)
    elif action=='native-ready':
        s,_=source(root,source_sha);frozen(root,s,source_sha,'native',int(sys.argv[4]),sys.argv[5],sys.argv[6])
    elif action=='own':own(root,source_sha,sys.argv[4],Path(sys.argv[5]),sys.argv[6])
    elif action=='carrier':carrier(root,source_sha,int(sys.argv[4]),sys.argv[5])
    elif action=='carrier-ready':carrier_ready(root,source_sha,int(sys.argv[4]),sys.argv[5],sys.argv[6])
    elif action=='timing-ready':timing_ready(root,source_sha,int(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7])
    elif action=='timing':timing(root,source_sha,int(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7],Path(sys.argv[8]),sys.argv[9])
    elif action=='compare':raise SystemExit(compare(root,source_sha,int(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7]))
    else:raise SystemExit('unknown bounded analysis action')
