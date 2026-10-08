#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze annotation environment; authenticate completed output against Task40."""
import datetime
import importlib.util
import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('ring26', ROOT/'docs/experiments/gemma26-swa-ring-transfer/own_freeze.py')
ring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ring)
BASE = '36811f4ac93975c490e9118b93506aa3880df298'
PREFIX = 'docs/experiments/gemma26-prefill-profile/'
SOURCES = ['benchmarks/CMakeLists.txt', 'benchmarks/gemma26_prefill_profile.cc',
           'benchmarks/gemma_prefill.cc', 'docs/experiments/gemma26-swa-ring-transfer/own_freeze.py',
           'docs/experiments/gemma-prefill-large/compare.py',
           'docs/experiments/gemma-dense31-c1-norm/own_freeze.py'] + [PREFIX+n for n in
           ['llama_prefill_profile.cc','profile_child.cc','reference.sh','native.sh',
            'validate.py','trace.py','container_retire.py','PROTOCOL.md']]
TASK40_OWN = 'a1f7cea38f1f4db2122502382735a34af52738e84add31cd0199b166b0ff25ad'
TASK40_SOURCE = '25e5e05a06cb31c22c0fce24882713ce21f2932e2383a31cf3ea673fb5443f9f'
TASK40_LOG = '604c5ccc2706f91c3b3368940c6c6636dd894e7922fc1cedc292ead40094affd'
TASK40_HEADS = {
 'native': ['1dc3ba34b12692c5c416740a869eb43e68a586e7b5b1fb4bb60f8105799880fd',
            'e07cbe2d33e5234fb1e7f90dbc3642a6775e59e8cadbd6fe4b13db646fc5e853'],
 'original': ['254e6cf57dea36d5da3be3e0d53157bb886924b61e6493050daa147e82a2eaa3',
              '48c4a6123708f182d28e6063f124b6b5facdbda881046be6b1f1e239d652f2a6']}
INTERFACE = '6311a17d35c6658ec4c99d5006283c7098054b5d99faa89e80ace2a6ca97c7c9'


def completion(path):
    record = json.loads(path.read_text())
    assert record['phase'] == 'child_exit0_after_source_verified_release'
    assert record['child_status'] == 0 and isinstance(record['child_pid'], int) and record['child_pid'] > 0
    return ring.identity(path)


def environment(root, scratch):
    sources = {n:ring.identity(root/n) for n in SOURCES}
    assert sources['benchmarks/gemma_prefill.cc']['sha256'] == '8fc6e4b377f5fa120d1dcc82f44531c2b201eb0462905fbe9ff1c68e45b772a2'
    ancestry = json.loads((scratch/'source-ancestry.json').read_text())
    assert ancestry['base'] == BASE and ancestry['tracked_production_diff'] == []
    assert ancestry['source_files'] == sources and set(ancestry['dirty_paths']).issubset(SOURCES)
    assert ring.identity(scratch/'ids.i32') == {'bytes':8227*4,'sha256':ring.IDS}
    headers = {n:ring.identity(scratch/'headers'/n) for n in ring.pinned.HEADERS}
    assert {n:v['sha256'] for n,v in headers.items()} == ring.pinned.HEADERS
    libraries = dict(line.split()[::-1] for line in (scratch/'libraries.sha256').read_text().splitlines())
    assert libraries == ring.LIBRARIES
    artifact = pathlib.Path.home()/'.local/share/llmp/m3-artifacts'/ring.ARTIFACT
    assert ring.identity(artifact/'manifest.json')['sha256'] == ring.ARTIFACT
    assert ring.identity(artifact/'index.json')['sha256'] == ring.INDEX
    interface_path = scratch/'nvtx-interface.json'
    assert ring.identity(interface_path)['sha256'] == INTERFACE
    interface = json.loads(interface_path.read_text())
    observer = pathlib.Path(interface['observer_root'])
    assert all(ring.identity(observer/n) == v for n,v in interface['headers'].items())
    assert all(ring.identity(pathlib.Path(n)) == v for n,v in interface['tools'].items())
    baseline = pathlib.Path.home()/'.local/share/llmp/gemma26-swa-ring-transfer'
    assert ring.identity(baseline/'native-frozen.json')['sha256'] == TASK40_OWN
    assert ring.identity(baseline/'source-identities.json')['sha256'] == TASK40_SOURCE
    baseline_log = pathlib.Path.home()/'.local/share/llmp/jobs/m35-gemma26-swa-ring-bookend1/log'
    assert ring.identity(baseline_log)['sha256'] == TASK40_LOG
    return {'base':BASE,'sources':sources,'ancestry':ancestry,'headers':headers,'libraries':libraries,
            'ids':ring.identity(scratch/'ids.i32'),'manifest':ring.identity(artifact/'manifest.json'),
            'index':ring.identity(artifact/'index.json'),'interface':ring.identity(interface_path),
            'production_source_manifest':ring.identity(scratch/'production-source-manifest.json'),
            'build_receipt':ring.identity(root/'build/spark-native/llmp-receipt.json'),
            'native_binary':ring.identity(root/'build/spark-native/benchmarks/llmp_gemma26_prefill_profile'),
            'original_binary':ring.identity(scratch/'llama_prefill_profile'),
            'child_binary':ring.identity(scratch/'profile_child'),
            'task40_own':ring.identity(baseline/'native-frozen.json'),
            'task40_source':ring.identity(baseline/'source-identities.json'),
            'task40_bookend_log':ring.identity(baseline_log)}


def tokens_equal(a,b):
    return [list(t) for t in a] == [list(t) for t in b]


def check_run(scratch, name, engine, job):
    assert engine in TASK40_HEADS and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',name)
    retirement = ring.retired(job)
    log = pathlib.Path.home()/'.local/share/llmp/jobs'/job/'log'
    runs = ring.old.runs(log,'all','26')
    assert len(runs) == 1 and runs[0]['engine'] == ('PREFILL_NATIVE' if engine == 'native' else 'PREFILL_REFERENCE')
    baseline_log = pathlib.Path.home()/'.local/share/llmp/jobs/m35-gemma26-swa-ring-bookend1/log'
    baseline_runs = ring.old.runs(baseline_log,'all','26')
    baseline = baseline_runs[1] if engine == 'native' else baseline_runs[0]
    assert tokens_equal(runs[0]['tokens'],baseline['tokens']) and len(runs[0]['tokens']) == 32
    # No time comparison: annotation and profiling are observer controls.
    assert {k:v for k,v in runs[0]['metrics'].items() if k not in ['prefill_seconds','decode_seconds']} == {
        k:v for k,v in baseline['metrics'].items() if k not in ['prefill_seconds','decode_seconds']}
    record = {'heads':{}}
    for n,sha in zip(['prefill.f32','final.f32'],TASK40_HEADS[engine],strict=True):
        _,actual = ring.old.head(scratch/name/n)
        assert actual == sha
        record['heads'][n] = ring.identity(scratch/name/n)
    if engine == 'native':
        actual = ring.record(scratch/name)
        old = json.loads((pathlib.Path.home()/'.local/share/llmp/gemma26-swa-ring-transfer/native-frozen.json').read_text())
        assert actual == old['records']['native-first']
        record.update(actual)
    else:
        container = scratch/(name+'-container-retired.json')
        container_record = json.loads(container.read_text())
        assert container_record['container_absent_after_checked_docker_query'] is True
        assert container_record['owner_label'] == 'gemma26-prefill-profile'
        assert container_record['name'] == 'llmp-gemma26-prefill-profile-'+name
        assert container_record['cid'] == (scratch/(name+'.cid')).read_text().strip()
        assert re.fullmatch('[0-9a-f]{64}',container_record['cid'])
        container_retirement = ring.identity(container)
        text = log.read_text()
        assert len(re.findall(r'creating\s+SWA KV cache, size = 2048 cells',text)) == 1
        assert text.count('creating non-SWA KV cache, size = 16384 cells') == 1
    return {'engine':engine,'output':name,'complete_heads_choices_and_native_state_match_task40':True,
            'completion':completion(scratch/(name+'-completion.json')),'official_retirement':retirement,
            'record':record,'container_retirement':container_retirement if engine == 'original' else None,
            'timing_is_observer_diagnostic_only':True}


def write(path, result):
    with path.open('x') as output:
        json.dump(result,output,indent=2,sort_keys=True); output.write('\n')
    path.chmod(0o444)
    print(json.dumps({'identity':ring.identity(path),**result},indent=2,sort_keys=True))


def selftest():
    assert tokens_equal([(7,11),(13,17)],[[7,11],[13,17]])
    assert not tokens_equal([(7,11)],[[7,12]])
    assert not tokens_equal([(7,11)],[[7,11],[9,10]])
    print('normalized identities and changed/incomplete choices refused')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        selftest()
    else:
        mode,root,scratch,job = sys.argv[1:5]
        root,scratch = pathlib.Path(root),pathlib.Path(scratch)
        env = environment(root,scratch)
        common = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if mode == 'source':
            assert len(sys.argv) == 5
            write(scratch/'source-identities.json',{**common,'environment':env,'official_retirement':ring.retired(job)})
        else:
            assert mode == 'result' and len(sys.argv) == 8
            expected,name,engine = sys.argv[5:8]
            pre = scratch/'source-identities.json'
            assert ring.identity(pre)['sha256'] == expected
            assert json.loads(pre.read_text())['environment'] == env
            write(scratch/(name+'-validated.json'),{**common,'source_freeze':ring.identity(pre),
                                                   **check_run(scratch,name,engine,job)})
