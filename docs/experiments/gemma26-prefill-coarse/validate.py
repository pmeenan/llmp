#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate an isolated three-unit overlay and native Task40 fidelity."""
import datetime
import importlib.util
import json
import pathlib
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('profile', ROOT/'docs/experiments/gemma26-prefill-profile/validate.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
ring = profile.ring
BASE = 'd08eea75fac191bdb0e6ad6d6535ea1363d71f23'
PREFIX = 'docs/experiments/gemma26-prefill-coarse/'
SOURCES = ['benchmarks/CMakeLists.txt', 'benchmarks/gemma26_prefill_coarse.cc',
           'benchmarks/gemma26_prefill_coarse.h', 'benchmarks/gemma26_prefill_coarse_scopes.cc',
           'benchmarks/gemma26_prefill_clock_probe.cc', 'benchmarks/gemma26_prefill_profile.cc',
           'benchmarks/gemma_prefill.cc', 'docs/experiments/gemma26-prefill-profile/validate.py',
           'docs/experiments/gemma26-prefill-profile/trace_v2.py',
           'docs/experiments/gemma26-swa-ring-transfer/own_freeze.py',
           'docs/experiments/gemma-prefill-large/compare.py',
           'docs/experiments/gemma-dense31-c1-norm/own_freeze.py'] + [PREFIX+n for n in
           ['apply_overlay.py', 'overlay-identities.json', 'instrumentation.patch',
            'profile_child.cc', 'native.sh', 'probe.sh', 'validate.py', 'analyze.py', 'PROTOCOL.md']]


def prepare(root, destination):
    """Local metadata only, before applying the overlay in another tree."""
    git = lambda *args: subprocess.check_output(['git', *args], cwd=root, text=True)
    assert git('rev-parse', 'HEAD').strip() == BASE
    dirty = git('status', '--porcelain', '--untracked-files=all').splitlines()
    allowed = set(SOURCES) | {PREFIX+'overlay-identities.json.license', PREFIX+'instrumentation.patch.license'}
    assert all(line[3:] in allowed for line in dirty), dirty
    paths = git('ls-tree', '-r', '--name-only', BASE).splitlines()
    paths = [p for p in paths if p.startswith(('src/', 'cmake/', 'third_party/', 'toolchains/', 'tests/support/'))
             or p in ['CMakeLists.txt', 'CMakePresets.json', 'mise.toml', 'mise.lock']]
    canonical = {}
    for p in paths:
        original = subprocess.check_output(['git', 'show', BASE+':'+p], cwd=root)
        actual = (root/p).read_bytes()
        assert original == actual, p
        canonical[p] = ring.identity(root/p)
    overlay = json.loads((root/PREFIX/'overlay-identities.json').read_text())
    assert overlay['base'] == BASE
    assert all(canonical[p] == value for p,value in overlay['original'].items())
    assert ring.identity(root/PREFIX/'instrumentation.patch') == overlay['patch']
    measured = {**canonical, **overlay['instrumented']}
    profile.write(destination, {'base':BASE, 'canonical':canonical, 'measured':measured,
                               'overlay':overlay, 'sources':{p:ring.identity(root/p) for p in SOURCES},
                               'dirty_paths':sorted(line[3:] for line in dirty)})


def environment(root, scratch):
    frame_path = scratch/'source-input-frame.json'
    frame = json.loads(frame_path.read_text())
    assert frame['base'] == BASE
    overlay = json.loads((root/PREFIX/'overlay-identities.json').read_text())
    assert frame['overlay'] == overlay and overlay['base'] == BASE
    assert set(overlay['original']) == set(overlay['instrumented']) == {
        'src/engine/gemma4_runner.cc', 'src/engine/planned.cc', 'src/engine/live_state.cc'}
    assert ring.identity(root/PREFIX/'instrumentation.patch') == overlay['patch']
    assert frame['measured'] == {**frame['canonical'], **overlay['instrumented']}
    assert all(ring.identity(root/p) == value for p,value in frame['measured'].items())
    sources = {p:ring.identity(root/p) for p in SOURCES}
    assert sources == frame['sources']
    assert sources['benchmarks/gemma_prefill.cc']['sha256'] == '8fc6e4b377f5fa120d1dcc82f44531c2b201eb0462905fbe9ff1c68e45b772a2'
    assert ring.identity(scratch/'ids.i32') == {'bytes':8227*4, 'sha256':ring.IDS}
    artifact = pathlib.Path.home()/'.local/share/jitllm/m3-artifacts'/ring.ARTIFACT
    assert ring.identity(artifact/'manifest.json')['sha256'] == ring.ARTIFACT
    assert ring.identity(artifact/'index.json')['sha256'] == ring.INDEX
    interface_path = scratch/'nvtx-interface.json'
    assert ring.identity(interface_path)['sha256'] == profile.INTERFACE
    interface = json.loads(interface_path.read_text())
    observer = pathlib.Path(interface['observer_root'])
    assert all(ring.identity(observer/p) == value for p,value in interface['headers'].items())
    assert all(ring.identity(pathlib.Path(p)) == value for p,value in interface['tools'].items())
    baseline = pathlib.Path.home()/'.local/share/jitllm/gemma26-swa-ring-transfer'
    assert ring.identity(baseline/'native-frozen.json')['sha256'] == profile.TASK40_OWN
    assert ring.identity(baseline/'source-identities.json')['sha256'] == profile.TASK40_SOURCE
    baseline_log = pathlib.Path.home()/'.local/share/jitllm/jobs/m35-gemma26-swa-ring-bookend1/log'
    assert ring.identity(baseline_log)['sha256'] == profile.TASK40_LOG
    probe = scratch/'probe-validated.json'
    probe_record = json.loads(probe.read_text())
    assert probe_record['known_busy_wait_same_thread_clock_and_nvtx_passed'] is True
    assert probe_record['source_frame'] == ring.identity(frame_path)
    assert probe_record['sources'] == sources
    binaries = {p:ring.identity(scratch/p) for p in ['profile_child']}
    binaries.update({p:ring.identity(root/'build/spark-native/benchmarks'/p) for p in
                     ['jitllm_gemma26_clock_probe', 'jitllm_gemma26_prefill_coarse']})
    assert probe_record['probe_binary'] == binaries['jitllm_gemma26_clock_probe']
    assert probe_record['child_binary'] == binaries['profile_child']
    return {'base':BASE, 'source_frame':ring.identity(frame_path), 'sources':sources,
            'overlay':overlay, 'measured_source_count':len(frame['measured']),
            'binaries':binaries, 'probe':ring.identity(probe), 'interface':ring.identity(interface_path),
            'ids':ring.identity(scratch/'ids.i32'), 'manifest':ring.identity(artifact/'manifest.json'),
            'index':ring.identity(artifact/'index.json'),
            'build_receipt':ring.identity(root/'build/spark-native/jitllm-receipt.json'),
            'task40_own':ring.identity(baseline/'native-frozen.json'),
            'task40_source':ring.identity(baseline/'source-identities.json'),
            'task40_log':ring.identity(baseline_log)}


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        assert len(sys.argv) == 4
        prepare(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    else:
        mode, root, scratch, job = sys.argv[1:5]
        root, scratch = pathlib.Path(root), pathlib.Path(scratch)
        env = environment(root, scratch)
        common = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if mode == 'source':
            assert len(sys.argv) == 5
            profile.write(scratch/'source-identities.json', {**common, 'environment':env,
                          'official_retirement':ring.retired(job)})
        else:
            assert mode == 'result' and len(sys.argv) == 7
            expected, name = sys.argv[5:7]
            source = scratch/'source-identities.json'
            assert ring.identity(source)['sha256'] == expected
            assert json.loads(source.read_text())['environment'] == env
            profile.write(scratch/(name+'-validated.json'), {**common, 'source_freeze':ring.identity(source),
                          **profile.check_run(scratch, name, 'native', job)})
