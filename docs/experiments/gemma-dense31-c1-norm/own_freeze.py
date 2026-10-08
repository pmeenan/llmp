#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Exclusive source/own freeze for closed dense31 scalar C1, before stock."""
import datetime
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True
historical = pathlib.Path(__file__).resolve().parents[1] / 'gemma-joined-serving/analyze.py'
spec = importlib.util.spec_from_file_location('historical_joined_analysis', historical)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
BASE = '0a1a0a6b4f5791573ae71358a55c17369fc5cbf8'
IDS = 'b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610'
ARTIFACT = '32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08'
INDEX = '9ae365a17ed73528262d7befdb505bf07e6695e03f0fa36f69e69099dbd5f34c'
PREFIX = 'docs/experiments/gemma-dense31-c1-norm/'
SOURCES = ['benchmarks/CMakeLists.txt', 'benchmarks/gemma_dense31_c1_norm.cc'] + [
    PREFIX+n for n in ['PROTOCOL.md', 'reference.sh', 'own_freeze.py', 'compare.py']
] + ['docs/experiments/gemma-joined-serving/'+n for n in ['analyze.py', 'llama_joined.cc']]
HEADERS = {
    'llama.h': 'fedb52ea9291c9900e637ed6ffec339919dd27c3dadc2c8cf216c552e0488dda',
    'ggml.h': '9ac7ee3d3a957e5f7b21b104d3e3dc50b1b106f808f9d6735ead950450bfeec6',
    'ggml-cpu.h': '316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e',
    'ggml-backend.h': '46d84cb998105f871240864fd0f55446939a2fe86c5c281afa63a010fb1f65a2',
    'ggml-opt.h': '3586de1bc8a934b5c72339e2b6937b0641e8f149b512231e666f67de0736eea2',
    'gguf.h': 'e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4',
    'ggml-alloc.h': '94e4cd069b9313b2ceb35dacec901981e0bb478d8bb31035b7126be091998c23',
    'ggml-cuda.h': '9cbfcea49f6da4d07f41dc22fa8dbf870825faaa2a68e34e65fe30ea6c41e042',
}


def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': analysis.digest(path)}


def retired(name):
    d = pathlib.Path.home()/'.local/share/llmp/jobs'/name
    final = json.loads((d/'final.json').read_text())
    assert final['state'] == 'done' and final['rc'] == 0
    return {n: identity(d/n) for n in ['final.json', 'job.json', 'log']}


def environment(root, scratch):
    sources = {n: identity(root/n) for n in SOURCES}
    ancestry = json.loads((scratch/'source-ancestry.json').read_text())
    assert ancestry['base'] == BASE and ancestry['tracked_production_diff'] == []
    assert set(ancestry['dirty_paths']).issubset(SOURCES) and ancestry['source_files'] == sources
    inputs = identity(scratch/'inputs.i32')
    assert inputs == {'bytes': 4096, 'sha256': IDS}
    analysis.read_inputs(scratch/'inputs.i32')
    headers = {n: identity(scratch/'headers'/n) for n in HEADERS}
    assert {n:x['sha256'] for n,x in headers.items()} == HEADERS
    artifact = pathlib.Path.home()/'.local/share/llmp/m3-artifacts'/ARTIFACT
    manifest, index = identity(artifact/'manifest.json'), identity(artifact/'index.json')
    assert manifest['sha256'] == ARTIFACT and index['sha256'] == INDEX
    libraries = dict(line.split()[::-1] for line in (scratch/'libraries.sha256').read_text().splitlines())
    assert set(libraries) == {'/app/libllama.so', '/app/libggml.so', '/app/libggml-base.so', '/app/libggml-cuda.so'}
    assert libraries['/app/libggml-cuda.so'] == '5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e'
    assert libraries['/app/libggml.so'] == 'f09875d0593d950f0b1f4f57803235cb06209f774c524c005fb192000d5d7418'
    assert libraries['/app/libggml-base.so'] == 'ed0a6a3f65472059789159569080d08e7f1dcc8bdd7a35d127efbd1b3f4e273e'
    return {'base': BASE, 'sources': sources, 'ancestry': ancestry, 'headers': headers,
            'inputs': inputs, 'manifest': manifest, 'index': index, 'libraries': libraries,
            'library_witness': identity(scratch/'libraries.sha256'),
            'build_receipt': identity(root/'build/spark-native/llmp-receipt.json'),
            'native_binary': identity(root/'build/spark-native/benchmarks/llmp_gemma_dense31_c1_norm'),
            'original_binary': identity(scratch/'llama_joined')}


def main(mode, root, scratch, job, expected=None):
    current = environment(root, scratch)
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'official_retirement': retired(job)}
    if mode == 'source':
        result['environment'] = current
        output = scratch/'source-identities.json'
    else:
        assert mode == 'own' and re.fullmatch('[0-9a-f]{64}', expected or '')
        pre = scratch/'source-identities.json'
        assert identity(pre)['sha256'] == expected
        assert json.loads(pre.read_text())['environment'] == current
        names = ['ordinary-control', 'candidate-first', 'candidate-repeat']
        records = {}
        for name in names:
            assert identity(scratch/name/'inputs.i32') == current['inputs']
            records[name] = analysis.record(scratch/name/'heads.f32', 1)
        assert records['candidate-first'] == records['candidate-repeat']
        log = (pathlib.Path.home()/'.local/share/llmp/jobs'/job/'log').read_text()
        policies = re.findall(r'C1_NATIVE variant=31 owners=1 mode=scalar policy=(ordinary|norm) seconds=[0-9.]+ completed_waves=32 completed_units=32 paid_gpu_groups=32 first_past=67 input_mode=supplied unequal_past=0 context=256 max_rows=128', log)
        assert policies == ['ordinary', 'norm', 'norm']
        assert log.count('C1_SELECTED phase=prefill-first-build owner=0 rows=64 segments=1') == 3
        assert log.count('C1_SELECTED phase=decode-first-build owner=0 rows=1 segments=1') == 3
        result.update({'source_identity': identity(pre), 'owners': 1, 'steps': 32, 'vocab': 262144,
                       'records': records, 'candidate_own_repeat_byte_exact': True,
                       'candidate_own_repeat_raw_noise': 0, 'control_own_noise_measured': False})
        output = scratch/'native-frozen.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    output.chmod(0o444)
    print(json.dumps({'freeze_identity': identity(output), **result}, indent=2, sort_keys=True))


if __name__ == '__main__':
    assert len(sys.argv) in (5, 6)
    main(sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4],
         sys.argv[5] if len(sys.argv) == 6 else None)
