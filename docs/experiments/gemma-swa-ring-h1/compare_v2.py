#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate H1's new-recipe bookend, two complete heads and native state."""
import importlib.util
import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('h1_own', pathlib.Path(__file__).with_name('own_freeze.py'))
own = importlib.util.module_from_spec(spec)
spec.loader.exec_module(own)


def tokens_equal(parsed, saved):
    return [list(t) for t in parsed] == saved


def selftest():
    parsed = [(7, 11), (13, 17)]
    assert tokens_equal(parsed, [[7, 11], [13, 17]])
    assert not tokens_equal(parsed, [[7, 11], [19, 17]])
    assert not tokens_equal(parsed, [[7, 11], [13, 19]])
    assert not tokens_equal(parsed, [[7, 11]])
    print('tuple/list identity, changed greedy/input and incomplete stream controls passed')


def compare(root, source_root, expected, job, analysis_sha):
    assert re.fullmatch('[0-9a-f]{64}', analysis_sha)
    analysis_identity = own.identity(pathlib.Path(__file__))
    assert analysis_identity['sha256'] == analysis_sha
    assert re.fullmatch('[0-9a-f]{64}', expected)
    assert own.identity(root/'native-frozen.json')['sha256'] == expected
    frozen = json.loads((root/'native-frozen.json').read_text())
    assert own.identity(root/'source-identities.json') == frozen['source_identity']
    pre = json.loads((root/'source-identities.json').read_text())
    assert own.environment(source_root, root) == pre['environment']
    retirement = own.retired(job)
    assert own.record(root/'native-first') == frozen['records']['native-first']
    assert own.record(root/'native-repeat') == frozen['records']['native-repeat']
    assert own.record(root/'native-bookend') == frozen['records']['native-first']
    log = pathlib.Path.home()/'.local/share/llmp/jobs'/job/'log'
    text = log.read_text()
    runs = own.old.runs(log, 'both', '31')
    assert [r['engine'] for r in runs] == ['PREFILL_REFERENCE','PREFILL_NATIVE','PREFILL_REFERENCE']
    assert all(r['metrics']['chunk'] == '256' for r in runs if r['engine'] == 'PREFILL_NATIVE')
    assert all(r['metrics'].get('swa_full') == 'false' and r['metrics'].get('kv_unified') == 'false'
               and r['metrics'].get('cache_recipe') == 'ring'
               for r in runs if r['engine'] == 'PREFILL_REFERENCE')
    assert len(re.findall(r'creating\s+SWA KV cache, size = 1280 cells', text)) == 2
    assert text.count('creating non-SWA KV cache, size = 16384 cells') == 2
    assert runs[0]['tokens'] == runs[2]['tokens']
    assert tokens_equal(runs[1]['tokens'], frozen['native_runs'][0]['tokens'])
    assert all([t[1] for t in r['tokens']] == [t[1] for t in runs[0]['tokens']] for r in runs)
    result = {'analysis_source_identity': analysis_identity, 'own_freeze': own.identity(root/'native-frozen.json'),
              'source_freeze': frozen['source_identity'], 'official_retirement': retirement,
              'scope': 'two complete heads; 32 incoming-head choices; initialized native state',
              'native_repeat_and_state_exact': True, 'reference_recipe': 'swa_full=false, kv_unified=false',
              'strict_argmax_differences_32': sum(a[0] != b[0] for a,b in zip(runs[1]['tokens'],runs[0]['tokens'],strict=True)),
              'margin_noise_calibrated': False, 'all_32_head_vectors_retained': False, 'heads': {}, 'paid': {}}
    for n in ['prefill.f32','final.f32']:
        native, ns = own.old.head(root/'native-bookend'/n)
        a, ash = own.old.head(root/'ring-first'/n)
        b, bsh = own.old.head(root/'ring-repeat'/n)
        assert ash == bsh
        metrics = own.old.compare(native,a)
        metrics.pop('argmax'); metrics.pop('reference_argmax')
        result['heads'][n] = {'native_sha256':ns,'ring_reference_sha256':ash,
                              'byte_exact':(root/'native-bookend'/n).read_bytes() == (root/'ring-first'/n).read_bytes(),
                              'reference_repeat_exact':True, **metrics}
    for stage in ['prefill','decode']:
        times = [float(r['metrics'][stage+'_seconds']) for r in runs]
        mean = (times[0]+times[2])/2
        result['paid'][stage] = {'seconds':times,'native_relative_to_reference_percent':100*(times[1]/mean-1)}
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        selftest()
    else:
        assert len(sys.argv) == 6
        compare(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
