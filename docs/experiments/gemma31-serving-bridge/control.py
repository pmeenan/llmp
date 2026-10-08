#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Workstation coordinator: separate supervised B jobs, own-before-cross, stop first failure."""
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = '/home/pmeenan/.local/share/llmp/gemma31-serving-bridge'
JOBS = '/home/pmeenan/.local/share/llmp/jobs'
ANALYSIS = ROOT + '/source/docs/experiments/gemma31-serving-bridge/analyze.py'
SUPERVISOR = '/home/pmeenan/.local/bin/spark-job'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def exclusive(path, data):
    with path.open('xb') as file:
        file.write(data)


def ssh(command, check=True, data=None):
    return subprocess.run(['ssh', 'spark-b', command], input=data, capture_output=True, check=check)


def read(path, limit=16 << 20):
    # Metadata only. No heads/state payloads leave B; full SHA/finite work is supervised there.
    code = ('import pathlib,sys; p=pathlib.Path(sys.argv[1]); '
            'assert p.is_file() and not p.is_symlink() and p.stat().st_size<=int(sys.argv[2]); '
            'sys.stdout.buffer.write(p.read_bytes())')
    return ssh('python3 -B -c ' + shlex.quote(code) + ' ' + shlex.quote(path) + ' ' + str(limit)).stdout


def write(path, data):
    code = 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.open("xb").write(sys.stdin.buffer.read())'
    ssh('python3 -B -c ' + shlex.quote(code) + ' ' + shlex.quote(path), data=data)


class Pipeline:
    def __init__(self, source, source_sha, output):
        assert re.fullmatch('[0-9a-f]{64}', source_sha)
        self.source, self.source_sha, self.output = source, source_sha, output
        output.mkdir(mode=0o700)
        self.index = 0
        self.completed = []

    def job(self, label, command):
        assert '\n' not in command and re.fullmatch('[a-z0-9-]+', label)
        self.index += 1
        name = 'm35-gemma31-serving2-' + label
        queue = ROOT + '/serving2-' + label + '.steps'
        data = (command + '\n').encode()
        exclusive(self.output / (label + '.steps'), data)
        write(queue, data)
        admission = ssh(SUPERVISOR + ' busy', check=False)
        exclusive(self.output / (label + '-busy.txt'), admission.stdout + admission.stderr)
        if admission.returncode != 0:
            raise RuntimeError('B is occupied; no competing job submitted')
        start = ('cd ' + shlex.quote(ROOT) + ' && ' + SUPERVISOR + ' start --gpu --name ' + name +
                 ' --timeout 600 --step-timeout 300 --stop-on-fail --steps ' + shlex.quote(queue))
        launched = ssh(start)
        exclusive(self.output / (label + '-start.txt'), launched.stdout + launched.stderr)
        # Official wait is authoritative; never poll completion markers or process names.
        waited = ssh(SUPERVISOR + ' wait ' + name, check=False)
        exclusive(self.output / (label + '-wait.txt'), waited.stdout + waited.stderr)
        records = {}
        directory = self.output / ('official-' + label)
        directory.mkdir(mode=0o700)
        for field, filename in [('job', 'job.json'), ('final', 'final.json'), ('steps', 'steps.json'), ('log', 'log')]:
            path = JOBS + '/' + name + '/' + filename
            data = read(path)
            exclusive(directory / filename, data)
            records[field] = dict(path=path, sha256=sha(data))
        final = json.loads((directory / 'final.json').read_bytes())
        steps = json.loads((directory / 'steps.json').read_bytes())['steps']
        assert len(steps) == 1 and steps[0]['cmd'] == command
        if not (waited.returncode == 0 and final['state'] == 'done' and final['rc'] == 0 and
                steps[0]['state'] == 'done' and steps[0]['rc'] == 0):
            raise RuntimeError('FIRST failure: ' + name + '; original records preserved')
        self.completed.append(name)
        return records

    def analysis(self, *args):
        return 'python3 -B ' + shlex.quote(ANALYSIS) + ' ' + ' '.join(
            shlex.quote(str(x)) for x in (*args[:1], ROOT, self.source_sha, *args[1:]))

    def producer(self, engine, count, mode, arm='once', native=None, reference=None, comparison=None, anchor=None):
        key = f'{engine}-c{count}-{mode}' + ('' if arm == 'once' else '-' + arm)
        bindings = dict(SOURCE_SHA256=self.source_sha, NATIVE_OWN_SHA256=native,
                        REFERENCE_OWN_SHA256=reference, COMPARISON_SHA256=comparison,
                        ANCHOR_SHA256=anchor)
        command = self.source['producer_commands'][key]
        for marker, value in bindings.items():
            if value is not None:
                command = command.replace('__' + marker + '__', value)
        assert '__' not in command
        admission = self.job(key, command)
        for name, value in [('native_own_sha256', native), ('reference_own_sha256', reference),
                            ('comparison_sha256', comparison), ('anchor_proof_sha256', anchor)]:
            if value is not None:
                admission[name] = value
        admission['output_log_sha256'] = sha(read(ROOT + '/' + key + '.log'))
        if engine == 'reference':
            retired = read(ROOT + '/' + key + '-container-retired.json')
            assert json.loads(retired)['container_absent_after_checked_docker_query'] is True
            admission['retirement_sha256'] = sha(retired)
            exclusive(self.output / (key + '-container-retired.json'), retired)
        exclusive(self.output / (key + '-admission.json'), dump(admission))
        return key, admission

    def freeze(self, key, admission):
        data = dump(admission)
        path = ROOT + '/' + key + '-admission.json'
        write(path, data)
        self.job(key + '-own', self.analysis('own', key, path, sha(data)))
        proof = read(ROOT + '/' + key + '-own.json')
        assert json.loads(proof)['complete_finite_repeat_exact'] is True
        exclusive(self.output / (key + '-own.json'), proof)
        return sha(proof)

    def compare(self, count, mode, native, reference):
        # Post-source authentication is separately supervised even when strict comparison fails.
        failure = None
        try:
            self.job(f'compare-c{count}-{mode}', self.analysis('compare', count, mode, native, reference))
        except Exception as error:
            failure = error
        finally:
            self.job(f'postauth-c{count}-{mode}', self.analysis('guard'))
        result = read(ROOT + f'/comparison-c{count}-{mode}.json')
        exclusive(self.output / f'comparison-c{count}-{mode}.json', result)
        if failure is not None:
            raise failure
        assert json.loads(result)['pass_all'] is True
        return sha(result)

    def pair(self, count, mode):
        key, admission = self.producer('native', count, mode)
        native = self.freeze(key, admission)
        anchor = None
        if mode == 'quality':
            self.job(f'anchors-c{count}', self.analysis('carrier', count, native))
            data = read(ROOT + f'/anchors-c{count}.json')
            exclusive(self.output / f'anchors-c{count}.json', data)
            anchor = sha(data)
        key, admission = self.producer('reference', count, mode, native=native, anchor=anchor)
        reference = self.freeze(key, admission)
        comparison = self.compare(count, mode, native, reference)
        return native, reference, comparison

    def timing(self, count, bindings):
        native, reference, comparison = bindings
        order = ['reference-first', 'native-first', 'native-repeat', 'reference-repeat']
        arms = {}
        for label in order:
            engine, arm = label.split('-')
            _, admission = self.producer(engine, count, 'cycle', arm, native, reference, comparison)
            arms[label] = admission
        data = dump(dict(order=order, arms=arms))
        path = ROOT + f'/timing-c{count}-admissions.json'
        write(path, data)
        self.job(f'timing-c{count}-exact', self.analysis('timing', count, native, reference, comparison, path, sha(data)))
        result = read(ROOT + f'/timing-c{count}.json')
        assert json.loads(result)['complete_paid_outputs_match_own'] is True
        exclusive(self.output / f'timing-c{count}.json', result)


if __name__ == '__main__':
    # Release is external: invoke only after root+fresh whole source/method review authorizes it.
    source_path, expected, output_path = sys.argv[1:]
    data = Path(source_path).read_bytes()
    assert sha(data) == expected and json.loads(data)['schema'] == 'gemma31-serving-bridge-v1'
    source = json.loads(data)
    caller = next(item for item in source['protected_files'] if item['path'].endswith('/gemma31-serving-bridge/control.py'))
    assert caller['sha256'] == sha(Path(__file__).read_bytes())
    pipeline = Pipeline(source, expected, Path(output_path))
    try:
        for count in (1, 4):
            bindings = pipeline.pair(count, 'quality')
            pipeline.timing(count, bindings)
        # Mandatory before adoption, while leaving earlier decisive screen evidence intact.
        pipeline.pair(1, 'corpus')
    except Exception:
        # Even producer/own failures retain their first records and get a source-only postauth.
        try:
            pipeline.job('failure-postauth', pipeline.analysis('guard'))
        finally:
            exclusive(pipeline.output / 'stopped.json', dump(dict(completed=pipeline.completed, stopped=True)))
        raise
    exclusive(pipeline.output / 'completed.json', dump(dict(completed=pipeline.completed,
        source_sha256=expected, representative_quality_corpus_and_bookends_complete=True,
        adoption=False, remaining='HTTP lifecycle/current stop and explicit reviewed adoption delta')))
