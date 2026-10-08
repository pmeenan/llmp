#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Native literal cohort controls; supervised on a Spark, raw results external."""
import argparse
import concurrent.futures as futures
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import subprocess
import threading
import time

HOME = pathlib.Path.home()
TARGET = 'c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93'
DS = 'cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac'
MTP = '8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40'
DSPARK = 'dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5'
MODEL = HOME / '.local/share/llmp/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6'

class Service:
    def __init__(self, args, path, spill, mtp):
        self.args, self.path, self.spill, self.mtp = args, path, spill, mtp

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        subprocess.run(['python3', '-B', str(self.args.preflight),
                        '105'], check=True)
        config = f'''schema_version = 2
[storage]
data_dir = "{self.path}/data"
installed = "{HOME}/.local/share/llmp/m3-artifacts"
[memory]
spill_budget_gib = {self.spill}
[client]
bind = ["127.0.0.1"]
port = {self.args.port}
model_turn_seconds = 1
[models.qwen]
artifact = "{DS if self.args.family == 'deepseek' else TARGET}"
speculation = {str(self.mtp).lower()}
context = {self.args.context}
prefill_chunk = 256
max_slots = 4
'''
        if self.args.family == 'qwen':
            config += f'tokenizer = "{MODEL}/tokenizer.json"\n'
            config += f'chat_template = "{MODEL}/chat_template.jinja"\n'
        else:
            config += 'wave_form = "speculative"\n'
        if self.mtp:
            config += f'drafter = "{DSPARK if self.args.family == "deepseek" else MTP}"\n'
        config += f'''[models.substitute]
artifact = "{TARGET if self.args.family == 'deepseek' else DS}"
speculation = false
context = 512
prefill_chunk = 256
max_slots = 1
'''
        if self.args.family == 'deepseek':
            config += f'tokenizer = "{MODEL}/tokenizer.json"\n'
            config += f'chat_template = "{MODEL}/chat_template.jinja"\n'
        (self.path / 'config.toml').write_text(config)
        self.log = open(self.path / 'service.log', 'w')
        self.process = subprocess.Popen([self.args.binary, '--config', str(self.path / 'config.toml'),
                                         '--anchor', str(self.path / 'anchor')],
                                        stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                assert self.process.poll() is None, 'service exited before ready'
                if 'llmp-runtime: ready' in (self.path / 'service.log').read_text():
                    return self
                time.sleep(.2)
            raise AssertionError('service readiness deadline')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        self.process.send_signal(signal.SIGTERM)
        try:
            code = self.process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise AssertionError('service required forced retirement')
        finally:
            self.log.close()
        (self.path / 'exit.json').write_text(json.dumps({'exit': code}))
        assert code == 0, f'service exit {code}'
        subprocess.run(['python3', '-B', str(self.args.preflight),
                        '105'], check=True)

def request(port, body, chat=False, expected=200):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=240)
    started = time.monotonic()
    connection.request('POST', '/v1/chat/completions' if chat else '/v1/completions',
                       json.dumps(body), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    record = {'status': response.status, 'body': json.loads(response.read()),
              'started': started, 'ended': time.monotonic()}
    connection.close()
    assert record['status'] == expected, record
    return record


def signature(record):
    return {key: record['body'][key] for key in ('choices', 'usage')}


def warm(args):
    return request(args.port, {'model': 'qwen', 'prompt': [40], 'max_tokens': 1,
                               'temperature': 0})


def screen(args):
    cases = [
        {'model': 'qwen', 'prompt': 'The capital of France is', 'max_tokens': 8,
         'temperature': 0, 'echo': True, 'logprobs': 3, 'prompt_logprobs': 2},
        {'model': 'qwen', 'prompt': [40, 41, 42], 'max_tokens': 0, 'prompt_logprobs': 0},
        {'model': 'qwen', 'prompt': 'The capital of France is', 'max_tokens': 8,
         'temperature': .7, 'seed': 7654, 'logprobs': 0},
        {'model': 'qwen', 'prompt': 'The capital of France is', 'max_tokens': 8,
         'temperature': 0},
        {'model': 'qwen', 'prompt': '<|im_start|>user\nReply with exactly one word: Hi.'
         '<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n',
         'max_tokens': 32, 'temperature': 0, 'logprobs': 2},
    ]
    records = {}
    candidate = args.binary
    for label, binary in [('serial', args.baseline), ('candidate', candidate)]:
        args.binary = binary
        with Service(args, args.out / label, 128, args.speculation):
            warm(args)
            records[label] = [request(args.port, body) for body in cases]
            if label == 'candidate':
                barrier = threading.Barrier(4)
                def run(index):
                    barrier.wait(timeout=10)
                    body = cases[0] if index < 3 else {
                        'model': 'qwen', 'messages': [{'role': 'user',
                        'content': 'Name the capital of France.'}], 'max_tokens': 8,
                        'temperature': 0}
                    return request(args.port, body, index == 3)
                with futures.ThreadPoolExecutor(max_workers=4) as executor:
                    pending = [executor.submit(run, index) for index in range(4)]
                    records['mixed'] = [task.result(timeout=180) for task in pending]
                mixed_exact = [signature(record) == signature(records[label][0])
                               for record in records['mixed'][:3]]
                assert all(mixed_exact), 'mixed literal scores/reply/usage differ from solo'
                for record in records['mixed'][:3]:
                    usage = record['body']['usage']
                    choice = record['body']['choices'][0]
                    rows = choice['logprobs']
                    assert len(rows['tokens']) == usage['total_tokens']
                    assert len(choice['prompt_logprobs']) == usage['prompt_tokens']
                    at = usage['prompt_tokens']
                    assert all(value == max(top.values()) for value, top in zip(
                        rows['token_logprobs'][at:], rows['top_logprobs'][at:]))
            (args.out / label / 'responses.json').write_text(json.dumps(records[label], indent=2))
    args.binary = candidate
    (args.out / 'mixed.json').write_text(json.dumps(records['mixed'], indent=2))
    exact = [signature(records['serial'][i]) == signature(records['candidate'][i])
             for i in range(len(cases))]
    generated = [r['body']['usage']['completion_tokens'] for r in records['candidate']]
    eos_scored = records['candidate'][-1]['body']['choices'][0]['finish_reason'] == 'stop'
    if args.family == 'qwen':
        assert eos_scored, 'the EOS control did not stop naturally'
    return {'exact_serial_choices_usage': exact, 'generated_tokens': generated,
            'eos_scored': eos_scored,
            'mixed_exact_solo': mixed_exact,
            'mixed_status': [r['status'] for r in records['mixed']], 'passed': all(exact)}


def boundary(args):
    args.context, args.speculation = 512, False
    candidate = args.binary
    args.binary = args.baseline
    with Service(args, args.out / 'serial-scored', 128, False):
        reference = request(args.port, {'model': 'qwen', 'prompt': [40] * 512,
                                       'max_tokens': 0, 'prompt_logprobs': 0})
        (args.out / 'serial-scored' / 'response.json').write_text(json.dumps(reference, indent=2))
    args.binary = candidate
    with Service(args, args.out / 'candidate', 128, False):
        records = [request(args.port, {'model': 'qwen', 'prompt': [40] * 512,
                   'max_tokens': 0, **({'prompt_logprobs': 0} if scored else {})})
                   for scored in (False, True)]
        refused = request(args.port, {'model': 'qwen', 'prompt': [40] * 513,
                                     'max_tokens': 0}, expected=400)
        assert refused['body']['error']['code'] == 'context_length_exceeded', refused
        assert records[0]['body']['usage'] == reference['body']['usage']
        assert records[0]['body']['choices'][0]['text'] == ''
        assert records[0]['body']['choices'][0]['finish_reason'] == 'stop'
        (args.out / 'candidate' / 'responses.json').write_text(json.dumps(records, indent=2))
    exact = signature(reference) == signature(records[1])
    return {'exact_serial_scored_choices_usage': exact, 'plain_matches_usage': True,
            'context': 512, 'past_context_status': 400, 'passed': exact}


def continuation(args):
    prefill = args.case == 'continuation-prefill'
    body = {'model': 'qwen', 'prompt': [40, 41, 42] * 48, 'max_tokens': 0,
            'prompt_logprobs': 2} if prefill else {
        'model': 'qwen', 'prompt': '<|im_start|>user\nWrite a very long numbered list of '
        'interesting ocean facts, at least 100 items.\n<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n', 'max_tokens': 128, 'temperature': 0,
        'logprobs': 2, 'echo': True}
    records = []
    for label in ('baseline', 'switched'):
        path = args.out / label
        with Service(args, path, 128, args.speculation):
            warm(args)
            if label == 'baseline':
                record = request(args.port, body)
            else:
                with futures.ThreadPoolExecutor(max_workers=1) as executor:
                    pending = executor.submit(request, args.port, body)
                    time.sleep(.3 if prefill else 2.5)
                    assert not pending.done(), 'original finished before substitute was queued'
                    substitute = request(args.port, {
                        'model': 'substitute', 'messages': [{'role': 'user', 'content': 'Hi.'}],
                        'max_tokens': 1, 'temperature': 0}, chat=True)
                    assert not pending.done(), 'substitute did not precede original retirement'
                    record = pending.result(timeout=180)
                    (path / 'substitute.json').write_text(json.dumps(substitute, indent=2))
                log = (path / 'service.log').read_text()
                pauses = re.findall(r"qwen's request paused for a model switch: (\d+) of (\d+) "
                                    r'prompt tokens processed, (\d+) generated', log)
                assert pauses, 'no literal pause witness'
                if prefill:
                    assert any(0 < int(at) < int(total) and int(generated) == 0
                               for at, total, generated in pauses), pauses
                else:
                    assert any(int(generated) > 0 for _, _, generated in pauses), pauses
                assert 'swap qwen -> substitute:' in log and 'swap substitute -> qwen:' in log
            (path / 'response.json').write_text(json.dumps(record, indent=2))
            records.append(record)
    exact = signature(records[0]) == signature(records[1])
    return {'exact_choices_usage': exact, 'usage': records[1]['body']['usage'],
            'pause_witness': pauses, 'passed': exact}


def throughput(args):
    body = {'model': 'qwen', 'prompt': '<|im_start|>user\nWrite a very long numbered list of '
            'interesting ocean facts, at least 100 items.\n<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n', 'max_tokens': 128, 'temperature': 0}
    candidate = args.binary
    cells = []
    for label, binary, counts in [('serial-before', args.baseline, (1, 4)),
                                   ('candidate', candidate, (1, 2, 4)),
                                   ('serial-after', args.baseline, (4,))]:
        args.binary = binary
        with Service(args, args.out / label, 128, args.speculation):
            warm(args)
            for count in counts:
                barrier = threading.Barrier(count)
                def run():
                    barrier.wait(timeout=10)
                    return request(args.port, body)
                with futures.ThreadPoolExecutor(max_workers=count) as executor:
                    pending = [executor.submit(run) for _ in range(count)]
                    records = [task.result(timeout=180) for task in pending]
                seconds = max(r['ended'] for r in records) - min(r['started'] for r in records)
                for record in records:
                    usage = record['body']['usage']
                    assert usage['completion_tokens'] == 128, record
                    assert usage['prompt_tokens_details']['cached_tokens'] == 0, record
                    assert record['body']['choices'][0]['finish_reason'] == 'length', record
                completed = sum(r['body']['usage']['completion_tokens'] for r in records)
                assert completed > 0
                cells.append({'name': label, 'concurrency': count, 'completed_tokens': completed,
                              'seconds': seconds, 'completed_tok_s': completed / seconds,
                              'signature_sha256': [hashlib.sha256(json.dumps(
                                  signature(r), sort_keys=True).encode()).hexdigest()
                                  for r in records]})
                (args.out / label / f'c{count}.json').write_text(json.dumps(records, indent=2))
    args.binary = candidate
    return {'cells': cells, 'passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--baseline', help='Pre-batching runtime; needed for screen/boundary/throughput')
    parser.add_argument('--library-dir', type=pathlib.Path,
                        help='CUDA shared libraries when a copied baseline loses its relative rpath')
    parser.add_argument('--preflight', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--port', type=int, default=18240)
    parser.add_argument('--context', type=int, default=16384)
    parser.add_argument('--family', choices=('qwen', 'deepseek'), default='qwen')
    parser.add_argument('--speculation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--case', required=True, choices=(
        'screen', 'boundary', 'continuation-prefill', 'continuation-generation', 'throughput'))
    args = parser.parse_args()
    if args.case in ('screen', 'boundary', 'throughput') and not args.baseline:
        parser.error('this case requires --baseline')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.library_dir:
        os.environ['LD_LIBRARY_PATH'] = str(args.library_dir)
    identity = {'binary_sha256': hashlib.sha256(pathlib.Path(args.binary).read_bytes()).hexdigest(),
                'harness_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    if args.baseline:
        identity['baseline_sha256'] = hashlib.sha256(pathlib.Path(args.baseline).read_bytes()).hexdigest()
    (args.out / 'identity.json').write_text(json.dumps(identity, indent=2))
    if args.case == 'screen':
        result = screen(args)
    elif args.case == 'boundary':
        result = boundary(args)
    elif args.case.startswith('continuation-'):
        result = continuation(args)
    else:
        result = throughput(args)
    result.update(identity)
    result['case'] = args.case
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert result['passed'], result


if __name__ == '__main__':
    main()
