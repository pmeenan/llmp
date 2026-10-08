#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Native HTTP model-turn controls; run on a Spark, keep raw receipts external."""
import argparse
import concurrent.futures as futures
import hashlib
import http.client
import json
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
MODEL = HOME / '.local/share/llmp/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6'

def request(port, model, prompt, tokens, first=None, stream=True):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': tokens, 'temperature': 0, 'stream': stream}
    if stream:
        body['stream_options'] = {'include_usage': True}
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=240)
    started = time.monotonic()
    connection.request('POST', '/v1/chat/completions', json.dumps(body),
                       {'Content-Type': 'application/json'})
    response = connection.getresponse()
    record = {'model': model, 'status': response.status, 'started': started, 'events': []}
    if response.status != 200 or not stream:
        record['body'] = json.loads(response.read())
    else:
        text, reasoning, finish, usage = [], [], None, None
        done = False
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            value = raw[6:].strip()
            if value == b'[DONE]':
                done = True
                break
            event = json.loads(value)
            record['events'].append({'time': time.monotonic(), 'event': event})
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                content = delta.get('content', '') or ''
                thought = delta.get('reasoning', delta.get('reasoning_content', '')) or ''
                text.append(content)
                reasoning.append(thought)
                if (content or thought) and first is not None:
                    first.set()
                finish = choice.get('finish_reason') or finish
        assert done, record
        record.update(text=''.join(text), reasoning=''.join(reasoning), finish=finish, usage=usage)
    record['ended'] = time.monotonic()
    connection.close()
    return record

def signature(record):
    assert record['status'] == 200, record
    assert record['usage'] is not None and record['finish'] is not None, record
    return {key: record[key] for key in ('text', 'reasoning', 'finish', 'usage')}

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
artifact = "{TARGET}"
speculation = {str(self.mtp).lower()}
context = 16384
prefill_chunk = 256
max_slots = 4
tokenizer = "{MODEL}/tokenizer.json"
chat_template = "{MODEL}/chat_template.jinja"
'''
        if self.mtp:
            config += f'drafter = "{MTP}"\n'
        config += f'''[models.substitute]
artifact = "{DS}"
speculation = false
context = 512
prefill_chunk = 256
max_slots = 1
'''
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

def cell(args, name, mode, reference=None):
    prefill = mode == 'prefill'
    count = 1 if prefill else args.slots
    mtp = prefill or args.slots == 1
    spill = 0 if mode == 'refusal' else 128
    path = args.out / name
    with Service(args, path, spill, mtp):
        warm = request(args.port, 'qwen', 'Say hello briefly.', 1, stream=False)
        assert warm['status'] == 200, warm
        # Keep first-prefill control well below long-context qualification.
        prompts = [('A list of numbered observations:\n' + '\n'.join(
            f'{i}: The silver bird crosses a quiet garden.' for i in range(420))) if prefill else
            f'Write a long numbered list of interesting facts about the ocean. Begin with item {i + 1}.'
            for i in range(count)]
        first = [threading.Event() for _ in range(count)]
        barrier = threading.Barrier(count)
        def run(index):
            barrier.wait(timeout=10)
            return request(args.port, 'qwen', prompts[index], 32 if prefill else 128, first[index])
        with futures.ThreadPoolExecutor(max_workers=count + 1) as executor:
            pending = [executor.submit(run, i) for i in range(count)]
            substitute = None
            if not name.startswith('baseline'):
                if prefill:
                    time.sleep(.2)
                    assert not first[0].is_set(), 'prefill ended before switch was queued'
                else:
                    for event in first:
                        assert event.wait(30), 'no generation to pause'
                substitute = request(args.port, 'substitute', 'Hi.', 1, stream=False)
                assert substitute['status'] == (503 if mode == 'refusal' else 200), substitute
                assert not any(task.done() for task in pending), 'substitute did not precede original retirement'
                if prefill:
                    assert not first[0].is_set(), 'first-prefill switch did not finish before first output'
                    expected = reference[0]['usage']['prompt_tokens']
                    log = (path / 'service.log').read_text()
                    pauses = re.findall(
                        r"qwen's request paused for a model switch: (\d+) of (\d+) "
                        r"prompt tokens processed, (\d+) generated", log)
                    progress = [(int(at), int(total), int(generated))
                                for at, total, generated in pauses]
                    assert any(warm['body']['usage']['prompt_tokens'] < at < expected and
                               total == expected and generated == 0
                               for at, total, generated in progress), (
                                   'no completed partial-prompt prefix observed at the swap')
                    (path / 'paused-progress.json').write_text(json.dumps(progress))
            records = [task.result(timeout=240) for task in pending]
        (path / 'requests.json').write_text(json.dumps({'warm': warm, 'substitute': substitute,
                                                     'requests': records}, indent=2))
        (path / 'spill-files.json').write_text(json.dumps([
            {'path': str(p.relative_to(path)), 'size': p.stat().st_size,
             'blocks': p.stat().st_blocks}
            for p in (path / 'data').rglob('*') if p.is_file()], indent=2))
        signatures = [signature(r) for r in records]
        if reference is not None:
            assert signatures == reference, 'reply/usage differs from uninterrupted control'
        if substitute is not None and mode != 'refusal':
            log = (path / 'service.log').read_text()
            assert 'swap qwen -> substitute:' in log and 'swap substitute -> qwen:' in log, 'no completed roundtrip'
        if mode == 'refusal':
            error = substitute.get('body', {}).get('error', {}).get('message', '')
            assert 'retained request state exceeds the spill budget' in error, 'wrong refusal reason'
        summary = {'name': name, 'rows': count, 'exact_reference': reference is not None,
                   'signatures_sha256': hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest(),
                   'usage': [r['usage'] for r in records],
                   'substitute_status': substitute['status'] if substitute else None}
        print(json.dumps(summary), flush=True)
        (path / 'summary.json').write_text(json.dumps(summary, indent=2))
        return signatures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--preflight', type=pathlib.Path, required=True,
                        help='Spark admission/retirement probe accepting minimum free GiB')
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--port', type=int, default=18239)
    parser.add_argument('--case', choices=('generation', 'prefill'), required=True)
    parser.add_argument('--slots', type=int, choices=(1, 4), default=4)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'binary.sha256').write_text(hashlib.sha256(pathlib.Path(args.binary).read_bytes()).hexdigest())
    if args.case == 'generation':
        reference = cell(args, 'baseline-generation', 'generation')
        if args.baseline_only:
            return
        cell(args, 'switched-generation', 'generation', reference)
        cell(args, 'refused-generation', 'refusal', reference)
    else:
        reference = cell(args, 'baseline-prefill', 'prefill')
        if args.baseline_only:
            return
        cell(args, 'switched-prefill', 'prefill', reference)

if __name__ == '__main__':
    main()
