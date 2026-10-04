# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate cache-path diagnostics without treating their clocks as performance."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from cache_control import OUTPUTS
from compare import accepted, check_verdicts, digest, read, require, rows

HISTORY = 'c0f9706146aca18113f1cd91cdc1e1107c18df34c72fdd6a56e3192dc6a7aa3c'
PROMPTS = [
    '306136a7dc2a5f66c94cbf4ac874e9011c0af79b79072731407a144bfba1fbd5',
    '53bda5259e6751681f80455d531d6dd89d4e255f8779b35446d59e2c245f963b',
    'a3261b4f298733484a629fe27f0e0109b4945e0d8d750d92ef8d85e855341e28',
    '184d7c02214ef662c0e927ca63f4e0091eb42a918a1f79423c1be34b7303ae1e',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(directory, schedule, two_paths=False, deterministic=False):
    receipt = read(directory / 'receipt.json')
    require(receipt['complete'] and receipt['body_complete'] and receipt['container_absent']
            and receipt['launch_reaped'] and not receipt['cleanup_errors']
            and receipt['timing_qualified'] is False and receipt['history_sha256'] == HISTORY,
            'complete unqualified reference and cleanup required')
    anchor_path = directory / 'anchors/receipt.json'
    anchors = read(anchor_path)
    require(sha(anchor_path) == receipt['anchor_receipt_sha256'] and anchors['complete']
            and anchors['all_requested_collected'], 'authenticated complete client receipt required')
    expected = '1' if deterministic else '0'
    actual = receipt['identity']['environment']
    require(all(actual[k] == expected for k in ('VLLM_QSA_DET_TOPK', 'VLLM_MOE_DET_FINALIZE')),
            'actual diagnostic determinism settings differ')
    label = directory.name
    require(label in ('mia-cold-p3', 'mia-cache-paths', 'mia-cache-det1'), 'measured collection name required')
    client_name = 'mia-cold-p3-client.py' if label == 'mia-cold-p3' else 'mia-cache-paths-client.py'
    require(receipt['controller_sha256'] == OUTPUTS[label + '-run.py']
            and receipt['client_sha256'] == OUTPUTS[client_name]
            and (not deterministic or receipt['helper_sha256'] == OUTPUTS['mia-det1-helper.py']),
            'measured controller/client/helper source pins required')
    expected_pins = {
        'observer_sha256': 'fc34ef7a55ed95204632006ebac1ec31a5b6f573f74d37b97b2cc9710ce8cab9',
        'launcher_sha256': 'c8353f04dc30e5d0f298217edc0ef52ec2f24bcfdb6c396e57a700468760e088',
        'overlay_sha256': 'f87e3bcec9453f41b289f02ed3cf82bc899ff105a0fd6a24d114d9bdc928b539',
        'source_sha256': '308d4171f892f27231655289ceb03cd4566533c7d26b8f98bd62ef0a34f986f0',
    }
    require(all(receipt[k] == value for k, value in expected_pins.items())
            and hashlib.sha256(json.dumps(receipt['launcher_pin'], sort_keys=True,
                separators=(',', ':')).encode()).hexdigest() == 'c7f57e14d386e0ad22b2c98239b7eb14fd30e6d02efdf5132f8af375f455b0e9'
            and receipt['identity']['image'] == 'sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8',
            'pinned observer, loader, launcher and image required')
    arguments = {'--max-model-len': '33792', '--max-num-seqs': '4', '--max-num-batched-tokens': '2048',
                 '--kv-cache-dtype': 'fp8', '--mamba-ssm-cache-dtype': 'bfloat16',
                 '--gpu-memory-utilization': '0.786', '--tensor-parallel-size': '1',
                 '--served-model-name': 'qwen3.8-flash-next', '--host': '127.0.0.1', '--port': str(receipt['port'])}
    require(receipt['identity']['arguments'] == arguments and anchors['container'] == receipt['container']
            and anchors['port'] == receipt['port']
            and anchors['identity']['pid'] == receipt['identity']['container_pid']
            and anchors['identity']['started'] == receipt['identity']['container_started_at'],
            'client and verified launch identity must agree')
    controls = receipt['controls']
    requests = anchors['requests']
    require(len(controls) == len(requests) == len(schedule)
            and [(c['position'], c['repeat']) for c in controls] == schedule,
            'exact cache-control schedule required')
    if two_paths:
        salts = [r['cache_salt'] for r in requests]
        require(salts[0] == salts[1] and salts[2] == salts[3] == salts[4]
                and salts[0] != salts[2], 'independent cache-salt paths required')
    used = set()
    result = []
    for index, (control, request) in enumerate(zip(controls, requests)):
        first = control['first_verify']
        position = control['position']
        require(request['complete'] and request['status'] == 200 and request['finish_reason'] == 'length'
                and request['usage']['prompt_tokens'] == 31743 + position
                and request['usage']['total_tokens'] == 31752 + position
                and request['usage']['prompt_tokens_details']['cached_tokens'] == control['cached_tokens']
                and (request['position'], request['repeat']) == (position, control['repeat'])
                and control['response_id'] == request['response_id']
                and (first['request_id'] == control['response_id']
                     or re.fullmatch(re.escape(control['response_id']) + r'-0-[0-9a-f]{8}', first['request_id']))
                and control['cached_tokens'] == request['cached_tokens']
                and control['prompt_i32_sha256'] == request['prompt_i32_sha256'] == PROMPTS[position]
                and first['position'] == first['prompt_tokens'] == request['prompt_tokens'] == 31743 + position
                and first['anchor_token'] == request['token_ids'][0]
                and len(request['token_ids']) == request['usage']['completion_tokens'] == 9
                and first['observer_sha256'] == receipt['observer_sha256']
                and first['request_id'] not in used, 'unique paid-work identity/history required')
        used.add(first['request_id'])
        name = first['logits_file']
        require(Path(name).name == name, 'trace basename required')
        values = rows(directory / 'traces' / name, first['logits_sha256'])
        check_verdicts(first, values)
        count = accepted(first)
        require(count == first['accepted'] and first['rejected'] == 3 - count
                and first['sampled_tokens'] == first['drafts'][:count] + [first['verdicts'][count]]
                and request['token_ids'][1:1 + first['kept']] == first['sampled_tokens'],
                'actual sampler must agree with complete target rows')
        cold = index in (0, 2) if two_paths else index == 0
        require(control['cached_tokens'] == (0 if cold else 31616), 'observed cold/warm counts differ')
        result.append(dict(position=position, repeat=control['repeat'], cached_tokens=control['cached_tokens'],
                           anchor=first['anchor_token'], drafts=first['drafts'], verdicts=first['verdicts'],
                           accepted=count, logits_sha256=first['logits_sha256']))
    return dict(receipt_sha256=sha(directory / 'receipt.json'), deterministic=deterministic, rows=result)


def native(directory):
    record = read(directory / 'spec.json')
    require(record['check'] == 'greedy' and record['problems'] == [] and record['prompt_token_ids']
            and record['runtime_prefill'] and record['draft_rows'] == 3 and record['draft_vocab'] == 47172
            and not record['adaptive_depth'] and record['window'] == 0, 'passing fixed-depth native controls required')
    cells = [c for c in record['results'] if c['check'] == 'greedy']
    require(len(cells) == 1 and cells[0]['prompt'] == 'p3', 'only native p3 required')
    cell = cells[0]
    first = cell['first_verify']
    require(len(cell['spec_tok_s']) == 2 and cell['generated'] == len(cell['spec_tokens']) == len(cell['plain_tokens']) == 9
            and cell['prompt_tokens'] == first['position'] == 31746
            and digest(cell['prompt_ids']) == cell['prompt_ids_sha256'] == PROMPTS[3], 'native complete identical input required')
    values = rows(directory / 'p3-first-verify.f32', first['logits_sha256'])
    check_verdicts(first, values)
    count = accepted(first)
    require(cell['step_trace'][0] == [31746, 3, 4, count + 1]
            and first['next_token'] == first['verdicts'][count]
            and cell['spec_tokens'][:count + 2] == [first['anchor_token']] + first['drafts'][:count] + [first['next_token']],
            'native first verify must match actual sampled prefix and paid step')
    return dict(spec_sha256=sha(directory / 'spec.json'), anchor=first['anchor_token'],
                drafts=first['drafts'], verdicts=first['verdicts'], accepted=accepted(first),
                logits_sha256=first['logits_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('cold', 'paths', 'native_before', 'native_small', 'native_after', 'out'):
        parser.add_argument('--' + option.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--det', type=Path)
    args = parser.parse_args()
    report = dict(complete=False, timing_qualified=False,
                  native_scope='chunk sizes, source identity and retirement require the archived supervisor commands/receipts',
                  cold=reference(args.cold, [(3, False), (3, True), (0, False), (1, False), (2, False)]),
                  paths=reference(args.paths, [(3, False), (3, True), (0, False), (3, False), (3, True)], True),
                  native_before=native(args.native_before), native_small=native(args.native_small),
                  native_after=native(args.native_after))
    if args.det is not None:
        report['det'] = reference(args.det, [(3, False), (3, True), (0, False), (3, False), (3, True)], True, True)
    require(report['native_before']['logits_sha256'] == report['native_after']['logits_sha256'],
            'native 4096-row bookends must be exact')
    for key in ('native_before', 'native_small', 'native_after'):
        require(report[key]['anchor'] == 1156, 'native p3 anchor differs')
    for key in ('cold', 'paths') + (('det',) if args.det is not None else ()):
        require(all(r['anchor'] == 1156 for r in report[key]['rows'] if r['position'] == 3), 'reference p3 anchor differs')
    report['complete'] = True
    with args.out.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
