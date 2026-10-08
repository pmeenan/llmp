#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline first-cut Gemma trace replay. Byte/time outputs are estimates."""
import argparse
from bisect import bisect_right
from collections import OrderedDict, Counter
import hashlib
import json
import math
from pathlib import Path
from run import IMAGE, REVISION, MODEL_SHA, TRACE_SHA

GIB = 1 << 30
EXTENT = 2 << 20
EXPERTS = 128
LAYERS = 30
TOP_K = 8
# Verified GGUF accounting in reference-setup. Hypothetical isolated closures:
# concatenate each expert's projections, then pad that closure to 2 MiB.
LOGICAL = [3717124] * 29 + [4336644]
SIZES = [math.ceil(n / EXTENT) * EXTENT for n in LOGICAL]
NON_EXPERT = math.ceil(2578661496 / EXTENT) * EXTENT
KV = 32768 * 220 * 1024  # full-SWA f16, including unused cells
# Conservative scenario allowances, not measured allocation peaks.
SCRATCH = 1 * GIB
STAGING = 8 << 20
HEADROOM = 2 * GIB
FIXED = NON_EXPERT + KV + SCRATCH + STAGING + HEADROOM
READ_BPS = 14.962131e9  # measured bulk sustained direct host-VMM bandwidth
SERIAL_EXTENT_US = 177  # measured 2 MiB queue-depth-one p50 upper run endpoint


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path, metadata, run):
    if sha(path) != run['routes_sha256']:
        raise ValueError('Route identity mismatch')
    batch = run['batch']
    prompt = metadata['prompt_tokens']
    decode = metadata['decode_tokens']
    if batch not in (64, 512) or prompt != 18339 or decode != 117:
        raise ValueError('Unexpected capture configuration')
    counts = [min(batch, prompt - pos) for pos in range(0, prompt, batch)] + [1] * decode
    records = []
    with path.open() as stream:
        for index, line in enumerate(stream):
            step, layer = divmod(index, LAYERS)
            if step >= len(counts) or len(line) > 128 * 1024:
                raise ValueError('Unexpected record/size')
            r = json.loads(line)
            phase = 'prefill' if step < math.ceil(prompt / batch) else 'decode'
            if (type(r.get('step')) is not int or type(r.get('layer')) is not int or
                    r.get('step') != step or r.get('layer') != layer or r.get('phase') != phase):
                raise ValueError('Missing, duplicate or reordered routing layer/phase')
            routes = r.get('routes')
            if not isinstance(routes, list) or len(routes) != counts[step]:
                raise ValueError('Wrong token count')
            for ids in routes:
                if (not isinstance(ids, list) or len(ids) != TOP_K or
                        any(type(n) is not int or not 0 <= n < EXPERTS for n in ids) or
                        len(set(ids)) != TOP_K):
                    raise ValueError('Invalid selected expert set')
            records.append(r)
    if len(records) != len(counts) * LAYERS:
        raise ValueError('Incomplete route trace')
    return records


def replay(records, total_budget):
    capacity = (total_budget - FIXED) // EXTENT * EXTENT
    if capacity < 0:
        return dict(feasible=False, reason='Fixed envelope exceeds budget')
    cache = OrderedDict()
    occupied = 0
    phases = {phase: dict(miss_bytes=0, logical_miss_bytes=0, groups=0,
                          selected_expert_uses=0, expert_misses=0)
              for phase in ('prefill', 'decode')}
    decode_bytes = Counter()
    peak = 0
    for r in records:
        layer = r['layer']
        # All selected dependencies remain leased until this layer/chunk completes.
        required = {(layer, expert) for ids in r['routes'] for expert in ids}
        need = len(required) * SIZES[layer]
        if need > capacity:
            return dict(feasible=False, reason='Selected layer/chunk closure cannot fit',
                        required_bytes=need, expert_capacity_bytes=capacity)
        missing = required - cache.keys()
        incoming = len(missing) * SIZES[layer]
        while occupied + incoming > capacity:
            victim = next(key for key in cache if key not in required)
            del cache[victim]
            occupied -= SIZES[victim[0]]
        for key in sorted(required):
            if key in cache:
                cache.move_to_end(key)
            else:
                cache[key] = None
                occupied += SIZES[layer]
        peak = max(peak, occupied)
        p = phases[r['phase']]
        p['miss_bytes'] += incoming
        p['logical_miss_bytes'] += len(missing) * LOGICAL[layer]
        p['groups'] += 1
        p['selected_expert_uses'] += len(required)
        p['expert_misses'] += len(missing)
        if r['phase'] == 'decode':
            decode_bytes[r['step']] += incoming
    for p in phases.values():
        p['bulk_no_overlap_seconds'] = p['miss_bytes'] / READ_BPS
        p['serial_extent_service_seconds'] = p['miss_bytes'] / EXTENT * SERIAL_EXTENT_US / 1e6
        p['extent_reads'] = p['miss_bytes'] // EXTENT
    values = sorted(decode_bytes.values())
    return dict(feasible=True, total_budget_bytes=total_budget, expert_capacity_bytes=capacity,
                peak_expert_bytes=peak, phases=phases,
                decode_token_miss_bytes=dict(mean=sum(values) / len(values),
                                            p50=values[(len(values)-1)//2],
                                            p95=values[int((len(values)-1)*.95)], max=max(values)),
                whole_model_resident_feasible=capacity >= sum(SIZES) * EXPERTS)


def locality(records):
    # Group-aware reuse distance: distinct experts accessed in STRICTLY intervening
    # token groups; both endpoint groups are excluded. Top-k peers have equal time.
    last_seen = [[-1] * EXPERTS for _ in range(LAYERS)]
    clocks = [0] * LAYERS
    hist = {p: [Counter() for _ in range(LAYERS)] for p in ('prefill', 'decode')}
    cold = {p: [0] * LAYERS for p in hist}
    decode = [[] for _ in range(LAYERS)]
    unions = []
    for r in records:
        layer, phase = r['layer'], r['phase']
        unions.append((phase, len({n for ids in r['routes'] for n in ids})))
        seen = last_seen[layer]
        for ids in r['routes']:
            ordered = sorted(seen)
            for n in ids:
                if seen[n] >= 0:
                    hist[phase][layer][EXPERTS - bisect_right(ordered, seen[n])] += 1
                else:
                    cold[phase][layer] += 1
            for n in ids:
                seen[n] = clocks[layer]
            clocks[layer] += 1
            if phase == 'decode':
                decode[layer].append(set(ids))
    per_layer = []
    for layer in range(LAYERS):
        entry = dict(layer=layer)
        for phase in hist:
            h = hist[phase][layer]
            n = sum(h.values())
            def quantile(q):
                remaining = int((n - 1) * q)
                for value, count in sorted(h.items()):
                    if remaining < count:
                        return value
                    remaining -= count
                return None
            entry[phase] = dict(cold=cold[phase][layer], reused=n,
                                distinct_distance_p50=quantile(.5), distinct_distance_p95=quantile(.95))
        per_layer.append(entry)
    # Causal train/test split. Learn layer-i -> layer-(i+1) associations on the
    # first half of decode tokens only; evaluate the later half, never training on it.
    split = len(decode[0]) // 2
    hits = baseline_hits = last_token_hits = total = 0
    for layer in range(1, LAYERS):
        counts = Counter(n for ids in decode[layer][:split] for n in ids)
        baseline = set(sorted(range(EXPERTS), key=lambda n: (-counts[n], n))[:TOP_K])
        pairs = [Counter() for _ in range(EXPERTS)]
        for previous, current in zip(decode[layer-1][:split], decode[layer][:split]):
            for n in previous:
                pairs[n].update(current)
        for t in range(split, len(decode[layer])):
            scores = Counter()
            for n in decode[layer-1][t]:
                scores.update(pairs[n])
            predicted = set(sorted(range(EXPERTS), key=lambda n: (-scores[n], n))[:TOP_K])
            actual = decode[layer][t]
            hits += len(predicted & actual)
            baseline_hits += len(baseline & actual)
            last_token_hits += len(decode[layer][t-1] & actual)
            total += len(actual)
    union_stats = {}
    for phase in hist:
        values = sorted(n for p, n in unions if p == phase)
        union_stats[phase] = dict(mean=sum(values)/len(values), p50=values[(len(values)-1)//2],
                                  p95=values[int((len(values)-1)*.95)], max=max(values))
    return dict(per_layer_reuse=per_layer, selected_union=union_stats,
                predictability=dict(training_tokens=split, test_tokens=len(decode[0])-split,
                                    selected_contributions=total, next_layer_recall=hits/total,
                                    static_layer_recall=baseline_hits/total,
                                    previous_token_same_layer_recall=last_token_hits/total))


def validate_metadata(metadata):
    for key, expected in [('image', IMAGE), ('engine_revision', REVISION),
                          ('model_sha256', MODEL_SHA), ('trace_sha256', TRACE_SHA)]:
        if metadata.get(key) != expected:
            raise ValueError('Capture provenance mismatch: ' + key)
    expected = [('b512-control',512,False), ('b512-routes',512,True),
                ('b64-control',64,False), ('b64-routes',64,True)]
    actual = [(r['name'],r['batch'],r['tracing']) for r in metadata['runs']]
    if actual != expected:
        raise ValueError('Both distinct batch captures and controls required')
    for i in (1,3):
        if metadata['runs'][i]['predictions_sha256'] != metadata['runs'][i-1]['predictions_sha256']:
            raise ValueError('Capture/control prediction mismatch')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    metadata = json.loads((a.capture / 'capture.json').read_text())
    validate_metadata(metadata)
    for run in metadata['runs']:
        if sha(a.capture / (run['name'] + '.predictions')) != run['predictions_sha256']:
            raise ValueError('Prediction identity mismatch')
    results = dict(fixed_bytes=FIXED, budget_components=dict(non_expert=NON_EXPERT, kv=KV,
                    scratch_allowance=SCRATCH, staging=STAGING, headroom_allowance=HEADROOM),
                   layout=dict(expert_payload_bytes=sum(LOGICAL)*EXPERTS,
                               expert_padded_bytes=sum(SIZES)*EXPERTS,
                               extent_bytes=EXTENT), runs=[])
    for run in metadata['runs']:
        if not run['tracing']:
            continue
        records = load(a.capture / (run['name'] + '.jsonl'), metadata, run)
        results['runs'].append(dict(name=run['name'], routes_sha256=run['routes_sha256'],
                                   locality=locality(records),
                                   budgets=[replay(records, n*GIB) for n in (12,14,16,18,20,22,24,28)]))
    if len(results['runs']) != 2:
        raise ValueError('Both batch captures required')
    with a.output.open('x') as f:
        json.dump(results, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
