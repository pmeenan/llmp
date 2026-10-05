#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate full heads; aggregate bounded host observations, not CUDA counts."""
import collections
import hashlib
import json
import pathlib
import re
import sys

HEAD_BYTES = 128 * 262144 * 4
HEAD_SHA = '5808306e0f261896500bdf9f3b44d4b1d0f18b34ba99d976da15a916aa8ff164'
ID_SHA = 'b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610'


def digest(path, size):
    if path.stat().st_size != size:
        raise ValueError('wrong complete file size: ' + str(path))
    result = hashlib.sha256()
    with path.open('rb') as file:
        while block := file.read(1 << 20):
            result.update(block)
    return result.hexdigest()


def tensor(line, role):
    match = re.search(r' ' + role + r'=([^=,\n]+),([A-Z_]+),([A-Za-z0-9_]+),ne=([0-9:]+),nb=([0-9:]+),data=(0x[0-9a-f]+|\(nil\)),bytes=([0-9]+)(?:,alloc=([0-9]+))?', line)
    if not match:
        raise ValueError('missing descriptor ' + role)
    name, op, dtype, ne, nb, address, size, allocation = match.groups()
    dims, strides = tuple(map(int, ne.split(':'))), tuple(map(int, nb.split(':')))
    if len(dims) != 4 or len(strides) != 4 or len(name) > 63:
        raise ValueError('unbounded descriptor')
    return {'name': name, 'op': op, 'type': dtype, 'ne': dims, 'nb': strides,
            'bytes': int(size), 'allocation': None if allocation is None else int(allocation)}


def number(line, field):
    match = re.search(r'\b' + field + r'=(-?\d+)\b', line)
    if not match:
        raise ValueError('missing field ' + field)
    return int(match[1])


def boolean(line, field):
    value = number(line, field)
    if value not in (0, 1):
        raise ValueError('non-Boolean ' + field)
    return value


def shape(desc, ne, dtype='f32'):
    if desc['ne'] != ne or desc['type'] != dtype:
        raise ValueError('outside closed tensor shape/type')


def coverage(graphs, phases):
    for rows in (4, 64, 65, 66, 67):
        evaluated = [g for g in graphs.values() if g['rows'] == rows and g['records']]
        if not evaluated:
            raise ValueError('no evaluated graph in required phase')
        for kind in ('route_candidate', 'reduction_gate'):
            for gid, graph in graphs.items():
                if graph['rows'] == rows and graph['records'] and phases.get((gid, kind), set()) != set(range(30)):
                    raise ValueError('incomplete thirty-layer ' + kind + ' graph=' + str(gid))


def layer(name):
    match = re.search(r'-(\d+)(?: |$)', name)
    if not match or not 0 <= int(match[1]) < 30:
        raise ValueError('missing closed Gemma26 layer: ' + name)
    return int(match[1])


def observations(lines, require_complete=False):
    graphs = {}
    gates = collections.Counter()
    products = collections.Counter()
    descriptors = {}
    phases = collections.defaultdict(set)
    recognized = collections.Counter()
    route_gates = collections.Counter()
    allowed_routes = collections.Counter()
    selected_routes = collections.Counter()
    allowed_reductions = collections.Counter()
    selected_reductions = collections.Counter()
    record_count = 0
    for line in lines:
        if not line.startswith('JITLLM_OBSERVE '):
            continue
        record_count += 1
        if record_count > 200000 or len(line) > 8192:
            raise ValueError('observation bound exceeded')
        gid = number(line, 'graph')
        kind = re.search(r'\bkind=([a-z_]+)', line).group(1)
        if kind == 'graph':
            if gid in graphs or gid < 1:
                raise ValueError('duplicate graph id')
            rows = number(line, 'router_rows')
            if rows not in (4, 64, 65, 66, 67):
                raise ValueError('outside closed graph rows')
            graphs[gid] = {'rows': rows, 'cuda_graph': boolean(line, 'cuda_graph'),
                           'update': boolean(line, 'update'), 'records': 0}
            continue
        if gid not in graphs:
            raise ValueError('record without graph')
        graph = graphs[gid]
        graph['records'] += 1
        phase = 'decode4' if graph['rows'] == 4 else 'prefill' + str(graph['rows'])
        if kind == 'product':
            dst, a, b = (tensor(line, r) for r in ('dst', 'a', 'b'))
            family = re.search(r'\bfamily=([A-Z_]+)', line).group(1)
            if family not in {'CUBLAS', 'MMVF', 'MMVF_TRANSPOSE', 'MMF', 'MMVQ', 'MMQ',
                               'SORTED_FALLBACK', 'MMVQ_FUSED', 'MMVF_FUSED'}:
                raise ValueError('unknown product family')
            routed = boolean(line, 'routed')
            key = (phase, dst['name'], family, routed, a['type'], a['ne'], b['ne'], dst['ne'])
            products[key] += 1
            descriptors[(phase, dst['name'], family)] = {'a': a, 'b': b, 'dst': dst}
            if routed:
                ids = tensor(line, 'ids')
                if ids['ne'][0] != 8 or dst['ne'][2] != graph['rows']:
                    raise ValueError('routed shape differs from graph rows')
            continue
        node = tensor(line, 'node')
        lid = layer(node['name'])
        phases[(gid, kind)].add(lid)
        if kind in ('route_candidate', 'route_gate', 'route_selected'):
            shape(node, (128, graph['rows'], 1, 1))
        if kind == 'route_candidate':
            outcome = 'recognized=' + str(boolean(line, 'recognized'))
            recognized[(gid, lid)] += boolean(line, 'recognized')
        elif kind == 'route_gate':
            for field in ('structural', 'shape', 'memory'):
                boolean(line, field)
            if number(line, 'count') != 10:
                raise ValueError('outside ten-node routing contract')
            outcome = ','.join(f'{f}={number(line, f)}' for f in ('structural', 'shape', 'memory', 'count'))
            shape(tensor(line, 'logits'), (128, graph['rows'], 1, 1))
            shape(tensor(line, 'weights'), (1, 8, graph['rows'], 1))
            shape(tensor(line, 'ids'), (8, graph['rows'], 1, 1), 'i32')
            route_gates[(gid, lid)] += 1
            allowed_routes[(gid, lid)] += all(boolean(line, f) for f in ('structural', 'shape', 'memory'))
        elif kind == 'reduction_gate':
            matched = boolean(line, 'matched')
            boolean(line, 'scale')
            shape(node, (2816, 8, graph['rows'], 1))
            outcome = ','.join(f'{f}={number(line, f)}' for f in ('matched', 'count', 'scale', 'memory'))
            if matched:
                boolean(line, 'memory')
                if number(line, 'count') != 17 or number(line, 'scale') != 1:
                    raise ValueError('outside scaled seventeen-node reduction')
                shape(tensor(line, 'experts'), (2816, 8, graph['rows'], 1))
                shape(tensor(line, 'scale'), (1, 8, graph['rows'], 1))
                shape(tensor(line, 'weights'), (1, 8, graph['rows'], 1))
                shape(tensor(line, 'dst'), (2816, graph['rows'], 1, 1))
                allowed_reductions[(gid, lid)] += boolean(line, 'memory')
            elif any(number(line, f) != v for f, v in [('count', 0), ('scale', 0), ('memory', -1)]):
                raise ValueError('invalid unmatched reduction fields')
        elif kind in ('route_selected', 'reduction_selected'):
            outcome = 'selected'
            (selected_routes if kind == 'route_selected' else selected_reductions)[(gid, lid)] += 1
        else:
            raise ValueError('unknown observer record')
        gates[(phase, lid, kind, outcome)] += 1
    if not graphs or not products or not gates:
        raise ValueError('empty observations')
    for rows in (4, 64, 65, 66, 67):
        if not any(g['rows'] == rows for g in graphs.values()):
            raise ValueError('missing phase')
    if require_complete:
        coverage(graphs, phases)
        if +recognized != +route_gates:
            raise ValueError('recognized routing omitted gate observations')
        if +allowed_routes != +selected_routes or +allowed_reductions != +selected_reductions:
            raise ValueError('selected outcomes differ from recorded actual gates')
    return {'host_graph_calls': len(graphs), 'host_observation_records': record_count,
            'host_graph_phase_counts': dict(collections.Counter(g['rows'] for g in graphs.values())),
            'host_evaluated_or_captured_calls': sum(g['records'] > 0 for g in graphs.values()),
            'host_graph_policy_calls': dict(collections.Counter(
                f"allowed={g['cuda_graph']},update={g['update']}" for g in graphs.values())),
            'fusion_decisions': [{'phase': p, 'layer': l, 'kind': k, 'outcome': o, 'observations': v}
                                 for (p, l, k, o), v in sorted(gates.items())],
            'product_decisions': [{'phase': p, 'node': n, 'family': f, 'routed': r,
                                   'weight_type': t, 'weight_ne': a, 'input_ne': b,
                                   'output_ne': d, 'observations': v}
                                  for (p, n, f, r, t, a, b, d), v in sorted(products.items())],
            'product_descriptors': [{'phase': p, 'node': n, 'family': f, **value}
                                    for (p, n, f), value in sorted(descriptors.items())]}


def main():
    if len(sys.argv) != 6:
        raise ValueError('LOG OBSERVED OLD_A OLD_B NEW_AGGREGATE')
    log, observed, old_a, old_b, destination = map(pathlib.Path, sys.argv[1:])
    proofs = {}
    for name, root in [('observed', observed), ('old_a', old_a), ('old_b', old_b)]:
        proofs[name] = digest(root / 'heads.f32', HEAD_BYTES)
        if proofs[name] != HEAD_SHA or digest(root / 'inputs.i32', 4096) != ID_SHA:
            raise ValueError('intrusive observation or different actual IDs: ' + name)
    if log.stat().st_size > 64 << 20:
        raise ValueError('unbounded log')
    lines = log.read_text().splitlines()
    if not any(l == 'OBSERVATION_CLIENT_EXIT0 explicit_public_backend_teardown=source_verified' for l in lines):
        raise ValueError('missing successful original teardown marker')
    result = observations(lines, require_complete=True)
    result.update({'complete_head_bytes': HEAD_BYTES, 'complete_rows': 128,
                   'complete_head_sha256': proofs, 'actual_input_sha256': ID_SHA,
                   'claim': 'host decisions only; no CUDA replay/kernel count or timing claim'})
    with destination.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
