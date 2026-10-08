#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate completed receipts and aggregate the interconnect baseline."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics


HOST_SIZES = [2**n for n in range(1, 24)]
NCCL_SIZES = [8 * 4**n for n in range(14)]


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def expected_host_names(phase):
    if phase == 'sweep-main':
        arms = ['ib_write_bw', 'ib_read_bw', 'ib_send_lat']
    else:
        arms = [f'{tool}-{direction}' for direction in ['forward', 'reverse', 'bidirectional']
                for tool in ['ib_write_bw', 'ib_read_bw']
                if not (direction == 'bidirectional' and tool == 'ib_read_bw')]
    return {f'{arm}-h{h}-r{rep}' for arm in arms for h in range(2) for rep in range(3)}


def expected_nccl_names():
    return {f'main-{op}-{arm}-r{rep}' for op in ['sendrecv', 'all_reduce', 'all_gather']
            for arm in ['auto', 'h0', 'h1'] for rep in range(3)}


def receipts_for(folder, expected):
    receipts = json.loads((folder / 'receipts.json').read_text())
    names = [r['name'] for r in receipts]
    check(len(names) == len(set(names)), f'{folder.name}: duplicate receipt identities')
    check(set(names) == expected, f'{folder.name}: missing or unexpected cases/repetitions')
    for rec in receipts:
        check(json.loads((folder / (rec['name'] + '.json')).read_text()) == rec,
              f"{rec['name']}: index differs from individual receipt")
    return receipts


def numeric_rows(content, columns, name, nccl=False):
    pattern = r'^\s*\d+\s+\d+\s+float\s+' if nccl else r'^\s*\d+\s+\d+\s+'
    rows = [line.split() for line in content.splitlines() if re.match(pattern, line)]
    check(all(len(row) == columns for row in rows), f'{name}: unexpected result columns')
    return rows


def finite(value, name, positive=False):
    value = float(value)
    check(math.isfinite(value) and (value > 0 if positive else value >= 0),
          f'{name}: invalid numeric metric {value}')
    return value


def parse_host(content, name, phase):
    bandwidth = '_bw' in name
    bidirectional = 'bidirectional' in name
    rows = numeric_rows(content, 5 if bandwidth else 9, name)
    sizes = [int(row[0]) for row in rows]
    expected = [2**23] * 3 if bidirectional else HOST_SIZES if phase == 'sweep-main' else [2**23]
    check(sizes == expected, f'{name}: missing, duplicate or unexpected message sizes')
    if bandwidth:
        check('BW average[Gb/sec]' in content, f'{name}: bandwidth units are not Gb/s')
    else:
        check('t_typical[usec]' in content and '99% percentile[usec]' in content,
              f'{name}: latency columns/units differ')
    if bidirectional:
        check('Local results:' in content and 'Remote results:' in content,
              f'{name}: directional results missing')
        check(content.index('Local results:') < content.index('Remote results:'),
              f'{name}: directional result order differs')
    parsed = []
    for i, raw in enumerate(rows):
        row = [finite(x, name) for x in raw]
        check(row[1] > 0 and row[1].is_integer(), f'{name}: invalid iteration count')
        if phase == 'sweep-main':
            check(row[1] == 1000, f'{name}: wrong sweep iteration count')
        if bandwidth:
            metrics = {'bw_gbps': row[3]}
        else:
            check(row[2] <= row[4] <= row[7] <= row[3], f'{name}: latency order differs')
            metrics = {'median_us': row[4], 'p99_us': row[7], 'max_us': row[3]}
        suffix = ['aggregate', 'tx', 'rx'][i] if bidirectional else 'oneway'
        parsed.append((int(row[0]), suffix, metrics))
    if bidirectional:
        check(abs(parsed[0][2]['bw_gbps'] - sum(r[2]['bw_gbps'] for r in parsed[1:])) <= .021,
              f'{name}: bidirectional total does not reconcile')
    return parsed


def parse_nccl(content, name):
    op = name.removeprefix('main-').rsplit('-', 2)[0]
    check('Out of bounds values : 0 OK' in content, f'{name}: failed/missing correctness summary')
    check(f'Collective test concluded: {op}_perf' in content, f'{name}: incomplete output')
    check('on spark-c4e2 device  0' in content and 'on spark-56f5 device  0' in content,
          f'{name}: expected ranks/devices absent')
    check('nccl-headers=23007 nccl-library=23007' in content and
          'nccl-tests version 2.20.0 (b4d5bee)' in content, f'{name}: engine/test version differs')
    check('warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0' in content,
          f'{name}: iteration/validation protocol differs')
    check('minBytes 8 maxBytes 536870912 step: 4(factor)' in content,
          f'{name}: message sweep differs')
    rows = numeric_rows(content, 13, name, nccl=True)
    # Pinned AllGatherGetCollByteCount rounds each rank's contribution to
    # 16 bytes; the requested 8-byte total becomes a zero-count no-op.
    sizes = [0, *NCCL_SIZES[1:]] if op == 'all_gather' else NCCL_SIZES
    check([int(row[0]) for row in rows] == sizes,
          f'{name}: missing, duplicate or unexpected message sizes')
    parsed = []
    for row in rows:
        check(int(row[1]) == int(row[0]) // (8 if op == 'all_gather' else 4),
              f'{name}: element count differs from the pinned operation')
        check(row[8] == '0', f'{name}: out-of-place data mismatch')
        variants = [('out', 5)]
        if op != 'sendrecv':
            check(row[12] == '0', f'{name}: in-place data mismatch')
            variants.append(('in', 9))
        for variant, offset in variants:
            metrics = {metric: finite(row[offset+i], name, positive=(i == 0))
                       for metric, i in [('us', 0), ('algbw_GBs', 1), ('busbw_GBs', 2)]}
            factor = .5 if op == 'all_gather' else 1
            check(abs(metrics['busbw_GBs'] - factor * metrics['algbw_GBs']) <= .011,
                  f'{name}: unexpected bandwidth definition')
            if int(row[0]) == 0:
                check(metrics['algbw_GBs'] == metrics['busbw_GBs'] == 0,
                      f'{name}: zero-count operation reports bandwidth')
            else:
                parsed.append((int(row[0]), variant, metrics))
    return parsed


def validate_host_args(rec, phase, devices):
    name = rec['name']
    tool = next(t for t in ['ib_write_bw', 'ib_read_bw', 'ib_send_lat'] if name.startswith(t))
    hca = int(name.rsplit('-h', 1)[1][0])
    args = rec['server_args']
    check(tool in args, f'{name}: wrong executable')
    at = args.index(tool)
    check(args[at+1] == '-d', f'{name}: missing explicit HCA')
    dev = args[at+2]
    check(devices.setdefault(hca, dev) == dev, f'{name}: HCA identity changed')
    if len(devices) == 2:
        check(devices[0] != devices[1], f'{name}: concurrent arms select the same HCA')
    opts = ['-a', '-n', '1000'] if phase == 'sweep-main' else ['-s', str(2**23), '-D', '8', '-f', '1']
    if tool.endswith('_bw'):
        opts += ['--report_gbits']
    if 'bidirectional' in name:
        opts += ['-b', '--report-both']
    check(args[at:] == [tool, '-d', dev, '-R', '-p', str(18700+hca), *opts],
          f'{name}: host test arguments differ from the protocol')
    check(rec['client_args'][:-1] == args, f'{name}: client/server options differ')
    check(rec['server'] != rec['client'], f'{name}: both processes target the same host')
    return dev


def validate_nccl_args(rec, devices):
    name = rec['name']
    op, arm, _ = name.removeprefix('main-').rsplit('-', 2)
    args = rec['args']
    binary = next((i for i, arg in enumerate(args) if arg.endswith('/' + op + '_perf')), None)
    check(binary is not None, f'{name}: wrong executable')
    check(args[binary+1:] == ['-b', '8', '-e', '512M', '-f', '4', '-g', '1',
                              '-w', '5', '-n', '20', '-c', '1'],
          f'{name}: NCCL test arguments differ from the protocol')
    check('-np' in args and args[args.index('-np')+1] == '2', f'{name}: wrong rank count')
    env = rec['environment']
    expected_hca = None if arm == 'auto' else '=' + devices[int(arm[1])] + ':1'
    check(env.get('NCCL_IB_HCA') == expected_hca, f'{name}: HCA selector differs from its arm')
    check(env.get('CUDA_DISABLE_PTX_JIT') == '1' and env.get('NCCL_SOCKET_IFNAME') == '=enP7s7',
          f'{name}: PTX/bootstrap configuration differs')
    check('NCCL_NET' not in env, f'{name}: main run unexpectedly forces a network transport')


def stats(values):
    return {'n': len(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}


def summarize(raw):
    result = {'host': {}, 'nccl': {}, 'counts': {}}
    devices = {}
    for phase in ['sweep-main', 'large', 'combined']:
        folder = raw / ('host-' + phase)
        receipts = receipts_for(folder, expected_host_names(phase))
        grouped = defaultdict(lambda: defaultdict(list))
        skew = defaultdict(list)
        paired = defaultdict(list)
        for rec in receipts:
            check(rec['client_rc'] == rec['server_rc'] == 0, f"{rec['name']}: unsuccessful trial")
            name = rec['name']
            dev = validate_host_args(rec, phase, devices)
            content = (folder / (name + '.client.log')).read_text()
            found = re.search(r'\bDevice\s*:\s*(\S+)', content)
            check(found is not None and found[1] == dev, f'{name}: logged HCA differs from receipt')
            rows = parse_host(content, name, phase)
            key = re.sub(r'-r\d+$', '', name)
            for size, suffix, metrics in rows:
                for metric, val in metrics.items():
                    grouped[f'{key}/{size}/{suffix}'][metric].append(val)
                    if phase == 'combined':
                        paired[(re.sub(r'-h\d', '', name), size, suffix, metric)].append(val)
            if phase == 'combined':
                launch = finite(rec['client_launch_unix'], name)
                check(finite(rec['started_unix'], name) <= launch <= finite(rec['finished_unix'], name),
                      f'{name}: invalid launch timestamps')
                skew[re.sub(r'-h\d', '', name)].append(launch)
        for key, metrics in grouped.items():
            check(all(len(v) == 3 for v in metrics.values()), f'{key}: unequal repeats')
        result['host'][phase] = {k: {m: stats(v) for m, v in fields.items()}
                                  for k, fields in sorted(grouped.items())}
        if phase == 'combined':
            check(all(len(v) == 2 for v in skew.values()), 'Incomplete concurrent HCA pair')
            result['combined_launch_skew_s'] = stats([max(v)-min(v) for v in skew.values()])
            sums = defaultdict(list)
            for (name, size, suffix, metric), values in paired.items():
                check(len(values) == 2, f'{name}: incomplete concurrent HCA pair')
                sums[(re.sub(r'-r\d+$', '', name), size, suffix, metric)].append(sum(values))
            result['combined_sum'] = {'/'.join(map(str, key)): stats(values)
                                      for key, values in sorted(sums.items())}
        result['counts']['host-' + phase] = len(receipts)
    folder = raw / 'nccl-main'
    receipts = receipts_for(folder, expected_nccl_names())
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in receipts:
        check(rec['rc'] == 0, f"{rec['name']}: unsuccessful trial")
        name = rec['name']
        validate_nccl_args(rec, devices)
        content = (folder / (name + '.log')).read_text()
        rows = parse_nccl(content, name)
        key = re.sub(r'-r\d+$', '', name).removeprefix('main-')
        for size, variant, metrics in rows:
            for metric, value in metrics.items():
                grouped[f'{key}/{size}/{variant}'][metric].append(value)
    for key, metrics in grouped.items():
        check(all(len(v) == 3 for v in metrics.values()), f'{key}: unequal repeats')
    result['nccl'] = {k: {m: stats(v) for m, v in fields.items()}
                      for k, fields in sorted(grouped.items())}
    result['counts']['nccl-main'] = len(receipts)
    result['excluded_zero_count_rows'] = {'all_gather': 9}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('raw', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--csv-out', type=Path)
    a = ap.parse_args()
    result = summarize(a.raw)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    if a.csv_out:
        with a.csv_out.open('w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['family', 'case', 'metric', 'runs', 'median', 'min', 'max'])

            def emit(family, case, metric, values):
                writer.writerow([family, case, metric, values['n'],
                                 *[format(values[k], '.9g') for k in ['median', 'min', 'max']]])

            for phase, cases in result['host'].items():
                for case, metrics in cases.items():
                    for metric, values in metrics.items():
                        emit('host-' + phase, case, metric, values)
            for case, metrics in result['nccl'].items():
                for metric, values in metrics.items():
                    emit('nccl', case, metric, values)
            for key, values in result['combined_sum'].items():
                case, metric = key.rsplit('/', 1)
                emit('combined-sum', case, metric, values)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
