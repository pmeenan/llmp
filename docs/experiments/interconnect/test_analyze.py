#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Adversarial receipt/log checks; fixtures are generated only in temporary dirs."""
import json
from pathlib import Path
import tempfile
import unittest

import analyze


DEVICES = ['rocep1s0f1', 'roceP2p1s0f1']


def host_log(name, phase, value=20):
    device = DEVICES[int(name.rsplit('-h', 1)[1][0])]
    device_header = f'Device : {device}\n'
    if 'ib_send_lat' in name:
        return (device_header + '#bytes #iterations t_min[usec] t_max[usec] t_typical[usec] '
                't_avg[usec] t_stdev[usec] 99% percentile[usec] 99.9% percentile[usec]\n' +
                ''.join(f'{size} 1000 1 8 2 3 1 7 8\n' for size in analyze.HOST_SIZES))
    header = device_header + '#bytes #iterations BW peak[Gb/sec] BW average[Gb/sec] MsgRate[Mpps]\n'
    if 'bidirectional' in name:
        return (header + f'8388608 1000 0 {2*value} 1\nLocal results:\n' + header +
                f'8388608 1000 0 {value} 1\nRemote results:\n' + header +
                f'8388608 1000 0 {value} 1\n')
    sizes = analyze.HOST_SIZES if phase == 'sweep-main' else [2**23]
    return header + ''.join(f'{size} 1000 0 {value} 1\n' for size in sizes)


def nccl_log(name, value=20):
    op = name.removeprefix('main-').rsplit('-', 2)[0]
    bus = value / 2 if op == 'all_gather' else value
    lines = [
        '# nccl-tests version 2.20.0 (b4d5bee) nccl-headers=23007 nccl-library=23007',
        '# nThread 1 nGpus 1 minBytes 8 maxBytes 536870912 step: 4(factor) '
        'warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0',
        '# Rank 0 on spark-c4e2 device  0', '# Rank 1 on spark-56f5 device  0',
    ]
    sizes = [0, *analyze.NCCL_SIZES[1:]] if op == 'all_gather' else analyze.NCCL_SIZES
    for size in sizes:
        wrong = 'N/A' if op == 'sendrecv' else '0'
        algbw, busbw = (value, bus) if size else (0, 0)
        lines.append(f'{size} {size//(8 if op == "all_gather" else 4)} float sum -1 '
                     f'{value} {algbw} {busbw} 0 {value} {algbw} {busbw} {wrong}')
    lines += ['# Out of bounds values : 0 OK', f'# Collective test concluded: {op}_perf']
    return '\n'.join(lines)


def complete_fixture(root):
    for phase in ['sweep-main', 'large', 'combined']:
        folder = root / ('host-' + phase)
        folder.mkdir()
        records = []
        for name in sorted(analyze.expected_host_names(phase)):
            rep = int(name[-1])
            hca = int(name.rsplit('-h', 1)[1][0])
            tool = next(t for t in ['ib_write_bw', 'ib_read_bw', 'ib_send_lat'] if name.startswith(t))
            opts = ['-a', '-n', '1000'] if phase == 'sweep-main' else ['-s', '8388608', '-D', '8', '-f', '1']
            if tool.endswith('_bw'):
                opts += ['--report_gbits']
            if 'bidirectional' in name:
                opts += ['-b', '--report-both']
            args = [tool, '-d', DEVICES[hca], '-R', '-p', str(18700+hca), *opts]
            rec = {'name': name, 'server_rc': 0, 'client_rc': 0,
                   'started_unix': 1, 'client_launch_unix': 2 + hca * .01,
                   'finished_unix': 10, 'server': 'spark', 'client': 'spark-b',
                   'server_args': args, 'client_args': args + ['10.100.208.2']}
            records.append(rec)
            (folder / (name + '.json')).write_text(json.dumps(rec))
            (folder / (name + '.client.log')).write_text(
                host_log(name, phase, [10, 50, 20][rep] + hca * 100))
        (folder / 'receipts.json').write_text(json.dumps(records))
    folder = root / 'nccl-main'
    folder.mkdir()
    records = []
    for name in sorted(analyze.expected_nccl_names()):
        op, arm, _ = name.removeprefix('main-').rsplit('-', 2)
        env = {'CUDA_DISABLE_PTX_JIT': '1', 'NCCL_SOCKET_IFNAME': '=enP7s7'}
        if arm != 'auto':
            env['NCCL_IB_HCA'] = '=' + DEVICES[int(arm[1])] + ':1'
        rec = {'name': name, 'rc': 0, 'environment': env,
               'args': ['mpirun', '-np', '2', '/runtime/' + op + '_perf',
                        '-b', '8', '-e', '512M', '-f', '4', '-g', '1',
                        '-w', '5', '-n', '20', '-c', '1']}
        records.append(rec)
        (folder / (name + '.json')).write_text(json.dumps(rec))
        (folder / (name + '.log')).write_text(nccl_log(name, [10, 50, 20][int(name[-1])]))
    (folder / 'receipts.json').write_text(json.dumps(records))


class ParserTests(unittest.TestCase):
    def test_latency_columns_are_not_confused_with_average_or_percentile(self):
        rows = analyze.parse_host(host_log('ib_send_lat-h0-r0', 'sweep-main'),
                                  'ib_send_lat-h0-r0', 'sweep-main')
        self.assertEqual(rows[0], (2, 'oneway', {'median_us': 2, 'p99_us': 7, 'max_us': 8}))

    def test_missing_or_duplicate_size_cannot_preserve_row_count(self):
        name = 'ib_write_bw-h0-r0'
        with self.assertRaises(ValueError):
            analyze.parse_host(host_log(name, 'sweep-main').replace('2 1000 ', '4 1000 ', 1),
                               name, 'sweep-main')
        name = 'main-all_reduce-auto-r0'
        with self.assertRaises(ValueError):
            analyze.parse_nccl(nccl_log(name).replace('8 2 float', '32 8 float', 1), name)

    def test_bandwidth_units_and_bidirectional_total_are_checked(self):
        name = 'ib_write_bw-bidirectional-h0-r0'
        content = host_log(name, 'large')
        self.assertEqual([r[2]['bw_gbps'] for r in analyze.parse_host(content, name, 'large')],
                         [40, 20, 20])
        for bad in [content.replace('Gb/sec', 'MB/sec'),
                    content.replace('1000 0 40 ', '1000 0 200 '),
                    content.replace('Local results:', 'Remote results:', 1)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                analyze.parse_host(bad, name, 'large')

    def test_nonfinite_and_negative_metrics_are_rejected(self):
        host_name = 'ib_write_bw-forward-h0-r0'
        nccl_name = 'main-all_reduce-auto-r0'
        for number in ['nan', 'inf', '-1']:
            with self.subTest(number=number):
                with self.assertRaises(ValueError):
                    analyze.parse_host(host_log(host_name, 'large').replace(' 20 ', f' {number} '),
                                       host_name, 'large')
                with self.assertRaises(ValueError):
                    analyze.parse_nccl(nccl_log(nccl_name).replace(' 20 ', f' {number} '), nccl_name)

    def test_sendrecv_unsupported_inplace_is_excluded(self):
        name = 'main-sendrecv-auto-r0'
        rows = analyze.parse_nccl(nccl_log(name), name)
        self.assertEqual(len(rows), 14)
        self.assertEqual({r[1] for r in rows}, {'out'})

    def test_allgather_zero_count_is_checked_but_not_reported_as_latency(self):
        name = 'main-all_gather-auto-r0'
        rows = analyze.parse_nccl(nccl_log(name), name)
        self.assertEqual(len(rows), 26)
        self.assertEqual(min(r[0] for r in rows), 32)
        with self.assertRaises(ValueError):
            analyze.parse_nccl(nccl_log(name).replace('0 0 float', '8 1 float', 1), name)

    def test_each_supported_correctness_column_must_pass(self):
        name = 'main-all_reduce-auto-r0'
        for suffix in ['20 20 20 9 20 20 20 0', '20 20 20 0 20 20 20 N/A']:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                analyze.parse_nccl(nccl_log(name).replace('20 20 20 0 20 20 20 0', suffix, 1), name)

    def test_truncated_wrong_version_and_unchecked_nccl_runs_fail(self):
        name = 'main-all_reduce-auto-r0'
        content = nccl_log(name)
        for bad in [content.replace('Collective test concluded:', 'truncated:'),
                    content.replace('validation: 1', 'validation: 0'),
                    content.replace('nccl-library=23007', 'nccl-library=23008'),
                    content.replace('(b4d5bee)', '(abcdef0)')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                analyze.parse_nccl(bad, name)


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        complete_fixture(self.root)

    def test_complete_three_repeat_aggregate_and_paired_sum(self):
        result = analyze.summarize(self.root)
        self.assertEqual(result['counts'], {'host-sweep-main': 18, 'host-large': 30,
                                            'host-combined': 30, 'nccl-main': 27})
        expected = {'n': 3, 'median': 20, 'min': 10, 'max': 50}
        self.assertEqual(result['host']['large']['ib_write_bw-forward-h0/8388608/oneway']['bw_gbps'],
                         expected)
        self.assertEqual(result['nccl']['all_reduce-auto/8/out']['us'], expected)
        self.assertEqual(result['combined_sum']['ib_write_bw-forward/8388608/oneway/bw_gbps'],
                         {'n': 3, 'median': 140, 'min': 120, 'max': 200})

    def test_duplicate_first_repeat_cannot_impersonate_three_repeats(self):
        for folder_name in ['host-sweep-main', 'host-large', 'host-combined', 'nccl-main']:
            path = self.root / folder_name / 'receipts.json'
            old = path.read_text()
            receipts = json.loads(old)
            path.write_text(json.dumps([r for r in receipts if r['name'].endswith('-r0')] * 3))
            with self.subTest(folder=folder_name), self.assertRaises(ValueError):
                analyze.summarize(self.root)
            path.write_text(old)

    def test_receipt_index_must_match_individual_record(self):
        folder = self.root / 'nccl-main'
        path = folder / 'main-all_reduce-auto-r0.json'
        rec = json.loads(path.read_text())
        rec['rc'] = 1
        path.write_text(json.dumps(rec))
        with self.assertRaises(ValueError):
            analyze.summarize(self.root)

    def test_mislabeled_hca_receipt_is_rejected_even_with_successful_log(self):
        folder = self.root / 'nccl-main'
        path = folder / 'receipts.json'
        records = json.loads(path.read_text())
        rec = next(r for r in records if r['name'] == 'main-all_reduce-h0-r0')
        rec['environment']['NCCL_IB_HCA'] = '=' + DEVICES[1] + ':1'
        path.write_text(json.dumps(records))
        (folder / (rec['name'] + '.json')).write_text(json.dumps(rec))
        with self.assertRaises(ValueError):
            analyze.summarize(self.root)

    def test_unknown_case_or_missing_repeat_is_not_accepted(self):
        path = self.root / 'nccl-main' / 'receipts.json'
        records = json.loads(path.read_text())
        for bad in [records[:-1], [{**records[0], 'name': 'main-all_reduce-h9-r0'}, *records[1:]]]:
            path.write_text(json.dumps(bad))
            with self.subTest(bad=bad[0]['name']), self.assertRaises(ValueError):
                analyze.summarize(self.root)


if __name__ == '__main__':
    unittest.main()
