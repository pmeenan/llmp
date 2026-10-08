#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, two-node host-buffer RDMA baseline. Raw outputs stay outside Git."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import time


def remote(host, args):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
            shlex.join(args)]


def run_pair(out, name, server, client, ip, device, tool, opts, port):
    args = ['timeout', '--kill-after=5', '90', 'stdbuf', '-oL', '-eL', tool, '-d', device,
            '-R', '-p', str(port), *opts]
    receipt = {'name': name, 'server': server, 'client': client,
               'server_args': args, 'client_args': args + [ip],
               'started_unix': time.time()}
    with (out / (name + '.server.log')).open('w') as sf, \
         (out / (name + '.client.log')).open('w') as cf:
        proc = subprocess.Popen(remote(server, args), stdout=sf, stderr=subprocess.STDOUT)
        try:
            time.sleep(0.7)
            receipt['client_launch_unix'] = time.time()
            child = subprocess.run(remote(client, args + [ip]), stdout=cf,
                                   stderr=subprocess.STDOUT, timeout=100)
            receipt['client_rc'] = child.returncode
            receipt['server_rc'] = proc.wait(timeout=100)
        except BaseException as exc:
            receipt['exception'] = repr(exc)
            receipt['remote_completion'] = 'unconfirmed; remote timeout bounds test to 95 seconds'
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            receipt['finished_unix'] = time.time()
            (out / (name + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt.get('client_rc') != 0 or receipt.get('server_rc') != 0:
        raise RuntimeError(f'{name}: failed, inspect raw logs in {out}')
    print(name, 'PASS', flush=True)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--server', default='spark')
    ap.add_argument('--client', default='spark-b')
    ap.add_argument('--server-ips', nargs=2, default=['10.100.208.2', '10.100.209.2'])
    ap.add_argument('--client-ips', nargs=2, default=['10.100.208.1', '10.100.209.1'])
    ap.add_argument('--devices', nargs=2, default=['rocep1s0f1', 'roceP2p1s0f1'])
    ap.add_argument('--phase', choices=['sweep', 'large', 'combined'], required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    for rep in range(3):
        if a.phase == 'sweep':
            for idx, dev in enumerate(a.devices):
                for tool in ['ib_write_bw', 'ib_read_bw', 'ib_send_lat']:
                    opts = ['-a', '-n', '1000']
                    if tool.endswith('_bw'):
                        opts += ['--report_gbits']
                    records.append(run_pair(a.out, f'{tool}-h{idx}-r{rep}', a.server,
                        a.client, a.server_ips[idx], dev, tool, opts, 18700 + idx))
        else:
            for direction in ['forward', 'reverse', 'bidirectional']:
                for tool in ['ib_write_bw', 'ib_read_bw']:
                    if direction == 'bidirectional' and tool == 'ib_read_bw':
                        continue
                    opts = ['-s', str(2**23), '-D', '8', '-f', '1', '--report_gbits']
                    if direction == 'bidirectional':
                        opts += ['-b', '--report-both']
                    server, client, ips = a.server, a.client, a.server_ips
                    if direction == 'reverse':
                        server, client, ips = client, server, a.client_ips
                    jobs = [(a.out, f'{tool}-{direction}-h{idx}-r{rep}', server,
                             client, ips[idx], dev, tool, opts, 18700 + idx)
                            for idx, dev in enumerate(a.devices)]
                    if a.phase == 'combined':
                        with concurrent.futures.ThreadPoolExecutor(2) as ex:
                            records.extend(ex.map(lambda job: run_pair(*job), jobs))
                    else:
                        records.extend(run_pair(*job) for job in jobs)
    (a.out / 'receipts.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
