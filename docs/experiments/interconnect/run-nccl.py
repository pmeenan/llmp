#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run pinned native nccl-tests on two configured Spark nodes over SSH."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = '/home/pmeenan/.local/share/llmp/interconnect'


def ssh(host, script, **kwargs):
    kwargs.setdefault('timeout', 30)
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           host, script], check=True, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hosts', nargs=2, default=['spark', 'spark-b'])
    ap.add_argument('--devices', nargs=2, default=['rocep1s0f1', 'roceP2p1s0f1'])
    ap.add_argument('--mode', choices=['pilot', 'main', 'socket'], default='main')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    for host in a.hosts:
        ssh(host, shlex.join(['mkdir', '-p', ROOT + '/nccl-logs']))
        with (a.out / (host + '.preflight.json')).open('w') as f:
            ssh(host, 'source ' + shlex.quote(ROOT + '/env.sh') + '\n' +
                shlex.join(['python3', ROOT + '/audit.py']), stdout=f)
    arms = [('auto', {})]
    if a.mode == 'main':
        arms += [(f'h{i}', {'NCCL_IB_HCA': '=' + dev + ':1'})
                 for i, dev in enumerate(a.devices)]
    if a.mode == 'socket':
        arms = [('socket', {'NCCL_NET': 'Socket', 'NCCL_SOCKET_IFNAME': '=enp1s0f1np1'})]
    repeats = 3 if a.mode == 'main' else 1
    operations = ['sendrecv', 'all_reduce', 'all_gather'] if a.mode == 'main' else ['all_reduce']
    receipts = []
    for rep in range(repeats):
        for arm, extra in arms:
            for op in operations:
                name = f'{a.mode}-{op}-{arm}-r{rep}'
                for host in a.hosts:
                    with (a.out / f'{name}.{host}.before.json').open('w') as f:
                        ssh(host, shlex.join(['python3', ROOT + '/snapshot.py']), stdout=f)
                env = {'NCCL_DEBUG': 'INFO',
                       'NCCL_DEBUG_SUBSYS': 'INIT,NET,GRAPH,ENV,ALLOC_HOST,ALLOC,REG',
                       'NCCL_DEBUG_FILE': f'{ROOT}/nccl-logs/{name}.%h.%p.log',
                       'NCCL_SOCKET_IFNAME': '=enP7s7', 'CUDA_DISABLE_PTX_JIT': '1', **extra}
                args = ['timeout', '--kill-after=10', '180', 'mpirun.openmpi',
                        '-np', '2', '-H', ','.join(h + ':1' for h in a.hosts),
                        '--bind-to', 'none', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp',
                        '--mca', 'btl_tcp_if_include', 'enP7s7',
                        '--mca', 'oob_tcp_if_include', 'enP7s7',
                        '--mca', 'plm_rsh_agent', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                        '--mca', 'orte_launch_agent', '/bin/bash ' + ROOT + '/orted-wrapper.sh']
                for key in ['PATH', 'LD_LIBRARY_PATH', 'OPAL_PREFIX', 'OPAL_LIBDIR',
                            'OPAL_DATADIR', 'OPAL_SYSCONFDIR', 'OMPI_MCA_mca_base_component_path']:
                    args += ['-x', key]
                for key, value in env.items():
                    args += ['-x', key + '=' + value]
                args += [ROOT + '/nccl-tests-runtime/' + op + '_perf',
                         '-b', '8', '-e', '512M' if a.mode == 'main' else '8M',
                         '-f', '4', '-g', '1', '-w', '5', '-n', '20', '-c', '1']
                script = 'source ' + shlex.quote(ROOT + '/env.sh') + '\n' + shlex.join(args)
                receipt = {'name': name, 'args': args, 'environment': env, 'start_unix': time.time()}
                with (a.out / (name + '.log')).open('w') as f:
                    try:
                        ssh(a.hosts[0], script, stdout=f, stderr=subprocess.STDOUT, timeout=200)
                        receipt['rc'] = 0
                    except subprocess.CalledProcessError as exc:
                        receipt['rc'] = exc.returncode
                    except BaseException as exc:
                        receipt['rc'] = None
                        receipt['exception'] = repr(exc)
                        receipt['remote_completion'] = 'unconfirmed; remote mpirun timeout is 190 seconds'
                        raise
                    finally:
                        receipt['end_unix'] = time.time()
                        (a.out / (name + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
                receipt['end_unix'] = time.time()
                for host in a.hosts:
                    with (a.out / f'{name}.{host}.after.json').open('w') as f:
                        ssh(host, shlex.join(['python3', ROOT + '/snapshot.py']), stdout=f)
                (a.out / (name + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
                if receipt['rc'] != 0:
                    raise RuntimeError(f'{name} failed, inspect {a.out}')
                receipts.append(receipt)
                print(name, 'PASS', flush=True)
    (a.out / 'receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')


if __name__ == '__main__':
    main()
