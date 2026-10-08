#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only node evidence; run through SSH and keep output outside Git."""
import json
from pathlib import Path
import subprocess
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {'error': str(exc)}


def command(args):
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'rc': p.returncode, 'text': p.stdout}


result = {'time_unix': time.time(), 'commands': {}, 'devices': {}}
for args in [['hostname'], ['uname', '-a'], ['ip', '-br', 'addr'], ['rdma', 'link'],
             ['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,power.draw',
              '--format=csv'], ['g++', '--version'],
             ['dpkg-query', '-W', 'perftest', 'rdma-core', 'libibverbs1', 'libucx0'],
             ['ps', '-eo', 'pid,comm,%cpu,%mem', '--sort=-%cpu']]:
    result['commands'][' '.join(args)] = command(args)
for dev in Path('/sys/class/infiniband').iterdir():
    d = {'bdf': str((dev/'device').resolve()), 'files': {}}
    for pattern in ['ports/1/counters/*', 'ports/1/hw_counters/*',
                    'device/current_link_*', 'ports/1/state', 'ports/1/rate',
                    'ports/1/phys_state']:
        for p in dev.glob(pattern):
            d['files'][str(p.relative_to(dev))] = read(p)
    d['gids'] = {}
    for p in (dev/'ports/1/gids').glob('*'):
        value = read(p)
        if value != '0000:0000:0000:0000:0000:0000:0000:0000':
            d['gids'][p.name] = {'gid': value,
              'type': read(dev/'ports/1/gid_attrs/types'/p.name),
              'netdev': read(dev/'ports/1/gid_attrs/ndevs'/p.name)}
    d['ibv_devinfo'] = command(['ibv_devinfo', '-d', dev.name])
    result['devices'][dev.name] = d
result['meminfo'] = read('/proc/meminfo')
result['vmstat'] = read('/proc/vmstat')
result['finished_unix'] = time.time()
print(json.dumps(result, indent=2))
