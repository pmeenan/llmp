#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify measured executable identities and retain a read-only process audit."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path('/home/pmeenan/.local/share/llmp/interconnect')
pins = json.loads(Path(__file__).with_name('pins.json').read_text())
files = {root/'nccl-runtime/lib/libnccl.so.2.30.7': pins['nccl']['runtime_sha256']}
files.update({root/'nccl-tests-runtime'/name: digest
              for name, digest in pins['nccl_tests']['sha256'].items()})
files.update({Path('/usr/bin')/name: digest for name, digest in pins['perftest_sha256'].items()})
result = {'time_unix': time.time(), 'files': {}, 'source': {}, 'ldd': {}, 'processes': []}
ok = True
for path, expected in files.items():
    with path.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    result['files'][str(path)] = {'expected': expected, 'actual': actual, 'matches': actual == expected}
    ok &= actual == expected
for name, pin in [('nccl', pins['nccl']), ('nccl-tests', pins['nccl_tests'])]:
    if (root/name/'.git').exists():
        commit = subprocess.check_output(['git', '-C', str(root/name), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(root/name), 'status', '--porcelain'], text=True)
        result['source'][name] = {'commit': commit, 'status': status}
        ok &= commit == pin['commit'] and not status
for name in pins['nccl_tests']['sha256']:
    p = subprocess.run(['ldd', str(root/'nccl-tests-runtime'/name)], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result['ldd'][name] = p.stdout
    ok &= p.returncode == 0 and 'not found' not in p.stdout
    resolved = re.search(r'libnccl\.so\.2\s+=>\s+(\S+)', p.stdout)
    ok &= bool(resolved) and Path(resolved.group(1)).resolve() == (root/'nccl-runtime/lib/libnccl.so.2.30.7').resolve()
for path in Path('/proc').glob('[0-9]*/comm'):
    try:
        comm = path.read_text().strip()
        if comm in ['ib_write_bw', 'ib_read_bw', 'ib_send_lat', 'all_reduce_perf',
                    'all_gather_perf', 'sendrecv_perf', 'orted', 'mpirun.openmpi']:
            result['processes'].append({'pid': int(path.parent.name), 'comm': comm})
    except OSError:
        pass
result['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH')
result['valid'] = bool(ok)
print(json.dumps(result, indent=2))
raise SystemExit(0 if ok else 1)
