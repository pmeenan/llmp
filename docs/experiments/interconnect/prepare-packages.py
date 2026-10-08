#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Download hashed MPI/build dependencies and extract them without installation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root', type=Path, default=Path('/home/pmeenan/.local/share/llmp/interconnect'))
a = ap.parse_args()
packages = a.root / 'packages'
packages.mkdir(parents=True, exist_ok=True)
root = a.root / 'mpi-root'
root.mkdir(exist_ok=True)
pins = json.loads(Path(__file__).with_name('pins.json').read_text())
subprocess.run(['apt-get', 'download', *[p['pin'] for p in pins['debs']]], cwd=packages, check=True)
for p in pins['debs']:
    path = packages / p['file']
    assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == p['sha256'], path
    subprocess.run(['dpkg-deb', '-x', str(path), str(root)], check=True)
