#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Download pinned external benchmark data; never execute checkpoint code."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('pins', type=Path)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    pins = json.loads(a.pins.read_text())
    def fetch(model, item):
        path = a.directory / model['id'] / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size == item['bytes'] and sha(path) == item['sha256']:
                return
            raise ValueError('Existing artifact mismatch: ' + str(path))
        partial = Path(str(path)+'.partial')
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{item['path']}"
        subprocess.run(['curl','-fL','--retry','5','--continue-at','-','--output',str(partial),url], check=True)
        if partial.stat().st_size != item['bytes'] or sha(partial) != item['sha256']:
            raise ValueError('Downloaded artifact mismatch: ' + str(partial))
        partial.rename(path)
        print('Verified '+str(path), flush=True)
    for model in pins['large_models']:
        path=a.directory/model['id']/'LICENSE'
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            url=f"https://huggingface.co/{model['base_repository']}/resolve/{model['base_revision']}/LICENSE"
            subprocess.run(['curl','-fL','--retry','3','--output',str(path),url],check=True)
        if sha(path)!=model['license_sha256']:raise ValueError('License identity mismatch')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(fetch,m,f) for m in pins['large_models'] for f in m['files']]
        for task in tasks:
            task.result()


if __name__ == '__main__':
    main()
