#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Apply/reverse the exact diagnostic patch in an explicitly supplied measurement tree."""
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

def identity(path):
    data = path.read_bytes()
    return {'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}

if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] in ['apply', 'reverse']
    mode,root = sys.argv[1],pathlib.Path(sys.argv[2]).resolve()
    records = json.loads((HERE/'overlay-identities.json').read_text())
    assert identity(HERE/'instrumentation.patch') == records['patch']
    before,after = ('original','instrumented') if mode == 'apply' else ('instrumented','original')
    assert all(identity(root/n)==v for n,v in records[before].items())
    args=['git','apply','--whitespace=error']
    if mode == 'reverse': args.append('--reverse')
    subprocess.run([*args,'--check',str(HERE/'instrumentation.patch')],cwd=root,check=True)
    subprocess.run([*args,str(HERE/'instrumentation.patch')],cwd=root,check=True)
    assert all(identity(root/n)==v for n,v in records[after].items())
    print(mode,'exact three-unit overlay authenticated')
