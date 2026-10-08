#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze the real base-plus-additive checkout before checksum source sync."""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

BASE = '23de091bf5c73450a8df8255eefe7b895dff1627'


def record(path, cap=256 << 20):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > cap:
        raise ValueError('unbounded/nonregular identity file')
    result = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while block := stream.read(1 << 20):
            size += len(block)
            result.update(block)
    if size != path.stat().st_size:
        raise ValueError('identity changed during scan')
    return {'bytes': size, 'sha256': result.hexdigest()}


def admitted_receipts(pairs):
    """Check externally fixed prior-gate identities before parsing any receipt."""
    raw, identities = {}, {}
    for name, (path, expected_sha) in pairs.items():
        path = pathlib.Path(path)
        if not re.fullmatch(r'[0-9a-f]{64}', expected_sha) or not path.is_file() or path.is_symlink():
            raise ValueError('invalid external receipt admission: ' + name)
        with path.open('rb') as stream:
            data = stream.read((8 << 20) + 1)
        actual = hashlib.sha256(data).hexdigest()
        if len(data) > (8 << 20) or actual != expected_sha:
            raise ValueError('receipt differs from prior gate: ' + name)
        raw[name] = data
        identities[name] = {'bytes': len(data), 'sha256': actual}
    decoded = {name: json.loads(data) for name, data in raw.items()}
    if any(not isinstance(value, dict) for value in decoded.values()):
        raise ValueError('receipt must be a bounded object')
    return decoded, identities


def main():
    if len(sys.argv) != 3:
        raise ValueError('CHECKOUT NEW_EXTERNAL_MANIFEST')
    root, output = map(pathlib.Path, sys.argv[1:])
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != BASE:
        raise ValueError('wrong isolated base')
    raw = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--cached',
                                   '--others', '--exclude-standard'])
    paths = sorted(set(p.decode() for p in raw.split(b'\0') if p))
    if len(paths) > 5000:
        raise ValueError('unbounded checkout')
    files = {}
    for path in paths:
        if pathlib.PurePosixPath(path).is_absolute() or '..' in pathlib.PurePosixPath(path).parts:
            raise ValueError('outside checkout')
        files[path] = record(root / path, 32 << 20)
    os.umask(0o077)
    with output.open('x') as stream:
        json.dump({'base': BASE, 'files': files}, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
