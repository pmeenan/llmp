# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Run separately under supervision; retain full hash proof for immutable inputs."""
import argparse, hashlib, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, required=True)
p.add_argument('--pins', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
os.umask(63)

def metadata(path):
    stat = path.stat()
    return {n: getattr(stat, 'st_' + n) for n in ['dev', 'ino', 'size', 'mtime_ns', 'ctime_ns']}
checkpoint = next((m for m in json.loads(a.pins.read_text())['models'] if m['id'] == 'qwen3.8-flash-next-nvfp4'))
if checkpoint['revision'] != '925d7be6c14c6c9442ef83e8f05b5a3c39304f69':
    raise RuntimeError('checkpoint pin differs')
files = {}
for record in checkpoint['files']:
    path = (a.model / record['path']).resolve()
    before = metadata(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != record['sha256'] or before['size'] != record['bytes'] or before != metadata(path):
        raise RuntimeError('checkpoint input changed ' + str(path))
    files[str(path)] = {'sha256': digest, 'stat': before}
for name, record in files.items():
    if metadata(Path(name)) != record['stat']:
        raise RuntimeError('checkpoint changed during authentication')
with a.out.open('x') as out:
    json.dump({'repository': checkpoint['repository'], 'revision': checkpoint['revision'], 'files': files, 'assumption': 'private standing checkpoint remains immutable; reference verifies file stats pre/post and mounts read-only'}, out, indent=2)
    out.write('\n')
print('QWEN_CHECKPOINT_AUTHENTICATED files=' + str(len(files)))
