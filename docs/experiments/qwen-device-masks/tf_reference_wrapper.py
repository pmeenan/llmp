# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib, json, runpy, sys
from pathlib import Path
harness = Path(sys.argv[1])
sys.argv = sys.argv[1:]
runpy.run_path(str(harness), run_name='__main__')
out = Path(sys.argv[sys.argv.index('--out') + 1])
libs = {}
for line in Path('/proc/self/maps').read_text().splitlines():
    fields = line.split()
    if len(fields) < 6:
        continue
    path = fields[-1]
    if path.startswith('/') and '.so' in path and any((s in path for s in ['libcublas', 'libcudart', '/tfhome/'])):
        p = Path(path)
        if p.is_file() and path not in libs:
            with p.open('rb') as f:
                libs[path] = hashlib.file_digest(f, 'sha256').hexdigest()
if not any(('libcublas.so' in p for p in libs)) or not any(('libcublasLt.so' in p for p in libs)):
    raise RuntimeError('actual TF cuBLAS payload not observed')
expected = {}
for stage in ['core', 'gdn', 'nvfp4']:
    receipt = json.loads((Path('/reference') / ('prepared-' + stage + '.json')).read_text())
    if receipt['source_commit'] != 'ed78d6fc204d89d90b045bf033d6551e7714f3a1':
        raise RuntimeError('prepared extension source differs')
    for value in receipt['extensions'].values():
        expected[value['path']] = value['sha256']
for path, digest in expected.items():
    if path not in libs or libs[path] != digest:
        raise RuntimeError('actual forward did not load prepared extension ' + path)
(out / 'loaded-libraries.json').write_text(json.dumps(libs, indent=2) + '\n')
print('TENSORFOLD_REFERENCE_WRAPPER_COMPLETE', flush=True)
