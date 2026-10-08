# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import argparse, hashlib, json, os, signal, subprocess, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
P = Path(__file__).parent
R = args.out.resolve()
H = Path.home() / '.local/share/llmp'
REF = H / 'llama-reference-v060'
SDK = H / 'sdk/aarch64-c09daba6ac31edee'
os.umask(63)

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(n, v):
    (R / n).write_text(json.dumps(v, indent=2) + '\n')

def guard(n):
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    mem = next((int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    save(n + '-guard.json', {'gpu_processes': gpu, 'MemAvailable_bytes': mem})
    if gpu or mem < 32 << 30:
        raise RuntimeError('stock build guard')

def interrupted(s, f):
    raise RuntimeError('stock build interrupted')
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
R.mkdir(exist_ok=False)
guard('before')
header = REF / 'headers-and-helpers.json'
receipt = REF / 'image-closure1/receipt.json'
headers = json.loads(header.read_text())
if headers['source_commit'] != 'd81235049384534c167caea52b85a694f6103d14' or sha(receipt) != '63ccae40a696fb574bedf1145a75f9576a0f7914efbd8081fc6b085babe3cf64':
    raise RuntimeError('stock source/image closure mismatch')
bound = {str(header): sha(header), str(receipt): sha(receipt), str(P / 'llama_mask_reference.cc'): sha(P / 'llama_mask_reference.cc'), str(Path(__file__)): sha(Path(__file__))}
for f, h in headers['files'].items():
    if f.startswith('headers/'):
        if sha(REF / f) != h:
            raise RuntimeError('stock header mismatch')
        bound[str(REF / f)] = h
closure = json.loads(receipt.read_text())
for n in ['libllama.so', 'libggml.so', 'libggml-base.so']:
    p = REF / 'image-closure1/lib' / n
    if sha(p) != closure['app_files'][n]['sha256']:
        raise RuntimeError('stock link payload mismatch')
    bound[str(p)] = sha(p)
out = R / 'llama_mask_reference'
cmd = [str(SDK / 'bin/clang++'), '--target=aarch64-linux-gnu', '-march=armv8-a', '--gcc-install-dir=' + str(SDK / 'gcc/aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/16'), '-std=c++23', '-O2', '-Wall', '-Wextra', '-Werror', '-static-libstdc++', '-static-libgcc']
cmd += ['-I' + str(REF / f) for f in ['headers/include', 'headers/src', 'headers/ggml/include', 'headers/ggml/src']]
cmd += [str(P / 'llama_mask_reference.cc'), '-L' + str(REF / 'image-closure1/lib'), '-Wl,-rpath,/app', '-Wl,--allow-shlib-undefined', '-lllama', '-lggml', '-lggml-base', '-o', str(out)]
proc = None
start = time.monotonic()
try:
    with (R / 'build.log').open('xb') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        proc.wait(timeout=90)
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
    save('process.json', {'args': cmd, 'returncode': proc.returncode if proc else None, 'elapsed_seconds': time.monotonic() - start})
    guard('after')
if proc.returncode:
    raise RuntimeError('stock helper compile failed')
for f, h in bound.items():
    if sha(Path(f)) != h:
        raise RuntimeError('stock source/binding changed')
bound[str(out)] = sha(out)
save('passed.json', {'source_commit': headers['source_commit'], 'files': bound, 'helper': str(out)})
print('LLAMA_MASK_REFERENCE_BUILT')
