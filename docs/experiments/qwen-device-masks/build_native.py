# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
import argparse, hashlib, json, os, signal, subprocess, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--tree', type=Path, required=True)
parser.add_argument('--source-manifest', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
T = args.tree.resolve()
R = args.out.resolve()
S = Path.home() / '.local/share/jitllm/sdk/aarch64-c09daba6ac31edee'
M = args.source_manifest.resolve()
E = dict(os.environ, JITLLM_TEST_DATA=str(T / 'tests/unit/data'), LD_LIBRARY_PATH=f'{T}/build/spark-native/lib/jitllm:{T}/build/spark-native/cublas')
os.umask(63)

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(n, v):
    with (R / n).open('x') as f:
        json.dump(v, f, indent=2)
        f.write('\n')

def source_check():
    for f, h in json.loads(M.read_text())['files'].items():
        if sha(T / f) != h:
            raise RuntimeError('source mismatch ' + f)

def guard(n):
    p = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    m = next((int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    save(n + '-guard.json', {'gpu_processes': p, 'MemAvailable_bytes': m})
    if p or m < 32 << 30:
        raise RuntimeError('host guard failed')

def interrupted(s, f):
    raise RuntimeError('build/control interrupted ' + str(s))
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

def run(n, cmd, timeout):
    start = time.monotonic()
    p = None
    record = {'args': cmd, 'timeout_seconds': timeout}
    try:
        with (R / (n + '.log')).open('xb') as log:
            p = subprocess.Popen(cmd, cwd=T, env=E, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            p.wait(timeout=timeout)
    finally:
        if p is not None and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
        record.update(returncode=p.returncode if p else None, elapsed_seconds=time.monotonic() - start)
        save(n + '-process.json', record)
    if p.returncode:
        raise RuntimeError(n + ' failed')
R.mkdir(parents=True, exist_ok=False)
source_check()
if not (T / 'tests/unit/data').is_dir():
    raise RuntimeError('JITLLM_TEST_DATA directory absent')
guard('before')
targets = ['jitllm-runtime', 'jitllm_qwen38_spec']
run('build', [str(S / 'bin/cmake'), '--build', str(T / 'build/spark-native'), '--target', *targets, '--parallel', '4'], 480)
binaries = [T / 'build/spark-native/benchmarks/jitllm_qwen38_spec', T / 'build/spark-native/src/runtime/jitllm-runtime']
receipt = T / 'build/spark-native/jitllm-receipt.json'
rc = json.loads(receipt.read_text())
if not rc['official'] or rc['sdk'] != S.name or (not rc['cuda']):
    raise RuntimeError('unapproved SDK build receipt')
expected = {'libcublas.so.13': 'ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b', 'libcublasLt.so.13': 'ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30'}
ldd = subprocess.check_output(['ldd', str(binaries[0])], env=E, text=True)
libs = {}
for line in ldd.splitlines():
    f = line.split()
    if f and f[0] in expected:
        p = Path(f[2]).resolve(strict=True)
        h = sha(p)
        if h != expected[f[0]]:
            raise RuntimeError('unapproved actual cuBLAS payload')
        libs[f[0]] = {'first_resolved_path': f[2], 'path': str(p), 'sha256': h}
if len(libs) != 2:
    raise RuntimeError('actual native cuBLAS closure incomplete')
binding = {'source_manifest_sha256': sha(M), 'launcher_sha256': sha(Path(__file__)), 'files': {str(p.relative_to(T)): sha(p) for p in [*binaries, receipt]}, 'native_ldd': ldd, 'native_cuBLAS': libs}
save('binding.json', binding)
guard('after')
source_check()
for f, h in binding['files'].items():
    if sha(T / f) != h:
        raise RuntimeError('qualified binary/receipt changed ' + f)
for v in libs.values():
    if sha(Path(v['path'])) != v['sha256']:
        raise RuntimeError('cuBLAS changed')
save('passed.json', binding)
print('QWEN_MASK_ADOPTION_BUILD_RETIRED runtime_and_benchmark=true')
