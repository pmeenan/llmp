# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Supervised C1 R/N/N/R, pinned current TensorFold or same-format llama.cpp."""
import argparse, hashlib, json, math, os, re, signal, struct, subprocess, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('family', choices=['tensorfold', 'gguf'])
parser.add_argument('--tree', type=Path, required=True)
parser.add_argument('--source-manifest', type=Path, required=True)
parser.add_argument('--native-build', type=Path, required=True)
parser.add_argument('--stock-build', type=Path)
parser.add_argument('--checkpoint-auth', type=Path)
parser.add_argument('--tf-preparation-receipts', type=Path)
parser.add_argument('--checkpoint-auth-sha256', default='e2ff99b2ea1405ca2c7162e86855eb885e68edd7fd9f4fe92abe8511214b4b1c')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--native-drafter', type=Path, help='explicit prepared TensorFold-comparison MTP artifact')
parser.add_argument('--native-spec-only', action='store_true', help='run native speculative arms only; retain public TF control outputs')
args = parser.parse_args()
if args.family != 'tensorfold' and (args.native_drafter or args.native_spec_only):
    parser.error('native drafter/spec-only controls require the TensorFold comparison')
P = Path(__file__).parent
T = args.tree.resolve()
H = Path.home() / '.local/share/jitllm'
R = args.out.resolve()
M = args.source_manifest.resolve()
Q = args.native_build.resolve()
REF = H / 'llama-reference-v060'
F = H / 'references/tensorfold/ed78d6fc'
INPUT = H / 'references/qwen-device-masks/inputs'
MODEL = H / 'models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6'
TF_IMAGE = 'sha256:c8dc97d6dab8995704b6c151715f11f84419775a193c96a5ed3399407ad41c20'
LLAMA_IMAGE = 'ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db'
B = T / 'build/spark-native/benchmarks/jitllm_qwen38_spec'
E = dict(os.environ, LD_LIBRARY_PATH=f'{T}/build/spark-native/lib/jitllm:{T}/build/spark-native/cublas')
os.umask(63)

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(n, v):
    with (R / n).open('x') as f:
        json.dump(v, f, indent=2)
        f.write('\n')

def guard(n):
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    mem = next((int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    save(n + '-guard.json', {'gpu_processes': gpu, 'MemAvailable_bytes': mem})
    if gpu or mem < 90 << 30:
        raise RuntimeError('reference host guard')

def source_check():
    if qualified['source_manifest_sha256'] != sha(M):
        raise RuntimeError('native source/build qualification differs')
    for name, h in json.loads(M.read_text())['files'].items():
        if sha(T / name) != h:
            raise RuntimeError('source changed ' + name)
    for name, h in qualified['files'].items():
        if sha(T / name) != h:
            raise RuntimeError('qualified binary changed ' + name)

def interrupted(s, f):
    raise RuntimeError('reference interrupted ' + str(s))
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

def run(name, cmd, timeout=140, container=None):
    proc = None
    started = time.monotonic()
    guard(name + '-before')
    try:
        with (R / (name + '.log')).open('xb') as log:
            proc = subprocess.Popen(cmd, cwd=T, env=E, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            proc.wait(timeout=timeout)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        if container:
            inspect = subprocess.run(['docker', 'inspect', container], capture_output=True, text=True)
            save(name + '-inspect.json', {'returncode': inspect.returncode, 'stdout': inspect.stdout, 'stderr': inspect.stderr})
            for action in [['stop', '-t', '5'], ['kill'], ['rm', '-f']]:
                subprocess.run(['docker', *action, container], capture_output=True, text=True)
            left = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + container + '$'], text=True).strip()
            save(name + '-retired.json', {'container_absent': not left})
            if left:
                raise RuntimeError('owned reference container remains')
        save(name + '-process.json', {'args': cmd, 'returncode': proc.returncode if proc else None, 'elapsed_seconds': time.monotonic() - started})
        guard(name + '-after')
    if proc.returncode:
        raise RuntimeError('reference application failed ' + name)
    return (R / (name + '.log')).read_text()

def payload(path, expected=None):
    data = path.read_bytes()
    if len(data) != 32 * 248320 * 4 or not all((math.isfinite(v) for v, in struct.iter_unpack('<f', data))):
        raise RuntimeError('incomplete/nonfinite fullheads ' + str(path))
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise RuntimeError('fullhead digest mismatch')
    return digest

def tokens(ids):
    if len(ids) != 32 or any((type(t) != int or not 0 <= t < 248320 for t in ids)):
        raise RuntimeError('32 valid choices absent')

def docker_base(name, image):
    if subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'], text=True).strip():
        raise RuntimeError('owned container exists')
    return ['docker', 'run', '--name', name, '--network', 'none', '--device', 'nvidia.com/gpu=all', '--ipc=host', '--mount', f'type=bind,src={R},dst=/out', '--mount', f'type=bind,src={INPUT},dst=/inputs,readonly']
R.mkdir(exist_ok=False)
qualified = json.loads(Q.read_text())
source_check()
guard('entry')
bound = {str(Path(__file__).resolve()): sha(Path(__file__)), str(M): sha(M), str(Q): sha(Q)}
for n, h in {'prompt.json': '204185510a2c6f05f7b9672a96e66de5d816d188832cde728b7d64952423c066', 'standing-literal-prompts.json': '6d3b3aded76477a6cb1c40a64c941c0cab73411b059b3cf548f6a9293ff15809', 'native-histories.json': '11250b3551f0492dd0a6b4f59f09ab6f438a6eb7474e9e6b31befb8c250c437b'}.items():
    if sha(INPUT / n) != h:
        raise RuntimeError('standing input mismatch ' + n)
    bound[str(INPUT / n)] = h
prompt = json.loads((INPUT / 'prompt.json').read_text())['decode'][0]['prompt_token_ids']
histories = json.loads((INPUT / 'native-histories.json').read_text())
if len(prompt) != 1536 or hashlib.sha256(struct.pack('<1536i', *prompt)).hexdigest() != '35ece15522faff9e2895774eb9146b83a7ad623de08b484ac7e6a8feeb29c35d':
    raise RuntimeError('literal IDs differ')
(R / 'prompt.txt').write_text('\n'.join(map(str, prompt)) + '\n')
bound[str(R / 'prompt.txt')] = sha(R / 'prompt.txt')
ldd = subprocess.check_output(['ldd', str(B)], env=E, text=True)
libs = {}
for line in ldd.splitlines():
    fields = line.split()
    if fields and fields[0] in ['libcublas.so.13', 'libcublasLt.so.13']:
        path = Path(fields[2]).resolve(strict=True)
        libs[fields[0]] = {'first_resolved_path': fields[2], 'path': str(path), 'sha256': sha(path)}
if libs != qualified['native_cuBLAS']:
    raise RuntimeError('actual qualified native cuBLAS changed')
for v in libs.values():
    bound[v['path']] = v['sha256']
paths = {'native': H / 'm3-artifacts/c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93', 'draft': H / 'm3-artifacts/8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40', 'gguf': H / 'qgguf-artifacts/5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78'}
if args.native_drafter:
    paths['draft'] = args.native_drafter.resolve()
for a in paths.values():
    if sha(a / 'manifest.json') != a.name or not (a / 'data').is_dir():
        raise RuntimeError('approved prepared artifact absent')
    for n in ['manifest.json', 'index.json']:
        bound[str(a / n)] = sha(a / n)
if bound[str(paths['gguf'] / 'index.json')] != 'd6c69d701ead0cbadc3a64f1f3c354a70ecac7179f61dbac1e1163fca0cd3550':
    raise RuntimeError('GGUF artifact index differs')
tokenizer = MODEL / 'tokenizer.json'
if sha(tokenizer) != '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3':
    raise RuntimeError('tokenizer differs')
bound[str(tokenizer)] = sha(tokenizer)
if args.family == 'tensorfold':
    inventory = F / 'tf-source.json'
    if sha(inventory) != 'f12e61f93dda091508445182bf06a36eb745028f13bebe2c2d834953e556bad3':
        raise RuntimeError('TF task pin differs')
    manifest = json.loads(inventory.read_text())
    if manifest['commit'] != 'ed78d6fc204d89d90b045bf033d6551e7714f3a1':
        raise RuntimeError('TF source commit differs')
    bound[str(inventory)] = sha(inventory)
    for n, h in manifest['files'].items():
        if sha(F / 'source' / n) != h:
            raise RuntimeError('TF source payload differs')
        bound[str(F / 'source' / n)] = h
    expected_receipts = {'authenticate': '16b626f7fa99d40e977fc757ac8c85b8c3ad6265b4dfaa0c97dd37f1b2699192', 'core': '8fb3027e9d47a5fbba040d77da059a6598485d5082b72a11615d09d7f7d59d20', 'gdn': 'db60170e7600f9f84ad76364b235e66a13694fcf699252f0567c52d25cd20e87', 'nvfp4': '6ba89dd30d76954b780b40b7bc47abe5bcaefc1c2521cc7c06840c3d90357266'}
    if args.tf_preparation_receipts:
        expected_receipts = json.loads(args.tf_preparation_receipts.read_text())
        if set(expected_receipts) != {'authenticate', 'core', 'gdn', 'nvfp4'}:
            raise RuntimeError('preparation descriptor must bind all four stages')
        bound[str(args.tf_preparation_receipts.resolve())] = sha(args.tf_preparation_receipts)
    for stage, expected in expected_receipts.items():
        receipt = F / ('prepared-' + stage + '.json')
        if sha(receipt) != expected:
            raise RuntimeError('frozen TF preparation receipt differs')
        v = json.loads(receipt.read_text())
        bound[str(receipt)] = sha(receipt)
        if v['source_commit'] != 'ed78d6fc204d89d90b045bf033d6551e7714f3a1' or v['source_manifest_sha256'] != sha(inventory):
            raise RuntimeError('TF prepared identity differs')
        for value in v.get('extensions', {}).values():
            path = H / 'tensorfold-0.6.2-nvfp4-home' / Path(value['path']).relative_to('/tfhome')
            if sha(path) != value['sha256']:
                raise RuntimeError('prepared extension changed')
            bound[str(path)] = value['sha256']
    harness = T / 'docs/experiments/qwen-device-masks/tensorfold_reference.py'
    wrapper = P / 'tf_reference_wrapper.py'
    bound[str(harness)] = sha(harness)
    bound[str(wrapper)] = sha(wrapper)
    checkpoint = next((m for m in json.loads((T / 'docs/experiments/fast-swap/pins.json').read_text())['models'] if m['id'] == 'qwen3.8-flash-next-nvfp4'))
    if checkpoint['revision'] != '925d7be6c14c6c9442ef83e8f05b5a3c39304f69':
        raise RuntimeError('checkpoint source pin differs')
    auth = args.checkpoint_auth.resolve() if args.checkpoint_auth else H / 'references/qwen-device-masks/tf-checkpoint-auth.json'
    if sha(auth) != args.checkpoint_auth_sha256:
        raise RuntimeError('retained full checkpoint authentication differs')
    retained = json.loads(auth.read_text())
    bound[str(auth)] = sha(auth)

    def checkpoint_stat_check():
        for f in checkpoint['files']:
            path = MODEL / f['path']
            proved = retained['files'][str(path)]
            stat = path.stat()
            if proved['sha256'] != f['sha256'] or stat.st_size != f['bytes'] or proved['stat'] != {n: getattr(stat, 'st_' + n) for n in ['dev', 'ino', 'size', 'mtime_ns', 'ctime_ns']}:
                raise RuntimeError('immutable authenticated checkpoint changed ' + f['path'])
    checkpoint_stat_check()
else:
    if args.stock_build is None:
        raise RuntimeError('--stock-build is required for GGUF')
    stock = json.loads(args.stock_build.read_text())
    bound[str(args.stock_build.resolve())] = sha(args.stock_build)
    bound[str(P / 'stock_reference_wrapper.sh')] = sha(P / 'stock_reference_wrapper.sh')
    for n, h in stock['files'].items():
        if sha(Path(n)) != h:
            raise RuntimeError('qualified stock helper/link closure differs')
        bound[n] = h
    shards = H / 'reference-models/Q/UD-IQ3_XXS'
    shard_ids = [('00001', 10946624, '268f81fdedf3149a538f252308927a4d5d1f6e062c178568a51e3b519744f8a8'), ('00002', 49567921344, 'cfe600b236b88c7fad1613a5ca5e83b9f2beb63cbd44c32b2be50a44747c695f'), ('00003', 32382955968, 'f1912ba34c79427d2295a58dcb2b732b5931af5bef7a373c60557a57d9ee7250')]
    for part, size, h in shard_ids:
        path = shards / ('Qwen3.8-Flash-Next-UD-IQ3_XXS-' + part + '-of-00003.gguf')
        if path.stat().st_size != size or sha(path) != h:
            raise RuntimeError('same-format raw GGUF differs')
        bound[str(path)] = h
save('binding.json', {'family': args.family, 'source_manifest_sha256': sha(M), 'qualified_build_sha256': sha(Q), 'files': bound, 'native_ldd': ldd, 'native_cuBLAS': libs, 'TF_image': TF_IMAGE, 'llama_image': LLAMA_IMAGE})
records = {}
own_oracle = {}
tf_libs = None
stock_linkage = None
for arm in ['reference1', 'native1', 'native2', 'reference2']:
    if arm.startswith('native'):
        modes = (['spec-lean'] if args.native_spec_only else ['plain', 'spec-lean']) if args.family == 'tensorfold' else ['plain']
        rows = {}
        for mode in modes:
            name = arm + '-' + mode
            out = R / name
            cmd = [str(B), '--qwen38-artifact', str(paths['native' if args.family == 'tensorfold' else 'gguf']), '--tokenizer', str(tokenizer), '--prompts', str(INPUT / 'prompt.json'), '--only', 'mask-short-1536', '--prompt-token-ids', 'on', '--context', '2048', '--prefill-chunk', '512', '--check', 'masks-lean' if mode == 'spec-lean' else 'masks-target', '--tokens', '32', '--draft', '3', '--draft-vocab', '65536', '--adaptive-depth', 'off', '--window', '0', '--graphs', 'on', '--device-masks', 'on', '--out', str(out)]
            if mode == 'spec-lean':
                cmd += ['--drafter', str(paths['draft'])]
            text = run(name, cmd, 100)
            if 'retirement failed:' in text or 'DONE ' + str(out) not in text:
                raise RuntimeError('native positive retirement absent')
            result = json.loads((out / 'spec.json').read_text())
            row = result['results'][0]
            if result['problems'] or result['adaptive_depth'] or result['window'] or (len(result['results']) != 1) or (row['mode'] != mode) or (row['generated'] != 32) or (row['prompt_tokens'] != 1536) or (not row['device_masks']) or row['host_mask_bytes'] or (row['target_device_masks'] <= 0) or (row['heads_teacher_forced'] != (mode == 'spec-lean')) or any((row[k] <= 0 for k in ['warm_plans', 'warm_graphs', 'warm_plan_bytes', 'target_state_bytes'])):
                raise RuntimeError('native actual mask/work/policy differs')
            if any(not math.isfinite(row[k]) or row[k]<=0 for k in ['seconds','decode_seconds']):
                raise RuntimeError('native paid metrics invalid')
            if row['prompt_ids_sha256'] != '35ece15522faff9e2895774eb9146b83a7ad623de08b484ac7e6a8feeb29c35d':
                raise RuntimeError('native conditioning differs')
            if mode == 'spec-lean' and (result['draft_head_rows'] != 47172 or row['draft_device_masks'] <= 0 or row['draft_state_bytes'] <= 0):
                raise RuntimeError('native lean MTP did not execute')
            if mode == 'plain' and (row['draft_state_bytes'] or result['drafter_read_bytes']):
                raise RuntimeError('native plain drafter unexpectedly active')
            tokens(row['tokens'])
            payload(out / (mode + '-heads.f32'), row['heads_sha256'])
            if args.family == 'tensorfold' and row['tokens'] != histories[mode]:
                raise RuntimeError('native teacher history changed')
            comparable = {k: row[k] for k in ['tokens', 'step_trace', 'heads_sha256', 'target_state_bytes', 'target_state_sha256', 'draft_state_bytes', 'draft_state_sha256', 'target_paths', 'draft_paths']}
            if mode in own_oracle and comparable != own_oracle[mode]:
                raise RuntimeError('native exact repeat disagrees')
            own_oracle[mode] = comparable
            rows[mode] = row
        records[arm] = rows
    else:
        name = 'jitllm-qwen-mask-' + args.family + '-' + arm
        out = R / arm
        if args.family == 'tensorfold':
            cmd = docker_base(name, TF_IMAGE) + ['--mount', f'type=bind,src={F},dst=/reference,readonly', '--mount', f'type=bind,src={MODEL},dst=/model,readonly', '--mount', f'type=bind,src={H}/tensorfold-0.6.2-nvfp4-home,dst=/tfhome', '--mount', f'type=bind,src={harness},dst=/harness.py,readonly', '--mount', f'type=bind,src={wrapper},dst=/wrapper.py,readonly', '--env', 'HOME=/tfhome', '--env', 'PYTHONPATH=/reference/package', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'TENSORFOLD_PREFILL_ROWS=512', '--env', 'MAX_JOBS=4', '--env', 'TORCH_CUDA_ARCH_LIST=12.1', '--entrypoint', 'python3', TF_IMAGE, '-B', '/wrapper.py', '/harness.py', '--model', '/model', '--inputs', '/inputs/prompt.json', '--histories', '/inputs/native-histories.json', '--out', '/out/' + arm]
            text = run(arm, cmd, 160, container=name)
            if 'TENSORFOLD_REFERENCE_COMPLETE' not in text or 'TENSORFOLD_REFERENCE_WRAPPER_COMPLETE' not in text:
                raise RuntimeError('TF positive app completion absent')
            result = json.loads((out / 'reference.json').read_text())
            binding = result['binding']
            if binding['version'] != '0.6.6' or binding['context'] != 2048 or binding['cache_slots'] != 2052 or (binding['prefill_rows'] != 512) or (not binding['prompt_fp8']) or (binding['witnessed_fp8_calls'] <= 0) or (binding['depth'] != 3) or (binding['confidence'] != 0) or (binding['kv_dtype'] != 'bf16') or binding['plain_graphs'] or (not binding['spec_graphs']) or (not binding['copy_drafts']) or (binding['actual_draft_vocab'] != 79591):
                raise RuntimeError('TF actual public policy differs')
            observed = json.loads((out / 'loaded-libraries.json').read_text())
            if tf_libs is not None and observed != tf_libs:
                raise RuntimeError('TF actual library payload changed across bookends')
            tf_libs = observed
            for mode, row in result['results'].items():
                if any(not math.isfinite(value) or value<=0 for value in [row['seconds'],row['stats']['prefill_s'],row['stats']['decode_s']]):
                    raise RuntimeError('TF paid metrics invalid')
                tokens(row['tokens'])
                payload(out / (mode + '-heads.f32'), row['heads_sha256'])
                if row['teacher_history'] != histories[mode] or row['stats']['cached'] or row['stats']['drafts'] != (mode == 'spec-lean'):
                    raise RuntimeError('TF conditioning/fresh-state policy differs')
            records[arm] = result
        else:
            history = '-'
            if arm == 'reference2':
                ids = records['native1']['plain']['tokens']
                path = R / 'native-quality-history.txt'
                path.write_text('\n'.join(map(str, ids)) + '\n')
                bound[str(path)] = sha(path)
                history = '/out/' + path.name
            cmd = docker_base(name, LLAMA_IMAGE) + ['--mount', f'type=bind,src={shards},dst=/model,readonly', '--mount', f'type=bind,src={stock['helper']},dst=/helper,readonly', '--mount', f'type=bind,src={P}/stock_reference_wrapper.sh,dst=/wrapper.sh,readonly', '--entrypoint', '/bin/sh', LLAMA_IMAGE, '/wrapper.sh', '/model/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf', '/out/prompt.txt', '/out/' + arm, history]
            text = run(arm, cmd, 140, container=name)
            if 'LLAMA_MASK_REFERENCE_COMPLETE' not in text or 'STOCK_REFERENCE_WRAPPER_COMPLETE' not in text:
                raise RuntimeError('llama positive app completion absent')
            if 'load_backend: loaded CUDA backend from /app/libggml-cuda.so' not in text:
                raise RuntimeError('stock did not load the bound CUDA backend')
            if 'load_tensors: offloaded 49/49 layers to GPU' not in text or not re.search('llama_context: flash_attn\\s*= enabled', text):
                raise RuntimeError('stock actual full-offload/FA policy differs')
            linkage = text.split('STOCK_REFERENCE_LINKAGE_BEGIN\n', 1)[1].split('STOCK_REFERENCE_LINKAGE_END', 1)[0]
            actual = {}
            for line in linkage.splitlines():
                fields = line.split()
                if len(fields) == 2 and len(fields[0]) == 64 and fields[1].startswith('/'):
                    actual[fields[1]] = fields[0]
            closure = json.loads((REF / 'image-closure1/receipt.json').read_text())
            for n in ['libggml-cuda.so', 'libllama.so', 'libggml.so', 'libggml-base.so']:
                if actual.get('/app/' + n) != closure['app_files'][n]['sha256']:
                    raise RuntimeError('actual stock backend payload differs')
            if not all((any((n in path for path in actual)) for n in ['libcublas.so', 'libcublasLt.so', 'libcudart.so'])):
                raise RuntimeError('actual stock library closure incomplete')
            if stock_linkage is not None and actual != stock_linkage:
                raise RuntimeError('stock library payload changed across bookends')
            stock_linkage = actual
            save(arm + '-libraries.json', {'ldd': linkage, 'actual_payloads': actual})
            result = json.loads((out / 'reference.json').read_text())
            tokens(result['tokens'])
            tokens(result['teacher_history'])
            payload(out / 'plain-heads.f32')
            payload(out / 'teacher-heads.f32')
            if result['context'] != 2048 or result['chunk'] != 512 or result['prompt_tokens'] != 1536 or (result['generated'] != 32) or (result['full_head_rows'] != 32):
                raise RuntimeError('llama work differs')
            if arm == 'reference2' and result['teacher_history'] != records['native1']['plain']['tokens']:
                raise RuntimeError('GGUF common-history conditioning differs')
            records[arm] = result
    save(arm + '-checked.json', records[arm])
source_check()
if args.family == 'tensorfold':
    checkpoint_stat_check()
for name, h in bound.items():
    if sha(Path(name)) != h:
        raise RuntimeError('reference binding changed ' + name)
guard('exit')
save('runs.json', records)
print('QWEN_MASK_FINAL_REFERENCE_RETIRED family=' + args.family + ' arms=4')
