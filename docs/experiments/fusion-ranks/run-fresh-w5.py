# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Execute one registered frozen-W5 cell; preserve failures and reap every child."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import resource
import signal
import socket
import subprocess
import time

os.umask(0o077)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
p = argparse.ArgumentParser()
p.add_argument('--prereg-commit', required=True)
p.add_argument('--case', required=True, choices=[f'{w}x{r}' for w in (256, 1024, 4096) for r in (1, 4, 32, 256)])
p.add_argument('--first-screen-review-sha256')
a = p.parse_args()
if not re.fullmatch(r'[0-9a-f]{40}', a.prereg_commit):
    p.error('--prereg-commit must name the committed registration')
if a.case != '1024x1' and not re.fullmatch(r'[0-9a-f]{64}', a.first_screen_review_sha256 or ''):
    p.error('remaining cells require the independently accepted first-screen review receipt')
root = pathlib.Path('/home/pmeenan/src/llmp')
build = root / 'build/spark-native'
staged = pathlib.Path('/tmp/u6-fresh-w5')
manifest = staged / 'source-manifest.json'
binary = build / 'benchmarks/llmp_fusion_ranks'
expected_binary = '1394ae098f4159787bec6dc4c417407f36fc6ac4679e022792a83eca6e9f7637'
expected_identity = '4c2b14cf0a66eebb8f433daefff9b58b4cfc0afdc082d6ba0e5a1ec3ac200cee'
expected_manifest = '70812bd25843e0c06dcf75af729a6fcc36259e96c7c16fbdd640b70eee84ca34'
expected_build = {
    'compiler_identity.json': 'aea97f6ac4defe73b6a2415a36b7ed260788006f295cce1fba0b35f76c263060',
    'compiler_identity_receipt.json': 'd9750b7372dcacdf0ca699459bd2bcbe3374c03db6b271a1d41154e3566a50f1',
    'compiler_objects.txt': '6c11b744b3a20468ff9a0fc2f0cbe5667f7d7270e8531a0d33c3485ad2189047',
}
family = pathlib.Path.home() / '.local/share/llmp/m35-u6-fusion-ranks/fresh-w5-r1'
out = family / a.case
out.mkdir(parents=True, exist_ok=False, mode=0o700)
width, rows = a.case.split('x')
receipt = {'prereg_commit': a.prereg_commit, 'case': a.case,
           'compiled_snapshot': 'frozen W5 integrated source, benchmark compiled on Spark A',
           'source_base': 'e9965820ceba09cedfd9395eb60f0ae6e6cc3599',
           'source_patch_sha256': 'a4b5d01d67254631b708c055267eab048969f5106c685561bb51ed2bcd2c2eed',
           'implementation_identity': expected_identity, 'started_unix': time.time(),
           'completed': False, 'retired': False, 'children': []}
child = None
launching = False
pending_signal = None
cleanup_deadline = None
deadline = time.monotonic() + 540


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save():
    temporary = out / 'launch.json.tmp'
    with temporary.open('w') as target:
        target.write(json.dumps(receipt, indent=2) + '\n')
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(out / 'launch.json')


def stop(signum, frame):
    global pending_signal
    receipt['cancelled_signal'] = signum
    if launching:
        pending_signal = signum
        return
    raise SystemExit(128 + signum)


def group_members(pgid):
    # The leader is our unreaped child until cleanup finishes, so its PID/PGID
    # cannot be recycled for an unrelated process group while we signal it.
    live, zombies = [], []
    for path in pathlib.Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / 'stat').read_text()
        except FileNotFoundError:
            continue
        fields = text[text.rfind(')')+2:].split()
        if int(fields[2]) == pgid:
            require(int(fields[3]) == pgid, 'owned process group changed session')
            (zombies if fields[0] in ('Z', 'X') else live).append(int(path.name))
    return live, zombies


def leader_status():
    # WNOWAIT deliberately retains the exited leader as the ownership anchor.
    status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if status is None:
        return None
    return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status


def wait_leader():
    while True:
        code = leader_status()
        if code is not None:
            return code
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(receipt['children'][-1]['command'], 540)
        time.sleep(min(0.05, remaining))


def retire():
    global cleanup_deadline
    if child is not None:
        entry = receipt['children'][-1]
        if not entry.get('group_retired'):
            live, zombies = group_members(child.pid)
            if live:
                if cleanup_deadline is None:
                    cleanup_deadline = time.monotonic() + 25
                term_deadline = cleanup_deadline - 5
                for signum, until in ((signal.SIGTERM, term_deadline), (signal.SIGKILL, cleanup_deadline)):
                    if not live:
                        break
                    # The unreaped leader keeps this PGID owned even if it exited.
                    entry['retirement_signal'] = signal.Signals(signum).name
                    try:
                        os.killpg(child.pid, signum)
                    except ProcessLookupError:
                        pass
                    while live and time.monotonic() < until:
                        time.sleep(min(0.05, max(0, until-time.monotonic())))
                        live, zombies = group_members(child.pid)
                live, zombies = group_members(child.pid)
            entry['live_group_members'] = live
            entry['zombie_group_members'] = zombies
            require(not live, 'owned process group still has live members after cleanup budget')
            require(leader_status() is not None, 'group leader exit is unconfirmed')
            # Keep the ownership anchor until retirement becomes permanent. A
            # pending cancellation is delivered only after cleanup knows never
            # to inspect or signal this now-reusable PGID again.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                entry['retired_returncode'] = child.wait(timeout=0)
                entry['group_retired'] = True
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    receipt['retired'] = all(c.get('group_retired') and c.get('retired_returncode') is not None
                             for c in receipt['children'])
    require(receipt['retired'], 'owned process-group retirement is unconfirmed')


def run_child(label, command, stdout_name, stderr_name, env=None):
    global child, launching
    require(time.monotonic() < deadline, 'registered 540-second child budget exhausted')
    with (out / stdout_name).open('wb') as stdout, (out / stderr_name).open('wb') as stderr:
        launching = True
        try:
            child = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr,
                                     env=env, start_new_session=True)
            receipt['children'].append({'label': label, 'command': command, 'pid': child.pid, 'pgid': child.pid})
            receipt['retired'] = False
        finally:
            launching = False
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal)
        save()
        code = wait_leader()
        unexpected, _ = group_members(child.pid)
        receipt['children'][-1]['unexpected_live_descendants'] = unexpected
        receipt['children'][-1]['returncode'] = code
        retire()
        save()
    require(code == 0, f'{label} exited {code}')
    require(not unexpected, f'{label} left live descendants after its leader exited')


def source_check(label):
    require(sha(manifest) == expected_manifest, 'source manifest differs')
    files = json.loads(manifest.read_text())
    require(len(files) == 2814, 'source inventory count differs')
    mismatches = [name for name, want in files.items()
                  if not (root / name).is_file() or sha(root / name) != want]
    receipt[label] = {'files': len(files), 'mismatches': mismatches}
    require(not mismatches, 'source inventory differs')


def build_check(label):
    hashes = {name: sha(build / 'src/kernels/ggml' / name) for name in expected_build}
    receipt[label] = hashes
    require(hashes == expected_build, 'frozen compiler config/inventory/receipt differs')


def identity_check(label):
    owned = build / 'src/kernels/ggml'
    command = ['/home/pmeenan/.local/bin/mise', 'exec', '--', 'python3',
               str(root / 'tools/compiler_identity.py'), 'check', '--config', str(owned / 'compiler_identity.json'),
               '--objects', str(owned / 'compiler_objects.txt'), '--output', str(owned / 'module_digest.cc'),
               '--receipt', str(owned / 'compiler_identity_receipt.json')]
    run_child(label, command, label+'.stdout', label+'.stderr')
    require((out / (label+'.stdout')).read_text().strip() == expected_identity,
            'current compiler-owned identity differs')


def postchecks():
    errors = {}
    for label, check in (
            ('source_after', lambda: source_check('source_after')),
            ('compiler_receipts_after', lambda: build_check('compiler_receipts_after')),
            ('binary_after', binary_after)):
        try:
            check()
        except BaseException as error:
            errors[label] = repr(error)
    reason = None
    if 'cancelled_signal' in receipt:
        reason = 'unavailable: cancelled'
    elif time.monotonic() >= deadline:
        reason = 'unavailable: registered 540-second budget exhausted'
    elif not receipt['retired'] or cleanup_deadline is not None:
        reason = 'unavailable: process retirement failed or cleanup budget already engaged'
    if reason:
        receipt['identity_after_status'] = reason
        errors['identity_after'] = reason
    else:
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            # This final child is still cancellable within the original budget;
            # do not let it inherit the outer cleanup's ignored termination.
            for sig in previous_handlers:
                signal.signal(sig, stop)
            # Do not execute changed checker source merely to gather evidence.
            files = json.loads(manifest.read_text())
            require(sha(manifest) == expected_manifest and
                    sha(root / 'tools/compiler_identity.py') == files['tools/compiler_identity.py'],
                    'frozen identity checker source is unavailable')
            identity_check('identity_after')
            receipt['identity_after_status'] = 'PASS'
        except BaseException as error:
            receipt['identity_after_status'] = repr(error)
            errors['identity_after'] = repr(error)
        finally:
            for sig in previous_handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                retire()
            except BaseException as error:
                receipt['retired'] = False
                errors['identity_after_retirement'] = repr(error)
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
    if errors:
        receipt['postcheck_errors'] = errors


def binary_after():
    receipt['binary_sha256_after'] = sha(binary)
    require(receipt['binary_sha256_after'] == expected_binary, 'binary changed')


save()
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    require(socket.gethostname().split('.')[0] == 'spark-c4e2', 'registration requires Spark A')
    receipt['hostname'] = socket.gethostname()
    receipt['helper_sha256'] = sha(pathlib.Path(__file__))
    receipt['manifest_sha256'] = sha(manifest)
    if a.case != '1024x1':
        review_path = staged / 'first-screen-review.json'
        require(sha(review_path) == a.first_screen_review_sha256, 'first-screen review receipt differs')
        review = json.loads(review_path.read_text())
        first = family / '1024x1'
        require(review['verdict'] == 'PASS' and review['prereg_commit'] == a.prereg_commit and
                review['first_launch_sha256'] == sha(first / 'launch.json') and
                review['first_samples_sha256'] == sha(first / 'samples.jsonl') and
                re.fullmatch(r'[0-9a-f]{64}', review['independent_review_sha256']),
                'first-screen review does not bind this registration and raw evidence')
        require(json.loads((first / 'launch.json').read_text())['completed'], 'first screen did not complete')
        receipt['first_screen_review'] = review
        receipt['first_screen_review_sha256'] = a.first_screen_review_sha256
    source_check('source_before')
    build_check('compiler_receipts_before')
    receipt['binary_sha256_before'] = sha(binary)
    require(receipt['binary_sha256_before'] == expected_binary, 'benchmark differs')
    identity_check('identity_before')
    run_child('hardware', ['/usr/bin/nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'],
              'hardware.txt', 'hardware.stderr')
    hardware = (out / 'hardware.txt').read_text().strip().splitlines()
    require(len(hardware) == 1 and hardware[0].split(',')[0].strip() == 'NVIDIA GB10', 'GPU inventory differs')
    env = dict(os.environ, LD_LIBRARY_PATH=str(build / 'cublas'))
    run_child('benchmark', ['/home/pmeenan/.local/bin/mise', 'exec', '--', str(binary), width, rows],
              'samples.jsonl', 'stderr.txt', env)
    records = [json.loads(line) for line in (out / 'samples.jsonl').read_text().splitlines()]
    summaries = [r for r in records if 'measurement' in r]
    require(len(summaries) == 1 and records[-1].get('event') == 'complete' and
            records[-1].get('validation_only') is False, 'missing complete timing receipt')
    summary = summaries[0]
    require(summary['device'] == 'NVIDIA GB10' and summary['arch'] == 121 and
            summary['multiprocessors'] == 48 and summary['driver'] == 13000 and summary['runtime'] == 13040,
            'measured device/driver/runtime differs from registration')
    receipt['summary_sha256'] = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
except BaseException as error:
    receipt['error'] = repr(error)
    raise
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        retire()
    except BaseException as error:
        receipt['retired'] = False
        receipt['retirement_error'] = repr(error)
    postchecks()
    try:
        for path in out.iterdir():
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
    except BaseException as error:
        receipt['flush_error'] = repr(error)
    receipt['ended_unix'] = time.time()
    receipt['completed'] = (receipt['retired'] and bool(receipt['children']) and
                            all(c.get('returncode') == 0 for c in receipt['children']) and
                            'summary_sha256' in receipt and not any(key in receipt for key in
                            ('error', 'retirement_error', 'postcheck_errors', 'flush_error')))
    save()
    if not receipt['completed'] and 'error' not in receipt:
        raise SystemExit('retirement or post-run identity check failed; see launch.json')
