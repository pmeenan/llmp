# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded offline controls of the registered helper's actual lifecycle functions."""
import ast
import hashlib
import json
import os
import pathlib
import signal
import sys
import tempfile
import threading
import time

helper = pathlib.Path(__file__).with_name('run-fresh-w5.py')
# Load the actual functions without running the fixed-path measurement CLI.
tree = ast.parse(helper.read_text())
functions = ast.Module(body=[node for node in tree.body if isinstance(
    node, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[])
ns = {}
exec(compile(functions, str(helper), 'exec'), ns)

with tempfile.TemporaryDirectory(prefix='u6-helper-control-') as temporary:
    root = pathlib.Path(temporary)

    def reset():
        ns.update(root=root, out=root, child=None, launching=False, pending_signal=None,
                  cleanup_deadline=None, deadline=time.monotonic()+5,
                  receipt={'children': [], 'retired': False, 'completed': False})

    reset()
    try:
        ns['run_child']('orphan-control', [sys.executable, '-c',
            'import os,time; p=os.fork(); os._exit(0) if p else time.sleep(60)'],
            'orphan.stdout', 'orphan.stderr')
        raise AssertionError('leader success hid a live descendant')
    except ValueError as error:
        assert 'live descendants' in str(error), error
    entry = ns['receipt']['children'][0]
    assert entry['unexpected_live_descendants'] and entry['returncode'] == 0
    assert entry['group_retired'] and entry['retired_returncode'] == 0
    assert not ns['group_members'](entry['pgid'])[0]
    # Permanently retired PGIDs must never be revisited or signalled.
    original_members = ns['group_members']
    ns['group_members'] = lambda pgid: (_ for _ in ()).throw(AssertionError('retired PGID reused'))
    ns['retire']()
    ns['group_members'] = original_members
    print('PASS exited leader with live descendant; owned group retired, no PGID reuse', flush=True)

    reset()
    original_term = signal.signal(signal.SIGTERM, ns['stop'])
    original_int = signal.signal(signal.SIGINT, ns['stop'])
    def cancel_twice():
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)
    sender = threading.Thread(target=cancel_twice)
    sender.start()
    try:
        ns['run_child']('cancel-control', [sys.executable, '-c', 'import time; time.sleep(60)'],
                        'cancel.stdout', 'cancel.stderr')
        raise AssertionError('cancellation was ignored')
    except SystemExit:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        ns['retire']()
        old_deadline = ns['cleanup_deadline']
        ns['retire']()
        assert ns['cleanup_deadline'] == old_deadline
        assert ns['receipt']['retired']
    finally:
        sender.join(timeout=1)
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)
    print('PASS repeated cancellation; one cleanup deadline and confirmed group retirement', flush=True)

    reset()
    original_term = signal.signal(signal.SIGTERM, ns['stop'])
    original_int = signal.signal(signal.SIGINT, ns['stop'])
    ns['child'] = ns['subprocess'].Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    child = ns['child']
    ns['receipt']['children'].append({'pid': child.pid, 'pgid': child.pid})
    ns['wait_leader']()
    real_wait = child.wait
    def reap_then_cancel(*args, **kwargs):
        result = real_wait(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)  # Exact post-waitpid/pre-marker seam.
        return result
    child.wait = reap_then_cancel
    try:
        ns['retire']()
        raise AssertionError('pending reap-seam cancellation was lost')
    except SystemExit:
        assert ns['receipt']['cancelled_signal'] == signal.SIGTERM
        assert ns['receipt']['children'][-1]['group_retired']
        ns['group_members'] = lambda pgid: (_ for _ in ()).throw(AssertionError('reaped PGID revisited'))
        real_killpg = os.killpg
        os.killpg = lambda *args: (_ for _ in ()).throw(AssertionError('reaped PGID signalled'))
        try:
            ns['retire']()
            assert ns['receipt']['retired']
        finally:
            os.killpg = real_killpg
            ns['group_members'] = original_members
    finally:
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)
    print('PASS real reap seam cancellation; permanent retirement precedes delivered signal', flush=True)

    reset()
    checker = root/'tools/compiler_identity.py'
    checker.parent.mkdir()
    checker.write_text('# offline fixture; never executed\n')
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps({'tools/compiler_identity.py': ns['sha'](checker)}))
    binary = root/'binary'
    binary.write_bytes(b'unchanged frozen fixture')
    calls = []
    ns.update(manifest=manifest, expected_manifest=ns['sha'](manifest), binary=binary,
              expected_binary=ns['sha'](binary))
    ns['receipt']['retired'] = True
    ns['source_check'] = lambda label: (_ for _ in ()).throw(ValueError('source changed'))
    ns['build_check'] = lambda label: calls.append(label)
    ns['identity_check'] = lambda label: calls.append(label)
    ns['postchecks']()
    assert ns['receipt']['binary_sha256_after'] == ns['sha'](binary)
    assert ns['receipt']['identity_after_status'] == 'PASS'
    assert calls == ['compiler_receipts_after', 'identity_after']
    assert list(ns['receipt']['postcheck_errors']) == ['source_after']
    calls.clear()
    ns['receipt']['cancelled_signal'] = signal.SIGTERM
    ns['postchecks']()
    assert ns['receipt']['identity_after_status'] == 'unavailable: cancelled'
    assert calls == ['compiler_receipts_after']
    del ns['receipt']['cancelled_signal']
    ns['deadline'] = time.monotonic()-1
    ns['postchecks']()
    assert 'budget exhausted' in ns['receipt']['identity_after_status']
    print('PASS independent postchecks; binary and compiler status survive source failure; cancellation/budget refusal explicit', flush=True)
    reset()
    ns['receipt']['retired'] = True
    def cancellable_postchecker(label):
        ns['run_child'](label, [sys.executable, '-c', 'import time; time.sleep(60)'],
                        'postchecker.stdout', 'postchecker.stderr')
    ns['identity_check'] = cancellable_postchecker
    original_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    original_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
    sender = threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM))
    sender.start()
    try:
        ns['postchecks']()
        assert ns['receipt']['cancelled_signal'] == signal.SIGTERM
        assert 'SystemExit' in ns['receipt']['identity_after_status']
        assert ns['receipt']['retired']
        assert ns['receipt']['children'][0]['group_retired']
        assert not ns['group_members'](ns['child'].pid)[0]
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    finally:
        sender.join(timeout=1)
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)
    print('PASS late postchecker cancellation; actual child retired and failure recorded', flush=True)
print('Offline helper controls PASS; no CUDA, host access or measurement', flush=True)
