#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded experiment cleanup inside spark-job's ten-second termination grace."""
import signal
import subprocess
import uuid

from contract import encoded


def container_name(kind):
    return 'llmp-u3-qwen-' + kind + '-' + uuid.uuid4().hex


def stop(signum, _frame):
    raise SystemExit(128 + signum)


def signals(handler):
    return {sig: signal.signal(sig, handler) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}


def restore_signals(previous):
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def retire_container(name):
    result = {'name': name, 'retired': False}
    try:
        removed = subprocess.run(['sudo', '-n', 'docker', 'rm', '-f', name],
                                 capture_output=True, text=True, timeout=4, check=False)
        result['remove_returncode'] = removed.returncode
        result['remove_stderr'] = removed.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        result['remove_error'] = f'{type(error).__name__}: {error}'
    # A failed inspect alone cannot distinguish an absent container from a broken daemon.
    # A successful exact-name listing with no ID positively confirms absence, including
    # --rm containers which disappeared before our explicit removal.
    try:
        checked = subprocess.run(['sudo', '-n', 'docker', 'ps', '-a', '--filter',
                                  f'name=^/{name}$', '--format', '{{.ID}}'],
                                 capture_output=True, text=True, timeout=1, check=False)
        result.update(check_returncode=checked.returncode, remaining_ids=checked.stdout.strip(),
                      check_stderr=checked.stderr)
        result['retired'] = checked.returncode == 0 and not checked.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        result['check_error'] = f'{type(error).__name__}: {error}'
    return result


def run_container(command, out, receipt, name):
    previous = signals(stop)
    try:
        try:
            with (out / 'container.log').open('wb') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            receipt['completed'] = True
        except BaseException as error:
            receipt['failure'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            signals(signal.SIG_IGN)
            retirement = retire_container(name)
            receipt['retirement'] = retirement
            receipt['completed'] = receipt.get('completed', False) and retirement['retired']
            (out / 'launch.json').write_bytes(encoded(receipt))
            if not retirement['retired']:
                raise RuntimeError('container retirement is unconfirmed; stop before another job')
    finally:
        restore_signals(previous)
