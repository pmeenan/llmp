#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Timeout-path regression checks using mocks; never contacts either node."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host = load('interconnect_host_runner', 'run-host.py')
nccl = load('interconnect_nccl_runner', 'run-nccl.py')


class RunnerFailureTests(unittest.TestCase):
    def test_host_timeout_records_failure_even_when_ssh_needs_killing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            proc = MagicMock()
            proc.poll.return_value = None
            proc.wait.side_effect = [subprocess.TimeoutExpired('ssh', 10), 0]
            with patch.object(host.subprocess, 'Popen', return_value=proc), \
                 patch.object(host.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ssh', 100)), \
                 patch.object(host.time, 'sleep'):
                with self.assertRaises(subprocess.TimeoutExpired):
                    host.run_pair(out, 'timeout-case', 'server', 'client', '10.0.0.1',
                                  'rdma0', 'ib_write_bw', [], 18700)
            proc.terminate.assert_called_once_with()
            proc.kill.assert_called_once_with()
            receipt = json.loads((out / 'timeout-case.json').read_text())
            self.assertIn('TimeoutExpired', receipt['exception'])
            self.assertIn('unconfirmed', receipt['remote_completion'])
            self.assertNotEqual(receipt.get('client_rc'), 0)
            self.assertGreaterEqual(receipt['finished_unix'], receipt['started_unix'])

    def test_nccl_timeout_writes_nonpassing_receipt_and_stops_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'nccl'

            def fake_ssh(hostname, script, **kwargs):
                if 'mpirun.openmpi' in script:
                    raise subprocess.TimeoutExpired('ssh', 200)

            with patch.object(nccl, 'ssh', side_effect=fake_ssh) as calls, \
                 patch.object(sys, 'argv', ['run-nccl.py', '--out', str(out), '--mode', 'pilot']):
                with self.assertRaises(subprocess.TimeoutExpired):
                    nccl.main()
            receipt = json.loads((out / 'pilot-all_reduce-auto-r0.json').read_text())
            self.assertIsNone(receipt['rc'])
            self.assertIn('TimeoutExpired', receipt['exception'])
            self.assertIn('unconfirmed', receipt['remote_completion'])
            self.assertFalse((out / 'receipts.json').exists())
            invocations = [call for call in calls.call_args_list if 'mpirun.openmpi' in call.args[1]]
            self.assertEqual(len(invocations), 1)


if __name__ == '__main__':
    unittest.main()
