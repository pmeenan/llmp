#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Mocked lifecycle failures only: no container, executable, GPU or network runs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from contract import IMAGE, MANIFEST, encoded, sha
import launch_reference
import lifecycle
import native_runner
import reference


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.umask = os.umask(0o022)

    def tearDown(self):
        os.umask(self.umask)
        self.temp.cleanup()

    def test_unique_container_names(self):
        a, b = lifecycle.container_name('reference'), lifecycle.container_name('reference')
        self.assertNotEqual(a, b)
        self.assertRegex(a, r'^llmp-u3-qwen-reference-[0-9a-f]{32}$')

    @patch('lifecycle.subprocess.run')
    def test_already_removed_container_needs_successful_absence_check(self, run):
        run.side_effect = [SimpleNamespace(returncode=1, stderr='not found'),
                           SimpleNamespace(returncode=0, stdout='', stderr='')]
        self.assertTrue(lifecycle.retire_container('owned')['retired'])
        self.assertEqual([c.kwargs['timeout'] for c in run.call_args_list], [4, 1])
        self.assertIn('name=^/owned$', run.call_args_list[1].args[0])

    @patch('lifecycle.subprocess.run')
    def test_remove_timeout_and_surviving_container_fail_closed(self, run):
        run.side_effect = [subprocess.TimeoutExpired('rm', 4),
                           SimpleNamespace(returncode=0, stdout='abc\n', stderr='')]
        result = lifecycle.retire_container('owned')
        self.assertFalse(result['retired'])
        self.assertIn('TimeoutExpired', result['remove_error'])

    @patch('lifecycle.subprocess.run')
    def test_daemon_failure_is_not_absence(self, run):
        run.side_effect = [SimpleNamespace(returncode=1, stderr='daemon unavailable'),
                           SimpleNamespace(returncode=1, stdout='', stderr='daemon unavailable')]
        self.assertFalse(lifecycle.retire_container('owned')['retired'])

    @patch('lifecycle.subprocess.run')
    def test_check_timeout_is_not_absence(self, run):
        run.side_effect = [SimpleNamespace(returncode=0, stderr=''),
                           subprocess.TimeoutExpired('ps', 1)]
        self.assertFalse(lifecycle.retire_container('owned')['retired'])

    @patch('lifecycle.retire_container', return_value={'retired': True})
    @patch('lifecycle.subprocess.run', side_effect=SystemExit(143))
    def test_cancellation_keeps_failure_and_retirement_receipt(self, run, retire):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(SystemExit):
            lifecycle.run_container(['never-executed'], self.root, {'completed': False}, 'owned')
        receipt = json.loads((self.root / 'launch.json').read_bytes())
        self.assertFalse(receipt['completed'])
        self.assertTrue(receipt['retirement']['retired'])
        self.assertIn('SystemExit', receipt['failure'])
        retire.assert_called_once_with('owned')
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    @patch('lifecycle.retire_container', return_value={'retired': False, 'check_error': 'timeout'})
    @patch('lifecycle.subprocess.run')
    def test_successful_collection_cannot_hide_failed_cleanup(self, run, retire):
        with self.assertRaises(RuntimeError):
            lifecycle.run_container(['never-executed'], self.root, {'completed': False}, 'owned')
        receipt = json.loads((self.root / 'launch.json').read_bytes())
        self.assertFalse(receipt['completed'])
        self.assertFalse(receipt['retirement']['retired'])

    def launch_argv(self, phase='short'):
        return ['launch_reference.py', '--checkpoint', str(self.root), '--inputs', str(self.root),
                '--protocol', str(self.root/'prereg.json'), '--out', str(self.root/'out'),
                '--prereg-commit', 'a'*40, '--phase', phase]

    @patch('launch_reference.subprocess.check_output')
    @patch('launch_reference.require_validation', side_effect=ValueError('stale'))
    @patch('launch_reference.frozen', return_value=({}, [], {'cases': []}, 'b'*64))
    def test_stale_validation_stops_before_any_docker_invocation(self, freeze, validate, docker):
        with patch.object(sys, 'argv', self.launch_argv()), self.assertRaises(ValueError):
            launch_reference.main()
        docker.assert_not_called()
        self.assertFalse((self.root/'out').exists())

    @patch('launch_reference.run_container')
    @patch('launch_reference.subprocess.check_output', return_value=encoded([{'Id': IMAGE, 'Architecture': 'arm64'}]))
    @patch('launch_reference.sha', return_value=MANIFEST)
    @patch('launch_reference.metadata')
    @patch('launch_reference.frozen', return_value=({}, [], {'cases': []}, 'b'*64))
    def test_existing_output_is_preserved_before_container_start(self, freeze, metadata, digest, inspect, run):
        (self.root/'llmp-archive.json').write_text('{}')
        (self.root/'out').mkdir()
        sentinel = self.root/'out'/'launch.json'
        sentinel.write_text('existing receipt')
        with patch.object(sys, 'argv', self.launch_argv('validate')), self.assertRaises(FileExistsError):
            launch_reference.main()
        run.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'existing receipt')

    @patch('reference.SourceAdapter')
    @patch('reference.metadata')
    @patch('reference.frozen', return_value=({'ids_sha256': 'a'*64, 'behavior_sha256': 'b'*64},
                                           [1, 2], {'cases': []}, 'c'*64))
    def test_failed_close_cannot_publish_validation_and_keeps_private_receipt(self, freeze, metadata, adapter):
        adapter.return_value.identity = {}
        adapter.return_value.engine.capacity_plan = {}
        adapter.return_value.validate.return_value = []
        adapter.return_value.close.side_effect = RuntimeError('close failed')
        with patch.object(sys, 'argv', self.launch_argv('validate')), self.assertRaises(RuntimeError):
            reference.main()
        out = self.root/'out'
        self.assertFalse((out/'validation.json').exists())
        self.assertFalse(json.loads((out/'receipt.json').read_bytes())['completed'])
        self.assertEqual(out.stat().st_mode & 0o777, 0o700)
        self.assertEqual((out/'receipt.json').stat().st_mode & 0o777, 0o600)

    def test_native_changed_executable_invalidates_collection_after_retirement(self):
        runtime = self.root/'runtime'
        runtime.write_bytes(b'original binary')
        artifact = self.root/'artifact'
        artifact.mkdir()
        for name in ('manifest.json', 'spec.json'):
            (artifact/name).write_bytes(b'{}')
        (self.root/'validation.json').write_bytes(b'{}')
        (self.root/'tools').mkdir()
        (self.root/'tools'/'speed_gate.py').write_bytes(b'helper receipt')
        spec = {'native_runtime_sha256': sha(runtime.read_bytes()), 'native_artifact': 'artifact',
                'native_manifest_sha256': sha(b'{}'), 'native_spec_sha256': sha(b'{}')}
        service = Mock()
        service.stop.return_value = {'signal': 'TERM', 'returncode': 0}
        service.proc.poll.return_value = 0
        factory = Mock(return_value=service)
        argv = ['native_runner.py', '--repo', str(self.root), '--runtime', str(runtime),
                '--checkpoint', str(self.root), '--store', str(self.root), '--inputs', str(self.root),
                '--protocol', str(self.root/'prereg.json'), '--validation', str(self.root/'validation.json'),
                '--out', str(self.root/'out'), '--prereg-commit', 'a'*40, '--phase', 'short']
        old_path = sys.path[:]
        try:
            with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'speed_gate': SimpleNamespace(Service=factory)}), \
                    patch('native_runner.frozen', return_value=(spec, [], {'cases': []}, 'b'*64)), \
                    patch('native_runner.metadata'), patch('native_runner.require_validation'), \
                    patch('native_runner.subprocess.run', side_effect=lambda *a, **k: runtime.write_bytes(b'changed')):
                with self.assertRaisesRegex(ValueError, 'changed during collection'):
                    native_runner.main()
        finally:
            sys.path[:] = old_path
        service.stop.assert_called_once_with(timeout=5)
        receipt = json.loads((self.root/'out'/'launch.json').read_bytes())
        self.assertFalse(receipt['completed'])
        self.assertNotEqual(receipt['runtime_sha256'], receipt['runtime_after_sha256'])
        self.assertTrue(json.loads((self.root/'out'/'retirement.json').read_bytes())['retired'])


if __name__ == '__main__':
    unittest.main()
