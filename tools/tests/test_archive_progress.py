# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Local controller telemetry: loss, stale sessions, stalled I/O and viewers."""
import hashlib
import io
import json
import pathlib
import socket
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import llmp_archive as arc
import llmp_archive_progress as progress


class Source(io.BytesIO):
    def __init__(self, data, start):
        super().__init__(data)
        self.start = start


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='llmp-progress-')
        self.base = pathlib.Path(self.temp.name)
        self.plan = [{'id': 'one', 'repo': 'test/model', 'commit': 'a' * 40, 'selected_bytes': 100}]
        self.state = {'active': 'one', 'status': 'running', 'tasks': {'one': {'status': 'running'}}}
        self.addCleanup(self.temp.cleanup)
        arc.cancel.clear()

    def controller(self):
        controller = progress.Controller(self.base, self.plan, self.state)
        self.addCleanup(controller.close)
        return controller

    def viewer(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(str(self.base / 'status.sock'))
        self.addCleanup(sock.close)
        file = sock.makefile('rb')
        self.addCleanup(file.close)
        return file

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.02)
        self.fail('telemetry did not arrive')

    def test_multiple_viewers_push_without_filesystem_reads(self):
        controller = self.controller()
        first, second = self.viewer(), self.viewer()
        with mock.patch.object(pathlib.Path, 'read_text', side_effect=AssertionError('storage read')):
            one, two = json.loads(first.readline()), json.loads(second.readline())
            self.assertEqual(one['active'], two['active'])
            updated = dict(self.state, tasks={'one': {'status': 'complete'}}, active=None)
            controller.update(state=updated)
            one, two = json.loads(first.readline()), json.loads(second.readline())
            self.assertEqual(one['verified'], 100)
            self.assertEqual(two['verified'], 100)
            self.assertTrue(one['live'])

    def test_no_serialization_without_viewer(self):
        controller = self.controller()
        with mock.patch.object(progress, 'encode', side_effect=AssertionError('no viewer')):
            time.sleep(.15)
            # The publisher remains alive; no dropped thread exception.
            self.assertTrue(controller.thread.is_alive())

    def test_sessions_and_task_switch_reject_stale_report(self):
        controller = self.controller()
        old = controller.begin('first')
        reporter = progress.Reporter(old['LLMP_ARCHIVE_PROGRESS_SOCKET'], old['LLMP_ARCHIVE_PROGRESS_SESSION'])
        self.addCleanup(reporter.close)
        entry = {'path': 'model', 'size': 100}
        reporter.plan([entry], set())
        reporter.file(entry, 'receiving', 25)
        self.wait_for(lambda: controller.snapshot()['transfer'] is not None)
        self.wait_for(lambda: controller.snapshot()['transfer']['received'] == 25)
        controller.begin('second')
        reporter.file(entry, 'receiving', 80)
        time.sleep(.6)
        self.assertIsNone(controller.snapshot()['transfer'])
        controller.update(state=dict(self.state, active=None))
        self.assertIsNone(controller.snapshot()['active'])

    def test_snapshot_corrects_lost_messages_and_range_reset(self):
        controller = self.controller()
        env = controller.begin('download')
        reporter = progress.Reporter(env['LLMP_ARCHIVE_PROGRESS_SOCKET'], env['LLMP_ARCHIVE_PROGRESS_SESSION'])
        self.addCleanup(reporter.close)
        entry = {'path': 'model', 'size': 100}
        reporter.plan([entry], set())
        reporter.file(entry, 'checking resumed prefix', 80, 40)
        self.wait_for(lambda: controller.snapshot()['transfer'] is not None
                      and controller.snapshot()['transfer']['received'] == 80)
        reporter.file(entry, 'receiving', 0)
        self.wait_for(lambda: controller.snapshot()['transfer']['received'] == 0)
        reporter.file(entry, 'verified', 100)
        self.wait_for(lambda: controller.snapshot()['transfer']['verified_files'] == 1)
        controller.end()
        self.assertEqual(controller.snapshot()['transfer']['phase'], 'recording result')

    def test_absent_controller_and_full_receive_queue_do_not_block(self):
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.addCleanup(receiver.close)
        endpoint = str(self.base / 'unused.sock')
        receiver.bind(endpoint)
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        sender.setblocking(False)
        for _ in range(100):
            try:
                sender.sendto(b'full', endpoint)
            except BlockingIOError:
                break
        reporter = progress.Reporter(endpoint, 'session')
        self.addCleanup(reporter.close)
        entry = {'path': 'model', 'size': 100}
        start = time.monotonic()
        for n in range(1000):
            reporter.file(entry, 'receiving', n % 100)
        self.assertLess(time.monotonic() - start, 1)
        reporter.close()
        absent = progress.Reporter(str(self.base / 'missing.sock'), 'session')
        absent.file(entry, 'receiving', 20)
        absent.close()

    def test_bad_datagram_does_not_kill_publisher(self):
        controller = self.controller()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        sock.sendto(b'not JSON', str(controller.input_path))
        sock.sendto(b'[' * 1500 + b'0' + b']' * 1500, str(controller.input_path))
        sock.sendto(json.dumps({'protocol': 1, 'received': None}).encode(), str(controller.input_path))
        self.assertEqual(json.loads(self.viewer().readline())['revisions'], 1)
        self.assertTrue(controller.thread.is_alive())

    def test_reporter_thread_resource_failure_does_not_fail_transfer(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.addCleanup(sock.close)
        with mock.patch.object(progress.socket, 'socket', return_value=sock), \
                mock.patch.object(progress.threading.Thread, 'start', side_effect=RuntimeError('no threads')):
            reporter = progress.Reporter(str(self.base / 'unused.sock'), 'session')
        self.addCleanup(reporter.close)
        self.assertIsNone(reporter.thread)
        self.assertIsNone(reporter.socket)
        self.assertEqual(sock.fileno(), -1)
        data = b'abcdefgh'
        entry = {'path': 'model', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'lfs': True}
        reporter.plan([entry], set())
        reporter.phase('transferring')
        status, digest = arc.transfer(self.base, entry, lambda start: Source(data[start:], start), 'new',
                                      progress=lambda *values: reporter.file(entry, *values))
        self.assertEqual(status, 'done')
        self.assertEqual(digest, entry['sha256'])
        self.assertEqual((self.base / 'model').read_bytes(), data)

    def test_duplicate_controller_does_not_unlink_active_socket(self):
        controller = self.controller()
        with self.assertRaises(BlockingIOError):
            progress.Controller(self.base, self.plan, self.state)
        self.assertEqual(json.loads(self.viewer().readline())['revisions'], 1)
        self.assertTrue(controller.thread.is_alive())

    def test_slow_or_disconnected_viewer_does_not_block_other_viewer(self):
        controller = self.controller()
        slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        slow.connect(str(controller.output_path))
        self.addCleanup(slow.close)
        original = controller.snapshot
        def large():
            result = original()
            result['padding'] = 'x' * 55000
            return result
        controller.snapshot = large
        healthy = self.viewer()
        self.assertEqual(json.loads(healthy.readline())['revisions'], 1)
        self.assertEqual(json.loads(healthy.readline())['revisions'], 1)
        slow.close()
        self.assertEqual(json.loads(healthy.readline())['revisions'], 1)
        self.assertTrue(controller.thread.is_alive())

    def test_transfer_reports_resume_reset_and_verification(self):
        data = b'abcdefgh'
        entry = {'path': 'model', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'lfs': True}
        (self.base / ('model' + arc.PARTIAL + 'old')).write_bytes(data[:4])
        reports = []
        def report(*args):
            reports.append(args)
        status, digest = arc.transfer(self.base, entry, lambda start: Source(data, 0), 'new',
                                      writer_locked=True, progress=report)
        self.assertEqual(status, 'done')
        self.assertEqual(digest, entry['sha256'])
        self.assertIn(('checking resumed prefix', 4, 4), reports)
        self.assertIn(('receiving', 0, 0), reports)
        self.assertEqual(reports[-1], ('verified', 8, 0))
        self.assertEqual((self.base / 'model').read_bytes(), data)

    def test_telemetry_failure_cannot_fail_transfer_or_hide_hash_mismatch(self):
        data = b'abcdefgh'
        entry = {'path': 'model', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'lfs': True}
        def broken(*args):
            raise OSError('viewer unavailable')
        self.assertEqual(arc.transfer(self.base, entry, lambda start: Source(data[start:], start),
                                     'new', progress=broken)[0], 'done')
        (self.base / 'model').unlink()
        reports = []
        with self.assertRaises(arc.Mismatch):
            arc.transfer(self.base, entry, lambda start: Source(b'XXXXXXXX', 0), 'bad',
                         progress=lambda *args: reports.append(args))
        self.assertEqual(reports[-1], ('hash mismatch; discarded', 0, 0))
        self.assertFalse((self.base / 'model').exists())


if __name__ == '__main__':
    unittest.main()
