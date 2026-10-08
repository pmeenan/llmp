#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Light metadata, byte semantics and owned-container refusal controls."""
import json
import pathlib
import runpy
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent
capture = runpy.run_path(str(ROOT / 'capture_freeze.py'))
compare = runpy.run_path(str(ROOT / 'compare.py'))
retire = runpy.run_path(str(ROOT / 'container_retire.py'))
generator = runpy.run_path(str(ROOT / 'capture_controller.py'))
release = runpy.run_path(str(ROOT / 'release_inputs.py'))
own = runpy.run_path(str(ROOT / 'own_freeze.py'))
original = runpy.run_path(str(ROOT / 'original_freeze.py'))
snapshot = runpy.run_path(str(ROOT / 'source_snapshot.py'))


def descriptors():
    lines = []
    for layer in (28, 29):
        for role, (size, dtype, ne, nb) in capture['ROLES'].items():
            lines.append('\t'.join((str(layer), role, str(size), str(dtype), '0x10000000',
                                     ':'.join(map(str, ne)), ':'.join(map(str, nb)))))
    return ('\n'.join(lines) + '\n').encode()


class Controls(unittest.TestCase):
    def test_external_receipt_checks_precede_any_parse(self):
        with tempfile.TemporaryDirectory() as name:
            path = pathlib.Path(name) / 'receipt'
            path.write_text('{"admitted":true}')
            expected = snapshot['record'](path)
            receipts, proofs = snapshot['admitted_receipts']({'pre': (path, expected['sha256'])})
            self.assertTrue(receipts['pre']['admitted'])
            self.assertEqual(proofs['pre'], expected)
            path.write_text('not even valid JSON')
            with mock.patch.object(json, 'loads', side_effect=AssertionError('must not parse replaced receipt')):
                with self.assertRaises(ValueError):
                    snapshot['admitted_receipts']({'pre': (path, expected['sha256'])})

    def test_comparator_self_consistent_output_receipt_replacements_refused(self):
        for changed in ('own', 'original'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as name:
                root = pathlib.Path(name)
                paths = {label: root / label for label in ('pre', 'own', 'capture', 'original')}
                output = root / 'output'
                output.write_bytes(struct.pack('<f', 1.0))
                records = {label: {'output': snapshot['record'](output)} for label in paths}
                for label, path in paths.items(): path.write_text(json.dumps(records[label]))
                expected = [snapshot['record'](path)['sha256'] for path in paths.values()]
                output.write_bytes(struct.pack('<f', 2.0))
                records[changed]['output'] = snapshot['record'](output)
                paths[changed].write_text(json.dumps(records[changed]))
                if changed == 'own':
                    records['original']['native_own_sha256'] = snapshot['record'](paths['own'])['sha256']
                    paths['original'].write_text(json.dumps(records['original']))
                argv = ['compare'] + [str(path) for path in paths.values()] + [str(root)] * 5 + ['route', str(root / 'aggregate')] + expected
                with mock.patch.object(sys, 'argv', argv), mock.patch.dict(compare['main'].__globals__,
                        {'read': mock.Mock(side_effect=AssertionError('must not read output payload'))}):
                    with self.assertRaises(ValueError): compare['main']()
                self.assertFalse((root / 'aggregate').exists())

    def test_reduction_release_fabricated_routing_receipt_refused_before_links(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            capture_path, own_path = root / 'capture', root / 'own'
            capture_path.write_text(json.dumps({'schema': 'gemma26-late-moe-capture-v1',
                                               'all_phase_dispatch_unchanged': True}))
            capture_sha = snapshot['record'](capture_path)['sha256']
            own_path.write_text(json.dumps({'schema': 'gemma26-late-moe-native-own-v1',
                'kind': 'route', 'capture_receipt_sha256': capture_sha, 'official_native_acquisition': {'admitted': True}}))
            own_sha = snapshot['record'](own_path)['sha256']
            own_path.write_text(json.dumps({'schema': 'gemma26-late-moe-native-own-v1',
                'kind': 'route', 'capture_receipt_sha256': capture_sha}))
            argv = ['release', str(capture_path), str(root), str(root / 'released'),
                    'reduce', capture_sha, str(own_path), own_sha]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(release['os'], 'link',
                    side_effect=AssertionError('must not expose original weights')):
                with self.assertRaises(ValueError): release['main']()
            self.assertFalse((root / 'released').exists())

    def test_route_release_replaced_capture_refused_before_links(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            path = root / 'capture'
            path.write_text('{"admitted":true}')
            expected = snapshot['record'](path)['sha256']
            path.write_text(json.dumps({'schema': 'gemma26-late-moe-capture-v1',
                                       'all_phase_dispatch_unchanged': True, 'files': {}}))
            argv = ['release', str(path), str(root), str(root / 'released'), 'route', expected]
            with mock.patch.object(sys, 'argv', argv):
                with self.assertRaises(ValueError): release['main']()
            self.assertFalse((root / 'released').exists())

    def test_each_freezer_rejects_replaced_prior_gate_before_work(self):
        with tempfile.TemporaryDirectory() as name:
            path = pathlib.Path(name) / 'pre'
            path.write_text('{"source":"admitted"}')
            expected = snapshot['record'](path)['sha256']
            path.write_text('{"source":"replaced"}')
            cases = [(capture, ['capture'] + [str(path)] * 8 + [expected]),
                     (own, ['own', str(path), str(path), str(path), 'route'] + [str(path)] * 6 + [expected, expected]),
                     (original, ['original'] + [str(path)] * 8 + [expected] * 3)]
            for namespace, argv in cases:
                with self.subTest(freezer=argv[0]), mock.patch.object(sys, 'argv', argv):
                    with self.assertRaises(ValueError): namespace['main']()

    def test_complete_fixed_descriptor_groups(self):
        self.assertEqual(len(capture['descriptors'](descriptors())), 14)

    def test_missing_duplicate_and_wrong_pitched_id(self):
        raw = descriptors()
        first, rest = raw.split(b'\n', 1)
        for altered in (rest, first + b'\n' + raw,
                        raw.replace(b'4:512:32768:32768', b'4:32:2048:2048', 1),
                        raw.replace(b'0x10000000', b'0x0', 1)):
            with self.assertRaises(ValueError): capture['descriptors'](altered)

    def test_descriptor_capacity_before_read(self):
        with tempfile.TemporaryDirectory() as name:
            path = pathlib.Path(name) / 'large'
            path.write_bytes(b'')
            with self.assertRaises(ValueError): capture['read'](path, 1 << 30)

    def test_signed_zero_is_not_byte_identity(self):
        plus = struct.pack('<f', 0.0) * (720896 // 4)
        minus = struct.pack('<f', -0.0) + plus[4:]
        result = compare['metric'](plus, minus, False)
        self.assertEqual(result['max_absolute_float_delta'], 0)
        self.assertFalse(result['float_bytes_exact'])

    def test_nonfinite_and_truncated_output_refused(self):
        data = struct.pack('<f', 1.0) * (720896 // 4)
        for bad in (data[:-4], struct.pack('<f', float('nan')) + data[4:],
                    struct.pack('<f', float('inf')) + data[4:]):
            with self.assertRaises(ValueError): compare['metric'](bad, data, False)

    def test_id_and_float_comparisons_remain_independent(self):
        ids = struct.pack('<i', 0) * 512
        weights = struct.pack('<f', 0.125) * 512
        result = compare['metric'](ids + weights, struct.pack('<i', 1) + ids[4:] + weights, True)
        self.assertEqual(result['selected_id_mismatches'], 1)
        self.assertTrue(result['float_bytes_exact'])
        self.assertEqual(result['normalized_sum_range_candidate'], [1, 1])

    def test_controller_insertion_is_reversible_and_closed(self):
        original = 'before\nunique_anchor\nafter\n'
        changed = generator['insert'](original, 'unique_anchor', 'copy_only();')
        stripped = changed.replace(generator['BEGIN'] + 'copy_only();\n' + generator['END'], '')
        self.assertEqual(stripped, original)
        for bad in ('missing', 'unique_anchor\nunique_anchor'):
            with self.assertRaises(ValueError): generator['insert'](bad, 'unique_anchor', 'copy();')

    def test_container_wrong_owner_refused_without_removal(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            cid = 'a' * 64
            (root / 'capture.cid').write_text(cid)
            calls = []
            def docker(*args):
                calls.append(args)
                if args[0] == 'ps': return cid
                if args[0] == 'inspect': return json.dumps([{'Id': cid, 'Name': '/wrong',
                    'Config': {'Labels': {'llmp.observer': 'another-task'}}}])
                raise AssertionError('must never remove a wrong owner')
            function = retire['retire']
            original = function.__globals__['docker']
            function.__globals__['docker'] = docker
            try:
                with self.assertRaises(ValueError): function(root, 'capture')
            finally:
                function.__globals__['docker'] = original
            self.assertFalse(any(call[0] == 'rm' for call in calls))

    def test_container_correct_owner_removed_and_absence_proved(self):
        with tempfile.TemporaryDirectory() as name:
            root = pathlib.Path(name)
            cid = 'b' * 64
            (root / 'capture.cid').write_text(cid)
            active = [True]
            def docker(*args):
                if args[0] == 'ps': return cid if active[0] else ''
                if args[0] == 'inspect': return json.dumps([{'Id': cid,
                    'Name': '/llmp-gemma26-late-moe-capture',
                    'Config': {'Labels': {'llmp.observer': 'gemma26-late-moe'}}}])
                if args == ('rm', '-f', cid): active[0] = False; return ''
                raise AssertionError('unexpected Docker command')
            function = retire['retire']
            original = function.__globals__['docker']
            function.__globals__['docker'] = docker
            try:
                function(root, 'capture')
                function(root, 'capture')  # Idempotent only after the same proof.
            finally:
                function.__globals__['docker'] = original
            self.assertTrue(json.loads((root / 'capture-container-retired.json').read_text())[
                'container_absent_after_checked_docker_query'])


if __name__ == '__main__':
    unittest.main()
