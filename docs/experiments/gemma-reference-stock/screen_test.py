#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-screen tests: signed zeros, sampled alignment and frozen evidence."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import pathlib
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScreenControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.quality, self.stock = self.root / 'quality', self.root / 'stock'
        (self.quality / 'input1').mkdir(parents=True)
        self.stock.mkdir()
        self.ids = struct.pack('<1024i', *([2]*1024))
        (self.quality / 'input1' / 'ids.i32').write_bytes(self.ids)
        self.raw = struct.pack('<4f', 0, 0, 0, 0)
        self.paths = {}
        for parent, name in [(self.quality, 'native-ordinary128'),
                             (self.quality, 'reference128-first'),
                             (self.stock, 'no-routing-first'),
                             (self.stock, 'no-routing-repeat')]:
            path = parent / name / 'logits.f32'
            path.parent.mkdir()
            path.write_bytes(self.raw*1024)
            self.paths[name] = path
        row_hash = hashlib.sha256(self.raw).hexdigest()
        freeze = json.dumps(dict(hashes=[[row_hash,row_hash]]*1024,
                                 argmax_ids=[0]*1024, p99_top2_margin_movement=0)).encode()
        (self.quality / 'native-noise-128-frozen.json').write_bytes(freeze)
        self.screen = load('screen_control', HERE / 'screen.py')
        self.screen.CALIBRATION = hashlib.sha256(freeze).hexdigest()
        digest = hashlib.sha256(self.raw*1024).hexdigest()
        self.screen.BASELINE_SHA = dict(native=digest, stock=digest)
        self.common = load('quality_control', HERE.parent / 'gemma-quality' / 'analyze.py')
        self.common.VOCAB = 4
        self.common.ID_SHA = hashlib.sha256(self.ids).hexdigest()

    def candidate(self, raw):
        for name in ['no-routing-first', 'no-routing-repeat']:
            self.paths[name].write_bytes(raw)

    def run_screen(self, policy='no_routing'):
        fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
        with mock.patch.object(self.screen.importlib.util, 'spec_from_file_location', return_value=fake_spec), \
             mock.patch.object(self.screen.importlib.util, 'module_from_spec', return_value=self.common), \
             mock.patch.object(sys, 'argv', ['screen', str(self.stock), str(self.quality), policy]), \
             contextlib.redirect_stdout(io.StringIO()):
            return self.screen.main()

    def result(self):
        return json.loads((self.stock / 'no-routing-screen.json').read_text())

    def test_signed_zero_changes_bytes_without_changing_raw_or_argmax(self):
        self.candidate(struct.pack('<4f', -0.0, 0, 0, 0)*1024)
        self.assertEqual(self.run_screen(), 0)
        result = self.result()
        self.assertEqual(result['summaries']['stock'],
                         dict(byte_exact_rows=0, strict_argmax_mismatches=0, max_raw_delta=0))
        self.assertEqual(result['first_stock_byte_change'], 0)
        self.assertIsNone(result['first_stock_argmax_change'])
        self.assertEqual(result['repeat_byte_exact_rows'], 1024)
        self.assertEqual([r['row'] for r in result['explicitly_selected_distribution_rows']],
                         [0,127,128,512,1023])
        self.assertNotIn('ppl', result)
        self.assertNotIn('target', result['explicitly_selected_distribution_rows'][-1])

    def test_first_changed_row_has_independent_target_nll_alignment(self):
        rows = [self.raw]*1024
        rows[42] = struct.pack('<4f', 0, 0, 3, 0)
        self.candidate(b''.join(rows))
        self.assertEqual(self.run_screen(), 0)
        result = self.result()
        self.assertEqual(result['first_stock_byte_change'], 42)
        self.assertEqual(result['first_stock_argmax_change'], 42)
        self.assertEqual(result['summaries']['stock']['strict_argmax_mismatches'], 1)
        detail = next(r for r in result['explicitly_selected_distribution_rows'] if r['row']==42)
        self.assertEqual(detail['target'], 2)
        self.assertAlmostEqual(detail['target_nll']['candidate'], math.log(math.exp(3)+3)-3,
                               places=14)

    def test_changed_repeat_refuses(self):
        self.paths['no-routing-repeat'].write_bytes(struct.pack('<4f', -0.0,0,0,0)*1024)
        with self.assertRaisesRegex(AssertionError, 'own repeat changed'):
            self.run_screen()
        self.assertFalse((self.stock / 'no-routing-screen.json').exists())

    def test_changed_calibration_refuses(self):
        (self.quality / 'native-noise-128-frozen.json').write_text('{}')
        with self.assertRaises(AssertionError):
            self.run_screen()
        self.assertFalse((self.stock / 'no-routing-screen.json').exists())

    def test_unknown_policy_refuses_before_input_read(self):
        with self.assertRaises(AssertionError):
            self.run_screen('undeclared')


if __name__ == '__main__':
    unittest.main()
