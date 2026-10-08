#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Small controls exercise the real analyzer with a four-entry vocabulary."""
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


class AnalysisControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.quality = self.root / 'quality'
        self.fusions = self.root / 'fusions'
        self.fusions.mkdir()
        (self.quality / 'input1').mkdir(parents=True)
        self.ids = struct.pack('<1024i', *([2]*1024))
        (self.quality / 'input1' / 'ids.i32').write_bytes(self.ids)
        self.positive = struct.pack('<4f', 0, 0, 0, 0)
        self.negative = struct.pack('<4f', -0.0, 0, 0, 0)
        self.paths = {}
        for parent, name, raw in (
            (self.quality, 'native-ordinary128', self.positive),
            (self.quality, 'reference128-first', self.negative),
            (self.fusions, 'norm-rope-first', self.positive),
            (self.fusions, 'norm-rope-repeat', self.positive),
            (self.fusions, 'norm-add-first', self.negative),
            (self.fusions, 'norm-add-repeat', self.negative),
        ):
            path = parent / name / 'logits.f32'
            path.parent.mkdir()
            path.write_bytes(raw*1024)
            self.paths[name] = path
        row_hash = hashlib.sha256(self.positive).hexdigest()
        freeze = json.dumps(dict(hashes=[[row_hash, row_hash]]*1024,
                                 argmax_ids=[0]*1024, p99_top2_margin_movement=0)).encode()
        (self.quality / 'native-noise-128-frozen.json').write_bytes(freeze)
        self.analyzer = load('fusion_control', HERE / 'analyze.py')
        self.analyzer.CALIBRATION = hashlib.sha256(freeze).hexdigest()
        self.analyzer.BASELINE_SHA = {
            'off': hashlib.sha256(self.positive*1024).hexdigest(),
            'on': hashlib.sha256(self.negative*1024).hexdigest(),
        }
        self.common = load('quality_control', HERE.parent / 'gemma-quality' / 'analyze.py')
        self.common.VOCAB = 4
        self.common.ID_SHA = hashlib.sha256(self.ids).hexdigest()

    def run_analysis(self, combined=False):
        fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
        with mock.patch.object(self.analyzer.importlib.util, 'spec_from_file_location', return_value=fake_spec), \
             mock.patch.object(self.analyzer.importlib.util, 'module_from_spec', return_value=self.common), \
             mock.patch.object(sys, 'argv', ['analyze', str(self.fusions), str(self.quality)] + (['both'] if combined else [])), \
             contextlib.redirect_stdout(io.StringIO()):
            return self.analyzer.main()

    def test_signed_zero_is_not_byte_identity_and_nll_uses_1023_targets(self):
        self.assertEqual(self.run_analysis(), 0)
        result = json.loads((self.fusions / 'comparison.json').read_text())
        self.assertEqual(result['scored_targets'], 1023)
        arm = result['summaries']['norm_add']
        self.assertEqual(arm['off']['max_raw_delta'], 0)
        self.assertEqual(arm['off']['byte_exact_rows'], 0)
        self.assertEqual(arm['on']['byte_exact_rows'], 1024)
        self.assertEqual(arm['repeat_byte_exact_rows'], 1024)
        self.assertAlmostEqual(arm['mean_nll'], math.log(4), places=14)
        self.assertAlmostEqual(arm['ppl'], 4, places=14)

    def test_combined_policy_uses_separate_complete_repeat_and_output(self):
        for name in ['both-first', 'both-repeat']:
            path = self.fusions / name / 'logits.f32'
            path.parent.mkdir()
            path.write_bytes(self.negative*1024)
        self.assertEqual(self.run_analysis(combined=True), 0)
        self.assertFalse((self.fusions / 'comparison.json').exists())
        result = json.loads((self.fusions / 'comparison-both.json').read_text())
        self.assertEqual(list(result['summaries']), ['both'])
        self.assertEqual(result['summaries']['both']['off']['byte_exact_rows'], 0)
        self.assertEqual(result['summaries']['both']['on']['byte_exact_rows'], 1024)

    def test_changed_repeat_refuses(self):
        self.paths['norm-add-repeat'].write_bytes(self.positive*1024)
        with self.assertRaisesRegex(AssertionError, 'own repeat changed'):
            self.run_analysis()
        self.assertFalse((self.fusions / 'comparison.json').exists())

    def test_changed_native_row_refuses_frozen_calibration(self):
        self.paths['native-ordinary128'].write_bytes(self.negative*1024)
        with self.assertRaises(AssertionError):
            self.run_analysis()
        self.assertFalse((self.fusions / 'comparison.json').exists())

    def test_changed_calibration_refuses(self):
        (self.quality / 'native-noise-128-frozen.json').write_text('{}')
        with self.assertRaises(AssertionError):
            self.run_analysis()
        self.assertFalse((self.fusions / 'comparison.json').exists())


if __name__ == '__main__':
    unittest.main()
