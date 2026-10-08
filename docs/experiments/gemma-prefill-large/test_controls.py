#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiny real parser/freeze controls; no model inference or quality gate."""
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Controls(unittest.TestCase):
    def test_actual_byte_identity_and_ties(self):
        module = load('compare')
        module.VOCAB = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head'
            path.write_bytes(struct.pack('<4f', 0., 0., 0., 0.))
            positive, a = module.head(path)
            path.write_bytes(struct.pack('<4f', -0., 0., 0., 0.))
            negative, b = module.head(path)
            self.assertNotEqual(a, b)
            result = module.compare(positive, negative)
            self.assertEqual(result['max_raw_delta'], 0)
            self.assertEqual(result['argmax'], 0)
            self.assertEqual(result['full_softmax_tv'], 0)

    def test_nonfinite_and_truncated_head_refused(self):
        for name in ['compare', 'freeze']:
            module = load(name)
            module.VOCAB = 4
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'head'
                for values in [(0., float('nan'), 0., 0.), (0., float('inf'), 0., 0.)]:
                    path.write_bytes(struct.pack('<4f', *values))
                    with self.assertRaises(AssertionError):
                        module.head(path)
                path.write_bytes(b'bad')
                with self.assertRaises(AssertionError):
                    module.head(path)

    def test_freeze_real_main_repeat_and_no_overwrite(self):
        module = load('freeze')
        module.VOCAB = 4
        module.ID_BYTES = 4
        module.PROFILES['31'] = (256, 1280, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = root / 'ids'
            ids.write_bytes(struct.pack('<i', 2))
            module.ID_SHA = hashlib.sha256(ids.read_bytes()).hexdigest()
            source = root / 'source.log'
            source.write_text('measured source identity\n')
            for arm in ['both256-first', 'both256-repeat']:
                path = root / arm
                path.mkdir()
                for head in ['prefill.f32', 'final.f32']:
                    (path / head).write_bytes(struct.pack('<4f', 0., 1., 2., 3.))
                (path / 'initialized-state.bin').write_bytes(b'bits')
                (path / 'state-layout.txt').write_text('gemma31-f16-kv-scalar-device-v1:16384:256:16384:1280\n')
            old = sys.argv
            sys.argv = ['freeze.py', str(root), str(ids), str(source), 'both', '31']
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                frozen = (root / 'both256-own-repeat-frozen.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    module.main()
                self.assertEqual(frozen, (root / 'both256-own-repeat-frozen.json').read_bytes())
                (root / 'both256-repeat' / 'initialized-state.bin').write_bytes(b'diff')
                with self.assertRaises(AssertionError):
                    module.main()
            finally:
                sys.argv = old

    def test_31_parser_and_count_policy_refusals(self):
        module = load('compare')
        prefixes = ''.join(f'PREFILL_TIMED_PREFIX appended={token} past={8193+i}\n'
                           for i, token in enumerate([236761, 108, 236913]))
        line = ('PREFILL_NATIVE prefill_seconds=1 prefill_rows=8192 prefill_chunks=32 '
                'intermediate_heads=31 decode_seconds=1 decode_chunks=32 timed_start=8195 '
                'completed=8227 context=16384 chunk=256 masks=device shared_vecq=0 '
                'row_products=0 norm_fused=0 rope_store=0 norm_rope=120 norm_add=120 '
                'gemma_route=0 gemma_reduce=0 prefill_norm_rope=120 prefill_norm_add=120 '
                'prefill_gemma_route=0 prefill_gemma_reduce=0\n')
        tokens = ''.join(f'PREFILL_TOKEN step={i} argmax=2 forced=2\n' for i in range(32))
        good = prefixes + line + tokens
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log'
            path.write_text(good)
            self.assertEqual(len(module.runs(path, 'both', '31')), 1)
            for bad in [good.replace('norm_rope=120', 'norm_rope=119'),
                        good.replace('past=8193', 'past=8192'),
                        good.replace('decode_chunks=32', 'decode_chunks=31'),
                        good.replace('intermediate_heads=31', 'intermediate_heads=30'),
                        good.replace('rope_store=0', 'rope_store=1'),
                        prefixes + line + tokens.rsplit('\n', 2)[0] + '\n']:
                path.write_text(bad)
                with self.assertRaises(AssertionError):
                    module.runs(path, 'both', '31')


if __name__ == '__main__':
    unittest.main()
