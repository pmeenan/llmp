#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Small source-identity and bounded metadata-parser controls."""
import hashlib
import pathlib
import sys
import tempfile
import unittest

import analyze
import observe_controller


def desc(role, name, op, dtype, ne):
    return f' {role}={name},{op},{dtype},ne={ne},nb=4:512:2048:2048,data=0x1000,bytes=2048,alloc=2048'


def fixture():
    lines = []
    for gid, rows in enumerate((4, 64, 65, 66, 67), 1):
        prefix = f'JITLLM_OBSERVE graph={gid}'
        lines += [f'{prefix} kind=graph nodes=100 cuda_graph=0 update=0 router_rows={rows}',
                  f'{prefix} kind=route_candidate recognized=1' +
                  desc('node', 'ffn_moe_probs-0', 'SOFT_MAX', 'f32', f'128:{rows}:1:1'),
                  f'{prefix} kind=route_selected' +
                  desc('node', 'ffn_moe_probs-0', 'SOFT_MAX', 'f32', f'128:{rows}:1:1'),
                  f'{prefix} kind=product family=MMVQ routed=1' +
                  desc('a', 'blk.0.ffn_gate_up_exps', 'NONE', 'q4_K', '2816:1408:128:1') +
                  desc('b', 'ffn_norm_2-0', 'MUL', 'f32', f'2816:1:{rows}:1') +
                  desc('ids', 'ffn_moe_topk-0', 'VIEW', 'i32', f'8:{rows}:1:1') +
                  desc('dst', 'ffn_moe_gate_up-0', 'MUL_MAT_ID', 'f32', f'1408:8:{rows}:1')]
    return lines


def complete_fixture():
    lines = []
    for gid, rows in enumerate((4, 64, 65, 66, 67), 1):
        prefix = f'JITLLM_OBSERVE graph={gid}'
        lines.append(f'{prefix} kind=graph nodes=100 cuda_graph=0 update=0 router_rows={rows}')
        for lid in range(30):
            lines += [f'{prefix} kind=route_candidate recognized=1' +
                      desc('node', f'ffn_moe_probs-{lid}', 'SOFT_MAX', 'f32', f'128:{rows}:1:1'),
                      f'{prefix} kind=route_gate structural=1 shape=1 memory=1 count=10' +
                      desc('node', f'ffn_moe_probs-{lid}', 'SOFT_MAX', 'f32', f'128:{rows}:1:1') +
                      desc('logits', f'ffn_moe_logits-{lid}', 'MUL', 'f32', f'128:{rows}:1:1') +
                      desc('weights', f'ffn_moe_weights-{lid}', 'RESHAPE', 'f32', f'1:8:{rows}:1') +
                      desc('ids', f'ffn_moe_topk-{lid}', 'VIEW', 'i32', f'8:{rows}:1:1'),
                      f'{prefix} kind=route_selected' +
                      desc('node', f'ffn_moe_probs-{lid}', 'SOFT_MAX', 'f32', f'128:{rows}:1:1'),
                      f'{prefix} kind=reduction_gate matched=0 count=0 scale=0 memory=-1' +
                      desc('node', f'ffn_moe_down_scaled-{lid}', 'MUL', 'f32', f'2816:8:{rows}:1')]
        lines.append(fixture()[3].replace('graph=1', f'graph={gid}').replace(':4:', f':{rows}:'))
    return lines


class Controls(unittest.TestCase):
    def test_original_byte_reconstruction(self):
        raw = pathlib.Path(sys.argv[1]).read_bytes()
        output = observe_controller.transform(raw)
        self.assertEqual(observe_controller.original(output.decode()).encode(), raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), observe_controller.PIN)

    def test_altered_original_refused(self):
        with self.assertRaises(ValueError):
            observe_controller.transform(b'changed upstream source')

    def test_mixed_case_quant_and_all_phases(self):
        result = analyze.observations(fixture())
        self.assertEqual(result['host_graph_calls'], 5)
        self.assertEqual(len(result['product_decisions']), 5)
        self.assertEqual(result['product_decisions'][0]['weight_type'], 'q4_K')

    def test_unknown_rows_refused(self):
        lines = fixture()
        lines[0] = lines[0].replace('router_rows=4', 'router_rows=12')
        with self.assertRaises(ValueError):
            analyze.observations(lines)

    def test_duplicate_and_orphan_graph_refused(self):
        with self.assertRaises(ValueError):
            analyze.observations([fixture()[0], fixture()[0]])
        with self.assertRaises(ValueError):
            analyze.observations(fixture()[1:])

    def test_wrong_route_or_product_rows_refused(self):
        for index in (1, 3):
            lines = fixture()
            lines[index] = lines[index].replace('128:4:1:1', '128:3:1:1').replace('1408:8:4:1', '1408:8:3:1')
            with self.assertRaises(ValueError):
                analyze.observations(lines)

    def test_truncated_complete_file_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'fixture'
            path.write_bytes(b'123')
            with self.assertRaises(ValueError):
                analyze.digest(path, 4)

    def test_all_thirty_layers_required(self):
        lines = complete_fixture()
        self.assertEqual(analyze.observations(lines, True)['host_graph_calls'], 5)
        with self.assertRaises(ValueError):
            analyze.observations(fixture(), True)
        for kind in ('route_candidate', 'reduction_gate'):
            altered = [l for l in lines if not ('graph=1 ' in l and 'kind=' + kind in l and '-29,' in l)]
            with self.assertRaises(ValueError):
                analyze.observations(altered, True)
        for kind in ('route_gate', 'route_selected'):
            altered = [l for l in lines if not ('graph=1 ' in l and 'kind=' + kind in l and '-29,' in l)]
            with self.assertRaises(ValueError):
                analyze.observations(altered, True)

    def test_each_evaluated_call_requires_complete_layers(self):
        lines = complete_fixture()
        extra = [l.replace('graph=1 ', 'graph=6 ') for l in lines if 'graph=1 ' in l]
        self.assertEqual(analyze.observations(lines + extra, True)['host_graph_calls'], 6)
        for kind in ('route_candidate', 'reduction_gate'):
            partial = [l for l in extra if not ('kind=' + kind in l and '-29,' in l)]
            with self.assertRaises(ValueError):
                analyze.observations(lines + partial, True)

    def test_unknown_boolean_family_and_matched_shape_refused(self):
        for old, new in [('recognized=1', 'recognized=2'), ('family=MMVQ', 'family=UNKNOWN'),
                         ('routed=1', 'routed=2'), ('cuda_graph=0', 'cuda_graph=3')]:
            with self.assertRaises(ValueError):
                analyze.observations([l.replace(old, new) for l in fixture()])
        lines = complete_fixture()
        index = next(i for i, l in enumerate(lines) if 'kind=reduction_gate' in l)
        lines[index] = lines[index].replace('matched=0 count=0 scale=0 memory=-1',
                                           'matched=1 count=17 scale=1 memory=1')
        for role, ne in [('experts', '2816:8:3:1'), ('scale', '1:8:4:1'),
                         ('weights', '1:8:4:1'), ('dst', '2816:4:1:1')]:
            lines[index] += desc(role, role + '-0', 'MUL', 'f32', ne)
        with self.assertRaises(ValueError):
            analyze.observations(lines, True)

    def test_scalar_scale_field_cannot_capture_tensor_role(self):
        line = ('JITLLM_OBSERVE graph=1 kind=reduction_gate matched=1 count=17 scale=1 memory=1' +
                desc('node', 'ffn_moe_down_scaled-0', 'MUL', 'f32', '2816:8:64:1') +
                desc('experts', 'node_56', 'MUL_MAT_ID', 'f32', '2816:8:64:1') +
                desc('scale', 'node_59', 'GET_ROWS', 'f32', '1:8:64:1'))
        self.assertEqual(analyze.tensor(line, 'scale')['name'], 'node_59')
        self.assertEqual(analyze.tensor(line, 'scale')['ne'], (1, 8, 64, 1))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('PINNED_ORIGINAL_SOURCE')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    result = unittest.TextTestRunner().run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
