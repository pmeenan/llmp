#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests. No torch, GPU, source engine, checkpoint or inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from contract import ActualTargets, PHASES, RECIPE, encoded, frozen, native_records, row_groups, sha, valid_ids
from reference import require_validation


class ContractTests(unittest.TestCase):
    def test_chunks_pair_each_next_actual_token_including_boundaries(self):
        ids = [42, 6, 93, 1, 15, 80, 2]
        groups = row_groups(ids, 1, [(0, 3), (3, 6)], 2)
        self.assertEqual(groups[0][2], [(0, 2, [1, 2], [6, 93]), (2, 3, [3], [1])])
        self.assertEqual(groups[1][2], [(0, 2, [4, 5], [15, 80]), (2, 3, [6], [2])])
        collector = ActualTargets(ids, 1, 0)
        for _, _, batches in groups:
            for _, _, positions, targets in batches:
                collector.add(positions, targets, [-0.5] * len(targets), [[]] * len(targets), [[]] * len(targets))
        self.assertEqual([r['target'] for r in collector.finish()], ids[1:])

    def test_depth_preserves_all_prefix_spans_but_scores_only_tail(self):
        ids = list(range(9216))
        # Source's even partition: 18 chunks for 9215 teacher rows at max512.
        spans = [(9215*j//18, 9215*(j+1)//18) for j in range(18)]
        groups = row_groups(ids, 8192, spans)
        self.assertEqual([(a, b) for a, b, _ in groups], spans)
        positions = [p for _, _, batches in groups for _, _, ps, _ in batches for p in ps]
        self.assertEqual(positions, list(range(8192, 9216)))
        self.assertEqual(len(positions), 1024)

    def test_empty_and_broken_prefix_partition_refused(self):
        for spans in ([], [(0, 1)], [(1, 3)], [(0, 2), (1, 3)], [(0, 4)], [(0, 0), (0, 3)]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                row_groups([1, 2, 3, 4], 1, spans)
        with self.assertRaises(ValueError):
            row_groups([1], 1, [])

    def test_empty_out_of_range_and_boolean_ids_refused(self):
        for ids in ([], [1], [1, True], [1, -1], [1, 10], '12'):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                valid_ids(ids, 10)
        valid_ids([9, 2], 10)  # no Llama BOS sentinel

    def test_capture_refuses_wrong_target_position_and_count(self):
        cases = [([1], [99], [-1], [[]], [[]]), ([2], [3], [-1], [[]], [[]]),
                 ([1], [2], [], [[]], [[]]), ([], [], [], [], []),
                 ([1], [2], [-1], [[4]], [[-2]])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                ActualTargets([7, 2, 3], 1, 0).add(*args)

    def test_nonfinite_positive_and_non_numeric_scores_refused(self):
        for bad in (float('nan'), float('inf'), -float('inf'), .01, True, None, '-1'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ActualTargets([7, 2], 1, 0).add([1], [2], [bad], [[]], [[]])

    def test_missing_duplicate_and_extra_targets_refused(self):
        c = ActualTargets([7, 2, 3], 1, 0)
        c.add([1], [2], [-1], [[]], [[]])
        with self.assertRaises(ValueError): c.finish()
        with self.assertRaises(ValueError): c.add([1], [2], [-1], [[]], [[]])
        c.add([2], [3], [-1], [[]], [[]])
        self.assertEqual(len(c.finish()), 2)
        with self.assertRaises(ValueError): c.add([3], [3], [-1], [[]], [[]])

    @staticmethod
    def response():
        return {'choices': [{'prompt_logprobs': [None, {'2': {'logprob': -1}}, {'3': {'logprob': -.5}}],
                             'prompt_token_ids': [7, 2, 3]}], 'usage': {'prompt_tokens': 3}}

    def test_native_actual_targets_and_unscored_first_input(self):
        result = native_records(self.response(), [7, 2, 3], 1, 0)
        self.assertEqual(result, [{'document': 0, 'position': 1, 'target': 2, 'nll': 1},
                                  {'document': 0, 'position': 2, 'target': 3, 'nll': .5}])

    def test_native_refusals_and_malformed_payloads(self):
        for mutation in range(7):
            r = self.response()
            if mutation == 0: r['error'] = 'refused'
            if mutation == 1: r['choices'] = []
            if mutation == 2: r['usage']['prompt_tokens'] = 2
            if mutation == 3: r['choices'][0]['prompt_token_ids'] = [7, 3, 2]
            if mutation == 4: r['choices'][0]['prompt_logprobs'][1] = {'9': {'logprob': -1}}
            if mutation == 5: r['choices'][0]['prompt_logprobs'][1] = {'2': None}
            if mutation == 6: r['choices'][0]['prompt_logprobs'][0] = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native_records(r, [7, 2, 3], 1, 0)

    def test_draft_never_permits_inference(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'protocol.json'
            p.write_bytes(encoded({'format': 'llmp-u3-qwen27-prereg/1', 'status': 'draft'}))
            with self.assertRaises(ValueError): frozen(p, Path(d), 'a' * 40)

    def test_validation_must_match_identity_and_both_controls(self):
        identity = {'protocol_sha256': 'a'*64, 'prereg_commit': 'b'*40}
        good = {'passed': True, 'identity': identity, 'checks': [
            {'prompt_tokens': n, 'normal_first_token': 8, 'adapter_first_token': 8,
             'head_1_vs_16_exact': True} for n in (513, 1024)]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'validation.json'
            p.write_bytes(encoded(good)); require_validation(p, identity)
            for mutation in range(5):
                bad = copy.deepcopy(good)
                if mutation == 0: bad['passed'] = False
                if mutation == 1: bad['identity'] = {}
                if mutation == 2: bad['checks'].pop()
                if mutation == 3: bad['checks'][0]['head_1_vs_16_exact'] = False
                if mutation == 4: bad['checks'][0]['adapter_first_token'] = 9
                p.write_bytes(encoded(bad))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): require_validation(p, identity)
            with self.assertRaises(ValueError): require_validation(None, identity)

    def test_pairing_bound_files_match_fixed_geometry(self):
        root = Path(__file__).parent
        for phase, (_, _, count, _) in PHASES.items():
            for own in (False, True):
                path = root / f'{phase}{"-repeat" if own else ""}-prereg.json'
                bounds = json.loads(path.read_bytes())['bounds']
                self.assertEqual(bounds, {'expected_targets': count,
                    'max_abs_ppl_delta_pct' if own else 'max_ppl_increase_pct': .1 if own else 1.0})
        self.assertIs(RECIPE['source_prefill_fp8'], False)


if __name__ == '__main__':
    unittest.main()
