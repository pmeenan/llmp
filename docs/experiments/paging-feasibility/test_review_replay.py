# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent regression challenges for session replay accounting."""
import copy
import unittest
from unittest.mock import patch
import switch_replay as replay
from locality_sessions import analyze

E = replay.EXTENT


def fixture():
    models = [dict(id=name, experts=2, topk=1, layers=1, expert_bytes=[E],
                   nonexpert_bytes=E, kv_bytes=E, normal_kv_bytes=E, context=100) for name in ('A', 'B')]
    captures = {}
    for name in ('A', 'B'):
        for variant in ('retained', 'recompute', 'normal_retained', 'normal_recompute'):
            captures[name, variant] = [dict(prompt=10, decode=1, reused=9,
                state_bytes=E, coverage_reset=False, groups=[dict(layer=0, step=0, phase='prefill', routes=[[0]]),
                                     dict(layer=0, step=1, phase='decode', routes=[[0]])])
                                     for _ in range(2)]
    timeline = [dict(model='A', turn=0, arrival_s=0),
                dict(model='B', turn=0, arrival_s=60),
                dict(model='A', turn=1, arrival_s=120)]
    return models, captures, timeline


class ReviewChallenges(unittest.TestCase):
    def test_normal_initial_admission_uses_normal_allocation(self):
        models, captures, timeline = fixture()
        models[0]['kv_bytes'] = 6 * replay.GIB
        models[0]['normal_kv_bytes'] = 940 * (1 << 20)
        with patch.object(replay, 'OVERHEAD', 0):
            result = replay.simulate(models, captures, timeline[:1], replay.GIB,
                                     'whole', 'resident', normal=True)
        self.assertTrue(result['feasible'])
        self.assertTrue(result['warm_start'])
        self.assertLessEqual(result['peak_occupancy_bytes'], replay.GIB)

    def test_normal_spill_recomputes_only_on_return(self):
        models, captures, timeline = fixture()
        captures['A', 'normal_recompute'][1]['reused'] = 0
        with patch.object(replay, 'OVERHEAD', 0):
            resident = replay.simulate(models, captures, timeline, 2 * replay.GIB,
                                       'whole', 'resident', normal=True)
            spill = replay.simulate(models, captures, timeline, 2 * replay.GIB,
                                    'whole', 'spill', normal=True)
        self.assertEqual(resident['requests'][2]['reused'], 9)
        self.assertEqual(spill['requests'][2]['reused'], 0)
        self.assertEqual(spill['requests'][2]['state_restored_bytes'], E + 56)

    def test_full_swa_snapshot_rollback_requires_recompute(self):
        models, captures, timeline = fixture()
        captures['A', 'retained'][0]['prompt'] = 100
        captures['A', 'retained'][0]['decode'] = 10
        captures['A', 'retained'][1]['reused'] = 50
        captures['A', 'recompute'][1]['reused'] = 0
        with patch.object(replay, 'OVERHEAD', 0):
            result = replay.simulate(models, captures, timeline, 16 * E,
                                     'partial', 'spill')
        returned = result['requests'][2]
        self.assertEqual(returned['reused'], 0)
        self.assertEqual(returned['capture_variant'], 'recompute')
        self.assertIn('SWA cells', returned['state_reuse_fallback'])
        self.assertGreater(returned['state_restored_bytes'], 0)

    def test_legacy_normal_retention_is_not_assumed_valid(self):
        models, captures, timeline = fixture()
        for request in captures['A', 'normal_retained']:
            del request['coverage_reset']
        captures['A', 'normal_recompute'][1]['reused'] = 0
        with patch.object(replay, 'OVERHEAD', 0):
            result = replay.simulate(models, captures, timeline, 16 * E,
                                     'partial', 'resident', normal=True)
        self.assertEqual(result['requests'][2]['capture_variant'], 'normal_recompute')
        self.assertEqual(result['requests'][2]['reused'], 0)

    def test_explicit_spill_recompute_preserves_restore_cost(self):
        models, captures, timeline = fixture()
        replay.mark_spill_recompute(models, ['A'])
        # This append-only return would otherwise be reusable.
        captures['A', 'retained'][1]['reused'] = 10
        captures['A', 'recompute'][1]['reused'] = 0
        with patch.object(replay, 'OVERHEAD', 0):
            result = replay.simulate(models, captures, timeline, 16 * E,
                                     'partial', 'spill')
        returned = result['requests'][2]
        self.assertEqual(returned['reused'], 0)
        self.assertEqual(returned['capture_variant'], 'recompute')
        self.assertIn('unvalidated', returned['state_reuse_fallback'])
        self.assertGreater(returned['state_restored_bytes'], 0)

    def test_unknown_spill_recompute_model_is_rejected(self):
        models, _, _ = fixture()
        with self.assertRaises(ValueError):
            replay.mark_spill_recompute(models, ['D'])

    def test_spill_ceiling_counts_write_before_restore(self):
        models, captures, timeline = fixture()
        for requests in captures.values():
            for request in requests:
                request['state_bytes'] = 5 * replay.GIB
        with patch.object(replay, 'OVERHEAD', 0):
            result = replay.simulate(models, captures, timeline, 16 * replay.GIB,
                                     'partial', 'spill')
        self.assertFalse(result['feasible'])
        self.assertIn('spill limit', result['reason'])

    def test_eager_partial_retention_preserves_inactive_weight(self):
        models, captures, timeline = fixture()
        with patch.object(replay, 'OVERHEAD', 0):
            eager = replay.simulate(models, captures, timeline, 7 * E,
                                    'retained_eager', 'resident')
            whole = replay.simulate(models, captures, timeline, 7 * E,
                                    'whole', 'resident')
        self.assertTrue(eager['feasible'] and whole['feasible'])
        self.assertLess(eager['requests'][2]['weight_read_bytes'],
                        whole['requests'][2]['weight_read_bytes'])
        self.assertEqual(eager['requests'][2]['decode_miss_bytes']['max'], 0)

    def test_predictor_does_not_train_on_test_requests(self):
        model = dict(layers=2, experts=2, topk=1, expert_bytes=[E, E])
        def request(target):
            return dict(groups=[dict(layer=0, phase='decode', routes=[[0]]),
                                dict(layer=1, phase='decode', routes=[[target]])])
        requests = [request(0), request(0), request(1), request(1)]
        result = analyze(copy.deepcopy(requests), model)['predictor']
        self.assertEqual(result['training_requests'], 2)
        self.assertEqual(result['selected_contributions'], 2)
        self.assertEqual(result['next_layer_recall'], 0)
        self.assertEqual(result['cold_prediction_bytes'], 2 * E)


if __name__ == '__main__':
    unittest.main()
