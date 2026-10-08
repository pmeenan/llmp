# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import replay


def record(step, ids, phase='decode', layer=0):
    return dict(step=step, phase=phase, layer=layer, routes=[ids])


class ReplayTest(unittest.TestCase):
    def small(self, records, capacity):
        with patch.multiple(replay, FIXED=0, EXTENT=1, SIZES=[2]*30, LOGICAL=[1]*30):
            return replay.replay(records, capacity)

    def test_hits_and_physical_padding(self):
        r = self.small([record(0, [1,2]), record(1, [1,2])], 4)
        self.assertEqual(r['phases']['decode']['miss_bytes'], 4)
        self.assertEqual(r['phases']['decode']['logical_miss_bytes'], 2)
        self.assertEqual(r['peak_expert_bytes'], 4)

    def test_protect_all_selected_hits_during_eviction(self):
        r = self.small([record(0,[1,2]), record(1,[1,3]), record(2,[1,3])], 4)
        self.assertEqual(r['phases']['decode']['miss_bytes'], 6)

    def test_reject_impossible_atomic_group(self):
        r = self.small([dict(step=0,phase='decode',layer=0,routes=[[1,2],[3,4]])], 6)
        self.assertFalse(r['feasible'])
        self.assertEqual(r['required_bytes'], 8)

    def test_global_lru_and_layer_identity(self):
        r = self.small([record(0,[1],layer=0), record(1,[1],layer=1),
                        record(2,[2],layer=0), record(3,[1],layer=0)], 4)
        self.assertEqual(r['phases']['decode']['miss_bytes'], 8)

    def test_fixed_envelope_rejection(self):
        self.assertFalse(replay.replay([], replay.FIXED-1)['feasible'])

    def test_full_residency_only_cold_misses(self):
        records = [record(t,list(range(8)),layer=layer) for t in range(2) for layer in range(30)]
        r = replay.replay(records, 28*replay.GIB)
        self.assertEqual(r['phases']['decode']['miss_bytes'], sum(replay.SIZES)*8)
        self.assertTrue(r['whole_model_resident_feasible'])


class LocalityTest(unittest.TestCase):
    def test_identical_groups_zero_distance(self):
        records = [record(t,list(range(8)),layer=layer) for t in range(4) for layer in range(30)]
        # Supply prefill too so both summary phases are nonempty.
        records = [record(0,list(range(8)),phase='prefill',layer=layer) for layer in range(30)] + records
        r = replay.locality(records)
        for layer in r['per_layer_reuse']:
            self.assertEqual(layer['decode']['distinct_distance_p50'],0)
            self.assertEqual(layer['decode']['distinct_distance_p95'],0)

    def test_distances_invariant_to_expert_relabeling(self):
        records = [record(t,list(range(t*3,t*3+8)),phase='prefill' if t==0 else 'decode',layer=layer)
                   for t in range(8) for layer in range(30)]
        renamed = [dict(r,routes=[[127-n for n in ids] for ids in r['routes']]) for r in records]
        self.assertEqual(replay.locality(records)['per_layer_reuse'],
                         replay.locality(renamed)['per_layer_reuse'])


class ProvenanceTest(unittest.TestCase):
    def test_identity_and_duplicate_batches_rejected(self):
        m = dict(image=replay.IMAGE,engine_revision=replay.REVISION,model_sha256=replay.MODEL_SHA,
                 trace_sha256=replay.TRACE_SHA,
                 runs=[dict(name=f'b{b}-'+('routes' if t else 'control'),batch=b,tracing=t,
                            predictions_sha256='a') for b,t in [(512,False),(512,True),(64,False),(64,True)]])
        replay.validate_metadata(m)
        bad = copy.deepcopy(m)
        bad['runs'][0:2] = bad['runs'][2:4]
        with self.assertRaises(ValueError): replay.validate_metadata(bad)
        bad = copy.deepcopy(m)
        bad['model_sha256'] = 'unknown'
        with self.assertRaises(ValueError): replay.validate_metadata(bad)
        bad = copy.deepcopy(m)
        bad['runs'][1]['predictions_sha256'] = 'changed'
        with self.assertRaises(ValueError): replay.validate_metadata(bad)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'routes.jsonl'
        self.metadata = dict(prompt_tokens=18339,decode_tokens=117)
        counts = [512]*35+[419]+[1]*117
        self.records = [dict(step=step,phase='prefill' if step<36 else 'decode',layer=layer,
                             routes=[list(range(8))]*count)
                        for step,count in enumerate(counts) for layer in range(30)]

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, records):
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        return replay.load(self.path,self.metadata,dict(batch=512,routes_sha256=replay.sha(self.path)))

    def test_complete(self):
        self.assertEqual(len(self.load(self.records)),4590)

    def test_missing_tail(self):
        with self.assertRaises(ValueError): self.load(self.records[:-1])

    def test_corruptions(self):
        for key,value in [('layer',1),('phase','decode'),('step',2),('routes',[[0]*8])]:
            records = copy.copy(self.records)
            records[0] = dict(records[0],**{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError): self.load(records)
        records = copy.copy(self.records)
        records[-1] = dict(records[-1],routes=[[True,1,2,3,4,5,6,7]])
        with self.assertRaises(ValueError): self.load(records)

    def test_identity_mismatch(self):
        self.path.write_text('')
        with self.assertRaises(ValueError):
            replay.load(self.path,self.metadata,dict(batch=512,routes_sha256='0'*64))


if __name__ == '__main__':
    unittest.main()
