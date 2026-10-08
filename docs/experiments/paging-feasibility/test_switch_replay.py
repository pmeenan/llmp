# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch
import switch_replay as replay

E=replay.EXTENT


def models():
    return [dict(id=m,experts=2,topk=1,layers=1,expert_bytes=[E],nonexpert_bytes=E,kv_bytes=E,context=100) for m in ('A','B')]


def captures():
    out={}
    for m in ('A','B'):
        for variant in ('retained','recompute','normal'):
            out[m,variant]=[dict(reused=0,prompt=1,decode=1,state_bytes=100 if m=='A' else 200,
                                groups=[dict(layer=0,step=0,phase='prefill',routes=[[0]]),
                                        dict(layer=0,step=1,phase='decode',routes=[[0]])]) for _ in range(2)]
    return out


class CacheTest(unittest.TestCase):
    def test_shared_extent_charged_once(self):
        m=models()[0];m['expert_bytes']=[E//2]
        layout=replay.Layout([m],'packed')
        self.assertEqual(layout.closures['A',0,0],layout.closures['A',0,1])
        cache=replay.Cache(layout,2*E)
        self.assertEqual(cache.ensure(layout.weights['A'],0),2*E)
        self.assertEqual(cache.ensure(layout.weights['A'],0),0)

    def test_protected_group_cannot_evict_own_hit(self):
        layout=replay.Layout(models(),'isolated');cache=replay.Cache(layout,2*E)
        a=('A','expert',0);b=('B','expert',0);c=('A','expert',1)
        cache.ensure({a,b},0);self.assertEqual(cache.ensure({a,c},0),E)
        self.assertEqual(set(cache.cache),{a,c})

    def test_impossible_lease_rejected(self):
        layout=replay.Layout(models(),'isolated');cache=replay.Cache(layout,2*E)
        with self.assertRaises(ValueError):cache.ensure(layout.weights['A'],0)


class SwitchTest(unittest.TestCase):
    def simulate(self,budget,policy='partial',state='resident'):
        timeline=[dict(model='A',turn=0,arrival_s=0),dict(model='B',turn=0,arrival_s=60),dict(model='A',turn=1,arrival_s=120)]
        with patch.object(replay,'OVERHEAD',0):return replay.simulate(models(),captures(),timeline,budget,policy,state)

    def test_partial_return_saves_bytes(self):
        p=self.simulate(5*E);w=self.simulate(5*E,'whole')
        self.assertTrue(p['feasible'] and w['feasible'])
        self.assertLess(p['requests'][2]['first_token_weight_bytes'],w['requests'][2]['first_token_weight_bytes'])
        self.assertLessEqual(p['peak_occupancy_bytes'],5*E)

    def test_whole_keeps_both_when_fit(self):
        r=self.simulate(8*E,'whole')
        self.assertEqual(r['requests'][2]['weight_read_bytes'],0)

    def test_spill_write_restore_and_peak(self):
        r=self.simulate(5*E,state='spill')
        self.assertEqual(r['requests'][1]['state_written_bytes'],120)
        self.assertEqual(r['requests'][2]['state_written_bytes'],220)
        self.assertEqual(r['requests'][2]['state_restored_bytes'],120)
        self.assertEqual(r['peak_spill_bytes'],340)
        self.assertEqual(r['requests'][2]['spill_occupancy_bytes'],220)

    def test_cold_paging_when_full_model_cannot_fit(self):
        r=self.simulate(3*E,state='spill')
        self.assertTrue(r['feasible']);self.assertFalse(r['warm_start'])
        self.assertFalse(self.simulate(3*E,'whole','spill')['feasible'])

    def test_fixed_state_envelope_rejects(self):
        self.assertFalse(self.simulate(E)['feasible'])

    def test_recompute_never_writes_state(self):
        r=self.simulate(5*E,state='recompute')
        self.assertTrue(r['feasible'])
        self.assertEqual(sum(t['state_written_bytes']+t['state_restored_bytes'] for t in r['requests']),0)


if __name__=='__main__':unittest.main()
