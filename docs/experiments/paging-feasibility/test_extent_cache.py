# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Challenge compact weighted residency against independent individual extents."""
from collections import Counter
import random
from types import SimpleNamespace
import unittest
from switch_replay import Cache, EXTENT

E=EXTENT
A=('A','base',0)
B=('B','base',0)
X=('B','expert',0)
Y=('A','expert',0)


class ExtentCacheTest(unittest.TestCase):
    def test_inactive_base_reclaims_only_needed_extents_and_keeps_age(self):
        cache=Cache(SimpleNamespace(sizes={A:5*E,X:E,Y:E}),6*E)
        cache.ensure({A,Y},0)
        self.assertEqual(cache.ensure({X},0),E)
        self.assertEqual(cache.cache[A],4*E)
        self.assertEqual(cache.used,6*E)
        self.assertEqual(list(cache.cache),[A,Y,X])
        self.assertEqual(cache.missing_bytes({A}),E)
        self.assertEqual(cache.ensure({A},0),E)
        self.assertEqual(cache.cache[A],5*E)
        self.assertNotIn(Y,cache.cache)

    def test_active_base_is_fully_protected(self):
        cache=Cache(SimpleNamespace(sizes={A:4*E,B:4*E,X:E}),8*E)
        cache.ensure({A,B},0)
        cache.ensure({A,X},0)
        self.assertEqual(cache.cache[A],4*E)
        self.assertEqual(cache.cache[B],3*E)
        self.assertEqual(cache.cache[X],E)
        self.assertEqual(cache.used,8*E)

    def test_discard_partial_base_subtracts_resident_bytes(self):
        cache=Cache(SimpleNamespace(sizes={A:4*E,X:E}),4*E)
        cache.ensure({A},0);cache.ensure({X},0)
        cache.discard({A})
        self.assertEqual(cache.used,E)
        self.assertEqual(cache.missing_bytes({A}),4*E)

    def test_randomized_individual_extent_oracle(self):
        randomizer=random.Random(84219)
        sizes={A:5*E,B:3*E,X:E,Y:E,('C','base',0):2*E,('C','expert',0):E}
        keys=list(sizes)
        for _ in range(100):
            budget=randomizer.randint(3,12)*E+randomizer.randrange(E)
            cache=Cache(SimpleNamespace(sizes=sizes),budget)
            oracle=[]
            for _ in range(50):
                required=set(randomizer.sample(keys,randomizer.randint(0,3)))
                if randomizer.randrange(8)==0:
                    cache.discard(required)
                    oracle=[extent for extent in oracle if extent[0] not in required]
                else:
                    fixed=randomizer.randint(0,2)*E
                    expanded=[(key,n) for key in sorted(required) for n in range(sizes[key]//E)]
                    if len(expanded)*E+fixed>budget:
                        with self.assertRaises(ValueError):cache.ensure(required,fixed)
                        continue
                    expected_miss=sum(extent not in oracle for extent in expanded)*E
                    while len(oracle)*E+expected_miss+fixed>budget:
                        victim=next(extent for extent in oracle if extent[0] not in required)
                        oracle.remove(victim)
                    oracle=[extent for extent in oracle if extent[0] not in required]+expanded
                    self.assertEqual(cache.ensure(required,fixed),expected_miss)
                    self.assertLessEqual(cache.used+fixed,budget)
                self.assertEqual(cache.used,len(oracle)*E)
                counts=Counter(key for key,_ in oracle)
                self.assertEqual(dict(cache.cache),{key:count*E for key,count in counts.items()})
                order=list(dict.fromkeys(key for key,_ in oracle))
                self.assertEqual(list(cache.cache),order)


if __name__=='__main__':unittest.main()
