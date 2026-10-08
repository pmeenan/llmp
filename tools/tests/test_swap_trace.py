# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The retained-backing swap trace: determinism, schema and reference checks on synthetic captures."""

import collections
import copy
import hashlib
import importlib.util
import json
import pathlib
import random
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parents[2] / "docs/experiments/retained-backing"
SPEC = importlib.util.spec_from_file_location("swap_trace", HERE / "swap_trace.py")
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)
MIB = 1 << 20


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def library():
    moe = dict(profile="moe", model="synthetic MoE", architecture="x", experts=4,
               dense=[dict(kind="table", layer=None, used=MIB, stored=MIB),
                      dict(kind="layer", layer=0, used=3 * MIB - 100, stored=3 * MIB),
                      dict(kind="layer", layer=1, used=2 * MIB + 8192, stored=2 * MIB + 8192),
                      dict(kind="head", layer=None, used=MIB + 4096, stored=MIB + 4096)],
               expert_layers=[dict(layer=0, count=4, used=MIB + 256, stored=MIB + 4096),
                              dict(layer=1, count=4, used=2 * MIB, stored=2 * MIB)])
    dense = dict(profile="dense", model="synthetic dense", architecture="y", experts=0,
                 dense=[dict(kind="layer", layer=0, used=4 * MIB, stored=4 * MIB),
                        dict(kind="head", layer=None, used=MIB, stored=MIB)],
                 expert_layers=[])
    return dict(format="llmp-retained-backing-library", version=1, chunk_bytes=2 * MIB, file_align=4096,
                profiles=[moe, dense])


def params():
    return {"format": "llmp-swap-trace-params", "version": 1,
            "seeds": {"primary": 11, "confirmation": 12},
            "ratios": [[5, 4], [2, 1]], "budget_granule": 2 * MIB, "shrink_bytes": 2 * MIB, "rounds": 2,
            "episode": {"roles": ["A", "A", "B", "A"], "arrival_s": [0, 10, 20, 30], "span_s": 40},
            "captures": {"c": "cap"}, "pools": {"moe": [["c", "M-b4-r1-s1-t1"]]},
            "library": [{"name": "moe-1", "profile": "moe"}, {"name": "moe-2", "profile": "moe", "layers": [1]},
                        {"name": "dense-1", "profile": "dense"}]}


def capture_events(rng, requests, layers=2, experts=4, topk=2, batch=4):
    lines = []
    for n, (prompt, reused, decode) in enumerate(requests):
        lines.append(dict(event="request", request=n, prompt=prompt, decode=decode, reused=reused))
        remaining, step = prompt - reused, 0
        while remaining or decode:
            phase = "prefill" if remaining else "decode"
            tokens = min(batch, remaining) if remaining else 1
            for layer in range(layers):
                routes = [rng.sample(range(experts), topk) for _ in range(tokens)]
                lines.append(dict(event="routes", request=n, step=step, layer=layer, phase=phase, routes=routes))
            lines.append(dict(event="step", request=n, step=step, phase=phase, tokens=tokens, us=10))
            if remaining:
                remaining -= tokens
            else:
                decode -= 1
            step += 1
        lines.append(dict(event="end", request=n, history=prompt + requests[n][2], state_bytes=1000))
    return "".join(json.dumps(x) + "\n" for x in lines)


class Fixture:
    """A synthetic capture tree, evidence record and frozen-trace sessions file."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        cap = self.root / "captures" / "cap"
        cap.mkdir(parents=True)
        events = cap / "M-b4-r1-s1-t1.jsonl"
        events.write_text(capture_events(random.Random(5), [(9, 0, 3), (12, 9, 2), (6, 0, 1)]))
        spec = dict(id="M", layers=2, experts=4, topk=2, expert_bytes=[MIB, 2 * MIB], context=4096)
        receipt = dict(spec=dict(models=[spec]),
                       runs=[dict(name="M-b4-r1-s1-t1", model="M", batch=4, reuse=1, swa=1, trace=True,
                                  events_sha256=sha(events))])
        (cap / "receipt.json").write_text(json.dumps(receipt))
        sessions = self.root / "sessions.json"
        sessions.write_text(json.dumps({"requests": [
            {"model": r, "arrival_s": t} for r, t in zip("AABA", (0, 10, 20, 30))]}))
        self.evidence = dict(captures={"cap": dict(external_directory="cap", receipt_file_sha256=sha(cap / "receipt.json"),
                                                   receipt=receipt)},
                             inputs={"aba": dict(sha256=sha(sessions))})
        self.params, self.library = params(), library()

    def generate(self, name, role="primary"):
        paths = {}
        for key in ("params", "library", "evidence"):
            paths[key] = self.root / f"{key}.json"
            paths[key].write_text(json.dumps(getattr(self, key)))
        out = self.root / name
        digest = trace.generate(paths["params"], paths["library"], paths["evidence"], self.root / "captures",
                                self.root / "sessions.json", role, out)
        return digest, out


class Stream(unittest.TestCase):
    def test_draws_are_pinned(self):
        # The stream is the contract a C++ replayer may reproduce.
        d = trace.Draws(20260926, "A")
        self.assertEqual([d.below(8) for _ in range(6)], [5, 1, 1, 5, 1, 4])
        self.assertEqual(trace.Draws(7, "B").permutation(range(6)), [5, 2, 1, 3, 4, 0])

    def test_every_model_leads_once_per_round(self):
        p = params()
        episodes = trace.schedule(p, 11)
        self.assertEqual(len(episodes), 6)
        for r in range(2):
            self.assertEqual(sorted(a for a, _ in episodes[3 * r:3 * r + 3]), ["dense-1", "moe-1", "moe-2"])
        self.assertTrue(all(a != b for a, b in episodes))

    def test_budget_rule(self):
        self.assertEqual(trace.budget_for(10 * (1 << 30) + 5, [5, 4], 1 << 30), 8 << 30)
        self.assertEqual(trace.budget_for(10 * (1 << 30), [3, 2], 1 << 30), 6 << 30)
        self.assertEqual(trace.budget_for(10 * (1 << 30), [2, 1], 1 << 30), 5 << 30)


class Generation(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.fx = Fixture(self.scratch.name)

    def test_deterministic_and_valid(self):
        first, a = self.fx.generate("a")
        second, b = self.fx.generate("b")
        self.assertEqual(first, second)
        for name in json.loads((a / "manifest.json").read_text())["files"]:
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
        summary = trace.verify(a)
        self.assertEqual(len(summary), 2)
        header, counts = trace.validate(a / "trace-r2-1.jsonl")
        self.assertEqual(header["budget_bytes"] % (2 * MIB), 0)
        self.assertGreater(counts["evict"], 0)
        self.assertEqual(counts["request"], 24)

    def test_confirmation_seed_differs(self):
        _, a = self.fx.generate("a", "primary")
        _, b = self.fx.generate("b", "confirmation")
        ma, mb = (json.loads((d / "manifest.json").read_text()) for d in (a, b))
        self.assertNotEqual(ma["access_sha256"], mb["access_sha256"])
        self.assertEqual((ma["seed"], mb["seed"]), (11, 12))

    def test_layer_subset_and_dense_models(self):
        _, a = self.fx.generate("a")
        with open(a / "trace-r5-4.jsonl") as f:
            records = [json.loads(x) for x in f]
        groups = [r for r in records if r.get("record") == "group"]
        subset = [g for g in groups if g["model"] == "moe-2"]
        self.assertEqual({g["layer"] for g in subset if g["kind"] == "expert"}, {1})
        self.assertEqual({(g["kind"], g["layer"]) for g in subset if g["kind"] != "expert"},
                         {("table", None), ("layer", 1), ("head", None)})
        dense = [r for r in records if r.get("ev") == "request" and r["model"] == "dense-1"]
        self.assertTrue(dense and all(r["source"] == "dense" for r in dense))
        # A shrink probe precedes every B request; its bytes return before the next A request.
        probes = [r for r in records if r.get("ev") in ("shrink", "grow", "request")]
        state = False
        for r in probes:
            if r["ev"] == "request":
                self.assertEqual(state, r["role"] == "B")
            else:
                self.assertEqual(r["ev"] == "shrink", not state)
                state = r["ev"] == "shrink"
        self.assertFalse(state)

    def test_inputs_are_checked(self):
        cases = []
        bad = copy.deepcopy(self.fx.evidence)
        bad["captures"]["cap"]["receipt"]["runs"][0]["events_sha256"] = "0" * 64
        cases.append(("evidence", bad))
        bad = copy.deepcopy(self.fx.evidence)
        bad["inputs"]["aba"]["sha256"] = "0" * 64
        cases.append(("evidence", bad))
        bad = copy.deepcopy(self.fx.params)
        bad["episode"]["arrival_s"] = [0, 10, 20, 31]
        cases.append(("params", bad))
        bad = copy.deepcopy(self.fx.params)
        bad["seeds"]["confirmation"] = 11
        cases.append(("params", bad))
        bad = copy.deepcopy(self.fx.library)
        bad["profiles"][0]["expert_layers"][1]["used"] = MIB  # capture's 2 MiB class no longer fits
        cases.append(("library", bad))
        for n, (key, value) in enumerate(cases):
            original = getattr(self.fx, key)
            setattr(self.fx, key, value)
            with self.subTest(n), self.assertRaises(ValueError):
                self.fx.generate(f"bad{n}")
            setattr(self.fx, key, original)

    def test_budget_too_small_is_refused(self):
        self.fx.params["ratios"] = [[100, 1]]
        with self.assertRaises(ValueError):
            self.fx.generate("small")


class Schema(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        _, out = Fixture(self.scratch.name).generate("a")
        self.path = out / "trace-r2-1.jsonl"
        self.lines = self.path.read_text().splitlines()

    def mutate(self, change):
        records = [json.loads(x) for x in self.lines]
        change(records)
        path = self.path.with_name("mutated.jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def first(self, records, ev):
        return next(r for r in records if r.get("ev") == ev)

    def test_mutations_are_rejected(self):
        def drop_restore(rs):
            rs.remove(self.first(rs, "restore"))

        def extra_evict(rs):
            i = next(n for n, r in enumerate(rs) if r.get("ev") == "evict")
            resident = set()
            for r in rs[:i]:
                if r.get("ev") == "restore":
                    resident.update(r["groups"])
                if r.get("ev") == "evict":
                    resident.difference_update(r["groups"])
            rs[i]["groups"] += sorted(resident - set(rs[i]["groups"]))[:1]

        def shrink_budget(rs):
            rs[0]["budget_bytes"] //= 2

        def header_key(rs):
            rs[0]["extra"] = 1

        def unaligned(rs):
            next(r for r in rs if r.get("record") == "group")["stored"] += 1

        def access_changed(rs):
            self.first(rs, "use")["groups"].reverse()

        def open_lease(rs):
            rs.remove(next(r for r in reversed(rs) if r.get("ev") == "release"))

        def never_grows(rs):
            rs.remove(next(r for r in reversed(rs) if r.get("ev") == "grow"))

        def shrink_ignored(rs):
            n = next(n for n, r in enumerate(rs) if r.get("ev") == "shrink" and rs[n - 1].get("ev") == "evict")
            del rs[n - 1]

        for change in (drop_restore, extra_evict, shrink_budget, header_key, unaligned, access_changed, open_lease,
                       never_grows, shrink_ignored):
            with self.subTest(change.__name__), self.assertRaises((ValueError, KeyError)):
                trace.validate(self.mutate(change))

    def test_unmutated_round_trip_passes(self):
        trace.validate(self.mutate(lambda rs: None))


class ReferenceOracle(unittest.TestCase):
    """The reference matches an independent list-based LRU over random streams."""

    def oracle(self, sizes, budget, events):
        order, protected, leases, out, shrunk = [], collections.Counter(), {}, [], 0
        for e in events:
            if e["ev"] == "grow":
                shrunk -= e["bytes"]
                out.append(None)
                continue
            if e["ev"] == "shrink":
                shrunk += e["bytes"]
            if e["ev"] == "release":
                ids = leases.pop(e["lease"])
                for i in ids:
                    protected[i] -= 1
                order = [i for i in order if i not in ids] + list(ids)
                out.append(None)
                continue
            ids = e.get("groups", [])
            missing = [i for i in ids if i not in order]
            used = sum(sizes[i] for i in order)
            evicted = []
            for i in list(order):
                if used + sum(sizes[j] for j in missing) <= budget - shrunk:
                    break
                if i in ids or protected[i] > 0:
                    continue
                order.remove(i)
                used -= sizes[i]
                evicted.append(i)
            order = [i for i in order if i not in ids] + list(ids)
            if e["ev"] == "lease":
                leases[e["lease"]] = ids
                for i in ids:
                    protected[i] += 1
            out.append((evicted, missing))
        return out

    def test_random_streams(self):
        rng = random.Random(3)
        for trial in range(200):
            sizes = [rng.choice((4096, 8192, 2 * MIB, 3 * MIB)) for _ in range(12)]
            budget = max(sizes) * rng.randint(3, 6)
            events, open_leases, lease, shrunk = [], [], 0, False
            for _ in range(40):
                roll = rng.random()
                if roll > 0.93:
                    events.append(dict(ev="grow" if shrunk else "shrink", bytes=max(sizes)))
                    shrunk = not shrunk
                elif roll < 0.15 and len(open_leases) < 2:
                    ids = sorted(rng.sample(range(12), 2))
                    events.append(dict(ev="lease", lease=lease, groups=ids))
                    open_leases.append(lease)
                    lease += 1
                elif roll < 0.3 and open_leases:
                    events.append(dict(ev="release", lease=open_leases.pop(0)))
                else:
                    events.append(dict(ev="use", groups=sorted(rng.sample(range(12), rng.randint(1, 3)))))
            groups = [dict(stored=s) for s in sizes]
            ref = trace.Reference(groups, budget)
            want = self.oracle(sizes, budget, events)
            for e, expected in zip(events, want):
                try:
                    got = ref.apply(e)
                except ValueError:
                    break  # this stream's leases outgrew the budget; the oracle does not model refusal
                if expected is None:
                    self.assertEqual(got, [])
                    continue
                evicted = next((x["groups"] for x in got if x["ev"] == "evict"), [])
                restored = next((x["groups"] for x in got if x["ev"] == "restore"), [])
                self.assertEqual((evicted, restored), expected, f"trial {trial}")
                self.assertLessEqual(ref.used, budget - ref.shrunk)


if __name__ == "__main__":
    unittest.main()
