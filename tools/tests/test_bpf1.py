# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""BP-F1's case derivation, calibration gate and session checks, without a GPU."""

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
REPO = pathlib.Path(__file__).resolve().parents[2]
P1 = REPO / "docs/experiments/backend-proof-p1"
P0 = REPO / "docs/experiments/backend-proof-p0"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


cases = module("bpf1_cases", P1 / "bpf1_cases.py")
session = module("bpf1_session", P1 / "bpf1_session.py")
stats = module("bpf1_stats", P1 / "bpf1_stats.py")
protocol = module("timing_protocol", P0 / "timing_protocol.py")

HASH = "ab" * 32


class CaseDerivation(unittest.TestCase):
    def test_the_checked_in_case_file_is_derived_from_the_recorded_plan(self):
        text, count = cases.render((P0 / "fp16-plan.json").read_bytes())
        self.assertEqual(count, 53)
        self.assertEqual(text, (P1 / "bpf1-cases.txt").read_text())

    def test_internal_namespace_tags_are_normalized(self):
        mangled = "_Z11k_bin_bcastIXadL_ZN42_INTERNAL_d5c41c42_11_binbcast_cu_6840010b6op_addEffEEfffJPKfEEvP"
        self.assertEqual(cases.normalize(mangled), "_Z11k_bin_bcastIXadL_ZN_INTERNAL_6op_addEffEEfffJPKfEEvP")
        self.assertEqual(cases.normalize("_Z9mul_mat_fI7__half2EEv"), "_Z9mul_mat_fI7__half2EEv")


class CalibrationGate(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.dir = pathlib.Path(self.scratch.name)
        self.calibration = self.dir / "calibration.json"
        self.calibration.write_text("{}\n")
        self.digest = hashlib.sha256(b"{}\n").hexdigest()
        self.harness = self.dir / "llmp_ggml_vmm_bench"
        self.harness.write_bytes(b"harness")
        self.cases = self.dir / "bpf1-cases.txt"
        self.cases.write_text("set s\n")
        self.registry = self.dir / "backend-proof.md"

    def register(self, *digests, harness=None, cases=None):
        harness = harness or session.sha256(self.harness)
        cases = cases or session.sha256(self.cases)
        self.registry.write_text("".join(f"BP-F1 calibration SHA-256: `{d}`\n" for d in digests) +
                                 f"BP-F1 harness SHA-256: `{harness}`\nBP-F1 cases SHA-256: `{cases}`\n")

    def check(self):
        return session.check_registration(["cuda-malloc", "host-vmm"], self.registry, self.calibration,
                                           self.harness, self.cases)

    def test_an_a_a_session_needs_no_registration(self):
        self.assertIsNone(session.check_registration(["cuda-malloc", "cuda-malloc"], None, None))

    def test_host_vmm_is_refused_until_the_calibration_is_registered(self):
        with self.assertRaises(SystemExit):
            session.check_registration(["cuda-malloc", "host-vmm"], None, None)
        self.registry.write_text("no registration yet\n")
        with self.assertRaises(SystemExit):
            session.check_registration(["cuda-malloc", "host-vmm"], self.registry, self.calibration)

    def test_host_vmm_is_refused_under_another_or_an_ambiguous_calibration(self):
        for registered in ([HASH], [self.digest, HASH]):
            self.register(*registered)
            with self.subTest(registered=registered), self.assertRaises(SystemExit):
                self.check()

    def test_host_vmm_is_refused_with_another_harness_or_case_file(self):
        for other in ({"harness": HASH}, {"cases": HASH}):
            self.register(self.digest, **other)
            with self.subTest(other=other), self.assertRaises(SystemExit):
                self.check()
        self.register(self.digest)
        with self.assertRaises(SystemExit):
            session.check_registration(["cuda-malloc", "host-vmm"], self.registry, self.calibration)
        self.registry.write_text(f"BP-F1 calibration SHA-256: `{self.digest}`\n")
        with self.assertRaises(SystemExit):
            self.check()

    def test_the_repository_registers_the_checked_in_calibration(self):
        registry = (REPO / "docs/backend-proof.md").read_text()
        self.assertEqual(session.registered_calibration(registry), session.sha256(P1 / "bpf1-calibration.json"))
        self.assertEqual(session.registered(registry, "cases"), session.sha256(P1 / "bpf1-cases.txt"))
        record = json.loads((P1 / "bpf1-timing.json").read_text())
        for name, s in record["sessions"].items():
            with self.subTest(session=name):
                identities = s["manifest"]["identities"]
                self.assertEqual(identities["harness_sha256"], session.registered(registry, "harness"))
                self.assertEqual(identities["cases_sha256"], session.registered(registry, "cases"))

    def test_host_vmm_runs_under_the_registered_calibration(self):
        self.register(self.digest, self.digest)  # stated twice, one value
        self.assertEqual(self.check(), self.digest)

    def register_v2(self, calibration, harness=None, cases=None):
        harness = harness or session.sha256(self.harness)
        cases = cases or session.sha256(self.cases)
        with open(self.registry, "a") as f:
            f.write(f"BP-F1 v2 calibration SHA-256: `{calibration}`\nBP-F1 v2 harness SHA-256: `{harness}`\n"
                    f"BP-F1 v2 cases SHA-256: `{cases}`\n")

    def check_device(self):
        return session.check_registration(["cuda-malloc", "device-vmm"], self.registry, self.calibration,
                                           self.harness, self.cases)

    def test_device_vmm_runs_only_under_rule_v2(self):
        self.register(self.digest)  # rule v1 matches: not enough for device VMM
        with self.assertRaises(SystemExit):
            self.check_device()
        for other in ({"calibration": HASH}, {"harness": HASH}, {"cases": HASH}):
            self.register(self.digest)
            self.register_v2(**{"calibration": self.digest, **other})
            with self.subTest(other=other), self.assertRaises(SystemExit):
                self.check_device()
        self.register(HASH)  # rule v1 no longer matches; v2 does
        self.register_v2(self.digest)
        self.assertEqual(self.check_device(), self.digest)
        with self.assertRaises(SystemExit):
            self.check()  # host VMM stays under rule v1
        self.assertEqual(session.registered_calibration(self.registry.read_text()), HASH)
        self.assertEqual(session.registered_calibration(self.registry.read_text(), "v2"), self.digest)

    def test_a_session_compares_one_vmm_kind(self):
        self.register(self.digest)
        self.register_v2(self.digest)
        with self.assertRaises(SystemExit):
            session.check_registration(["host-vmm", "device-vmm"], self.registry, self.calibration,
                                       self.harness, self.cases)

    def test_the_repository_registers_rule_v2(self):
        registry = (REPO / "docs/backend-proof.md").read_text()
        self.assertEqual(session.registered_calibration(registry, "v2"),
                         session.sha256(P1 / "bpf1-v2-calibration.json"))
        self.assertEqual(session.registered(registry, "cases", "v2"), session.sha256(P1 / "bpf1-cases.txt"))
        self.assertNotEqual(session.registered(registry, "harness", "v2"), session.registered(registry, "harness"))
        for record_name in ("bpf1-v2-timing.json", "bpf1-v2-comparison.json"):
            record = json.loads((P1 / record_name).read_text())
            self.assertEqual(record["rule"]["calibration_sha256"], session.registered_calibration(registry, "v2"))
            for name, s in record["sessions"].items():
                with self.subTest(record=record_name, session=name):
                    identities = s["manifest"]["identities"]
                    self.assertEqual(identities["harness_sha256"], session.registered(registry, "harness", "v2"))
                    self.assertEqual(identities["cases_sha256"], session.registered(registry, "cases", "v2"))
        comparison = json.loads((P1 / "bpf1-v2-comparison.json").read_text())
        for name, s in comparison["sessions"].items():
            with self.subTest(comparison=name):
                self.assertEqual(s["manifest"]["arms"], {"A": "cuda-malloc", "B": "device-vmm"})
                self.assertEqual(s["manifest"]["calibration_sha256"], session.registered_calibration(registry, "v2"))


class SessionChecks(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.dir = pathlib.Path(self.scratch.name) / "c1"
        self.dir.mkdir()
        self.write_session("cuda-malloc")

    def write_session(self, candidate, calibration=None, scale=1.0):
        order = session.ORDERS["primary"]
        manifest = {"session": "c1", "order": order, "arms": {"A": "cuda-malloc", "B": candidate},
                    "calibration_sha256": calibration, "ended": "now",
                    "identities": {"harness_sha256": "cd" * 32, "cases_sha256": "ef" * 32},
                    "blocks": [{"block": b} for b in order.split()]}
        (self.dir / "manifest.json").write_text(json.dumps(manifest))
        for label in stats.BLOCKS:
            factor = scale if label[0] == "B" else 1.0
            self.write_block(label, {
                "memory": manifest["arms"][label[0]], "calibration_sha256": calibration if label[0] == "B" else None,
                "set": "s", "cases": [self.case(name, factor * (1 + 0.001 * int(label[1]))) for name in ("x", "y")]})

    def case(self, name, value):
        return {"name": name, "rows": 1, "sets": 2, "set_bytes": 256, "graph_launches_verified": 10,
                "eager_output_fnv1a64": "00ff", "final_output_fnv1a64": "00ff",
                "graph_replay_us": [value] * 31, "stream_us": [value * 1.1] * 31}

    def write_block(self, label, block):
        (self.dir / f"{label}.json").write_text(json.dumps(block))

    def block(self, label):
        return json.loads((self.dir / f"{label}.json").read_text())

    def test_a_session_summary_feeds_the_approved_rule(self):
        summary = stats.summarize(self.dir)
        self.assertEqual(summary["summary"]["blocks"], {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]})
        self.assertAlmostEqual(summary["cases"][0]["ratio"], 1.0)
        calibration = protocol.calibrate([summary, copy.deepcopy(summary)])
        self.assertEqual(sorted(calibration["sigma"]), ["s|x|1", "s|y|1"])
        self.assertTrue(protocol.apply(calibration, [summary])["stage_passes"])

    def test_incomplete_or_inconsistent_blocks_are_rejected(self):
        edits = {
            "a missing sample": lambda b: b["cases"][0]["graph_replay_us"].pop(),
            "a nonpositive sample": lambda b: b["cases"][0]["stream_us"].__setitem__(0, 0.0),
            "another memory kind": lambda b: b.__setitem__("memory", "host-vmm"),
            "an output that changed": lambda b: b["cases"][1].__setitem__("final_output_fnv1a64", "0100"),
            "an output that differs from other blocks": lambda b: [
                c.__setitem__(k, "0100") for c in b["cases"][:1]
                for k in ("eager_output_fnv1a64", "final_output_fnv1a64")],
            "another case list": lambda b: b["cases"].pop(),
            "no verified launches": lambda b: b["cases"][0].__setitem__("graph_launches_verified", 0),
        }
        original = self.block("B3")
        for what, edit in edits.items():
            block = copy.deepcopy(original)
            edit(block)
            self.write_block("B3", block)
            with self.subTest(what), self.assertRaises(ValueError):
                stats.summarize(self.dir)
        self.write_block("B3", original)
        (self.dir / "B3.json").unlink()
        with self.assertRaises(FileNotFoundError):
            stats.summarize(self.dir)

    def test_an_unfinished_session_is_rejected(self):
        manifest = json.loads((self.dir / "manifest.json").read_text())
        manifest["failed"] = {"block": "A2"}
        (self.dir / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            stats.summarize(self.dir)

    def test_host_vmm_blocks_need_the_registered_calibration(self):
        self.write_session("host-vmm", HASH, scale=1.2)
        identities = f"BP-F1 harness SHA-256: `{'cd' * 32}`\nBP-F1 cases SHA-256: `{'ef' * 32}`\n"
        registry = f"BP-F1 calibration SHA-256: `{HASH}`\n" + identities
        with self.assertRaises(ValueError):
            stats.summarize(self.dir)  # no registry
        for wrong in (registry.replace("ab", "12"), registry.replace("cd", "12"), registry.replace("ef", "12"),
                      f"BP-F1 calibration SHA-256: `{HASH}`\n"):
            with self.subTest(registry=wrong), self.assertRaises(ValueError):
                stats.summarize(self.dir, wrong)
        self.assertAlmostEqual(stats.summarize(self.dir, registry)["cases"][0]["ratio"], 1.2)

    def test_device_vmm_blocks_need_the_rule_v2_registration(self):
        self.write_session("device-vmm", HASH, scale=1.05)
        v1 = f"BP-F1 calibration SHA-256: `{HASH}`\nBP-F1 harness SHA-256: `{'cd' * 32}`\n" \
             f"BP-F1 cases SHA-256: `{'ef' * 32}`\n"
        v2 = v1.replace("BP-F1 ", "BP-F1 v2 ")
        with self.assertRaises(ValueError):
            stats.summarize(self.dir)  # no registry
        for wrong in (v1, v2.replace("ab", "12"), v2.replace("cd", "12"), v2.replace("ef", "12")):
            with self.subTest(registry=wrong), self.assertRaises(ValueError):
                stats.summarize(self.dir, wrong)
        self.assertAlmostEqual(stats.summarize(self.dir, v2)["cases"][0]["ratio"], 1.05)


if __name__ == "__main__":
    unittest.main()
