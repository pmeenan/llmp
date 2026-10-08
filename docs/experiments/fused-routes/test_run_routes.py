# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_routes


def saved_comparisons():
    return json.loads((Path(__file__).with_name("results.json")).read_text())["final_design"]["comparisons"]


class Acceptance(unittest.TestCase):
    def test_recorded_controls_pass_with_nonzero_diagnostic_differences(self):
        for comparisons in saved_comparisons().values():
            self.assertGreater(comparisons["legacy_split_vs_untraced"]["logit_hashes"]["different_records"], 0)
            run_routes.require_equal_controls(comparisons)

    def test_each_required_comparison_rejects_a_difference(self):
        paths = [(name, kind, "different_records")
                 for name in ("fusion_boundary_vs_untraced", "fusion_boundary_nofusion_vs_untraced_nofusion")
                 for kind in ("predictions", "logit_hashes")]
        paths += [("untraced_nofusion_vs_reference_untraced_predictions", "different_records")]
        paths += [("boundary_routes_nofusion_vs_reference_legacy_routes", key)
                  for key in ("different_events", "different_ordered_rows", "different_expert_sets")]
        for path in paths:
            with self.subTest(path=path):
                comparisons = copy.deepcopy(saved_comparisons()["A"])
                row = comparisons
                for key in path[:-1]:
                    row = row[key]
                row[path[-1]] = 1
                with self.assertRaisesRegex(ValueError, "Required equality control failed"):
                    run_routes.require_equal_controls(comparisons)

    def test_empty_missing_or_malformed_controls_fail(self):
        for change in ("empty_outputs", "empty_routes", "missing_control", "missing_count", "false_count"):
            with self.subTest(change=change):
                comparisons = copy.deepcopy(saved_comparisons()["A"])
                if change == "empty_outputs":
                    comparisons["fusion_boundary_vs_untraced"]["logit_hashes"]["records"] = 0
                elif change == "empty_routes":
                    comparisons["boundary_routes_nofusion_vs_reference_legacy_routes"]["rows"] = 0
                elif change == "missing_control":
                    del comparisons["fusion_boundary_vs_untraced"]
                elif change == "missing_count":
                    del comparisons["fusion_boundary_vs_untraced"]["predictions"]["different_records"]
                else:
                    comparisons["fusion_boundary_vs_untraced"]["predictions"]["different_records"] = False
                with self.assertRaises((ValueError, KeyError)):
                    run_routes.require_equal_controls(comparisons)

    def run_mocked_capture(self, root, changed_boundary=False):
        """Run the real CLI orchestration/comparers with only hardware calls replaced."""
        dirs = {name: root / name for name in ("models", "source", "inputs", "reference", "output")}
        for name, directory in dirs.items():
            if name != "output":
                directory.mkdir()
        (dirs["models"] / "model.bin").write_bytes(b"model fixture")
        (dirs["inputs"] / "tokens").write_text("token fixture")
        model = {"id": "A", "files": [{"path": "model.bin", "sha256": run_routes.sha(dirs["models"] / "model.bin")}],
                 "tokens": "tokens", "tokens_sha256": run_routes.sha(dirs["inputs"] / "tokens"),
                 "layers": 1, "experts": 2, "topk": 1, "context": 32}
        spec = root / "spec.json"
        spec.write_text(json.dumps({"models": [model]}))
        route = {"event": "routes", "request": 0, "step": 0, "phase": "decode", "layer": 0, "routes": [[1]]}
        reference_runs = []
        for trace in (0, 1):
            name = f"A-b512-r1-s0-t{trace}"
            predictions = dirs["reference"] / (name + ".predictions")
            events = dirs["reference"] / (name + ".jsonl")
            predictions.write_text("0 7\n")
            events.write_text(json.dumps(route) + "\n")
            reference_runs.append({"name": name, "predictions_sha256": run_routes.sha(predictions),
                                   "events_sha256": run_routes.sha(events)})
        (dirs["reference"] / "receipt.json").write_text(json.dumps({
            "runs": reference_runs, "cuda_disable_fusion": True, "cuda_disable_graphs": True}))
        pins = copy.deepcopy(run_routes.PINS)
        pins["workload"].update(spec_sha256=run_routes.sha(spec), models=["A"])
        pins["reference"]["headers"] = {}

        def fake_container(docker, base, tail, **kwargs):
            if "/output/capture" not in tail or "stdout" not in kwargs:
                (dirs["output"] / "capture").write_bytes(b"fake compiled harness")
                return
            event_arg = next(x for x in tail if x.endswith("/events.jsonl"))
            arm = dirs["output"] / Path(event_arg).parent.name
            mode = run_routes.ARMS[arm.name.removeprefix("A-")][0]
            events = {"event": "library", "mode": mode, "libllama": "/app/libllama.so.0"}
            (arm / "events.jsonl").write_text(json.dumps(events) + "\n" + json.dumps(route) + "\n")
            changed = changed_boundary and arm.name == "A-fusion-boundary"
            # Keep argmax unchanged so a token-only check cannot conceal the logit failure.
            (arm / "predictions").write_text("0 7\n")
            (arm / "logits").write_text("0 0 " + ("bb" if changed else "aa") + "\n")
            kwargs["stdout"].write("offloaded 1/1 layers to GPU\n")

        argv = ["run_routes.py", str(spec), *(str(dirs[n]) for n in ("models", "source", "inputs", "reference", "output"))]
        output = io.StringIO()
        old_umask = os.umask(0o077)
        try:
            with patch.object(run_routes, "PINS", pins), patch.object(run_routes, "container", fake_container), \
                 patch.object(run_routes.subprocess, "check_output", return_value="fake GPU"), \
                 patch("sys.argv", argv), contextlib.redirect_stdout(output):
                if changed_boundary:
                    with self.assertRaisesRegex(ValueError, "fusion_boundary_vs_untraced/logit_hashes"):
                        run_routes.main()
                else:
                    run_routes.main()
        finally:
            os.umask(old_umask)
        receipt = json.loads((dirs["output"] / "receipt.json").read_text())
        return output.getvalue().splitlines(), receipt

    def test_main_preserves_failure_receipt_and_never_reports_complete(self):
        with tempfile.TemporaryDirectory() as root:
            messages, receipt = self.run_mocked_capture(Path(root), changed_boundary=True)
            self.assertNotIn("complete", messages)
            comparison = receipt["comparisons"]["A"]["fusion_boundary_vs_untraced"]
            self.assertEqual(comparison["predictions"]["different_records"], 0)
            self.assertEqual(comparison["logit_hashes"]["different_records"], 1)

    def test_main_reports_complete_after_all_controls_pass(self):
        with tempfile.TemporaryDirectory() as root:
            messages, receipt = self.run_mocked_capture(Path(root))
            self.assertEqual(messages[-1], "complete")
            self.assertEqual(len(receipt["arms"]), 5)


if __name__ == "__main__":
    unittest.main()
