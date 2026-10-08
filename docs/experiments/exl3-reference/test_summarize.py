# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete, mislabeled and internally inconsistent reference bundles."""

import copy
import io
import unittest

import numpy as np

from checks import summary
from summarize import numerical_detail, validate_kernels, validate_logits, validate_model


FIXTURE = {"repository": "owner/model-4.0bpw", "revision": "revision",
           "published_sha256": "0" * 64, "per_tensor_bits": {"4.0": 1, "8.0": 1},
           "reference_expectations": {"vocab_size": 2,
                                      "real_kernel_shapes": {"projection": {"k": 128, "n": 128, "K": 4}}}}
PROTOCOL_HASH, RUNNER_HASH = "1" * 64, "2" * 64
PROTOCOL = {"performance_blocks": 2, "prefill_tokens": [2, 3], "decode_prefix_tokens": 2,
            "trials_per_block": 3, "decode_tokens": 2, "numerical_prefix_tokens": [2, 3],
            "numerical_suffix_tokens": 2, "numerical_repeats": 2, "full_prefill_logits_up_to": 2,
            "kernel_rows": [1, 145, 1024], "kernel_real_keys": ["projection"],
            "kernel_synthetic_shapes": [[128, 128]], "kernel_synthetic_rates": [4],
            "kernel_repeats": 3, "kernel_replays_per_sample": 10, "seed": 20260922}


def manifest(mode):
    return {"fixture": FIXTURE["repository"], "revision": FIXTURE["revision"],
            "weight_sha256": FIXTURE["published_sha256"], "profile": "optimized", "mode": mode,
            "protocol_sha256": PROTOCOL_HASH, "harness_sha256": RUNNER_HASH,
            "numerical_passed": True, "kernel_repeat_passed": True}


def exact(positions, width=2):
    return {"finite": True, "exact": True, "elements": positions * width,
            "max_abs": 0.0, "rms": 0.0, "top1_equal": positions, "positions": positions}


def model_bundle():
    rows, numerical = [], []
    for block in range(2):
        for prefix in PROTOCOL["prefill_tokens"]:
            times, last = [1., 2., 3.], [1., 1., 1.]
            ttft = [a + b for a, b in zip(times, last)]
            rows.append({"kind": "prefill", "prefix_tokens": prefix, "block": block,
                         "prefill_ms": times, "last_prompt_token_ms": last,
                         "ttft_proxy_ms": ttft, "gpu_event_prefill_ms": times,
                         "summary": {"prefill_ms": summary(times), "ttft_proxy_ms": summary(ttft),
                                     "gpu_event_prefill_ms": summary(times)}})
        times = [1., 1., 2., 2., 3., 3.]
        rows.append({"kind": "decode", "prefix_tokens": 2, "block": block, "tokens_per_trial": 2,
                     "token_ms": times, "gpu_event_ms": times, "request_ms": [2., 4., 6.],
                     "summary": summary(times, np.asarray(times).reshape(3, 2)),
                     "request_summary": summary([2., 4., 6.])})
    for prefix in PROTOCOL["numerical_prefix_tokens"]:
        comparisons = [{"kind": "suffix_repeat", **exact(2)} for _ in range(2)]
        comparisons += [{"kind": "suffix_restore", **exact(2)}]
        if prefix == 2:
            comparisons += [{"kind": "prefill_repeat", **exact(2)} for _ in range(2)]
        numerical.append({"prefix": prefix, "suffix_tokens": 2, "passed": True,
                          "poison_verified": True, "restored_storage_exact": True,
                          "snapshot_bytes": 32, "snapshot_sha256": ["3" * 64],
                          "comparisons": comparisons})
    return manifest("model"), rows, numerical


SYMBOLS = {"packed": ["void exl3_gemv_kernel<4, false, 1, 0, 0, false, false>(__half const*)"],
           "reconstruct": ["void reconstruct_kernel<4, 1, false>(__half*)",
                           "nvjet_sm121_hsh_mma_64x80x64_5_16x80x64_tmaAB_bz_NNNN"],
           "fused_reconstruct": ["void reconstruct_had_kernel<4, 1, false>(__half*)",
                                 "nvjet_sm121_hsh_mma_64x80x64_5_16x80x64_tmaAB_bz_NNNN"]}


def profile(path, symbols=None):
    names = SYMBOLS[path] if symbols is None else symbols
    return {"active_device_us": 1., "profiled_invocations": 10,
            "events": {name: {"calls": 10, "duration_us": 10. / len(names)} for name in names}}


def kernel_bundle(synthetic=False):
    name = "synthetic-128-128-K4" if synthetic else "projection"
    rows = []
    for count in PROTOCOL["kernel_rows"]:
        path = "packed" if count <= 144 else ("reconstruct" if count < 1024 else "fused_reconstruct")
        rows.append({"name": name, "k": 128, "n": 128, "K": 4, "rows": count, "path": path,
                     "repeat": exact(count, 128), "capture_check": exact(count, 128),
                     "versus_reconstruction": exact(count, 128), "graph_replay_us": [1., 2., 3.],
                     "summary": summary([1., 2., 3.]), "profile": profile(path)})
    return manifest("synthetic" if synthetic else "kernels"), rows


def check_model(bundle):
    validate_model(*bundle, FIXTURE, "optimized", PROTOCOL, PROTOCOL_HASH, RUNNER_HASH)


def check_kernels(bundle, synthetic=False):
    validate_kernels(*bundle, FIXTURE, "synthetic" if synthetic else "4.0bpw",
                     PROTOCOL, PROTOCOL_HASH, RUNNER_HASH)


class BundleValidation(unittest.TestCase):
    def test_complete_bundles_pass(self):
        check_model(model_bundle())
        check_kernels(kernel_bundle())
        check_kernels(kernel_bundle(True), True)

    def test_incomplete_or_duplicate_model_matrix_fails(self):
        for target in (1, 2):
            for operation in ("empty", "missing", "duplicate"):
                with self.subTest(target=target, operation=operation):
                    bundle = model_bundle()
                    rows = bundle[target]
                    if operation == "empty": rows.clear()
                    elif operation == "missing": rows.pop()
                    else: rows.append(copy.deepcopy(rows[0]))
                    with self.assertRaises(ValueError): check_model(bundle)

    def test_manifest_identity_fields_are_required(self):
        for key in ("fixture", "revision", "weight_sha256", "profile", "mode", "protocol_sha256", "harness_sha256"):
            with self.subTest(key=key):
                bundle = model_bundle()
                bundle[0][key] = "different"
                with self.assertRaises(ValueError): check_model(bundle)

    def test_sample_counts_and_values_are_checked(self):
        for values in ([], [1., 2.], [1., 2., float("nan")], [1., 2., 0.]):
            bundle = model_bundle()
            bundle[1][0]["prefill_ms"] = values
            with self.assertRaises(ValueError): check_model(bundle)

    def test_stored_summary_cannot_override_raw_samples(self):
        bundle = model_bundle()
        bundle[1][0]["summary"]["prefill_ms"]["median"] = .1
        with self.assertRaises(ValueError): check_model(bundle)
        kernels = kernel_bundle()
        kernels[1][0]["summary"]["median_ci95"] = [0., .1]
        with self.assertRaises(ValueError): check_kernels(kernels)

    def test_request_and_ttft_components_are_checked(self):
        bundle = model_bundle()
        bundle[1][0]["ttft_proxy_ms"][0] += 1
        with self.assertRaises(ValueError): check_model(bundle)
        bundle = model_bundle()
        bundle[1][2]["request_ms"][0] += 1
        with self.assertRaises(ValueError): check_model(bundle)

    def test_pass_boolean_cannot_hide_bad_numerical_detail(self):
        for key, value in (("finite", False), ("exact", False), ("rms", 1.), ("positions", 1)):
            with self.subTest(key=key):
                bundle = model_bundle()
                bundle[2][0]["comparisons"][0][key] = value
                with self.assertRaises(ValueError): check_model(bundle)
        bundle = model_bundle()
        bundle[2][0]["comparisons"].pop()
        with self.assertRaises(ValueError): check_model(bundle)
        bundle = model_bundle()
        bundle[2][0]["restored_storage_exact"] = False
        with self.assertRaises(ValueError): check_model(bundle)

    def test_incomplete_or_misidentified_kernel_matrix_fails(self):
        for change in ("empty", "missing", "duplicate", "shape", "rate", "rows", "profile", "nonfinite", "capture"):
            with self.subTest(change=change):
                bundle = kernel_bundle(True)
                if change == "empty": bundle[1].clear()
                elif change == "missing": bundle[1].pop()
                elif change == "duplicate": bundle[1].append(copy.deepcopy(bundle[1][0]))
                elif change == "shape": bundle[1][0]["k"] = 256
                elif change == "rate": bundle[1][0]["K"] = 5
                elif change == "rows": bundle[1][0]["rows"] = 2
                elif change == "profile": bundle[0]["profile"] = "attention_eager"
                elif change == "nonfinite": bundle[1][0]["repeat"]["finite"] = False
                else: bundle[1][0]["capture_check"]["exact"] = False
                with self.assertRaises(ValueError): check_kernels(bundle, True)

    def test_profiler_evidence_is_required(self):
        bundle = kernel_bundle()
        bundle[1][0]["profile"]["events"] = {}
        with self.assertRaises(ValueError): check_kernels(bundle)
        bundle = kernel_bundle()
        bundle[1][0]["profile"]["active_device_us"] = 2.
        with self.assertRaises(ValueError): check_kernels(bundle)

    def test_profiled_symbols_must_confirm_plan_and_rate(self):
        cases = {"declared packed, reconstruction ran": (0, SYMBOLS["reconstruct"]),
                 "declared reconstruct, fused ran": (1, SYMBOLS["fused_reconstruct"]),
                 "mixed plans": (0, SYMBOLS["packed"] + SYMBOLS["reconstruct"]),
                 "no EXL3 kernel": (0, ["nvjet_sm121_hsh_mma_64x80x64_5_16x80x64_tmaAB_bz_NNNN"]),
                 "wrong rate": (0, ["void exl3_gemv_kernel<5, false, 1, 0, 0, false, false>(__half const*)"])}
        for label, (index, symbols) in cases.items():
            with self.subTest(label=label):
                bundle = kernel_bundle()
                bundle[1][index]["profile"] = profile(bundle[1][index]["path"], symbols)
                with self.assertRaises(ValueError): check_kernels(bundle)

    def test_consistently_wrong_real_projection_is_rejected(self):
        for change in ("k", "n", "K"):
            with self.subTest(change=change):
                bundle = kernel_bundle()
                for row in bundle[1]:
                    row[change] *= 2
                    if change == "n":
                        for name in ("repeat", "capture_check", "versus_reconstruction"):
                            row[name]["elements"] = row["rows"] * row["n"]
                with self.assertRaises(ValueError): check_kernels(bundle)

    def test_model_comparisons_require_complete_vocabulary(self):
        bundle = model_bundle()
        for row in bundle[2]:
            for check in row["comparisons"]:
                check["elements"] = check["positions"]
        with self.assertRaises(ValueError): check_model(bundle)

    def test_cross_plan_drift_is_allowed_but_invalid_diagnostics_fail(self):
        drift = {**exact(2), "exact": False, "max_abs": .25, "rms": .125, "top1_equal": 1}
        numerical_detail(drift, 2, 4, exact=False)
        for key, value in (("finite", False), ("exact", True), ("exact", "false"), ("rms", .5)):
            with self.subTest(key=key, value=value):
                changed = {**drift, key: value}
                with self.assertRaises(ValueError): numerical_detail(changed, 2, 4, exact=False)

    def test_raw_logit_shape_and_input_are_checked(self):
        ids = np.array([[1000 + ((i * 37 + PROTOCOL["seed"]) % 29000) for i in range(4)]], dtype=np.int64)
        arrays = {"ids": ids, "suffix": np.ones((1, 2, 2), dtype=np.float32),
                  "prefill": np.ones((1, 2, 2), dtype=np.float32)}
        for change in (None, "ids", "shape", "finite", "missing", "vocabulary"):
            with self.subTest(change=change):
                sample = copy.deepcopy(arrays)
                if change == "ids": sample["ids"][0, 0] += 1
                elif change == "shape": sample["suffix"] = np.ones((1, 1, 2), dtype=np.float32)
                elif change == "finite": sample["suffix"][0, 0, 0] = np.nan
                elif change == "missing": del sample["prefill"]
                elif change == "vocabulary":
                    sample["suffix"] = np.ones((1, 2, 3), dtype=np.float32)
                    sample["prefill"] = np.ones((1, 2, 3), dtype=np.float32)
                detail = model_bundle()[2][0]
                if change == "vocabulary":
                    for check in detail["comparisons"]:
                        check["elements"] = check["positions"] * 3
                stream = io.BytesIO()
                np.savez(stream, **sample)
                stream.seek(0)
                with np.load(stream, allow_pickle=False) as bundle:
                    if change is None:
                        validate_logits(bundle, 2, PROTOCOL, detail, FIXTURE)
                    else:
                        with self.assertRaises(ValueError):
                            validate_logits(bundle, 2, PROTOCOL, detail, FIXTURE)


if __name__ == "__main__":
    unittest.main()
