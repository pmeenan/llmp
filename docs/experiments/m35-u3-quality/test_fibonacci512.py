#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Supplement request, response, identity and owned-launch controls; no inference."""
import copy
import hashlib
import json
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import fibonacci512 as fib
import native
import reference


def response(finish="stop", count=300):
    return {"choices": [{"text": "def fib(n):\n    return n\n", "finish_reason": finish}],
            "usage": {"prompt_tokens": len(fib.IDS), "completion_tokens": count,
                      "total_tokens": len(fib.IDS) + count,
                      "prompt_tokens_details": {"cached_tokens": 0}}}


def config(root):
    return {"schema_version": 2, "storage": {"installed": str(root / "store"),
                                             "data_dir": str(root / "fresh-state")},
            "models": {"llama33": {
                "artifact": "9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee",
                "context": 12288, "max_slots": 1, "prefill_chunk": 512,
                "kv_cache_encoding": "f16", "temperature": 0.0, "top_p": 1.0, "top_k": 0,
                "tokenizer": str(root / "tokenizer.json"), "chat_template": str(root / "template")}},
            "client": {"bind": "127.0.0.1", "port": 18974}}


class SupplementContract(unittest.TestCase):
    def test_request_only_allowance_changes(self):
        self.assertEqual(len(fib.IDS), 49)
        self.assertEqual(fib.body(), {"model": "llama33", "prompt": fib.IDS,
                                     "max_tokens": 512, "temperature": 0, "top_p": 1,
                                     "add_special_tokens": False, "stream": False})
        args = reference.server_args()
        for key, value in (("--dtype", "float16"), ("--kv-cache-dtype", "float16"),
                           ("--generation-config", "vllm"), ("--max-num-seqs", "1"),
                           ("--max-num-batched-tokens", "512"), ("--max-model-len", "12288")):
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn("--no-enable-prefix-caching", args)

    def test_actual_reference_precision_before_request(self):
        log = ("Resolved architecture: LlamaForCausalLM\n"
               "Initializing a V1 LLM engine (v0.1.dev20073+g8e685d198) "
               "dtype=torch.float16, kv_cache_dtype=float16, quantization=modelopt_fp4, "
               "speculative_config=None, trust_remote_code=False, enable_prefix_caching=False,\n"
               "Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM\n"
               "Using FLASH_ATTN attention backend out of potential backends: "
               "['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].\n"
               "Using FlashAttention version 2\n")
        fib.reference_startup(log)
        for old, new in (("torch.float16", "torch.bfloat16"),
                         ("kv_cache_dtype=float16", "kv_cache_dtype=fp8"),
                         ("modelopt_fp4", "modelopt"),
                         ("NvFp4LinearKernel", "W4A16LinearKernel"),
                         ("version 2", "version 3")):
            with self.subTest(field=old), self.assertRaises(ValueError):
                fib.reference_startup(log.replace(old, new))
        with self.assertRaises(ValueError):
            fib.reference_startup(log + log)
        # Appending a conflicting selection previously passed despite the
        # positive expected markers still being present.
        for extra in ("Resolved architecture: Qwen3ForCausalLM",
                      "Using MarlinLinearKernel for NVFP4 GEMM",
                      "Using HummingLinearKernel for NVFP4 GEMM",
                      "Using W4A16LinearKernel for NVFP4 GEMM",
                      "Using FlashAttention version 3",
                      "Using TritonAttention backend",
                      "Using TRITON_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN'].",
                      "Falling back to Marlin for NVFP4 GEMM"):
            with self.subTest(appended=extra), self.assertRaises(ValueError):
                fib.reference_startup(log + extra + "\n")

    def test_original_hash_and_override_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                fib.inputs(path)
            with self.assertRaises(ValueError):
                fib.inputs(path, fib.sha(path))

    def test_finish_is_not_semantic_pass(self):
        for reason, count, status in (("stop", 300, "complete_finish"),
                                      ("length", 512, "truncated")):
            got = fib.validate_response(response(reason, count))
            self.assertEqual(got["completion_status"], status)
            self.assertEqual(got["semantic_verdict"], "pending_independent_review")

    def test_invalid_usage_and_missing_finish_refused(self):
        alterations = [
            lambda x: x["choices"][0].pop("finish_reason"),
            lambda x: x["choices"][0].update(finish_reason="content_filter"),
            lambda x: x["choices"][0].update(text=""),
            lambda x: x["choices"].append(x["choices"][0]),
            lambda x: x["usage"].update(prompt_tokens=48),
            lambda x: x["usage"].update(completion_tokens=513),
            lambda x: x["usage"].update(completion_tokens=True),
            lambda x: x["usage"].update(total_tokens=0),
            lambda x: x["usage"]["prompt_tokens_details"].update(cached_tokens=49),
            lambda x: x["choices"][0].update(finish_reason="length"),
        ]
        for change in alterations:
            obj = response()
            change(obj)
            with self.subTest(response=obj), self.assertRaises(ValueError):
                fib.validate_response(obj)

    def test_failed_response_preserved_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fib, "inputs"), \
                mock.patch.object(fib.urllib.request, "urlopen") as send:
            raw = b'{"choices": []}'
            send.return_value.__enter__.return_value.read.return_value = raw
            out = Path(tmp) / "new"
            with self.assertRaises(ValueError):
                fib.collect("http://127.0.0.1", Path("inputs"), out)
            self.assertEqual((out / "response.json").read_bytes(), raw)
            self.assertFalse(json.loads((out / "receipt.json").read_text())["collection_complete"])
            self.assertEqual(send.call_count, 1)
            with self.assertRaises(FileExistsError):
                fib.collect("http://127.0.0.1", Path("inputs"), out)
            self.assertEqual(send.call_count, 1)

    def test_native_recipe_and_fresh_state(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fib, "require_sha"), \
                mock.patch.object(fib, "source_metadata"):
            root = Path(tmp)
            original = config(root)
            fib.native_preflight(root / "runtime", original, root)
            for key, value in (("drafter", "other"), ("kv_cache_encoding", "bf16"),
                               ("temperature", 0.5), ("top_k", 40)):
                altered = copy.deepcopy(original)
                altered["models"]["llama33"][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    fib.native_preflight(root / "runtime", altered, root)
            (root / "fresh-state").mkdir()
            with self.assertRaises(ValueError):
                fib.native_preflight(root / "runtime", original, root)

    def test_configured_tokenizer_cannot_hide_behind_valid_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "tokenizer.json"
            valid.write_bytes(b"authenticated tokenizer")
            different = root / "other-tokenizer.json"
            different.write_bytes(b"wrong tokenizer")
            original_require = fib.require_sha
            def require(path, expected):
                if Path(path) in {valid, different}:
                    return original_require(path, expected)
                return expected
            with mock.patch.object(fib, "METADATA", {"tokenizer.json": fib.sha(valid)}), \
                    mock.patch.object(fib, "require_sha", side_effect=require):
                good = config(root)
                fib.native_preflight(root / "runtime", good, root)
                good["models"]["llama33"]["tokenizer"] = str(different)
                with self.assertRaisesRegex(ValueError, "registered identity differs"):
                    fib.native_preflight(root / "runtime", good, root)

    def test_native_main_single_request_and_retirement_on_collection_error(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runtime = root / "runtime"
                runtime.write_bytes(b"fixture executable")
                conf = root / "config"
                conf.write_bytes(b"fixture config")
                service = mock.Mock()
                service.stop.return_value = {"signal": "TERM", "returncode": 0}
                factory = mock.Mock(return_value=service)
                argv = ["native.py", "--repo", str(root), "--runtime", str(runtime),
                        "--config", str(conf), "--out", str(root / "out"),
                        "--phase", "fibonacci512", "--behavior-inputs", "frozen",
                        "--behavior-sha256", fib.INPUT_SHA]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.dict(sys.modules, {"speed_gate": types.SimpleNamespace(Service=factory)}), \
                        mock.patch.object(native.tomllib, "loads", return_value=config(root)), \
                        mock.patch.object(native.signal, "signal"), \
                        mock.patch.object(native.subprocess, "check_output", return_value=b"libc => /lib/libc.so"), \
                        mock.patch.object(fib, "inputs"), mock.patch.object(fib, "native_preflight"), \
                        mock.patch.object(fib, "source_metadata", return_value={}), \
                        mock.patch.object(fib, "RUNTIME_SHA", fib.sha(runtime)), \
                        mock.patch.object(fib, "collect", side_effect=ValueError("bad reply") if fail else None) as collect:
                    if fail:
                        with self.assertRaisesRegex(ValueError, "bad reply"):
                            native.main()
                    else:
                        native.main()
                    collect.assert_called_once_with("http://127.0.0.1:18974", Path("frozen"), root / "out/fibonacci512")
                    service.wait_ready.assert_called_once_with(timeout=420)
                    service.stop.assert_called_once_with(timeout=20)
                    self.assertEqual(factory.call_args.args[-1], {})
                    receipt = json.loads((root / "out/launch.json").read_text())
                    self.assertEqual(receipt["runtime_sha256"], receipt["runtime_sha256_after"])
                    self.assertEqual((root / "out/launch.json").stat().st_mode & 0o777, 0o600)

    def test_reference_single_request_and_owned_cleanup_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_bytes(b"fixture")
            (checkpoint / "llmp-archive.json").write_text(json.dumps({
                "repo": "nvidia/Llama-3.3-70B-Instruct-NVFP4",
                "commit": "f792b707002f2c613e0729c6b37bbcedc4efa94e"}))
            pins = root / "docs/experiments/fast-swap/mia-launch.json"
            pins.parent.mkdir(parents=True)
            pins.write_text(json.dumps({"image": reference.IMAGE, "loader_files": {}}))
            argv = ["reference.py", "--repo", str(root), "--checkpoint", str(checkpoint),
                    "--out", str(root / "out"), "--phase", "fibonacci512",
                    "--behavior-inputs", "frozen", "--behavior-sha256", fib.INPUT_SHA]
            def run(command, **_kwargs):
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            with mock.patch.object(sys, "argv", argv), mock.patch.object(reference.Path, "home", return_value=root), \
                    mock.patch.object(reference, "CONFIG_SHA256", hashlib.sha256(b"fixture").hexdigest()), \
                    mock.patch.object(reference.signal, "signal"), \
                    mock.patch.object(reference.socket, "create_connection", side_effect=OSError), \
                    mock.patch.object(reference.subprocess, "run", side_effect=run) as process, \
                    mock.patch.object(reference.subprocess, "check_output", return_value=b"[]"), \
                    mock.patch.object(reference.urllib.request, "urlopen") as health, \
                    mock.patch.object(fib, "inputs"), mock.patch.object(fib, "source_metadata", return_value={}), \
                    mock.patch.object(fib, "require_sha"), \
                    mock.patch.object(fib, "reference_startup", return_value={}), \
                    mock.patch.object(fib, "collect", side_effect=ValueError("bad reply")) as collect:
                health.return_value.__enter__.return_value.status = 200
                with self.assertRaisesRegex(ValueError, "bad reply"):
                    reference.main()
                collect.assert_called_once()
                self.assertTrue(any("rm" in call.args[0] for call in process.call_args_list))
                receipt = json.loads((root / "out/retirement.json").read_text())
                self.assertTrue(receipt["absent"])
                self.assertEqual(receipt["remove"]["returncode"], 0)


class SupplementRetirement(unittest.TestCase):
    def result(self, returncode=0, stdout="", stderr=""):
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def test_repeated_signals_cannot_interrupt_cleanup(self):
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            for sig in before:
                signal.raise_signal(sig)
            self.assertLessEqual(kwargs["timeout"], 5)
            return self.result()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(reference.subprocess, "run", side_effect=run):
            out = Path(tmp)
            with self.assertRaises(SystemExit) as caught:
                with reference.supplement_owner("owned-test", out):
                    signal.raise_signal(signal.SIGTERM)
            self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
            self.assertEqual(len(commands), 3)
            self.assertIn("name=^/owned-test$", commands[-1])
            receipt = json.loads((out / "retirement.json").read_text())
            self.assertTrue(receipt["absent"])
            self.assertIn("SystemExit", receipt["operation_failure"]["error"])
        self.assertEqual({sig: signal.getsignal(sig) for sig in before}, before)

    def test_positive_absence_and_conflicting_daemon_results(self):
        for query in (self.result(), self.result(1, stderr="daemon unavailable"),
                      self.result(0, stdout="still-owned-container\n")):
            with self.subTest(query=query), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(reference.subprocess, "run",
                                      side_effect=[self.result(), self.result(), query]):
                out = Path(tmp)
                should_pass = query.returncode == 0 and not query.stdout
                if should_pass:
                    with reference.supplement_owner("owned-test", out):
                        pass
                else:
                    with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                        with reference.supplement_owner("owned-test", out):
                            pass
                receipt = json.loads((out / "retirement.json").read_text())
                self.assertEqual(receipt["absent"], should_pass)
                self.assertEqual(receipt["absence_check"]["stderr"], query.stderr)

    def test_removal_timeout_still_verifies_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(reference.subprocess, "run", side_effect=[
                    self.result(), subprocess.TimeoutExpired("docker rm", 5, stderr=b"daemon slow"),
                    self.result()]) as process:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                with reference.supplement_owner("owned-test", out):
                    pass
            self.assertEqual(process.call_count, 3)
            receipt = json.loads((out / "retirement.json").read_text())
            self.assertTrue(receipt["absent"])
            self.assertIn("TimeoutExpired", receipt["remove"]["error"])
            self.assertEqual(receipt["remove"]["stderr"], "daemon slow")

    def test_log_open_failure_still_removes_and_verifies(self):
        original_open = Path.open
        def open_file(path, *args, **kwargs):
            if path.name == "container.log":
                raise OSError("disk full")
            return original_open(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Path, "open", open_file), \
                mock.patch.object(reference.subprocess, "run", return_value=self.result()) as process:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                with reference.supplement_owner("owned-test", out):
                    pass
            self.assertEqual(process.call_count, 2)
            receipt = json.loads((out / "retirement.json").read_text())
            self.assertTrue(receipt["absent"])
            self.assertIn("disk full", receipt["logs"]["error"])

    def test_absence_timeout_remains_unconfirmed(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(reference.subprocess, "run", side_effect=[
                    self.result(), self.result(), subprocess.TimeoutExpired("docker ps", 1)]):
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                with reference.supplement_owner("owned-test", out):
                    pass
            receipt = json.loads((out / "retirement.json").read_text())
            self.assertFalse(receipt["absent"])
            self.assertIn("TimeoutExpired", receipt["absence_check"]["error"])


if __name__ == "__main__":
    unittest.main()
