# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The model archive tools (docs/model-archive.md) on synthetic repositories, no network.

A fake Hub serves in-memory files with their sizes, SHA-256 and Git blob IDs;
a temporary directory stands in for the NAS. Covers download, verification,
resume and adoption, a live writer's claim, concurrent writers of one
revision, catalog compaction under concurrent writers, relations, header
detection, staging, migration by move, the Hub client's token handling
against a loopback server, and that nothing the tools write is a link.
"""

import contextlib
import errno
import fcntl
import hashlib
import http.server
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import signal
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import unittest.mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import llmp_archive as arc  # noqa: E402
import llmp_archive_detect as detect  # noqa: E402
import llmp_hub as hublib  # noqa: E402


def load_tool(name):
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(TOOLS / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


archive_model = load_tool("archive-model")
archive_catalog = load_tool("archive-catalog")
stage_model = load_tool("stage-model")
REAL_SLEEP = arc.sleep
arc.sleep = lambda seconds: time.sleep(min(seconds, 0.01))
archive_model.WAIT_POLL = 0.05

COMMIT = "a" * 40
TOKEN = "hf_secretSECRETsecret123"


def gguf_bytes(file_type=15, quantized_by="Unsloth"):
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [(s("general.architecture") + struct.pack("<I", 8) + s("qwen3")),
          (s("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type)),
          (s("general.quantized_by") + struct.pack("<I", 8) + s(quantized_by)),
          (s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 3) + s("a") + s("b") + s("c"))]
    tensors = [(s("blk.0.w") + struct.pack("<I", 2) + struct.pack("<QQ", 256, 4) + struct.pack("<IQ", 12, 0)),
               (s("norm") + struct.pack("<I", 1) + struct.pack("<Q", 32) + struct.pack("<IQ", 0, 576))]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv) + b"".join(tensors)
    header += b"\0" * (-len(header) % 32)
    return header + bytes(range(256)) * 2 + b"\x11" * 64 + b"\x22" * 128  # 576 + 128 bytes of data


def safetensors_bytes(tensors):
    header, data = {}, b""
    for name, dtype, shape, nbytes in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + nbytes]}
        data += b"\x01" * nbytes
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + data


def git_oid(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, data, start, fail_after=None):
        super().__init__(data[start:] if fail_after is None else data[start:fail_after])
        self.start = start


class FakeHub:
    """Serves `files` (path -> bytes) for one repository; `lfs` marks LFS paths."""

    def __init__(self, repo, files, lfs=(), card=None, tags=(), commit=COMMIT, corrupt=()):
        self.repo, self.files, self.lfs, self.commit = repo, files, set(lfs), commit
        self.card, self.tags, self.corrupt = card or {}, list(tags), set(corrupt)
        self.requests, self.info_calls = [], 0
        self.lock = threading.Lock()

    def scrub(self, text):
        return text

    def revision_info(self, repo, revision):
        self.info_calls += 1
        siblings = []
        for path, data in sorted(self.files.items()):
            entry = {"rfilename": path, "size": len(data), "blobId": git_oid(data)}
            if path in self.lfs:
                entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "pointerSize": 130}
            siblings.append(entry)
        return {"id": self.repo, "sha": self.commit, "siblings": siblings, "cardData": self.card,
                "tags": self.tags, "gated": False, "private": False, "lastModified": "2026-10-01T00:00:00.000Z"}

    def open_file(self, repo, commit, path, start=0):
        with self.lock:
            self.requests.append((path, start))
        data = self.files[path]
        if path in self.corrupt:
            data = bytes([data[0] ^ 1]) + data[1:]
        return FakeResponse(data, start)


class ArchiveCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.root = self.tmp / "nas"
        self.out = io.StringIO()

    def tearDown(self):
        self.assert_no_links(self.tmp)
        self._tmp.cleanup()

    def assert_no_links(self, top):
        for dirpath, dirnames, filenames in os.walk(top):
            for name in dirnames + filenames:
                self.assertFalse(os.path.islink(os.path.join(dirpath, name)), f"a link: {dirpath}/{name}")

    def archive(self, hub, *args, init=True):
        flags = ["--init-root"] if init else []
        with contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.out):
            return archive_model.main([*args, *flags, "--archive", str(self.root)], hub=hub)


def gguf_repo():
    files = {"README.md": b"# card\n", "config.json": b"{}", ".gitattributes": b"*.gguf filter=lfs\n",
             "M-Q4_K_M.gguf": gguf_bytes(15), "M-Q8_0.gguf": gguf_bytes(7),
             "UD-Q2_K_XL/M-UD-Q2_K_XL-00001-of-00002.gguf": gguf_bytes(10),
             "UD-Q2_K_XL/M-UD-Q2_K_XL-00002-of-00002.gguf": gguf_bytes(10),
             "mmproj-F16.gguf": gguf_bytes(1), "weights.bin": b"pickle"}
    lfs = [p for p in files if p.endswith((".gguf", ".bin"))]
    return FakeHub("unsloth/M-GGUF", files, lfs, card={"base_model": "org/M", "license": "apache-2.0"},
                   tags=["base_model:quantized:org/M", "gguf"])


class DownloadTest(ArchiveCase):
    def test_archives_a_variant_with_verified_files_and_a_manifest(self):
        hub = gguf_repo()
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0, self.out.getvalue())
        rev = self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"
        m = arc.read_manifest(rev)
        present = sorted(f["path"] for f in m["files"] if f["present"])
        self.assertEqual(present, [".gitattributes", "M-Q4_K_M.gguf", "README.md", "config.json"])
        self.assertFalse((rev / "weights.bin").exists())  # pickles only when named
        for f in m["files"]:
            if f["present"]:
                self.assertEqual(hashlib.sha256((rev / f["path"]).read_bytes()).hexdigest(), f["sha256"])
        self.assertEqual(m["hub"]["base_models"], [{"repo": "org/M", "relation": "quantized", "source": "tags"}])
        self.assertEqual(m["hub"]["license"], "apache-2.0")
        variant = next(v for v in m["variants"] if v["id"] == "M-Q4_K_M")
        self.assertEqual(variant["detected"]["scheme"], "gguf:Q4_K_M")
        self.assertEqual(variant["detected"]["from"], "files")
        self.assertEqual(variant["detected"]["architecture"], "qwen3")
        self.assertEqual(variant["detected"]["quantizer"], "Unsloth")
        self.assertEqual(variant["detected"]["encodings"], {"F32": 128, "Q4_K": 576})
        self.assertEqual(variant["detected"]["parameters"], 1056)
        split = next(v for v in m["variants"] if v["id"] == "UD-Q2_K_XL/M-UD-Q2_K_XL")
        self.assertEqual((split["present"], split["detected"]["from"], split["detected"]["name_hint"]),
                         (False, "names", "UD-Q2_K_XL"))
        self.assertEqual(next(v for v in m["variants"] if v["id"] == "mmproj-F16")["role"], "projector")
        self.assertEqual(list(rev.glob(f"*{arc.PARTIAL}*")), [])
        catalog = json.loads((self.root / "catalog/catalog.json").read_text())
        self.assertEqual(catalog["repositories"][0]["model"], "org/M")
        self.assertIn("M-Q4_K_M", (self.root / "catalog/catalog.md").read_text())

        # Already archived: no Hub call, no download.
        hub2 = gguf_repo()
        self.assertEqual(self.archive(hub2, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0)
        self.assertEqual((hub2.info_calls, hub2.requests), (0, []))

        # A second variant joins the same revision.
        self.assertEqual(self.archive(hub2, "unsloth/M-GGUF", "--include", "UD-Q2_K_XL/*"), 0, self.out.getvalue())
        m = arc.read_manifest(rev)
        self.assertTrue(next(v for v in m["variants"] if v["id"] == "UD-Q2_K_XL/M-UD-Q2_K_XL")["present"] is True)
        self.assertTrue(next(f for f in m["files"] if f["path"] == "M-Q4_K_M.gguf")["present"])

    def test_several_gguf_variants_need_a_choice(self):
        self.assertEqual(self.archive(gguf_repo(), "unsloth/M-GGUF"), 2)
        self.assertIn("--all-variants", self.out.getvalue())
        self.assertFalse((self.root / "hf").exists())

    def test_a_hash_mismatch_leaves_nothing(self):
        hub = gguf_repo()
        hub.corrupt = {"M-Q4_K_M.gguf"}
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF@main", "--include", "M-Q4_K_M.gguf"), 1)
        self.assertIn("SHA-256", self.out.getvalue())
        rev = self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"
        self.assertFalse((rev / "M-Q4_K_M.gguf").exists())
        self.assertEqual(list(rev.glob(f"*{arc.PARTIAL}*")), [])
        m = arc.read_manifest(rev)  # what verified is listed; the bad file never is
        self.assertFalse(next(f for f in m["files"] if f["path"] == "M-Q4_K_M.gguf")["present"])

    def test_a_small_file_is_checked_by_its_git_blob_id(self):
        hub = gguf_repo()
        hub.corrupt = {"config.json"}
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 1)
        self.assertIn("Git blob ID", self.out.getvalue())

    def test_a_stale_partial_is_adopted_and_resumed_by_range(self):
        hub = gguf_repo()
        data = hub.files["M-Q4_K_M.gguf"]
        rev = self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"
        arc.open_root(self.root, create=True)
        rev.mkdir(parents=True)
        partial = rev / f"M-Q4_K_M.gguf{arc.PARTIAL}otherhost.1.dead"
        partial.write_bytes(data[:300])
        old = time.time() - arc.STALE_SECONDS - 10
        os.utime(partial, (old, old))
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0, self.out.getvalue())
        self.assertIn(("M-Q4_K_M.gguf", 300), hub.requests)
        self.assertEqual((rev / "M-Q4_K_M.gguf").read_bytes(), data)
        self.assertFalse(partial.exists())

    def test_a_recent_abandoned_partial_is_resumed_under_the_writer_lock(self):
        hub = gguf_repo()
        rev = self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"
        arc.open_root(self.root, create=True)
        rev.mkdir(parents=True)
        data = hub.files["M-Q4_K_M.gguf"]
        (rev / f"M-Q4_K_M.gguf{arc.PARTIAL}otherhost.1.dead").write_bytes(data[:300])
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf", "--no-wait"), 0)
        self.assertIn(("M-Q4_K_M.gguf", 300), hub.requests)
        self.assertEqual((rev / "M-Q4_K_M.gguf").read_bytes(), data)

    def test_no_wait_does_not_contact_the_hub_or_write_a_revision_when_locked(self):
        hub = gguf_repo()
        arc.open_root(self.root, create=True)
        with arc.writer_lock(self.root):
            self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--no-wait"), 3)
        self.assertEqual((hub.info_calls, hub.requests), (0, []))
        self.assertFalse((self.root / "hf").exists())

    def test_a_cut_off_transfer_resumes(self):
        hub = gguf_repo()
        data = hub.files["M-Q4_K_M.gguf"]
        calls = []

        def flaky(repo, commit, path, start=0):
            calls.append(start)
            return FakeResponse(data, start, fail_after=200 if len(calls) == 1 else None)
        hub.open_file = flaky
        entry = {"path": "M-Q4_K_M.gguf", "size": len(data), "lfs": True,
                 "sha256": hashlib.sha256(data).hexdigest(), "git_oid": None}
        rev = self.tmp / "rev"
        rev.mkdir()
        status, digest = arc.transfer(rev, entry, lambda s: hub.open_file("r", COMMIT, entry["path"], s), "w.1.x")
        self.assertEqual((status, calls), ("done", [0, 200]))
        self.assertEqual((rev / "M-Q4_K_M.gguf").read_bytes(), data)

    def test_concurrent_writers_of_one_revision_converge(self):
        hub = gguf_repo()
        results = []

        def go(pattern):
            results.append(archive_model.main(["unsloth/M-GGUF@main", "--include", pattern, "--archive",
                                               str(self.root), "--no-compact"], hub=hub))
        arc.open_root(self.root, create=True)
        with contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.out):
            threads = [threading.Thread(target=go, args=(p,)) for p in ("M-Q4_K_M.gguf", "M-Q8_0.gguf", "UD-*")]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(results, [0, 0, 0], self.out.getvalue())
        m = arc.read_manifest(self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}")
        present = {f["path"] for f in m["files"] if f["present"]}
        self.assertTrue({"M-Q4_K_M.gguf", "M-Q8_0.gguf", "UD-Q2_K_XL/M-UD-Q2_K_XL-00002-of-00002.gguf"} <= present)

    def test_refuses_paths_the_share_cannot_hold(self):
        hub = FakeHub("o/r", {"a:b.json": b"{}", "model.safetensors": safetensors_bytes([("w", "BF16", [2], 4)])})
        self.assertEqual(self.archive(hub, "o/r"), 1)
        self.assertIn("Windows-reserved", self.out.getvalue())
        hub = FakeHub("o/r", {"Config.json": b"{}", "config.json": b"{}"})
        self.assertEqual(self.archive(hub, "o/r"), 1)
        self.assertIn("differ only in case", self.out.getvalue())

    def test_migrates_a_local_copy_by_move(self):
        st = safetensors_bytes([("layer.weight", "U32", [4, 2], 32), ("layer.scales", "BF16", [4, 1], 8)])
        files = {"config.json": json.dumps({"model_type": "qwen3",
                                            "quantization": {"bits": 4, "group_size": 32}}).encode(),
                 "model.safetensors": st}
        hub = FakeHub("Vontra/X-MLX-4bit", files, lfs=["model.safetensors"])
        old = self.root / "huggingface-mlx" / f"Vontra--X-MLX-4bit@{COMMIT}"
        arc.open_root(self.root, create=True)
        old.mkdir(parents=True)
        for path, data in files.items():
            (old / path).write_bytes(data)
        (old / "SOURCE.txt").write_text("huggingface.co/Vontra/X-MLX-4bit\n")
        self.assertEqual(self.archive(hub, f"Vontra/X-MLX-4bit@{COMMIT}", "--from", str(old), "--move"), 0,
                         self.out.getvalue())
        self.assertEqual(hub.requests, [])
        self.assertEqual(sorted(p.name for p in old.iterdir()), ["SOURCE.txt"])
        m = arc.read_manifest(self.root / "hf/Vontra" / f"X-MLX-4bit@{COMMIT}")
        self.assertEqual({f["fetched"]["via"] for f in m["files"]}, {"move"})
        self.assertEqual(m["variants"][0]["detected"]["scheme"], "mlx:affine-4bit-g32")

    def test_refuses_a_non_archive_directory(self):
        self.root.mkdir()
        (self.root / "something").write_text("x")
        self.assertEqual(self.archive(gguf_repo(), "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 1)
        self.assertIn("not empty and not a model archive", self.out.getvalue())


class RobustnessTest(ArchiveCase):
    def setUp(self):
        super().setUp()
        arc.cancel.clear()
        self.addCleanup(arc.cancel.clear)

    def one(self, data=None, lfs=True, **extra):
        data = data or bytes(range(256)) * 8
        entry = {"path": "m.gguf", "size": len(data), "lfs": lfs, "git_oid": git_oid(data),
                 "sha256": hashlib.sha256(data).hexdigest() if lfs else None, **extra}
        rev = self.tmp / "rev"
        rev.mkdir(exist_ok=True)
        return data, entry, rev

    def test_a_short_write_is_continued_not_counted(self):
        data, entry, rev = self.one()
        real, calls = os.pwrite, []

        def short(fd, buf, offset):
            calls.append(len(buf))
            return real(fd, bytes(buf)[:100], offset) if len(calls) == 1 else real(fd, buf, offset)
        with unittest.mock.patch.object(os, "pwrite", short):
            status, _ = arc.transfer(rev, entry, lambda s: arc.file_opener(self.write_src(data))(s), "w.1.x")
        self.assertEqual(status, "done")
        self.assertEqual((rev / "m.gguf").read_bytes(), data)

    def write_src(self, data):
        src = self.tmp / "src.bin"
        src.write_bytes(data)
        return src

    def test_locked_transfer_skips_periodic_identity_checks_but_checks_before_publish(self):
        data, entry, rev = self.one()
        src = self.write_src(data)
        with unittest.mock.patch.object(arc, "IDENTITY_CHECK_BYTES", 1), \
                unittest.mock.patch.object(arc, "_same_file", wraps=arc._same_file) as same:
            self.assertEqual(arc.transfer(rev, entry, arc.file_opener(src), "w.1.x",
                                          writer_locked=True)[0], "done")
        self.assertEqual(same.call_count, 1)
        self.assertEqual((rev / "m.gguf").read_bytes(), data)

    def test_unlocked_transfer_keeps_live_claim_and_periodic_checks(self):
        data, entry, rev = self.one()
        other = rev / f"m.gguf{arc.PARTIAL}other.1.x"
        other.write_bytes(data[:300])
        src = self.write_src(data)
        self.assertEqual(arc.transfer(rev, entry, arc.file_opener(src), "w.1.x")[0], "busy")
        other.unlink()
        with unittest.mock.patch.object(arc, "IDENTITY_CHECK_BYTES", 1), \
                unittest.mock.patch.object(arc, "_same_file", wraps=arc._same_file) as same:
            self.assertEqual(arc.transfer(rev, entry, arc.file_opener(src), "w.1.x")[0], "done")
        self.assertGreater(same.call_count, 1)

    def test_locked_resume_rejects_corrupt_prefix_before_publication(self):
        data, entry, rev = self.one()
        other = rev / f"m.gguf{arc.PARTIAL}other.1.x"
        other.write_bytes(b"bad" + data[3:300])
        with self.assertRaises(arc.Mismatch):
            arc.transfer(rev, entry, arc.file_opener(self.write_src(data)), "w.1.x", writer_locked=True)
        self.assertFalse((rev / "m.gguf").exists())
        self.assertEqual(list(rev.iterdir()), [])

    def test_lock_is_cross_process_and_released_after_errors(self):
        arc.open_root(self.root, create=True)
        code = ("import sys,pathlib;sys.path.insert(0,sys.argv[1]);import llmp_archive as a\n"
                "try:\n with a.writer_lock(pathlib.Path(sys.argv[2]),no_wait=True): pass\n"
                "except a.WriterBusy: sys.exit(3)")
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with arc.writer_lock(self.root):
                blocked = subprocess.run([sys.executable, "-c", code, str(TOOLS), str(self.root)],
                                         capture_output=True)
                self.assertEqual(blocked.returncode, 3, blocked.stderr)
                raise RuntimeError("stop")
        self.assertFalse((self.root / arc.WRITER_GUARD).exists())
        free = subprocess.run([sys.executable, "-c", code, str(TOOLS), str(self.root)], capture_output=True)
        self.assertEqual(free.returncode, 0, free.stderr)

    def test_crashed_local_owner_is_recovered_without_a_timeout(self):
        arc.open_root(self.root, create=True)
        code = ("import sys,pathlib,os;sys.path.insert(0,sys.argv[1]);import llmp_archive as a\n"
                "with a.writer_lock(pathlib.Path(sys.argv[2])): os._exit(0)")
        crash = subprocess.run([sys.executable, "-c", code, str(TOOLS), str(self.root)], capture_output=True)
        self.assertEqual(crash.returncode, 0, crash.stderr)
        self.assertTrue((self.root / arc.WRITER_GUARD).is_dir())
        messages = []
        with arc.writer_lock(self.root, no_wait=True, log=messages.append):
            self.assertTrue(any("dead local" in m for m in messages))
        self.assertFalse((self.root / arc.WRITER_GUARD).exists())

    def test_forced_interrupt_keeps_guard_until_all_worker_threads_exit(self):
        # The actual CLI has a transfer blocked in read() after observing the
        # first signal. A second signal must kill the process, not unwind a join
        # and release its guard while that worker can still write the partial.
        code = textwrap.dedent("""\
            import pathlib, sys
            sys.path.insert(0, sys.argv[1])
            from test_archive import archive_model, FakeHub, FakeResponse, arc
            base = pathlib.Path(sys.argv[2])
            class BlockedResponse(FakeResponse):
                def read(self, size=-1):
                    (base / 'reading').write_text('ready')
                    if not arc.cancel.wait(10):
                        raise RuntimeError('first signal not received')
                    (base / 'cancelled').write_text('worker still alive')
                    sys.stdin.buffer.read(1)
                    return super().read(size)
            class BlockedHub(FakeHub):
                def open_file(self, repo, commit, path, start=0):
                    return BlockedResponse(self.files[path], start)
            hub = BlockedHub('org/M', {'m.gguf': b'abcd'}, lfs=['m.gguf'])
            sys.exit(archive_model.main(['org/M', '--include', 'm.gguf', '--jobs', '1',
                                        '--archive', str(base / 'nas'), '--init-root'], hub=hub))
            """)
        env = dict(os.environ, LLMP_ARCHIVE_PROGRESS_SOCKET="", LLMP_ARCHIVE_PROGRESS_SESSION="")
        proc = subprocess.Popen([sys.executable, "-B", "-c", code, str(TOOLS / "tests"), str(self.tmp)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        try:
            def wait_for(name):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if (self.tmp / name).exists():
                        return
                    if proc.poll() is not None:
                        self.fail(f"writer exited before {name}: {proc.communicate()!r}")
                    time.sleep(0.01)
                self.fail(f"writer did not reach {name}")

            wait_for("reading")
            guard = self.root / arc.WRITER_GUARD
            owner = (guard / arc.WRITER_OWNER).read_bytes()
            proc.send_signal(signal.SIGINT)
            wait_for("cancelled")
            self.assertIsNone(proc.poll())
            with self.assertRaises(arc.WriterBusy):
                with arc.writer_lock(self.root, no_wait=True):
                    self.fail("entered while the cancelled worker could still write")
            proc.send_signal(signal.SIGINT)
            self.assertEqual(proc.wait(timeout=5), 128 + signal.SIGINT)
            self.assertEqual((guard / arc.WRITER_OWNER).read_bytes(), owner)
            messages = []
            with arc.writer_lock(self.root, no_wait=True, log=messages.append):
                self.assertTrue(any("dead local" in line for line in messages))
            self.assertFalse(guard.exists())
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=5)

    def test_guard_with_live_foreign_or_missing_owner_is_never_stolen_by_age(self):
        arc.open_root(self.root, create=True)
        guard = self.root / arc.WRITER_GUARD
        guard.mkdir()
        owner_path = guard / arc.WRITER_OWNER
        for owner in (None, arc._writer_owner(), dict(arc._writer_owner(), boot="another-boot")):
            if owner is None:
                owner_path.unlink(missing_ok=True)
            else:
                owner_path.write_bytes(arc.dumps(owner))
            os.utime(guard, (0, 0))
            with self.assertRaises(arc.WriterBusy):
                with arc.writer_lock(self.root, no_wait=True):
                    self.fail("stole live or unconfirmed ownership")
            self.assertTrue(guard.is_dir())

    def test_recovery_requires_the_same_local_mutex_and_mount_namespace(self):
        with tempfile.TemporaryFile() as mutex:
            mine = arc._writer_owner(mutex.fileno())
        dead = dict(mine, start=str(int(mine["start"]) + 1))
        self.assertTrue(arc._owner_dead(dead, mine))
        self.assertFalse(arc._owner_dead(dict(dead, local_mutex=[0, 0]), mine))
        self.assertFalse(arc._owner_dead(dict(dead, mount_namespace=-1), mine))

    def test_local_mutex_does_not_depend_on_temporary_directory_settings(self):
        arc.open_root(self.root, create=True)
        with arc.writer_lock(self.root):
            owner = arc.loads_strict((self.root / arc.WRITER_GUARD / arc.WRITER_OWNER).read_bytes(), "owner")
            info = pathlib.Path(f"/tmp/llmp-archive-writer-{os.getuid()}.lock").stat()
            self.assertEqual(owner["local_mutex"], [info.st_dev, info.st_ino])
            with unittest.mock.patch.dict(os.environ, TMPDIR=str(self.tmp), TEMP=str(self.tmp), TMP=str(self.tmp)):
                with self.assertRaises(arc.WriterBusy):
                    with arc.writer_lock(self.root, no_wait=True):
                        self.fail("different temporary environment bypassed the mutex")

    def test_waiting_writer_stops_on_cancellation(self):
        arc.open_root(self.root, create=True)
        with arc.writer_lock(self.root):
            def stop_waiting(seconds):
                arc.cancel.set()
            with unittest.mock.patch.object(arc, "sleep", stop_waiting):
                with self.assertRaisesRegex(arc.ArchiveError, "interrupted while waiting"):
                    with arc.writer_lock(self.root):
                        self.fail("entered a held lock")

    def test_unsupported_local_locks_fail_closed(self):
        arc.open_root(self.root, create=True)
        with unittest.mock.patch.object(fcntl, "flock", side_effect=OSError(errno.EOPNOTSUPP, "no locks")):
            with self.assertRaises(OSError):
                with arc.writer_lock(self.root):
                    self.fail("entered without the local recovery mutex")
        self.assertFalse((self.root / arc.WRITER_GUARD).exists())

    def test_a_full_disk_stops_at_once_and_the_partial_is_adopted_by_the_rerun(self):
        data, entry, rev = self.one()
        src = self.write_src(data)
        real, naps = os.pwrite, []
        arc.sleep = naps.append

        def full(fd, buf, offset):
            if offset >= 1024:
                raise OSError(28, "No space left on device")
            return real(fd, buf, offset)
        try:
            with unittest.mock.patch.object(arc, "CHUNK", 512), unittest.mock.patch.object(os, "pwrite", full):
                with self.assertRaises(arc.WriteFailed) as caught:
                    arc.transfer(rev, entry, arc.file_opener(src), "w.1.x")
            self.assertIn("No space left", str(caught.exception))
            self.assertEqual(naps, [])  # not retried like a network failure
            partial = rev / f"m.gguf{arc.PARTIAL}w.1.x"
            self.assertEqual(partial.stat().st_size, 1024)
            self.assertEqual(partial.stat().st_mtime, 0)  # released: the next run need not wait 15 minutes
            self.assertFalse((rev / "m.gguf").exists())
            status, _ = arc.transfer(rev, entry, arc.file_opener(src), "w.2.y")
        finally:
            arc.sleep = lambda seconds: time.sleep(min(seconds, 0.01))
        self.assertEqual(status, "done")
        self.assertEqual((rev / "m.gguf").read_bytes(), data)
        self.assertEqual(list(rev.glob(f"*{arc.PARTIAL}*")), [])

    def test_a_failure_before_any_byte_leaves_no_partial(self):
        data, entry, rev = self.one()

        def refuse(start):
            raise hublib.HubError("HTTP 401", 401)
        with self.assertRaises(hublib.HubError):
            arc.transfer(rev, entry, refuse, "w.1.x")
        self.assertEqual(list(rev.iterdir()), [])

    def test_an_adopted_partial_shows_as_live_at_once(self):
        data, entry, rev = self.one()
        partial = rev / f"m.gguf{arc.PARTIAL}old.1.dead"
        partial.write_bytes(data[:100])
        os.utime(partial, (1000, 1000))
        src = self.write_src(data)
        seen = []

        def opener(start):
            seen.append(time.time() - next(rev.glob(f"*{arc.PARTIAL}*")).stat().st_mtime)
            return arc.file_opener(src)(start)
        self.assertEqual(arc.transfer(rev, entry, opener, "w.1.x")[0], "done")
        self.assertLess(seen[0], 60)  # a third writer must not take it for stale while the prefix is read

    def test_the_loser_of_an_adoption_race_waits_instead_of_downloading(self):
        data, entry, rev = self.one()
        partial = rev / f"m.gguf{arc.PARTIAL}old.1.dead"
        partial.write_bytes(data[:100])
        os.utime(partial, (1000, 1000))
        with unittest.mock.patch.object(os, "rename", side_effect=FileNotFoundError):
            status, _ = arc.transfer(rev, entry, lambda s: self.fail("downloaded"), "w.1.x")
        self.assertEqual(status, "busy")
        self.assertEqual(list(rev.glob(f"*{arc.PARTIAL}w.1.x")), [])

    def test_progress_resets_the_retry_budget(self):
        data, entry, rev = self.one()
        calls = []

        def drip(start):
            calls.append(start)
            return FakeResponse(data, start, fail_after=start + 400)  # 2048 bytes: six cut-offs
        status, _ = arc.transfer(rev, entry, drip, "w.1.x", retries=2)
        self.assertEqual(status, "done")
        self.assertGreater(len(calls), 3)
        self.assertEqual((rev / "m.gguf").read_bytes(), data)

    def test_a_file_without_any_identity_is_refused(self):
        data, entry, rev = self.one(lfs=False)
        entry["git_oid"] = None
        with self.assertRaises(arc.Mismatch):
            arc.transfer(rev, entry, arc.file_opener(self.write_src(data)), "w.1.x")
        self.assertFalse((rev / "m.gguf").exists())
        self.assertEqual(list(rev.glob(f"*{arc.PARTIAL}*")), [])

    def test_an_interrupt_releases_the_partial(self):
        data, entry, rev = self.one()
        src = self.write_src(data)

        class Cutting:
            start = 0
            n = 0

            def read(self, n):
                self.n += 1
                if self.n == 2:
                    arc.cancel.set()
                return data[(self.n - 1) * 100:self.n * 100]

            def close(self):
                pass
        with self.assertRaises(arc.ArchiveError) as caught:
            arc.transfer(rev, entry, lambda s: Cutting(), "w.1.x")
        self.assertIn("interrupted", str(caught.exception))
        partial = rev / f"m.gguf{arc.PARTIAL}w.1.x"
        self.assertEqual(partial.stat().st_mtime, 0)
        arc.cancel.clear()
        self.assertEqual(arc.transfer(rev, entry, arc.file_opener(src), "w.2.y")[0], "done")

    def test_an_include_that_matches_nothing_is_a_usage_error(self):
        self.assertEqual(self.archive(gguf_repo(), "unsloth/M-GGUF", "--include", "Q4_K_M/*"), 2)
        self.assertIn("matches no file", self.out.getvalue())
        self.assertFalse((self.root / "hf").exists())

    def test_a_missing_root_is_never_created_without_init_root(self):
        self.assertEqual(self.archive(gguf_repo(), "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf", init=False), 1)
        self.assertIn("--init-root", self.out.getvalue())
        self.assertFalse(self.root.exists())

    def test_the_roots_parents_are_never_created(self):
        self.root = self.tmp / "mnt" / "llm" / "nas"  # the mount point is not there
        self.assertEqual(self.archive(gguf_repo(), "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 1)
        self.assertIn("parent directory does not exist", self.out.getvalue())
        self.assertFalse((self.tmp / "mnt").exists())

    def test_directories_are_never_made_above_the_base(self):
        gone = self.tmp / "share"
        with self.assertRaises(FileNotFoundError):
            arc.make_dirs(gone, gone / "hf" / "org")
        self.assertFalse(gone.exists())
        gone.mkdir()
        arc.make_dirs(gone, gone / "hf" / "org")
        arc.make_dirs(gone, gone / "hf" / "org")
        self.assertTrue((gone / "hf" / "org").is_dir())

    def test_a_vanished_revision_directory_is_not_recreated_by_a_transfer(self):
        data, entry, rev = self.one()
        entry["path"] = "sub/m.gguf"
        rev.rmdir()
        with self.assertRaises(FileNotFoundError):
            arc.transfer(rev, entry, arc.file_opener(self.write_src(data)), "w.1.x")
        self.assertFalse(rev.exists())

    def test_a_signal_cuts_a_backoff_short(self):
        arc.cancel.set()
        started = time.monotonic()
        REAL_SLEEP(30)
        self.assertLess(time.monotonic() - started, 5)

    def test_a_failed_refetch_of_a_vanished_file_is_not_listed_present(self):
        hub = gguf_repo()
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0)
        rev = self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"
        (rev / "M-Q4_K_M.gguf").unlink()  # gone from the share (or deleted by --repair as bad)
        (rev / "README.md").unlink()
        real = hub.open_file

        def flaky(repo, commit, path, start=0):
            if path == "M-Q4_K_M.gguf":
                raise hublib.HubError("M-Q4_K_M.gguf: HTTP 403", 403)
            return real(repo, commit, path, start)
        hub.open_file = flaky
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 1)
        m = arc.read_manifest(rev)
        for f in m["files"]:
            if f["present"]:
                self.assertTrue((rev / f["path"]).is_file(), f["path"])
        self.assertTrue(next(f for f in m["files"] if f["path"] == "README.md")["present"])  # fetched again
        self.assertFalse(next(f for f in m["files"] if f["path"] == "M-Q4_K_M.gguf")["present"])

    def test_a_malformed_http_reply_is_a_retryable_hub_error(self):
        import http.client

        class Opener:
            def open(self, request, timeout=None):
                raise http.client.BadStatusLine("HTTP/1.1 garbage")
        with self.assertRaises(hublib.HubError) as caught:
            hublib.Hub("http://x", TOKEN, opener=Opener()).open_file("o/r", COMMIT, "f", 0)
        self.assertTrue(caught.exception.retryable)

    def test_a_manifest_entry_whose_file_is_gone_is_fetched_again(self):
        hub = gguf_repo()
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0)
        rev = self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"
        (rev / "M-Q4_K_M.gguf").unlink()
        hub2 = gguf_repo()
        self.assertEqual(self.archive(hub2, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0, self.out.getvalue())
        self.assertEqual([p for p, _ in hub2.requests], ["M-Q4_K_M.gguf"])
        self.assertEqual((rev / "M-Q4_K_M.gguf").read_bytes(), hub2.files["M-Q4_K_M.gguf"])

    def test_one_failed_file_does_not_lose_the_others(self):
        hub = gguf_repo()
        real = hub.open_file

        def flaky(repo, commit, path, start=0):
            if path == "README.md":
                raise hublib.HubError("README.md: HTTP 403", 403)
            return real(repo, commit, path, start)
        hub.open_file = flaky
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 1)
        self.assertIn("README.md: HTTP 403", self.out.getvalue())
        self.assertIn("rerun to resume", self.out.getvalue())
        rev = self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"
        m = arc.read_manifest(rev)
        self.assertEqual(sorted(f["path"] for f in m["files"] if f["present"]),
                         [".gitattributes", "M-Q4_K_M.gguf", "config.json"])
        hub.open_file = real
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0)
        self.assertTrue(next(f for f in arc.read_manifest(rev)["files"] if f["path"] == "README.md")["present"])

    def test_a_gated_repository_without_a_token_fails_clearly_and_leaves_nothing(self):
        hub = gguf_repo()

        def gated(repo, commit, path, start=0):
            raise hublib.HubError(f"{path}: HTTP 401 (gated or private: put HF_TOKEN in .env)", 401)
        hub.open_file = gated
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 1)
        self.assertIn("HTTP 401", self.out.getvalue())
        self.assertFalse((self.root / "hf/unsloth").exists())
        self.assertFalse((self.root / "catalog").exists())

    def test_nothing_is_written_when_the_share_goes_away(self):
        hub = gguf_repo()
        real, calls = hub.open_file, []

        def vanishing(repo, commit, path, start=0):
            calls.append(path)
            if len(calls) == 2:
                (self.root / arc.ROOT_MARKER).unlink()  # the share dropped
            return real(repo, commit, path, start)
        hub.open_file = vanishing
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf", "--jobs", "1"), 1)
        self.assertIn("archive root is gone", self.out.getvalue())
        self.assertFalse((self.root / "catalog").exists())

    def test_a_failed_compaction_does_not_fail_a_finished_archive(self):
        with unittest.mock.patch.object(arc, "compact", side_effect=PermissionError("held open")):
            self.assertEqual(self.archive(gguf_repo(), "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0)
        self.assertIn("catalog not compacted", self.out.getvalue())
        self.assertIsNotNone(arc.read_manifest(self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"))

    def test_a_compaction_leaves_a_file_it_cannot_delete(self):
        arc.open_root(self.root, create=True)
        for n in range(3):
            arc.write_event(self.root, "family", {"repo": f"o/r{n}", "label": "x"}, "t")
        real = pathlib.Path.unlink

        def held(self_, missing_ok=False):
            if self_.parent.name == "events":
                raise PermissionError("held open")
            return real(self_, missing_ok=missing_ok)
        report = []
        with unittest.mock.patch.object(pathlib.Path, "unlink", held):
            arc.compact(self.root, report, force_pack=True)
        self.assertTrue(any("left" in line for line in report))
        self.assertEqual(json.loads((self.root / "catalog/catalog.json").read_text())["events"], 3)

    def test_unreadable_values_in_a_header_never_block_the_manifest(self):
        config = b'{"model_type": "x", "quantization_config": {"quant_method": "exl3", "bits": NaN}}'
        files = {"config.json": config, "model.safetensors": safetensors_bytes([("a.trellis", "U16", [4], 8)])}
        hub = FakeHub("o/r", files, lfs=["model.safetensors"])
        self.assertEqual(self.archive(hub, "o/r"), 0, self.out.getvalue())
        m = arc.read_manifest(self.root / "hf/o" / f"r@{COMMIT}")
        self.assertEqual(m["variants"][0]["detected"]["from"], "names")
        self.assertTrue(m["variants"][0]["detected"]["notes"])

    def test_catalog_show_on_a_new_archive_is_not_an_error(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(archive_catalog.main(["--archive", str(self.tmp / "nothing"), "show", "q"]), 0)
        self.assertIn("no archive", out.getvalue())
        (self.tmp / "other").mkdir()
        (self.tmp / "other" / "x").write_text("x")
        with contextlib.redirect_stderr(out):
            self.assertEqual(archive_catalog.main(["--archive", str(self.tmp / "other"), "show"]), 1)


class CatalogTest(ArchiveCase):
    def test_concurrent_events_and_compactions_lose_nothing(self):
        arc.open_root(self.root, create=True)
        old = (arc.PACK_LOOSE_THRESHOLD, arc.PACK_COUNT_THRESHOLD)
        arc.PACK_LOOSE_THRESHOLD, arc.PACK_COUNT_THRESHOLD = 3, 2
        written = []
        try:
            def worker(n):
                for i in range(15):
                    written.append(arc.write_event(self.root, "family", {"repo": f"o/r{n}-{i}", "label": "x"},
                                                   "test")["id"])
                    if i % 3 == 0:
                        arc.compact(self.root)
            threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            events, _, _ = arc.read_all_events(self.root, [])
            self.assertEqual(set(events), set(written))
            snapshot = arc.compact(self.root, force_pack=True)
            self.assertEqual(snapshot["events"], 90)
            self.assertEqual(len(list((self.root / "catalog/packs").iterdir())), 1)
            self.assertEqual(list((self.root / "catalog/events").iterdir()), [])
        finally:
            arc.PACK_LOOSE_THRESHOLD, arc.PACK_COUNT_THRESHOLD = old

    def test_a_read_during_another_compaction_still_sees_every_event(self):
        arc.open_root(self.root, create=True)
        ids = {arc.write_event(self.root, "family", {"repo": f"o/r{n}", "label": "x"}, "t")["id"] for n in range(3)}
        real, state = arc._json_files, {"first": True, "inside": False}

        def listing(directory):
            result = real(directory)
            if state["first"] and not state["inside"]:
                state["first"], state["inside"] = False, True
                arc.compact(self.root, force_pack=True)  # another compactor, right after this read's first listing
                state["inside"] = False
            return result
        with unittest.mock.patch.object(arc, "_json_files", listing):
            events, _, _ = arc.read_all_events(self.root, [])
        self.assertEqual(set(events), ids)

    def test_invalid_events_are_quarantined(self):
        arc.open_root(self.root, create=True)
        arc.write_event(self.root, "family", {"repo": "o/r", "label": "x"}, "test")
        (self.root / "catalog/events/bad.json").write_text('{"format": "llmp-archive-event", "x": 1, "x": 2}')
        report = []
        snapshot = arc.compact(self.root, report)
        self.assertEqual(snapshot["events"], 1)
        self.assertTrue((self.root / "catalog/quarantine/bad.json").exists())
        self.assertIn("duplicate key", report[0])

    def test_relations_group_quantizations_under_their_model_and_family(self):
        arc.open_root(self.root, create=True)
        base_st = safetensors_bytes([("w", "BF16", [4, 4], 32)])
        repos = [FakeHub("Qwen/Q-Base", {"model.safetensors": base_st}, ["model.safetensors"]),
                 FakeHub("Qwen/Q", {"model.safetensors": base_st}, ["model.safetensors"],
                         card={"base_model": "Qwen/Q-Base"}, tags=["base_model:finetune:Qwen/Q-Base"]),
                 FakeHub("Mia/Q-NVFP4", {"model.safetensors": safetensors_bytes(
                     [("w", "U8", [4, 2], 8), ("w_scale", "F8_E4M3", [4, 1], 4)]),
                     "hf_quant_config.json": json.dumps({"producer": {"name": "modelopt", "version": "0.3"},
                                                         "quantization": {"quant_algo": "NVFP4"}}).encode()},
                         ["model.safetensors"], card={"base_model": ["Qwen/Q"]}),
                 FakeHub("unsloth/Q-GGUF", {"Q-Q4_K_M.gguf": gguf_bytes()}, ["Q-Q4_K_M.gguf"],
                         card={"base_model": "Qwen/Q"}, tags=["base_model:quantized:Qwen/Q"])]
        for hub in repos:
            self.assertEqual(self.archive(hub, hub.repo), 0, self.out.getvalue())
        snapshot = json.loads((self.root / "catalog/catalog.json").read_text())
        by_repo = {r["repo"]: r for r in snapshot["repositories"]}
        self.assertEqual(by_repo["Mia/Q-NVFP4"]["relation"], "quantized")  # inferred from its scheme
        self.assertEqual(by_repo["Mia/Q-NVFP4"]["relation_source"], "inferred")
        self.assertEqual({r: by_repo[r]["model"] for r in by_repo},
                         {"Qwen/Q-Base": "Qwen/Q-Base", "Qwen/Q": "Qwen/Q", "Mia/Q-NVFP4": "Qwen/Q",
                          "unsloth/Q-GGUF": "Qwen/Q"})
        self.assertEqual({by_repo[r]["family"] for r in by_repo}, {"Qwen/Q-Base"})
        model = next(m for m in snapshot["models"] if m["repo"] == "Qwen/Q")
        self.assertEqual(model["derived_from"], {"repo": "Qwen/Q-Base", "relation": "finetune"})
        self.assertEqual(len(model["variants"]), 3)
        nv = next(r for r in snapshot["revisions"] if r["repo"] == "Mia/Q-NVFP4")["variants"][0]["detected"]
        self.assertEqual((nv["scheme"], nv["bits"], nv["quantizer"], nv["notes"]),
                         ("modelopt:NVFP4", 4, "modelopt 0.3", []))
        # An owner override moves a repository.
        with contextlib.redirect_stdout(self.out):
            self.assertEqual(archive_catalog.main(["--archive", str(self.root), "relation", "unsloth/Q-GGUF",
                                                   "Qwen/Q-Base", "quantized"]), 0)
            self.assertEqual(archive_catalog.main(["--archive", str(self.root), "show", "q4_k_m"]), 0)
        snapshot = json.loads((self.root / "catalog/catalog.json").read_text())
        self.assertEqual(next(r for r in snapshot["repositories"] if r["repo"] == "unsloth/Q-GGUF")["model"],
                         "Qwen/Q-Base")
        self.assertIn("1 of 4 variants", self.out.getvalue())

    def test_rebuild_restores_the_catalog_from_manifests(self):
        hub = gguf_repo()
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0)
        import shutil
        shutil.rmtree(self.root / "catalog")
        (self.root / "hf/unsloth/M-GGUF@" f"{'b' * 40}").mkdir()  # an abandoned download
        with contextlib.redirect_stdout(self.out):
            self.assertEqual(archive_catalog.main(["--archive", str(self.root), "rebuild"]), 0)
        self.assertIn("no manifest", self.out.getvalue())
        snapshot = json.loads((self.root / "catalog/catalog.json").read_text())
        self.assertEqual([r["commit"] for r in snapshot["revisions"]], [COMMIT])


class StageTest(ArchiveCase):
    def test_the_hub_token_is_not_passed_to_ssh_or_rsync(self):
        seen = []
        with unittest.mock.patch.dict(os.environ, {"HF_TOKEN": TOKEN, "HUGGING_FACE_HUB_TOKEN": TOKEN}), \
                unittest.mock.patch.object(stage_model.subprocess, "run",
                                           side_effect=lambda cmd, **kw: seen.append(kw["env"]) or
                                           unittest.mock.Mock(returncode=0)):
            stage_model.run(["ssh", "peer", "true"])
        self.assertNotIn("HF_TOKEN", seen[0])
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", seen[0])

    def test_stages_a_variant_verified_with_the_manifest_last(self):
        hub = gguf_repo()
        self.assertEqual(self.archive(hub, "unsloth/M-GGUF", "--include", "M-Q4_K_M.gguf"), 0)
        dest = self.tmp / "store"
        with contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.out):
            self.assertEqual(stage_model.main(["unsloth/M-GGUF", "--variant", "M-Q4_K_M", "--dest", str(dest),
                                               "--archive", str(self.root), "--host", "spark"]), 0,
                             self.out.getvalue())
        staged = dest / "unsloth" / f"M-GGUF@{COMMIT[:8]}"
        rev = self.root / "hf/unsloth" / f"M-GGUF@{COMMIT}"
        self.assertEqual((staged / arc.MANIFEST).read_bytes(), (rev / arc.MANIFEST).read_bytes())
        self.assertEqual((staged / "M-Q4_K_M.gguf").read_bytes(), hub.files["M-Q4_K_M.gguf"])
        snapshot = json.loads((self.root / "catalog/catalog.json").read_text())
        self.assertEqual(snapshot["revisions"][0]["staged"][0]["host"], "spark")
        # A variant that is not archived is refused; a corrupted archive file is named.
        with contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.out):
            self.assertEqual(stage_model.main(["unsloth/M-GGUF", "--variant", "M-Q8_0", "--dest", str(dest),
                                               "--archive", str(self.root)]), 1)
            self.assertIn("not present on the archive", self.out.getvalue())
            data = bytearray((rev / "M-Q4_K_M.gguf").read_bytes())
            data[-1] ^= 1
            (rev / "M-Q4_K_M.gguf").write_bytes(bytes(data))
            other = self.tmp / "other"
            self.assertEqual(stage_model.main(["unsloth/M-GGUF", "--variant", "M-Q4_K_M", "--dest", str(other),
                                               "--archive", str(self.root)]), 1)
        self.assertIn("SHA-256", self.out.getvalue())
        self.assertFalse((other / "unsloth" / f"M-GGUF@{COMMIT[:8]}" / arc.MANIFEST).exists())


class _Handler(http.server.BaseHTTPRequestHandler):
    seen = []
    redirect_to = ""
    payload = b"0123456789" * 100

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        _Handler.seen.append((self.server.server_port, self.path, self.headers.get("Authorization"),
                              self.headers.get("Range")))
        if self.path.startswith("/api/models/gated/r/"):
            self.send_response(401)
            self.end_headers()
        elif self.path.startswith("/api/models/"):
            body = json.dumps({"id": "o/r", "sha": COMMIT, "siblings": []}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif "/resolve/" in self.path:
            self.send_response(302)
            self.send_header("Location", _Handler.redirect_to + "/blob")
            self.end_headers()
        else:
            start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0])
            body = _Handler.payload[start:]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(_Handler.payload) - 1}/{len(_Handler.payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


class HubClientTest(unittest.TestCase):
    def test_token_goes_to_the_hub_only_and_never_into_messages(self):
        servers = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler) for _ in range(2)]
        for s in servers:
            threading.Thread(target=s.serve_forever, daemon=True).start()
        try:
            hub_port, cdn_port = (s.server_port for s in servers)
            _Handler.redirect_to = f"http://localhost:{cdn_port}"
            _Handler.seen.clear()
            hub = hublib.Hub(f"http://127.0.0.1:{hub_port}", TOKEN)
            self.assertEqual(hub.revision_info("o/r", "main")["sha"], COMMIT)
            response = hub.open_file("o/r", COMMIT, "dir/model.gguf", 100)
            self.assertEqual((response.start, response.read()), (100, _Handler.payload[100:]))
            response.close()
            auth = {(port, path.split("?")[0]): a for port, path, a, _ in _Handler.seen}
            self.assertEqual(auth[(hub_port, "/api/models/o/r/revision/main")], f"Bearer {TOKEN}")
            self.assertEqual(auth[(hub_port, f"/o/r/resolve/{COMMIT}/dir/model.gguf")], f"Bearer {TOKEN}")
            self.assertIsNone(auth[(cdn_port, "/blob")])  # not forwarded across the redirect
            with self.assertRaises(hublib.HubError) as caught:
                hub.revision_info("gated/r", "main")
            self.assertEqual(caught.exception.status, 401)
            self.assertNotIn(TOKEN, str(caught.exception))
            self.assertEqual(hub.scrub(f"x {TOKEN} y"), "x <token> y")
        finally:
            for s in servers:
                s.shutdown()
                s.server_close()

    def test_token_from_the_environment(self):
        self.assertEqual(hublib.token_from_env({"HF_TOKEN": " t "}), "t")
        self.assertEqual(hublib.token_from_env({"HUGGING_FACE_HUB_TOKEN": "u"}), "u")
        self.assertIsNone(hublib.token_from_env({}))

    def test_a_malformed_token_is_refused_without_echoing_it(self):
        for bad in ("hf_ab\ncd", "hf_ab cd", "hf_\u2019x"):
            with self.assertRaises(hublib.HubError) as caught:
                hublib.token_from_env({"HF_TOKEN": bad})
            self.assertNotIn("hf_", str(caught.exception))

    def test_an_unexpected_exception_prints_no_traceback_and_no_token(self):
        class Boom(FakeHub):
            def revision_info(self, repo, revision):
                raise ValueError(f"Invalid header value {TOKEN!r}")

            def scrub(self, text):
                return text.replace(TOKEN, "<token>")
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            self.assertEqual(archive_model.main(["o/r", "--init-root", "--archive", tmp], hub=Boom("o/r", {})), 1)
        self.assertNotIn(TOKEN, out.getvalue())
        self.assertNotIn("Traceback", out.getvalue())
        self.assertIn("internal error", out.getvalue())

    def test_the_cli_never_prints_the_token(self):
        class Failing(FakeHub):
            def revision_info(self, repo, revision):
                raise hublib.HubError(f"boom {TOKEN}")

            def scrub(self, text):
                return text.replace(TOKEN, "<token>")
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            self.assertEqual(archive_model.main(["o/r", "--init-root", "--archive", tmp], hub=Failing("o/r", {})), 1)
        self.assertNotIn(TOKEN, out.getvalue())
        self.assertIn("<token>", out.getvalue())


class DetectTest(unittest.TestCase):
    def test_gguf_header_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "x.gguf"
            path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1 << 40, 0))
            with self.assertRaises(detect.HeaderError):
                detect.read_gguf(path)
            path.write_bytes(gguf_bytes()[:40])
            with self.assertRaises(detect.HeaderError):
                detect.read_gguf(path)

    def test_name_hints(self):
        self.assertEqual(detect._name_hint(["UD-Q2_K_XL/M-UD-Q2_K_XL-00001-of-00002.gguf"]), "UD-Q2_K_XL")
        self.assertEqual(detect._name_hint(["m-IQ3_XXS.gguf"]), "IQ3_XXS")


if __name__ == "__main__":
    unittest.main()
