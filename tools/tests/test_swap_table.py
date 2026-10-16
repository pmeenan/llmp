# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/swap_table.py's configuration, plan, run and checker on synthetic reports, no GPU.

The synthetic report has the fields and values of a real `swap-table --report` (the fixed
baseline's M3 table and the Gemma near-8K reports, docs/experiments/m36-baselines)."""

import contextlib
import copy
import hashlib
import io
import json
import os
import pathlib
import stat
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

import importlib.util

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("swap_table", ROOT / "tools/swap_table.py")
st = importlib.util.module_from_spec(spec)
sys.modules["swap_table"] = st
spec.loader.exec_module(st)

GIB = 1 << 30
H1, H2 = "a1" * 32, "b2" * 32


def make_row(pair, name, case, total=6.0, peak=110 * GIB):
    parsed = st.parse_row(name)
    a_llm = not pair.startswith("image")
    row = {"pair": pair, "name": name, "total": total, "first_output": 0.2, "digest_seconds": 0.0,
           "plan_seconds": 0.0, "demand_seconds": 0.0, "peak_bytes": peak, "exact": True,
           "state_exact": True, "output": "ab" * 32, "context_ids_sha256": H1,
           "initialized_state_bytes": 0, "state_before_sha256": "", "state_restored_sha256": "",
           "reference_logits_sha256": "", "continued_logits_sha256": "",
           "continuation_vocabulary": 0, "continuation_logit_rows": 0,
           "continuation_logit_bytes": 0, "graphs_kept": 0,
           "graphs": {"eager": 0, "captured": 0, "replayed": 0}, "parts": {}}
    if parsed["direction"] == "B->A" and a_llm:
        if parsed["context"] > 0:
            row.update(initialized_state_bytes=663330816, state_before_sha256=H1,
                       state_restored_sha256=H1, reference_logits_sha256=H2,
                       continued_logits_sha256=H2, continuation_vocabulary=248320,
                       continuation_logit_rows=case.continue_tokens,
                       output=",".join(str(100 + i) for i in range(case.continue_tokens)))
        if parsed["kind"] == "prepared":
            row["graphs_kept"] = 5
            if parsed["context"] > 0:
                row["graphs"] = {"eager": 0, "captured": 0, "replayed": 10}
            else:
                row["graphs"] = {"eager": 2, "captured": 0, "replayed": 0}
        elif parsed["context"] > 0:
            row["graphs"] = {"eager": 5, "captured": 3, "replayed": 2}
    return row


def make_report(case_name, pairs=None, **kw):
    case = st.CASES[case_name]
    rows = [make_row(p, n, case, **kw) for p, n in st.expected_rows(case, case.context_tokens)
            if pairs is None or p.replace(" <-> ", ":") in pairs]
    return {"command": "swap-table", "handoff": True, "context_tokens": case.context_tokens,
            "continue": case.continue_tokens, "cycles": 2,
            "context_sha256": case.context_sha256, "budget": 117290979884,
            "mem_available_start": 125002280960, "mem_available_low": 5565186048,
            "problems": [], "positioning": [], "rows": rows}


def control_line(pin=st.sg.IMAGE_PIXELS_SHA256):
    return f"== image <-> deepseek (image as A)\ncontrol: image generated, pixels {pin}\n"


def write_run(base, name, report, table="", runtime="c0" * 32):
    d = pathlib.Path(base) / name
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps(report))
    (d / "table.out").write_text(table)
    (d / "runtime.sha256").write_text(f"{runtime}  /runtime\n")
    return str(d)


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def m3(self, mutate=None, table=None):
        report = make_report("m3")
        if mutate:
            mutate(report)
        return st.check([write_run(self.dir, f"r{len(list(self.dir.iterdir()))}", report,
                                   table if table is not None else control_line())], "m3")


class Rows(unittest.TestCase):
    def test_the_m3_table_is_32_rows_over_six_pairs(self):
        rows = st.expected_rows(st.CASES["m3"], 8192)
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({p for p, _ in rows}), 6)
        self.assertEqual(len([r for r in rows if r[0].startswith("image")]), 8)
        self.assertEqual(len(set(rows)), 32)

    def test_gemma_cases_are_one_pair_of_six_rows(self):
        for name in ("gemma2", "gemma3", "gemma26", "gemma31"):
            rows = st.expected_rows(st.CASES[name], st.CASES[name].context_tokens)
            self.assertEqual(len(rows), 6)
            self.assertEqual({p for p, _ in rows}, {f"{name} <-> qwen38"})

    def test_row_names_parse(self):
        self.assertEqual(st.parse_row("B->A, prepared (8192 context tokens)"),
                         {"direction": "B->A", "kind": "prepared", "context": 8192,
                          "image": False})
        self.assertEqual(st.parse_row("A->B, prepared (0 context)")["context"], 0)
        self.assertTrue(st.parse_row("A->B, first use (image)")["image"])
        self.assertIsNone(st.parse_row("sideways"))


class Check(Fixture):
    def test_a_complete_exact_table_passes(self):
        report = self.m3()
        self.assertEqual(report["status"], "PASS", report["problems"])
        self.assertEqual(report["row_count"], 32)
        self.assertEqual(report["max_peak_bytes"], 110 * GIB)

    def test_the_table_may_come_from_several_jobs(self):
        full = make_report("m3")
        a = dict(full, rows=[r for r in full["rows"] if r["pair"].startswith("qwen3.8")])
        b = dict(full, rows=[r for r in full["rows"] if not r["pair"].startswith("qwen3.8")])
        report = st.check([write_run(self.dir, "a", a), write_run(self.dir, "b", b,
                                                                   control_line())], "m3")
        self.assertEqual(report["status"], "PASS", report["problems"])

    def test_missing_and_duplicate_rows(self):
        def drop(r):
            r["rows"].pop(3)
        report = self.m3(drop)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("missing" in p for p in report["problems"]))
        report = st.check([write_run(self.dir, "p", dict(make_report("m3"), rows=make_report(
            "m3")["rows"][:6]), control_line())], "m3", partial=True)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(len(report["missing"]), 26)

        def duplicate(r):
            r["rows"].append(copy.deepcopy(r["rows"][0]))
        self.assertTrue(any("more than once" in p for p in self.m3(duplicate)["problems"]))

    def test_inexact_rows_fail(self):
        for field in ("exact", "state_exact"):
            with self.subTest(field=field):
                report = self.m3(lambda r: r["rows"][4].update({field: False}))
                self.assertEqual(report["status"], "FAIL")

    def test_runtime_problems_fail(self):
        report = self.m3(lambda r: r["problems"].append("the regenerated image differs"))
        self.assertTrue(any("regenerated image" in p for p in report["problems"]))

    def test_restored_state_hash_must_equal(self):
        def alter(r):
            row = next(x for x in r["rows"] if x["state_before_sha256"])
            row["state_restored_sha256"] = H2
        self.assertTrue(any("restored state" in p for p in self.m3(alter)["problems"]))

        def blank(r):
            row = next(x for x in r["rows"] if x["state_before_sha256"])
            row["state_before_sha256"] = row["state_restored_sha256"] = ""
        self.assertTrue(any("restored state" in p for p in self.m3(blank)["problems"]))

    def test_continuation_tokens_and_logits_must_equal(self):
        def logits(r):
            row = next(x for x in r["rows"] if x["reference_logits_sha256"])
            row["continued_logits_sha256"] = H1
        self.assertTrue(any("continued logits" in p for p in self.m3(logits)["problems"]))

        def count(r):
            row = next(x for x in r["rows"] if x["reference_logits_sha256"])
            row["continuation_logit_rows"] = 8
        self.assertTrue(any("logit rows" in p for p in self.m3(count)["problems"]))

        def tokens(r):
            row = next(x for x in r["rows"] if x["reference_logits_sha256"])
            row["output"] = "1,2,3"
        self.assertTrue(any("16 tokens" in p for p in self.m3(tokens)["problems"]))

        def vocabulary(r):
            row = next(x for x in r["rows"] if x["reference_logits_sha256"])
            row["continuation_vocabulary"] = 0
        self.assertTrue(any("vocabulary" in p for p in self.m3(vocabulary)["problems"]))

    def test_gemma_continuations_are_64_tokens(self):
        case = st.CASES["gemma31"]
        report = make_report("gemma31")
        d = write_run(self.dir, "g", report)
        self.assertEqual(st.check([d], "gemma31")["status"], "PASS")
        row = next(x for x in report["rows"] if x["reference_logits_sha256"])
        self.assertEqual(len(row["output"].split(",")), case.continue_tokens)
        report["continue"] = 16
        d2 = write_run(self.dir, "g2", report)
        self.assertEqual(st.check([d2], "gemma31")["status"], "FAIL")

    def test_prepared_rows_must_replay_graphs(self):
        def nothing_replayed(r):
            row = next(x for x in r["rows"] if x["name"] == "B->A, prepared (8192 context tokens)"
                       and x["pair"] == "qwen3.8 <-> deepseek")
            row["graphs"] = {"eager": 0, "captured": 0, "replayed": 0}
        self.assertTrue(any("not replayed" in p for p in self.m3(nothing_replayed)["problems"]))

        def recaptured(r):
            row = next(x for x in r["rows"] if x["name"] == "B->A, prepared (8192 context tokens)")
            row["graphs"]["captured"] = 1
        self.assertTrue(any("not replayed" in p for p in self.m3(recaptured)["problems"]))

        def none_kept(r):
            row = next(x for x in r["rows"] if x["name"] == "B->A, prepared (0 context)")
            row["graphs_kept"] = 0
        self.assertTrue(any("no graphs kept" in p for p in self.m3(none_kept)["problems"]))

    def test_image_rows_need_no_graphs(self):
        self.assertEqual(self.m3()["status"], "PASS")  # image A rows keep none

    def test_time_limits(self):
        def prepared(r):
            row = next(x for x in r["rows"] if "prepared" in x["name"])
            row["total"] = 20.5
        self.assertTrue(any("prepared limit" in p for p in self.m3(prepared)["problems"]))

        def first_use(r):
            row = next(x for x in r["rows"] if "first use" in x["name"])
            row["total"] = 41.0
        self.assertTrue(any("first use limit" in p for p in self.m3(first_use)["problems"]))

        def edge(r):
            for row in r["rows"]:
                row["total"] = 20.0 if "prepared" in row["name"] else 40.0
        self.assertEqual(self.m3(edge)["status"], "PASS")
        report = make_report("m3", total=25.0)
        d = write_run(self.dir, "late", report, control_line())
        self.assertEqual(st.check([d], "m3", prepared_limit=30.0, first_use_limit=30.0)["status"],
                         "PASS")

    def test_image_controls_must_equal_the_pin(self):
        report = self.m3(table=control_line("f" * 64))
        self.assertTrue(any("not the pin" in p for p in report["problems"]))
        self.assertEqual(self.m3(table="")["status"], "FAIL")  # image A, no control line
        d = write_run(self.dir, "pin", make_report("m3"), control_line("f" * 64))
        self.assertEqual(st.check([d], "m3", image_pin="f" * 64)["status"], "PASS")

    def test_the_planned_image_a_job_reads_no_context_text(self):
        # plan's third job (image:deepseek, image:qwen3.8) has no LLM A: the
        # runtime reads no context text and reports the empty text's hash.
        full = make_report("m3")
        image = [r for r in full["rows"] if r["pair"].startswith("image")]
        llm = [r for r in full["rows"] if not r["pair"].startswith("image")]
        empty = hashlib.sha256(b"").hexdigest()
        report = st.check([write_run(self.dir, "llm", dict(full, rows=llm)),
                           write_run(self.dir, "image", dict(full, rows=image,
                                                             context_sha256=empty),
                                     control_line())], "m3")
        self.assertEqual(report["status"], "PASS", report["problems"])
        # An LLM A that hashed another text still fails.
        other = dict(full, rows=llm, context_sha256=empty)
        report = st.check([write_run(self.dir, "llm2", other),
                           write_run(self.dir, "image2", dict(full, rows=image),
                                     control_line())], "m3")
        self.assertTrue(any("context text" in p for p in report["problems"]))

    def test_runs_must_agree_on_runtime_and_context(self):
        full = make_report("m3")
        a = dict(full, rows=[r for r in full["rows"] if r["pair"].startswith("qwen3.8")])
        b = dict(full, rows=[r for r in full["rows"] if not r["pair"].startswith("qwen3.8")])
        report = st.check([write_run(self.dir, "a", a), write_run(
            self.dir, "b", b, control_line(), runtime="d0" * 32)], "m3")
        self.assertTrue(any("2 runtimes" in p for p in report["problems"]))
        other = dict(make_report("m3"), context_sha256="e" * 64)
        self.assertTrue(any("context text" in p for p in st.check(
            [write_run(self.dir, "c", other, control_line())], "m3")["problems"]))
        wrong = dict(make_report("m3"), context_tokens=4096)
        self.assertTrue(any("context tokens" in p for p in st.check(
            [write_run(self.dir, "w", wrong, control_line())], "m3")["problems"]))

    def test_unreadable_reports_fail(self):
        d = self.dir / "none"
        d.mkdir()
        report = st.check([str(d)], "m3")
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("report.json" in p for p in report["problems"]))

    def test_worst_rows_and_peaks_are_reported(self):
        def shape(r):
            for row in r["rows"]:
                row["total"] = 3.0
            llm = [x for x in r["rows"] if "image" not in x["pair"]]
            llm[1]["total"] = 9.5
            llm[2]["total"] = 9.0
            llm[2]["peak_bytes"] = 118 * GIB
            next(x for x in r["rows"] if "image" in x["pair"])["total"] = 15.0
        report = self.m3(shape)
        self.assertEqual(report["worst_llm_llm"]["total"], 9.5)
        self.assertEqual(report["worst_prepared_llm_llm"]["total"], 9.0)
        self.assertEqual(report["max_peak_bytes"], 118 * GIB)
        self.assertTrue(all("peak_bytes" in r for r in report["rows"]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            st.print_check(report)
        self.assertIn("worst LLM<->LLM row: 9.500 s", out.getvalue())
        self.assertIn("PASS", out.getvalue())

    def test_command_exit_status(self):
        d = write_run(self.dir, "ok", make_report("m3"), control_line())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(st.main(["check", d, "--case", "m3"]), 0)
            self.assertEqual(st.main(["check", d, "--case", "m3", "--prepared-limit", "1"]), 1)
        d2 = write_run(self.dir, "part", dict(make_report("m3"), rows=make_report(
            "m3")["rows"][:6]), control_line())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(st.main(["check", d2, "--case", "m3", "--partial"]), 2)
            self.assertEqual(st.main(["check", d2, "--case", "m3"]), 1)
            st.main(["check", d, "--case", "m3", "--json"])


class Configuration(unittest.TestCase):
    def render(self, case, ids=None, **kw):
        return tomllib.loads(st.render_config(st.CASES[case], ids or st.load_ids(None),
                                              "/store", "/data", "/share", **kw))

    def test_the_m3_configuration_is_the_baselines(self):
        cfg = self.render("m3")
        self.assertEqual(cfg["storage"], {"data_dir": "/data", "installed": "/store"})
        ds = cfg["models"]["deepseek"]
        self.assertEqual((ds["artifact"], ds["drafter"], ds["wave_form"], ds["context"],
                          ds["max_slots"], ds["speculation"]),
                         (st.sg.DSV4, st.sg.DSPARK, "speculative", 262144, 4, True))
        q = cfg["models"]["qwen3.8"]
        self.assertEqual((q["artifact"], q["drafter"], q["context"], q["draft_vocab"],
                          q["prefill_chunk"]), (st.sg.QWEN, st.sg.QWEN_MTP, 33792, 65536, 4096))
        self.assertTrue(q["tokenizer"].startswith("/share/models/"))
        self.assertEqual(cfg["models"]["image"], {"composition": st.sg.IMAGE})

    def test_gemma_cases_pair_with_a_one_slot_qwen(self):
        cfg = self.render("gemma3")
        self.assertEqual(cfg["models"]["gemma3"], {"artifact": st.sg.GEMMA3, "context": 8448,
                                                   "prefill_chunk": 128, "max_slots": 1})
        q = cfg["models"]["qwen38"]
        self.assertEqual((q["artifact"], q["context"], q["max_slots"]),
                         (st.sg.QWEN, 8448, 1))
        self.assertNotIn("drafter", q)
        self.assertEqual(self.render("gemma26")["models"]["gemma26"]["prefill_chunk"], 1024)

    def test_ids_override_the_defaults_and_settings_add_keys(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "ids.json"
        path.write_text(json.dumps({"deepseek": {"artifact": "x" * 64},
                                    "qwen3.8": {"settings": {"depth_cost_ratio": 1.16,
                                                             "max_slots": 2}},
                                    "image": {"composition": "y" * 64}}))
        cfg = self.render("m3", st.load_ids(str(path)))
        self.assertEqual(cfg["models"]["deepseek"]["artifact"], "x" * 64)
        self.assertEqual(cfg["models"]["deepseek"]["drafter"], st.sg.DSPARK)  # kept
        self.assertEqual(cfg["models"]["qwen3.8"]["depth_cost_ratio"], 1.16)
        self.assertEqual(cfg["models"]["qwen3.8"]["max_slots"], 2)
        self.assertEqual(cfg["models"]["image"]["composition"], "y" * 64)

    def test_bad_id_files_are_refused(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "ids.json"
        for content in ([], {"deepseek": []}, {"deepseek": {"artefact": "x"}}):
            path.write_text(json.dumps(content))
            with self.subTest(content=content), self.assertRaises(SystemExit):
                st.load_ids(str(path))

    def test_a_template_replaces_the_rendering(self):
        text = st.render_config(st.CASES["m3"], st.load_ids(None), "/store", "/data", "/share",
                                'data_dir = "{data}"\ninstalled = "{installed}"\nx = "{share}"\n')
        self.assertEqual(tomllib.loads(text), {"data_dir": "/data", "installed": "/store",
                                               "x": "/share"})
        with self.assertRaises(SystemExit):
            st.render_config(st.CASES["m3"], {}, "/store", "/data", "/share", "nothing")


class PlanAndRun(Fixture):
    def argv(self, command, *extra):
        return [command, "--case", "m3", "--runtime", "/rt/llmp-runtime", "--store", "/store",
                "--out", str(self.dir / "out"), *extra]

    def test_plan_splits_the_table_into_jobs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            st.main(self.argv("plan"))
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("--name m3-1 --pairs qwen3.8:deepseek,qwen3.8:image", lines[0])
        pairs = [p for line in lines for p in line.rsplit("--pairs ", 1)[1].split(",")]
        self.assertEqual(pairs, list(st.M3_PAIRS))
        self.assertTrue(all(" run " in line and "--runtime /rt/llmp-runtime" in line
                            for line in lines))

    def test_plan_passes_options_through_and_gemma_is_one_job(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            st.main(self.argv("plan", "--pairs-per-job", "6", "--image-pin", "f" * 64,
                              "--ids", "/ids.json"))
        (line,) = out.getvalue().splitlines()
        self.assertIn("--name m3 ", line)
        self.assertIn("--image-pin " + "f" * 64, line)
        self.assertIn("--ids /ids.json", line)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            st.main(["plan", "--case", "gemma31", "--runtime", "/r", "--store", "/s",
                     "--out", "/o"])
        self.assertEqual(len(out.getvalue().splitlines()), 1)
        self.assertIn("--pairs gemma31:qwen38", out.getvalue())

    def test_swap_commands_match_the_baselines(self):
        args = mock.Mock(runtime="/rt", image_pin=None)
        run_dir = pathlib.Path("/out/m3-1")
        m3 = st.swap_command(st.CASES["m3"], args, run_dir, "qwen3.8:image",
                             pathlib.Path("/share"), pathlib.Path("/inputs"))
        self.assertEqual(m3[:7], ["/rt", "--config", "/out/m3-1/node.toml", "--anchor",
                                  "/out/m3-1/enrollment", "swap-table", "--pairs"])
        joined = " ".join(m3)
        for part in ("--context-tokens 8192", "--continue 16", "--cycles 2", "--zero-context on",
                     "--handoff on", "--image-expect " + st.sg.IMAGE_PIXELS_SHA256,
                     "--image-noise /share/references/qwen-image-2.1/native/ref1/"
                     "latents_init.bf16", "--report /out/m3-1/report.json"):
            self.assertIn(part, joined)
        self.assertNotIn("--plain", m3)
        gemma = st.swap_command(st.CASES["gemma3"], args, run_dir, "gemma3:qwen38",
                                pathlib.Path("/share"), pathlib.Path("/inputs"))
        joined = " ".join(gemma)
        for part in ("--context-tokens 8192", "--continue 64", "--plain",
                     "--context-text /inputs/gemma-context.txt"):
            self.assertIn(part, joined)
        self.assertNotIn("--image-expect", gemma)
        args.image_pin = "f" * 64
        self.assertIn("f" * 64, st.swap_command(st.CASES["m3"], args, run_dir, "a:b",
                                                pathlib.Path("/s"), pathlib.Path("/i")))

    def fake_runtime(self, report, table):
        """A script standing in for llmp-runtime: copies the synthetic outputs where asked."""
        (self.dir / "report.src").write_text(json.dumps(report))
        (self.dir / "table.src").write_text(table)
        path = self.dir / "fake-runtime"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import os, shutil, sys\n"
            f"shutil.copy('{self.dir}/report.src', sys.argv[sys.argv.index('--report') + 1])\n"
            f"sys.stdout.write(open('{self.dir}/table.src').read())\n"
            "sys.stderr.write('PTX ' + os.environ.get('CUDA_DISABLE_PTX_JIT', '') + ' '"
            " + sys.argv[sys.argv.index('--pairs') + 1] + '\\n')\n"
            "os.makedirs(os.path.dirname(sys.argv[sys.argv.index('--config') + 1]) + '/data/x',"
            " exist_ok=True)\n"
            "sys.exit(int(os.environ.get('FAKE_STATUS', '0')))\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return str(path)

    def test_run_writes_the_run_directory_and_the_checker_reads_it(self):
        runtime = self.fake_runtime(make_report("m3"), control_line())
        argv = ["run", "--case", "m3", "--runtime", runtime, "--store", str(self.dir / "store"),
                "--share", str(self.dir / "share"), "--out", str(self.dir / "out"),
                "--name", "m3-all"]
        with mock.patch.object(st, "pinned") as pinned, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(st.main(argv), 0)
        self.assertEqual(pinned.call_count, 2)  # the context text and the latents
        run_dir = self.dir / "out" / "m3-all"
        self.assertTrue({"node.toml", "report.json", "runtime.sha256", "table.log",
                         "table.out"} <= {p.name for p in run_dir.iterdir()})
        self.assertFalse((run_dir / "data").exists())  # removed afterwards
        self.assertIn("PTX 1 " + ",".join(st.M3_PAIRS), (run_dir / "table.log").read_text())
        cfg = tomllib.loads((run_dir / "node.toml").read_text())
        self.assertEqual(cfg["storage"]["data_dir"], str(run_dir / "data"))
        self.assertEqual(stat.S_IMODE(run_dir.stat().st_mode), 0o700)
        self.assertTrue((run_dir / "runtime.sha256").read_text().startswith(
            st.sha256_file(pathlib.Path(runtime))))
        self.assertEqual(st.check([str(run_dir)], "m3")["status"], "PASS")
        # An existing run directory is never reused.
        with mock.patch.object(st, "pinned"), self.assertRaises(SystemExit):
            st.main(argv)

    def test_a_failing_runtime_fails_the_run_and_unknown_pairs_are_refused(self):
        runtime = self.fake_runtime(make_report("m3"), "")
        argv = ["run", "--case", "m3", "--runtime", runtime, "--store", "/s",
                "--share", str(self.dir / "share"), "--out", str(self.dir / "out"),
                "--name", "bad"]
        with mock.patch.object(st, "pinned"), mock.patch.dict(os.environ, {"FAKE_STATUS": "3"}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(st.main(argv), 3)
        with mock.patch.object(st, "pinned"), self.assertRaises(SystemExit):
            st.main(argv[:-1] + ["other", "--pairs", "deepseek:deepseek"])

    def test_unpinned_inputs_are_refused_before_anything_runs(self):
        runtime = self.fake_runtime(make_report("m3"), "")
        (self.dir / "inputs").mkdir()
        (self.dir / "inputs" / st.CONTEXT_M3).write_text("not the text")
        with self.assertRaises(SystemExit):
            st.main(["run", "--case", "m3", "--runtime", runtime, "--store", "/s",
                     "--inputs", str(self.dir / "inputs"), "--out", str(self.dir / "out"),
                     "--name", "x"])
        self.assertFalse((self.dir / "out").exists())


if __name__ == "__main__":
    unittest.main()
