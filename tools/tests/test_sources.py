# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the source lock and its tools (no downloads, builds or SDK needed).

tests/sources/ checks the same mechanism end to end with the SDK's CMake.
"""

import contextlib
import gzip
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import unittest
import zipfile
from unittest import mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import llmp_sources as srclib  # noqa: E402
from llmp_sources import SourceError  # noqa: E402


def load_script(name: str):
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(TOOLS / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


prepare_sources = load_script("prepare-sources")
setup = load_script("setup")


def component(**changes) -> dict:
    comp = {
        "version": "1.0", "kind": "archive", "category": "implementation", "tier": "core", "use": "product",
        "machine": "target",
        "upstream": {"repository": "https://example.invalid/x", "commit": "a" * 40},
        "archive": {"file": "x-1.0.tar.gz", "urls": ["https://example.invalid/x-1.0.tar.gz"],
                    "sha256": "b" * 64, "size": 10},
        "patches": [], "tree_sha256": "c" * 64, "depends": [],
        "cmake": {"subdirectory": "", "options": {"X_TESTS": "OFF"}, "platform_packages": ["Threads"],
                  "targets": ["x::x"]},
        "license": {"expression": "MIT", "files": ["LICENSE"], "scope": "all", "evidence": "headers"},
        "verification": "checked",
    }
    comp.update(changes)
    return comp


def lock(components: dict, modules: dict | None = None) -> dict:
    return {"schema": 1, "modules": modules or {}, "components": components}


class CheckedInLock(unittest.TestCase):
    def test_validates_and_uses_https_only(self):
        data = srclib.load_lock()  # raises SourceError listing any problem
        for cid, comp in data["components"].items():
            for url in comp["archive"]["urls"]:
                self.assertTrue(url.startswith("https://"), f"{cid}: {url}")

    def test_core_selection(self):
        self.assertEqual(srclib.select(srclib.load_lock(), []),
                         ["cutlass", "ggml", "ds4", "exllamav3", "googletest", "tomlplusplus"])
        self.assertEqual(srclib.select(srclib.load_lock(), [], cuda=True),
                         srclib.select(srclib.load_lock(), []))
        self.assertEqual(srclib.select(srclib.load_lock(), [], cuda=False),
                         ["cutlass", "exllamav3", "ggml", "googletest", "tomlplusplus"])

    def test_mise_tasks(self):
        tasks = tomllib.loads((REPO / "mise.toml").read_text())["tasks"]
        self.assertEqual(tasks["setup"]["run"], "python3 tools/setup")
        self.assertEqual(tasks["prepare"]["run"], "python3 tools/prepare-sources")


class ExllamaV3Component(unittest.TestCase):
    """The locked ExLlamaV3 subset: the pin, and a keep set that is exactly the native linear's closure.

    That is upstream's GEMM compilation units and what llmpalooza's instance unit (llmp/
    llmp_exl3_kernels.cu, added by patch 0002) includes: the GEMV kernel's header (core since
    D-080) and the reconstruction, Hadamard and bias-add sources that patch 0003 reduces to their
    kernels. The closure test reads the prepared tree (`mise run prepare`) and is skipped where
    none exists.
    """

    COMMIT = "6b84a21b6f1e5da3f291b9e1019061f0de788279"
    EXT = "exllamav3/exllamav3_ext/"
    # The compilation units the build compiles: the mcg codebook (cb1) at the M2 fixtures' rates,
    # and llmpalooza's instance unit.
    UNITS = [EXT + "quant/comp_units/exl3_comp_unit_4_cb1.cu", EXT + "quant/comp_units/exl3_comp_unit_5_cb1.cu",
             EXT + "quant/comp_units/exl3_comp_unit_6_cb1.cu", EXT + "quant/comp_units/exl3_comp_unit_8_cb1.cu",
             "llmp/llmp_exl3_kernels.cu"]
    # Kept sources the instance unit includes rather than compiles on their own.
    INCLUDED_SOURCES = [EXT + "add.cu", EXT + "quant/hadamard.cu", EXT + "quant/reconstruct.cu"]
    # Upstream files the core component does not keep: the ATen host wrappers (llmpalooza's launchers
    # replace them, with recorded copies of what they decide), bits_k.cuh's c10 include, the int8
    # GEMV and the MoE kernels.
    EXCLUDED = ["quant/exl3_gemv.cu", "quant/exl3_gemv.cuh", "quant/comp_units/exl3_gemv_half_inst.cu",
                "quant/exl3_gemv_int8.cu", "quant/exl3_gemv_int8_kernel.cuh", "quant/exl3_moe_coop_kernel.cuh",
                "quant/exl3_moe_coop.cu", "quant/exl3_gemm.cu", "quant/exl3_kernel_map.cu",
                "quant/exl3_devctx.cu", "quant/bits_k.cuh", "quant/reconstruct.cuh", "quant/hadamard.cuh",
                "add.cuh", "hgemm.cu"]

    def setUp(self):
        self.comp = srclib.load_lock()["components"]["exllamav3"]
        self.keep = self.comp["archive"]["keep"]

    def test_pin(self):
        comp = self.comp
        self.assertEqual(comp["upstream"]["commit"], self.COMMIT)
        self.assertEqual(comp["archive"]["urls"],
                         [f"https://github.com/turboderp-org/exllamav3/archive/{self.COMMIT}.tar.gz"])
        self.assertEqual(comp["archive"]["sha256"],
                         "63c3c7fe4adc281753d1dfd37110f772775a77a6dce72be068b3c23e07a915b0")
        self.assertEqual(comp["archive"]["size"], 8628152)
        self.assertEqual((comp["category"], comp["tier"], comp["use"]), ("implementation", "core", "test"))
        self.assertNotIn("module", comp)
        self.assertEqual(comp["license"]["expression"], "MIT")
        self.assertEqual(comp["cmake"]["targets"], ["llmp_exl3_headers", "llmp_exl3_cuda"])

    def test_patches_carry_their_license(self):
        for patch in self.comp["patches"]:
            path = REPO / "third_party" / patch["path"]
            self.assertTrue(path.parent.name == "exllamav3", patch["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), patch["sha256"])
            self.assertTrue(path.with_name(path.name + ".license").is_file(), patch["path"])

    def test_keep_holds_the_gemv_kernel_but_no_host_wrapper(self):
        self.assertIn("LICENSE", self.keep)
        for path in self.keep:
            self.assertNotIn("moe", path)
            self.assertNotIn("int8", path)
            if "gemv" in path:
                self.assertEqual(path, self.EXT + "quant/exl3_gemv_kernel.cuh")
            # Single files only: a kept directory would take whatever upstream adds to it.
            self.assertTrue(path == "LICENSE" or (path.startswith(self.EXT) and path.endswith((".cu", ".cuh", ".h"))),
                            path)
        for path in self.EXCLUDED:
            self.assertNotIn(self.EXT + path, self.keep)
        units = [p for p in self.keep if p.endswith(".cu")]
        self.assertEqual(sorted(units), sorted([u for u in self.UNITS if u.startswith(self.EXT)] +
                                               self.INCLUDED_SOURCES))

    # The MoE cooperative kernels and their derivatives, which still need their own audit (D-080); no
    # kept file may name one. The GEMV and the QTIP citation in its header are cleared (D-080).
    GATED = re.compile(r"moe_coop|cooperative_moe", re.I)
    # Calls that end the process (D-066; a served process never exits from inside a kernel library):
    # host exits, aborts, asserts and fatal signals, and device traps, breakpoints and asserts
    # (compiler builtins and cooperative_groups' _CG_ABORT among them), as calls or as PTX.
    EXITING = re.compile(r"\b(?:std::)?(?:_?exit|_Exit|quick_exit|abort|terminate|assert|__assert\w*|__trap"
                         r"|__builtin_\w*(?:trap|abort)|_CG_ABORT|__brkpt|raise|kill)\s*\(|\b(?:trap|brkpt)\s*;")

    @classmethod
    def include_closure(cls, tree: pathlib.Path, units: list[str], roots: list[str]) -> tuple[set[str], list[str]]:
        """The tree files the units reach, as the compiler resolves includes, and what is wrong with them.

        A quoted include is looked up beside the including file, then in the include roots; an angle
        include in the roots only (then in the toolkit, which is not the tree). Both kinds are followed
        wherever they resolve inside the tree: a kept file an angle include reaches is compiled as
        surely as a quoted one.
        """
        seen, pending, problems = set(), list(units), []
        while pending:
            rel = pending.pop()
            if rel in seen:
                continue
            seen.add(rel)
            text = (tree / rel).read_text()
            if cls.GATED.search(text):
                problems.append(f"{rel} names a gated file or source ({cls.GATED.search(text)[0]})")
            for quote, name in re.findall(r'^\s*#\s*include\s*(["<])([^">]+)[">]', text, re.M):
                if cls.GATED.search(name) or name.startswith(("ATen/", "c10/", "torch/", "pybind11/")):
                    problems.append(f"{rel} includes {name}")
                bases = ([os.path.dirname(rel)] if quote == '"' else []) + roots
                found = next((os.path.normpath(os.path.join(base, name)) for base in bases
                              if (tree / os.path.normpath(os.path.join(base, name))).is_file()), None)
                if found is not None:
                    pending.append(found)
                elif quote == '"':
                    problems.append(f"{rel} includes \"{name}\", which is not kept")
        return seen, problems

    def test_include_closure_follows_angle_includes_into_the_tree(self):
        # A kept file reaching a gated header through the include root (-I), with angle brackets,
        # is found and refused, as a quoted include is.
        with tempfile.TemporaryDirectory() as tmp:
            tree = pathlib.Path(tmp)
            for rel, text in {self.EXT + "quant/unit.cu": '#include <cuda_fp16.h>\n#include "../util.cuh"\n',
                              self.EXT + "util.cuh": "#include <quant/other.cuh>\n",
                              self.EXT + "quant/other.cuh": "#pragma once\n#include <quant/deep.cuh>\n",
                              self.EXT + "quant/deep.cuh": "#pragma once\n"}.items():
                (tree / rel).parent.mkdir(parents=True, exist_ok=True)
                (tree / rel).write_text(text)
            seen, problems = self.include_closure(tree, [self.EXT + "quant/unit.cu"], [self.EXT.rstrip("/")])
            self.assertEqual(problems, [])
            self.assertEqual(seen, {self.EXT + p for p in ("quant/unit.cu", "util.cuh", "quant/other.cuh",
                                                           "quant/deep.cuh")})
            (tree / self.EXT / "quant/deep.cuh").write_text("#pragma once\n#include <quant/exl3_moe_coop_kernel.cuh>\n")
            (tree / self.EXT / "quant/exl3_moe_coop_kernel.cuh").write_text("// cooperative_moe dispatch\n")
            seen, problems = self.include_closure(tree, [self.EXT + "quant/unit.cu"], [self.EXT.rstrip("/")])
            self.assertIn(self.EXT + "quant/exl3_moe_coop_kernel.cuh", seen)
            self.assertTrue(any("includes quant/exl3_moe_coop_kernel.cuh" in p for p in problems), problems)
            self.assertTrue(any("names a gated file or source (cooperative_moe)" in p for p in problems), problems)

    def test_keep_is_exactly_the_compiled_closure(self):
        tree = srclib.prepared_dir(srclib.SOURCES_DIR, "exllamav3", self.comp)
        if not tree.is_dir():
            self.skipTest(f"{tree} is not prepared (mise run prepare)")
        self.assertEqual(srclib.tree_digest(tree), self.comp["tree_sha256"])
        # The build exposes one include root, the extension's directory, and compiles exactly the units.
        build = (tree / "llmp" / "CMakeLists.txt").read_text()
        self.assertIn('set(ext "${CMAKE_CURRENT_SOURCE_DIR}/../exllamav3/exllamav3_ext")', build)
        self.assertEqual(re.findall(r"target_include_directories\(([^)]*)\)", build),
                         ['llmp_exl3_headers INTERFACE "${ext}" "${CMAKE_CURRENT_SOURCE_DIR}"'])
        self.assertIn('foreach(bits IN ITEMS 4 5 6 8)', build)
        self.assertIn('exl3_comp_unit_${bits}_cb1.cu', build)
        self.assertIn('list(APPEND units "${CMAKE_CURRENT_SOURCE_DIR}/llmp_exl3_kernels.cu")', build)
        # Every include, quoted or angle, that resolves in the tree is followed; nothing gated is reached.
        seen, problems = self.include_closure(tree, self.UNITS, [self.EXT.rstrip("/")])
        self.assertEqual(problems, [])
        # Llmpalooza's own files (patch 0002) are the build's, not upstream's kept ones.
        self.assertEqual(sorted({p for p in seen if not p.startswith("llmp/")} | {"LICENSE"}), self.keep)
        # The three kept sources are included, never compiled as units of their own.
        for source in self.INCLUDED_SOURCES:
            self.assertNotIn(source.removeprefix(self.EXT), build)

    def test_exiting_pattern_finds_every_way_to_end_the_process(self):
        # Host exits, aborts and asserts, and device traps and asserts: a trapped kernel leaves the
        # context unusable, so the process is as good as ended. Each is found where upstream would
        # write it; static_assert and prose are not calls.
        for snippet in ("if (abort) exit(code);", "std::exit (1);", "_exit(1);", "_Exit(1);", "quick_exit(0);",
                        "abort();", "std::abort();", "std::terminate();", "assert(x == 1);",
                        "__assert_fail(a, b, c, d);", "__assertfail(a, b, c, d, e);", "__trap();",
                        "__builtin_trap();", "__brkpt();", 'asm volatile ("trap;");', 'asm("brkpt;");',
                        "__builtin_abort();", "__builtin_debugtrap();", '__builtin_verbose_trap("a", "b");',
                        "__assert_perror_fail(e, f, l, fn);", "_CG_ABORT();", "raise(SIGABRT);",
                        "std::raise(SIGTERM);", "kill(getpid(), SIGKILL);"):
            with self.subTest(snippet=snippet):
                self.assertRegex(snippet, self.EXITING)
        for snippet in ("static_assert(sizeof(int) == 4);", "// assert x is dtype T", "bool abort = true;",
                        "// a served process never exits from inside a kernel library",
                        "int exit_code = 0;", "trapezoid(x);", "// raise the limit", "skill(x);",
                        "atexit_count(x);"):
            with self.subTest(snippet=snippet):
                self.assertNotRegex(snippet, self.EXITING)

    CONTRACT = "src/kernels/exl3/launch_contract.h"

    def test_launch_contract_covers_every_kept_kernel(self):
        # A kept kernel waits on other blocks (cooperative_groups' grid sync, which traps unless the
        # launch was cooperative, and spin barriers, which hang unless every block is co-resident),
        # so the launch contract the launchers include names each, and the kernel tables include it.
        tree = srclib.prepared_dir(srclib.SOURCES_DIR, "exllamav3", self.comp)
        if not tree.is_dir():
            self.skipTest(f"{tree} is not prepared (mise run prepare)")
        kernels = set()
        for rel in self.keep:
            # Line continuations join a macro's lines, as the preprocessor does.
            text = (tree / rel).read_text().replace("\\\n", " ")
            for match in re.finditer(r"\b__global__\b", text):
                end = min(i for i in (text.find("{", match.end()), text.find(";", match.end()), len(text)) if i >= 0)
                declarator = re.sub(r"__launch_bounds__\s*\([^)]*\)", "", text[match.end():end])
                kernels.add(re.search(r"(\w+)\s*\(", declarator)[1])
            # add.cu defines its kernels through a macro whose fourth argument names each.
            if re.search(r"^#define KERNEL_DEF\(", text, re.M):
                kernels.discard("kernel")
                kernels.update(re.findall(r"^KERNEL_DEF\(\s*\w+,\s*\w+,\s*\w+,\s*(\w+)\s*,", text, re.M))
        self.assertTrue({"exl3_gemm_kernel", "exl3_mgemm_kernel", "exl3_gemv_kernel", "reconstruct_kernel",
                         "reconstruct_had_kernel", "had_hf_r_128_kernel", "had_ff_r_128_kernel",
                         "add_kernel_hhh"} <= kernels, kernels)
        self.assertEqual(len(kernels), 24, sorted(kernels))
        contract = (REPO / self.CONTRACT).read_text()
        for kernel in kernels:
            self.assertRegex(contract, rf"\b{kernel}\b")
        include = f'#include "{self.CONTRACT.removeprefix("src/")}"'
        self.assertIn(include, (REPO / "tests/unit/exl3_tables.h").read_text())
        self.assertIn(include, (REPO / "src/kernels/exl3/launch.h").read_text())

    def test_no_kept_file_can_end_the_process(self):
        # Patch 0001 removes upstream's exiting error checks from util.cuh outright: they are inline
        # host code in a header, so any translation unit that reaches the header without a guard
        # macro (or undefines one) would compile them. No kept file, patched, may call one.
        tree = srclib.prepared_dir(srclib.SOURCES_DIR, "exllamav3", self.comp)
        if not tree.is_dir():
            self.skipTest(f"{tree} is not prepared (mise run prepare)")
        for rel in self.keep:
            text = (tree / rel).read_text()
            found = [m[0] for m in self.EXITING.finditer(text)]
            self.assertEqual(found, [], f"{rel} can end the process: {found}")
        self.assertNotRegex((tree / self.EXT / "util.cuh").read_text(), r"#\s*define\s+(?:cuda|cublas)_check")


class Validation(unittest.TestCase):
    def problems(self, data: dict, base: pathlib.Path = REPO) -> str:
        return "\n".join(srclib.validate_lock(data, base))

    def test_a_valid_lock_has_no_problems(self):
        self.assertEqual(self.problems(lock({"x": component()})), "")

    def test_unclassified_and_disallowed_licenses(self):
        comp = component()
        del comp["license"]
        self.assertIn("license must be an object", self.problems(lock({"x": comp})))
        comp = component(license={"files": ["LICENSE"], "scope": "s", "evidence": "e"})
        self.assertIn("unclassified components are rejected", self.problems(lock({"x": comp})))
        for expression in ("GPL-3.0-only", "MIT AND GPL-2.0-only", "LGPL-2.1-only", "CC-BY-NC-4.0",
                           "LicenseRef-NVIDIA-cutlass-dsl", "MIT AND LicenseRef-unknown"):
            comp = component(license=dict(component()["license"], expression=expression))
            self.assertIn("outside D-017's core allowlist", self.problems(lock({"x": comp})))
        comp = component(license=dict(component()["license"], expression="MIT OR GPL-2.0-only"))
        self.assertIn("joined by ' AND '", self.problems(lock({"x": comp})))

    def test_any_recognized_permissive_license_is_core(self):
        # D-091: a permissive license needs no decision of its own, CUB's and Thrust's among them.
        for expression in ("BSL-1.0", "BSD-3-Clause AND BSL-1.0 AND Apache-2.0", "ISC", "Zlib", "0BSD",
                           "Unicode-3.0", "MPL-2.0"):
            with self.subTest(expression=expression):
                comp = component(license=dict(component()["license"], expression=expression))
                self.assertEqual(self.problems(lock({"x": comp})), "")

    def test_optional_components_need_a_declared_module(self):
        comp = component(tier="optional", license=dict(component()["license"], expression="AGPL-3.0-only"))
        self.assertIn("names a declared module", self.problems(lock({"x": comp})))
        comp["module"] = "m"
        self.assertEqual(self.problems(lock({"x": comp}, {"m": {"description": "d"}})), "")
        self.assertIn("no component belongs to it",
                      self.problems(lock({"x": component()}, {"m": {"description": "d"}})))
        self.assertIn("only optional components belong to a module",
                      self.problems(lock({"x": component(module="m")}, {"m": {"description": "d"}})))

    def test_dependencies(self):
        modules = {"m": {"description": "d"}, "n": {"description": "d"}}
        self.assertIn("cannot depend on the optional", self.problems(lock(
            {"x": component(depends=["y"]), "y": component(tier="optional", module="m")}, {"m": modules["m"]})))
        self.assertIn("not in the lock", self.problems(lock({"x": component(depends=["nope"])})))
        self.assertIn("from another module", self.problems(lock(
            {"y": component(tier="optional", module="m"),
             "z": component(tier="optional", module="n", depends=["y"])}, modules)))
        self.assertIn("dependency cycle", self.problems(lock(
            {"a": component(depends=["b"]), "b": component(depends=["a"])})))

    def test_unsupported_kinds_and_identities(self):
        for change, text in (
                ({"kind": "vendored"}, "kind must be one of archive"),
                ({"machine": "build"}, "machine must be one of target"),
                ({"category": "build-tool"}, "category must be one of implementation"),
                ({"tree_sha256": "abc"}, "tree_sha256"),
                ({"upstream": {"repository": "r", "commit": "main"}}, "full commit"),
                ({"archive": dict(component()["archive"], urls=["http://example.invalid/x.tgz"])},
                 "is not an https://"),
                ({"archive": dict(component()["archive"], file="../x.tgz")}, "plain file name"),
                ({"cmake": dict(component()["cmake"], options={"X": "a b"})}, "plain string values"),
                ({"cmake": dict(component()["cmake"], subdirectory="../up")}, "relative path"),
                ({"cmake": dict(component()["cmake"], targets=[])}, "must name what llmpalooza links"),
                ({"verification": ""}, "how the pin was checked")):
            with self.subTest(change=change):
                self.assertIn(text, self.problems(lock({"x": component(**change)})))
        self.assertIn("ids are lowercase", self.problems(lock({"X": component()})))
        self.assertIn("schema must be 1", self.problems({"schema": 2, "components": {}}))

    def test_malformed_scalar_values_are_diagnostics(self):
        for change, text in (
                ({"tier": "optional", "module": []}, "names a declared module"),
                ({"tier": "optional", "module": {}}, "names a declared module"),
                ({"archive": dict(component()["archive"], size=True)}, "positive byte count"),
                ({"archive": dict(component()["archive"], urls=["https://[broken"])}, "is not an https://"),
                ({"archive": dict(component()["archive"], urls=["https:missing-host"])}, "is not an https://"),
                ({"tree_sha256": "c" * 64 + "\n"}, "tree_sha256"),
                ({"cmake": dict(component()["cmake"], options={"X\n": "OFF"})}, "plain string values"),
                ({"cmake": dict(component()["cmake"], subdirectory="unsafe;path")}, "relative path"),
                ({"license": dict(component()["license"], files=["../LICENSE"])}, "license.files"),
                ({"license": dict(component()["license"], notices=["/NOTICE"])}, "license.notices")):
            with self.subTest(change=change):
                self.assertIn(text, self.problems(lock({"x": component(**change)})))
        self.assertIn("schema must be 1", self.problems(dict(lock({"x": component()}), schema=True)))
        self.assertIn("ids are lowercase", self.problems(lock({"x\n": component()})))

    def test_options_cannot_rebind_cmake_or_llmp_variables_or_name_paths(self):
        for options, text in (
                ({"CMAKE_PROJECT_INCLUDE": "x"}, "may not set CMAKE_*"),
                ({"cmake_sysroot": "x"}, "may not set CMAKE_*"),
                ({"LLMP_PYTHON": "x"}, "may not set"),
                ({"FETCHCONTENT_FULLY_DISCONNECTED": "OFF"}, "may not set"),
                ({"BUILD_SHARED_LIBS": "ON"}, "may not set"),
                ({"_LLMP_SOURCES_SCRIPTS": "x"}, "cmake.options must map"),
                ({"X_TESTS": "/tmp/evil.cmake"}, "plain string values"),
                ({"X_TESTS": "a;b"}, "plain string values")):
            with self.subTest(options=options):
                comp = component(cmake=dict(component()["cmake"], options=options))
                self.assertIn(text, self.problems(lock({"x": comp})))
        comp = component(cmake=dict(component()["cmake"], options={"gtest_build_tests": "OFF", "X_LEVEL": "3.1"}))
        self.assertEqual(self.problems(lock({"x": comp})), "")

    def test_archive_keep(self):
        def keep(value):
            return lock({"x": component(archive=dict(component()["archive"], keep=value))})

        self.assertEqual(self.problems(keep(["LICENSE", "ggml", "include/x.h"])), "")
        for value, text in (
                ("ggml", "must list relative paths"),
                ([], "must list relative paths"),
                ([1], "must list relative paths"),
                ([""], "must list relative paths"),
                (["/ggml"], "must list relative paths"),
                (["../ggml"], "must list relative paths"),
                (["ggml/../x"], "must list relative paths"),
                (["ggml/"], "must list relative paths"),
                (["./ggml"], "must list relative paths"),
                (["web/[id]"], "must list relative paths"),
                (["ggml", "LICENSE"], "must be sorted, without duplicates"),
                (["ggml", "ggml"], "must be sorted, without duplicates"),
                (["ggml", "ggml/src"], "entry 'ggml/src' lies inside 'ggml'")):
            with self.subTest(value=value):
                self.assertIn(text, self.problems(keep(value)))
        self.assertEqual(self.problems(keep(["ggml", "ggml-extra"])), "")  # a shared prefix is not nesting

    def test_duplicate_json_keys_are_rejected(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = root / "sources.lock.json"
        path.write_text(json.dumps(lock({"x": component()})).replace('"tier": "core"',
                                                                   '"tier": "optional", "tier": "core"'))
        with self.assertRaisesRegex(SourceError, "duplicate JSON key 'tier'"):
            srclib.load_lock(path)

    def test_patches_must_exist_and_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            (base / "p.patch").write_text("patch")
            good = hashlib.sha256(b"patch").hexdigest()
            self.assertEqual(self.problems(lock({"x": component(patches=[{"path": "p.patch", "sha256": good}])}),
                                           base), "")
            self.assertIn("does not match its recorded sha256", self.problems(
                lock({"x": component(patches=[{"path": "p.patch", "sha256": "0" * 64}])}), base))
            self.assertIn("is missing", self.problems(
                lock({"x": component(patches=[{"path": "q.patch", "sha256": good}])}), base))
            self.assertIn("must stay under", self.problems(
                lock({"x": component(patches=[{"path": "../p.patch", "sha256": good}])}), base))

    def test_patch_symlinks_cannot_escape_the_lock_directory(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = root / "lock"
        base.mkdir()
        outside = root / "outside.patch"
        outside.write_bytes(b"patch")
        (base / "escape.patch").symlink_to(outside)
        self.assertIn("must stay under", self.problems(lock({"x": component(patches=[{
            "path": "escape.patch", "sha256": hashlib.sha256(b"patch").hexdigest()}])}), base))

    def test_excluded_module_patches_are_not_read(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = root / "sources.lock.json"
        data = lock({"x": component(), "o": component(tier="optional", module="m", patches=[{
            "path": "missing.patch", "sha256": "0" * 64}])}, {"m": {"description": "optional module"}})
        path.write_text(json.dumps(data))
        with mock.patch.object(srclib, "sha256_file", side_effect=AssertionError("read an excluded patch")):
            self.assertEqual(srclib.load_lock(path, modules=[]), data)
        for modules in (None, ["m"]):
            with self.subTest(modules=modules), self.assertRaisesRegex(SourceError, "patch missing.patch is missing"):
                srclib.load_lock(path, modules=modules)


class Selection(unittest.TestCase):
    MODULES = {"m": {"description": "d"}}

    def test_core_then_modules_in_dependency_order(self):
        data = lock({"b": component(depends=["a"]), "a": component(),
                     "o": component(tier="optional", module="m", depends=["b"])}, self.MODULES)
        self.assertEqual(srclib.select(data, []), ["a", "b"])
        self.assertEqual(srclib.select(data, ["m"]), ["a", "b", "o"])
        self.assertEqual(srclib.license_profile(["m"]), "core+m")
        self.assertEqual(srclib.license_profile([]), "core")

    def test_unknown_modules_and_unselected_dependencies_are_refused(self):
        data = lock({"a": component()}, {})
        with self.assertRaisesRegex(SourceError, "unknown module"):
            srclib.select(data, ["m"])
        data = lock({"a": component(depends=["o"]), "o": component(tier="optional", module="m")}, self.MODULES)
        with self.assertRaisesRegex(SourceError, "does not select"):
            srclib.select(data, [])

    def test_cuda_condition_preserves_preparation_and_legacy_selection(self):
        data = lock({"a": component(), "b": component(requires_cuda=True, depends=["a"]),
                     "c": component(requires_cuda=False)})
        self.assertEqual(srclib.select(data, []), ["a", "b", "c"])
        self.assertEqual(srclib.select(data, [], cuda=True), ["a", "b", "c"])
        self.assertEqual(srclib.select(data, [], cuda=False), ["a", "c"])
        data["components"]["a"]["depends"] = ["b"]
        data["components"]["b"]["depends"] = []
        with self.assertRaisesRegex(SourceError, "does not select"):
            srclib.select(data, [], cuda=False)

    def test_cuda_condition_requires_a_boolean(self):
        for value in (None, 0, 1, "ON", "OFF", [], {}):
            with self.subTest(value=value):
                problems = srclib.validate_lock(lock({"a": component(requires_cuda=value)}),
                                               pathlib.Path("."), check_patch_files=False)
                self.assertTrue(any("requires_cuda must be a boolean" in p for p in problems), problems)

    def test_prepared_dir_names_the_tree(self):
        self.assertEqual(srclib.prepared_dir(pathlib.Path("/s"), "x", component()), pathlib.Path("/s/x-" + "c" * 16))


class TreeDigest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "b").mkdir()
        (self.root / "b" / "y.txt").write_text("y\n")
        (self.root / "a.txt").write_text("a\n")
        (self.root / "empty").mkdir()

    def test_matches_the_documented_format(self):
        lines = "".join(f"{hashlib.sha256(data).hexdigest()} - {name}\n"
                        for name, data in (("a.txt", b"a\n"), ("b/y.txt", b"y\n")))
        self.assertEqual(srclib.tree_digest(self.root), hashlib.sha256(lines.encode()).hexdigest())

    def test_content_names_and_executable_bits_count(self):
        base = srclib.tree_digest(self.root)
        (self.root / "a.txt").chmod(0o755)
        executable = srclib.tree_digest(self.root)
        self.assertNotEqual(base, executable)
        (self.root / "a.txt").chmod(0o644)
        self.assertEqual(srclib.tree_digest(self.root), base)
        (self.root / "b" / "y.txt").write_text("z\n")
        self.assertNotEqual(srclib.tree_digest(self.root), base)

    def test_links_and_unsupported_names_are_refused(self):
        (self.root / "link").symlink_to("a.txt")
        with self.assertRaisesRegex(SourceError, "symbolic links"):
            srclib.tree_digest(self.root)
        (self.root / "link").unlink()
        for name in ("semi;colon", "[bracket]", "new\nline", "trailing\n"):
            path = self.root / name
            path.write_text("x")
            with self.subTest(name=name), self.assertRaisesRegex(SourceError, "unsupported file name"):
                srclib.tree_digest(self.root)
            path.unlink()

    def test_an_empty_tree_is_refused(self):
        empty = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (empty / "only-a-directory").mkdir()
        with self.assertRaisesRegex(SourceError, "has no files"):
            srclib.tree_digest(empty)

    def test_missing_and_symlink_roots_are_refused(self):
        for path in (self.root / "absent", self.root / "a.txt"):
            with self.subTest(path=path), self.assertRaisesRegex(SourceError, "not a prepared directory"):
                srclib.tree_digest(path)
        link = self.root / "link"
        link.symlink_to(self.root / "b", target_is_directory=True)
        with self.assertRaisesRegex(SourceError, "not a prepared directory"):
            srclib.tree_digest(link)

    def test_unreadable_subdirectories_do_not_disappear_from_the_digest(self):
        path = self.root / "b"
        path.chmod(0)
        try:
            if os.geteuid() == 0:
                self.skipTest("root bypasses the unreadable-directory fixture")
            with self.assertRaisesRegex(SourceError, "cannot scan prepared tree"):
                srclib.tree_digest(self.root)
        finally:
            path.chmod(0o755)


class Patches(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "src").mkdir()
        (self.root / "src" / "f.h").write_text("one\ntwo\nthree\n")

    def test_modifies_creates_and_deletes(self):
        (self.root / "gone.txt").write_text("bye\n")
        srclib.apply_patch(self.root, (
            "diff --git a/src/f.h b/src/f.h\n--- a/src/f.h\n+++ b/src/f.h\n"
            "@@ -1,3 +1,4 @@\n one\n-two\n+TWO\n+2.5\n three\n"
            "--- /dev/null\n+++ b/new/n.txt\n@@ -0,0 +1,2 @@\n+a\n+b\n\\ No newline at end of file\n"
            "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"))
        self.assertEqual((self.root / "src" / "f.h").read_text(), "one\nTWO\n2.5\nthree\n")
        self.assertEqual((self.root / "new" / "n.txt").read_text(), "a\nb")
        self.assertFalse((self.root / "gone.txt").exists())

    def test_keeps_crlf_and_mode(self):
        path = self.root / "src" / "w.txt"
        path.write_bytes(b"a\r\nb\r\n")
        path.chmod(0o755)
        srclib.apply_patch(self.root, "--- a/src/w.txt\n+++ b/src/w.txt\n@@ -1,2 +1,2 @@\n a\r\n-b\r\n+c\r\n")
        self.assertEqual(path.read_bytes(), b"a\r\nc\r\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o755)

    def test_refuses_mismatches_and_unsafe_patches(self):
        for text, message in (
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1,2 +1,2 @@\n one\n-TWO\n+2\n", "context does not match"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -9,1 +9,1 @@\n-x\n+y\n", "past the end"),
                ("--- a/../x\n+++ b/../x\n@@ -1 +1 @@\n-a\n+b\n", "leaves the source tree"),
                ("--- src/f.h\n+++ src/f.h\n@@ -1 +1 @@\n-one\n+1\n", "must start with a/ or b/"),
                ("--- a/src/f.h\n+++ b/src/g.h\n@@ -1 +1 @@\n-one\n+1\n", "renames are not supported"),
                ("diff --git a/src/f.h b/src/f.h\nold mode 100644\nnew mode 100755\n", "not supported"),
                ("--- /dev/null\n+++ b/src/f.h\n@@ -0,0 +1 @@\n+x\n", "which exists"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1,3 +1,3 @@\n one\n-two\n", "ends inside a hunk"),
                ("just text\n", "changes no files")):
            with self.subTest(message=message), self.assertRaisesRegex(SourceError, message):
                srclib.apply_patch(self.root, text)

    def test_refuses_to_patch_through_a_link(self):
        outside = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (outside / "f.h").write_text("one\n")
        (self.root / "linked").symlink_to(outside)
        with self.assertRaisesRegex(SourceError, "symbolic link"):
            srclib.apply_patch(self.root, "--- a/linked/f.h\n+++ b/linked/f.h\n@@ -1 +1 @@\n-one\n+1\n")
        self.assertEqual((outside / "f.h").read_text(), "one\n")

    def test_hunk_positions_and_line_counts_are_exact(self):
        for text, message in (
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +99 @@\n-one\n+1\n", "new hunk position"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n+unrecorded\n", "unexpected line after"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1", "truncated hunk line"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n\\ anything\n", "unknown newline marker"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n\\ No newline at end of file\n",
                 "before the end"),
                ("diff --git a/new.sh b/new.sh\nnew file mode 100755\n--- /dev/null\n+++ b/new.sh\n"
                 "@@ -0,0 +1 @@\n+x\n", "not supported")):
            with self.subTest(message=message), self.assertRaisesRegex(SourceError, message):
                srclib.apply_patch(self.root, text)
            self.assertEqual((self.root / "src" / "f.h").read_text(), "one\ntwo\nthree\n")

    def test_lines_between_files_must_be_git_headers(self):
        base = "diff --git a/src/f.h b/src/f.h\nindex 1..2 100644\n--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n"
        with self.assertRaisesRegex(SourceError, "unexpected 'stray text' between files"):
            srclib.apply_patch(self.root, base + "stray text\n")
        self.assertEqual((self.root / "src" / "f.h").read_text(), "1\ntwo\nthree\n")  # staged callers discard

    def test_preamble_and_signature_are_skipped(self):
        srclib.apply_patch(self.root, "From 0000 Mon Sep 17 00:00:00 2001\nSubject: change\n\nrename nothing\n---\n"
                                      " src/f.h | 2 +-\n\ndiff --git a/src/f.h b/src/f.h\n--- a/src/f.h\n"
                                      "+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n-- \n2.43.0\n")
        self.assertEqual((self.root / "src" / "f.h").read_text(), "1\ntwo\nthree\n")

    def test_old_file_final_newline_is_exact(self):
        path = self.root / "src" / "last.txt"
        header = "--- a/src/last.txt\n+++ b/src/last.txt\n@@ -1 +1 @@\n"
        for original, removed in (("a\n", "-a\n\\ No newline at end of file\n"), ("a", "-a\n")):
            path.write_text(original)
            with self.subTest(original=original), self.assertRaisesRegex(SourceError, "context does not match"):
                srclib.apply_patch(self.root, header + removed + "+b\n")
            self.assertEqual(path.read_text(), original)
        path.write_text("a")
        srclib.apply_patch(self.root, header + "-a\n\\ No newline at end of file\n+b\n")
        self.assertEqual(path.read_bytes(), b"b\n")


class Archives(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))

    def tar(self, *members: tarfile.TarInfo, name="a.tar.gz") -> pathlib.Path:
        path = self.tmp / name
        with tarfile.open(path, "w:gz") as tar:
            data = b"x\n"
            info = tarfile.TarInfo("x-1.0/file.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            for member in members:
                tar.addfile(member)
        return path

    def member(self, name: str, kind: bytes, link: str = "") -> tarfile.TarInfo:
        info = tarfile.TarInfo(name)
        info.type, info.linkname = kind, link
        return info

    def test_plain_files_and_directories_pass(self):
        srclib.check_archive(self.tar(self.member("x-1.0/dir", tarfile.DIRTYPE)))

    def test_links_specials_and_unsafe_paths_are_refused(self):
        for member, text in (
                (self.member("x-1.0/escape", tarfile.SYMTYPE, "../../outside"), "not a file or directory"),
                (self.member("x-1.0/hard", tarfile.LNKTYPE, "x-1.0/file.txt"), "not a file or directory"),
                (self.member("x-1.0/pipe", tarfile.FIFOTYPE), "not a file or directory"),
                # libarchive extracts a regular-file member with a link name as a hard link.
                (self.member("x-1.0/linked", tarfile.REGTYPE, "x-1.0/file.txt"), "not a file or directory"),
                (self.member("../x-1.0/up.txt", tarfile.REGTYPE), "unsafe or unsupported path"),
                (self.member("/abs.txt", tarfile.REGTYPE), "unsafe or unsupported path"),
                (self.member("x-1.0/semi;colon", tarfile.REGTYPE), "unsafe or unsupported path")):
            with self.subTest(name=member.name), self.assertRaisesRegex(SourceError, text):
                srclib.check_archive(self.tar(member))

    def test_zip_archives_are_refused(self):
        path = self.tmp / "a.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("x-1.0/file.txt", "x\n")
        with self.assertRaisesRegex(SourceError, "is not a tar archive"):
            srclib.check_archive(path)

    def test_duplicate_paths_and_members_inside_files_are_refused(self):
        data = tarfile.TarInfo("x-1.0/file.txt")
        with self.assertRaisesRegex(SourceError, "same path"):
            srclib.check_archive(self.tar(data))  # self.tar adds x-1.0/file.txt too
        with self.assertRaisesRegex(SourceError, "inside a file member"):
            srclib.check_archive(self.tar(self.member("x-1.0/file.txt/inner", tarfile.REGTYPE)))

    def archive(self, kind: str, files: list[str], specials: tuple[str, ...] = ()) -> pathlib.Path:
        """A tar archive of empty files, plus symbolic links at specials."""
        path = self.tmp / f"k.{kind}"
        path.unlink(missing_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            for name in files:
                tar.addfile(tarfile.TarInfo(name))
            for name in specials:
                tar.addfile(self.member(name, tarfile.SYMTYPE, "../../outside"))
        return path

    KEPT = ["x-1.0/LICENSE", "x-1.0/ggml/src/a.c", "x-1.0/ggml/include/a.h"]
    WEB = "x-1.0/tools/ui/src/routes/(chat)/chat/[id]/+page.svelte"

    def test_keep_allows_unsupported_names_only_where_discarded(self):
        for kind in ("tar.gz",):
            with self.subTest(kind=kind):
                path = self.archive(kind, self.KEPT + [self.WEB, "x-1.0/web dir/@{x}!"])
                srclib.check_archive(path, ["LICENSE", "ggml"])
                with self.assertRaisesRegex(SourceError, "unsafe or unsupported path"):
                    srclib.check_archive(path)
                with self.assertRaisesRegex(SourceError, r"member '.*\[id\].*' has an unsafe"):
                    srclib.check_archive(path, ["LICENSE", "ggml", "tools"])
                path = self.archive(kind, self.KEPT + ["x-1.0/ggml/[id].c"])
                with self.assertRaisesRegex(SourceError, "unsafe or unsupported path"):
                    srclib.check_archive(path, ["ggml"])

    def test_keep_still_refuses_links_and_unsafe_discarded_names(self):
        for kind in ("tar.gz",):
            for files, specials, text in (
                    (self.KEPT, ("x-1.0/tools/link",), "not a file or directory"),
                    (self.KEPT + ["x-1.0/tools/back\\slash"], (), "unsafe or unsupported path"),
                    (self.KEPT + ["x-1.0/tools/new\nline"], (), "unsafe or unsupported path"),
                    (self.KEPT + ["x-1.0/tools/caf\u00e9"], (), "unsafe or unsupported path"),
                    (self.KEPT + ["x-1.0/tools/../../up"], (), "unsafe or unsupported path"),
                    (self.KEPT + ["/abs"], (), "unsafe or unsupported path")):
                with self.subTest(kind=kind, files=files, specials=specials):
                    with self.assertRaisesRegex(SourceError, text):
                        srclib.check_archive(self.archive(kind, files, specials), ["LICENSE", "ggml"])
        with self.assertRaisesRegex(SourceError, "not a file or directory"):
            srclib.check_archive(self.tar(self.member("x-1.0/tools/fifo", tarfile.FIFOTYPE)), ["file.txt"])

    def test_every_kept_path_must_name_a_file(self):
        for kind in ("tar.gz",):
            with self.subTest(kind=kind):
                path = self.archive(kind, self.KEPT)
                for keep, missing in ((["ggml", "src"], "src"), (["LICENSE/x"], "LICENSE/x"), (["ggm"], "ggm"),
                                      (["x-1.0"], "x-1.0")):
                    with self.assertRaisesRegex(SourceError, f"archive.keep entry '{missing}' names no file"):
                        srclib.check_archive(path, keep)
                srclib.check_archive(path, ["ggml/include/a.h", "ggml/src"])
        path = self.tar(self.member("x-1.0/empty", tarfile.DIRTYPE))
        with self.assertRaisesRegex(SourceError, "'empty' names no file"):
            srclib.check_archive(path, ["empty"])

    def test_keep_paths_are_relative_to_the_tree_population_makes(self):
        # A single top-level directory is stripped (a top-level .DS_Store is
        # dropped with it); anything else unpacks as it is.
        srclib.check_archive(self.archive("tar.gz", self.KEPT + [".DS_Store"]), ["ggml"])
        path = self.archive("tar.gz", ["LICENSE", "ggml/a.c", "web/[id]/x"])
        srclib.check_archive(path, ["LICENSE", "ggml"])
        path = self.archive("tar.gz", ["x-1.0/ggml/a.c", "y-1.0/web/[id]/x"])
        srclib.check_archive(path, ["x-1.0/ggml"])
        with self.assertRaisesRegex(SourceError, "'ggml' names no file"):
            srclib.check_archive(path, ["ggml"])

    def test_an_unpacked_tree_holds_only_plain_files_and_directories(self):
        root = self.tmp / "unpacked"
        (root / "d").mkdir(parents=True)
        (root / "d" / "a.txt").write_text("a")
        srclib.check_unpacked(root)
        os.link(root / "d" / "a.txt", root / "hard.txt")
        with self.assertRaisesRegex(SourceError, "hard.txt"):
            srclib.check_unpacked(root)
        (root / "hard.txt").unlink()
        (root / "d" / "link").symlink_to(self.tmp)
        with self.assertRaisesRegex(SourceError, "link"):
            srclib.check_unpacked(root)

    def test_unpacking_writes_only_kept_paths_from_the_checked_members(self):
        path = self.archive("tar.gz", self.KEPT + [self.WEB, "x-1.0/tools/other.txt"])
        dest = self.tmp / "unpacked"
        srclib.unpack_archive(path, dest, ["LICENSE", "ggml/src"])
        self.assertEqual(sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")),
                         ["LICENSE", "ggml", "ggml/src", "ggml/src/a.c"])
        with self.assertRaisesRegex(SourceError, "exists"):
            srclib.unpack_archive(path, dest, ["LICENSE"])
        # Without keep, everything under the stripped top directory.
        whole = self.tmp / "whole"
        srclib.unpack_archive(self.archive("tar.gz", self.KEPT), whole)
        self.assertTrue((whole / "ggml" / "include" / "a.h").is_file())
        # What the check refuses is never written.
        refused = self.tmp / "refused"
        with self.assertRaisesRegex(SourceError, "not a file or directory"):
            srclib.unpack_archive(self.archive("tar.gz", self.KEPT, ("x-1.0/tools/link",)), refused)
        self.assertEqual(list(refused.iterdir()), [])

    def test_truncated_sparse_and_timeless_members_are_refused(self):
        # tarfile ends an archive at the first unreadable header; the rest
        # would silently go missing.
        whole = self.archive("tar.gz", self.KEPT)
        data = gzip.decompress(whole.read_bytes())
        cut = self.tmp / "cut.tar"
        cut.write_bytes(data[:512 * 3])
        with self.assertRaisesRegex(SourceError, "does not end"):
            srclib.check_archive(cut)
        cut.write_bytes(data[:512 * 3] + bytes(512))  # one zero block, not two
        with self.assertRaisesRegex(SourceError, "does not end"):
            srclib.check_archive(cut)
        bad = bytearray(data)
        bad[512 + 148:512 + 156] = b"0000000\0"  # member 2's checksum: tarfile stops there
        cut.write_bytes(bytes(bad))
        with self.assertRaisesRegex(SourceError, "data or excess padding"):
            srclib.check_archive(cut)
        broken = self.tmp / "broken.tar.gz"
        broken.write_bytes(whole.read_bytes()[:-4])  # the gzip trailer cut off
        with self.assertRaisesRegex(SourceError, "cannot read the archive"):
            srclib.check_archive(broken)
        padded = self.tmp / "padded.tar"
        padded.write_bytes(data + bytes(17 << 20))
        with self.assertRaisesRegex(SourceError, "excess padding"):
            srclib.check_archive(padded)
        sparse = tarfile.TarInfo("x-1.0/sparse")
        sparse.pax_headers = {"GNU.sparse.size": str(2**40), "GNU.sparse.map": "0,0"}
        with self.assertRaisesRegex(SourceError, "not a file or directory"):
            srclib.check_archive(self.tar(sparse))
        timeless = tarfile.TarInfo("x-1.0/timeless")
        timeless.pax_headers = {"mtime": "1e400"}
        path = self.tar(timeless)
        with self.assertRaisesRegex(SourceError, "modification time"):
            srclib.check_archive(path)

    def test_unreadable_archives_are_refused(self):
        path = self.tmp / "junk.tar.gz"
        path.write_bytes(b"not an archive")
        with self.assertRaisesRegex(SourceError, "is not a tar"):
            srclib.check_archive(path)


class Fetch(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.origin = self.tmp / "x-1.0.tar.gz"
        self.origin.write_bytes(b"archive bytes")
        self.cache = self.tmp / "cache"
        self.archive = {"file": "x-1.0.tar.gz", "urls": [self.origin.as_uri()],
                        "sha256": hashlib.sha256(b"archive bytes").hexdigest(), "size": len(b"archive bytes")}

    def fetch(self, archive=None):
        return srclib.fetch(self.cache, "x", archive or self.archive, log=lambda _: None)

    def test_fetches_once_into_the_shared_download_cache(self):
        path = self.fetch()
        self.assertEqual(path, self.cache / "downloads" / self.archive["sha256"] / "x-1.0.tar.gz")
        self.assertEqual(path.read_bytes(), b"archive bytes")
        with mock.patch.object(srclib.urllib.request, "urlopen", side_effect=AssertionError("fetched again")):
            self.assertEqual(self.fetch(), path)

    def test_a_damaged_cache_entry_is_replaced(self):
        path = self.fetch()
        path.write_bytes(b"archive bytez")
        self.assertEqual(self.fetch().read_bytes(), b"archive bytes")

    def test_wrong_or_oversized_bytes_are_refused(self):
        self.origin.write_bytes(b"archive bytez")
        with self.assertRaisesRegex(SourceError, "not the locked"):
            self.fetch()
        self.origin.write_bytes(b"archive bytes and more")
        with self.assertRaisesRegex(SourceError, "more than the locked"):
            self.fetch()
        self.assertEqual([p.name for p in (self.cache / "downloads" / self.archive["sha256"]).iterdir()], [])


class PrepareSources(unittest.TestCase):
    def test_a_modified_prepared_tree_is_an_error(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        tree = root / "x"
        tree.mkdir()
        (tree / "f").write_text("f\n")
        comp = component(tree_sha256=srclib.tree_digest(tree))
        self.assertTrue(prepare_sources.check_prepared(tree, "x", comp))
        self.assertFalse(prepare_sources.check_prepared(root / "absent", "x", comp))
        (tree / "f").write_text("g\n")
        with self.assertRaisesRegex(SourceError, "was modified"):
            prepare_sources.check_prepared(tree, "x", comp)



class Setup(unittest.TestCase):
    def steps(self, *args: str) -> list[list[str]]:
        calls = []
        with mock.patch.object(setup, "run", side_effect=lambda cmd: calls.append(cmd[1:]) or 0), \
                mock.patch.object(sys, "argv", ["setup", *args]):
            self.assertEqual(setup.main(), 0)
        return [[pathlib.Path(c[0]).name, *c[1:]] for c in calls]

    def test_prepares_sources_after_the_sdk(self):
        self.assertEqual(self.steps(), [["setup-toolchain"], ["prepare-sources"]])
        self.assertEqual(self.steps("--jobs", "4"), [["setup-toolchain", "--jobs", "4"], ["prepare-sources"]])
        self.assertEqual(self.steps("--dry-run"), [["setup-toolchain", "--dry-run"], ["prepare-sources", "--dry-run"]])

    def test_sdk_only_actions_and_failures_stop_there(self):
        for flag in ("--print-root", "--list", "--prune"):
            self.assertEqual(self.steps(flag), [["setup-toolchain", flag]])
        with mock.patch.object(setup, "run", return_value=1) as run, mock.patch.object(sys, "argv", ["setup"]):
            self.assertEqual(setup.main(), 1)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
