# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the build inventory audit, without a compiler or SDK."""

import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "tests/sources/check_closure.py"
SPEC = importlib.util.spec_from_file_location("check_closure", SCRIPT)
closure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(closure)


class Inventory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.source = self.root / "repo"
        self.build = self.source / "build/cpu"
        self.build.mkdir(parents=True)
        self.sdk = self.root / "sdk"
        self.sdk.mkdir()
        self.prepared = self.source / "build/sources"
        self.tree = self.prepared / ("selected-" + "a" * 16)
        self.tree.mkdir(parents=True)
        self.component_output = "third_party/selected-" + "a" * 16
        self.receipt = {"components": [{"id": "selected", "source_tree": "a" * 64,
                                         "source": str(self.tree)}]}
        (self.build / "CMakeCache.txt").write_text(f"LLMP_SOURCES_DIR:PATH={self.prepared}\n")
        self.commands = [{"directory": str(self.build), "file": str(self.tree / "source.cc"),
                          "output": str(self.build / "object.o"),
                          "command": "clang++ -O2 -fno-exceptions -c source.cc -o object.o"}]
        self.targets = {"object.o": "CXX_COMPILER__example", "app": "CXX_EXECUTABLE_LINKER__app"}
        self.recording_rules = set(self.targets.values())
        self.default_inputs = set(self.targets)
        self.deps = {"object.o": [str(self.tree / "source.cc")], "app": ["object.o"]}
        self.stale = set()
        self.link_rule = "CXX_EXECUTABLE_LINKER__app"
        self.link_inputs = "object.o"
        self.link_lines = ""
        self.extra_edges = ""
        self.symbols = ""  # what the fake llvm-nm -A -u reports
        self.sections = ""  # what the fake llvm-readelf -S reports
        self.closures = {}  # what the fake ninja -t inputs reports for one target

    def ninja(self, command, **kwargs):
        if command[0].endswith(("llvm-nm", "llvm-readelf")):
            out = self.symbols if command[0].endswith("llvm-nm") else self.sections
            return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")
        tool = command[command.index("-t") + 1]
        if tool == "targets":
            out = "".join(f"{name}: {rule}\n" for name, rule in self.targets.items())
        elif tool == "inputs" and command[-1] == "all":
            out = "\0".join(sorted(self.default_inputs)) + "\0"
        elif tool == "inputs":
            out = "".join(f"{path}\0" for path in self.closures.get(command[-1], []))
        elif tool == "deps":
            out = ""
            for name in command[command.index("deps") + 1:]:
                if name in self.deps:
                    state = "STALE" if name in self.stale else "VALID"
                    out += f"{name}: #deps {len(self.deps[name])}, deps mtime 1 ({state})\n"
                    out += "".join(f"    {path}\n" for path in self.deps[name]) + "\n"
                else:
                    out += f"{name}: deps not found\n"
        else:
            self.fail(f"unexpected Ninja tool: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")

    def check(self, expected=0, message=None):
        (self.build / "llmp-receipt.json").write_text(json.dumps(self.receipt))
        (self.build / "compile_commands.json").write_text(json.dumps(self.commands))
        (self.build / "build.ninja").write_text(
            f"build app: {self.link_rule} {self.link_inputs}\n{self.link_lines}\n{self.extra_edges}")
        (self.build / "CMakeFiles").mkdir(exist_ok=True)
        (self.build / "CMakeFiles/rules.ninja").write_text("".join(
            f"rule {rule}\n" + ("  depfile = $DEP_FILE\n  deps = gcc\n" if rule in self.recording_rules else "")
            + "  command = compiler-or-linker\n\n" for rule in set(self.targets.values())))
        argv = [str(SCRIPT), "--build-dir", str(self.build), "--source-dir", str(self.source),
                "--sdk", str(self.sdk), "--ninja", "ninja", "--cross"]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(closure.subprocess, "run", self.ninja), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = closure.main()
        self.assertEqual(result, expected, stderr.getvalue())
        if message:
            self.assertIn(message, stderr.getvalue())

    def test_valid_complete_inventory(self):
        self.check()

    def test_absolute_unselected_component_output_rejected(self):
        self.deps["app"].append(str(self.build / "third_party/unselected-old/lib.a"))
        self.check(1, "from no component in the receipt")

    def test_relative_unselected_component_output_rejected(self):
        self.deps["app"].append("third_party/unselected-old/lib.a")
        self.check(1, "from no component in the receipt")

    def test_alias_to_unselected_component_output_rejected(self):
        (self.build / "alias").symlink_to(self.build / "third_party/unselected-old", target_is_directory=True)
        self.deps["app"].append("alias/lib.a")
        self.check(1, "from no component in the receipt")

    def test_absolute_selected_component_output_accepted(self):
        self.deps["app"].append(str(self.build / self.component_output / "lib.a"))
        self.check()

    def test_missing_compile_dependencies_rejected(self):
        del self.deps["object.o"]
        self.check(1, "object.o has no current dependency record")

    def test_stale_link_dependencies_rejected(self):
        self.stale.add("app")
        self.check(1, "app has no current dependency record")

    def test_gnu_link_without_dependency_record_accepted(self):
        self.recording_rules.remove(self.link_rule)
        del self.deps["app"]
        self.check()

    def test_gnu_link_without_depfile_still_checks_inputs_and_flags(self):
        self.recording_rules.remove(self.link_rule)
        del self.deps["app"]
        self.link_inputs += " " + str(self.root / "external.o")
        self.check(1, "outside the source tree")
        self.link_inputs = "object.o"
        self.link_lines = "  LINK_FLAGS = -Wl,-lexternal\n"
        self.check(1, "not a declared platform library")

    def test_unbuilt_excluded_target_does_not_require_dependencies(self):
        self.targets["unused.o"] = "CXX_COMPILER__excluded"
        self.commands.append({**self.commands[0], "output": str(self.build / "unused.o")})
        self.check()

    def test_unbuilt_excluded_component_is_not_counted_as_used(self):
        other = self.prepared / "unused"
        self.receipt["components"].append({"id": "unused", "source_tree": "b" * 64, "source": str(other)})
        self.targets["unused.o"] = "CXX_COMPILER__excluded"
        self.commands.append({**self.commands[0], "file": str(other / "source.cc"),
                              "output": str(self.build / "unused.o")})
        self.extra_edges = (f"build third_party/unused-{'b' * 16}/unused.a: "
                            f"CXX_STATIC_LIBRARY_LINKER__unused third_party/unused-{'b' * 16}/unused.o\n")
        self.check(1, "the receipt lists unused, which nothing compiles or links")

    def test_cuda_link_flags_library_rejected(self):
        self.link_rule = self.targets["app"] = "CUDA_EXECUTABLE_LINKER__app"
        del self.deps["app"]
        self.link_lines = f"  LINK_FLAGS = {self.root / 'libexternal.a'}\n"
        self.check(1, "outside the source tree")

    def test_forwarded_library_flag_rejected(self):
        self.link_lines = "  LINK_LIBRARIES = -Wl,-lexternal\n"
        self.check(1, "not a declared platform library")

    def test_split_library_flag_rejected(self):
        self.link_lines = "  LINK_LIBRARIES = -l external\n"
        self.check(1, "not a declared platform library")

    def test_long_forwarded_library_flag_rejected(self):
        for option in ("-Wl,--library=external", "-Xlinker --library -Xlinker external"):
            with self.subTest(option=option):
                self.link_lines = f"  LINK_FLAGS = {option}\n"
                self.check(1, "not a declared platform library")

    def test_cuda_device_linker_forwarding_rejected(self):
        self.link_rule = self.targets["app"] = "CUDA_EXECUTABLE_DEVICE_LINKER__app"
        del self.deps["app"]
        for option in ("-Xlinker=-lexternal", "--linker-options=-lexternal",
                       "--linker-options --start-group,-lexternal,--end-group",
                       "-Xlinker --start-group,-lexternal,--end-group"):
            with self.subTest(option=option):
                self.link_lines = f"  LINK_FLAGS = {option}\n"
                self.check(1, "not a declared platform library")

    def test_platform_libraries_and_non_input_flags_accepted(self):
        self.link_lines = ("  LINK_FLAGS = -Xlinker -rpath -Xlinker /runtime-only "
                           "-Wl,-z,now,--soname,libexample.so\n"
                           "  LINK_LIBRARIES = -Wl,-lc -l m -Xlinker --library=dl\n")
        self.check()

    def test_link_library_with_spaces_and_dollar_signs(self):
        library = self.build / "own library$.a"
        self.targets["own library$.a"] = "CXX_STATIC_LIBRARY_LINKER__own"
        self.link_inputs = str(library).replace("$", "$$").replace(" ", "$ ")
        self.link_lines = '  LINK_LIBRARIES = "own library$$.a"\n'
        self.check()

    def test_build_tree_file_no_rule_produced_rejected(self):
        # A module's leftover, or a file a script wrote at configure.
        self.deps["object.o"].append(str(self.build / "leftover.h"))
        self.check(1, "a file in the build tree that no build rule or selected component produced")

    def test_build_tree_rule_outputs_and_component_outputs_accepted(self):
        self.targets["generated/config.h"] = "CUSTOM_COMMAND"
        self.deps["object.o"] += [str(self.build / "generated/config.h"),
                                  str(self.build / self.component_output / "configured.h")]
        self.check()

    ELF_OBJECT = b"\x7fELF\x02\x01\x01" + bytes(9) + (1).to_bytes(2, "little")

    def test_linked_object_without_compile_record_is_audited_for_exceptions(self):
        side = self.build / "side.o"
        side.write_bytes(self.ELF_OBJECT)
        self.targets["side.o"] = "CUSTOM_COMMAND"
        self.link_inputs = "object.o side.o"
        self.deps["app"].append("side.o")
        self.symbols = f"{side}:                  U __cxa_throw\n"
        self.check(1, f"{side} was compiled with exceptions: it has __cxa_throw")

    def test_objects_and_archives_are_found_by_content_not_name(self):
        for name, data in (("side.O", self.ELF_OBJECT), ("sideobj", self.ELF_OBJECT),
                           ("libside.lib", b"!<arch>\n"), ("side.bc", b"BC\xc0\xde")):
            with self.subTest(name=name):
                side = self.build / name
                side.write_bytes(data)
                self.targets[name] = "CUSTOM_COMMAND"
                self.link_inputs = f"object.o {name}"
                self.symbols = f"{side}:                  U __gxx_personality_v0\n"
                self.check(1, f"{side} was compiled with exceptions")

    def test_linker_scripts_and_other_files_are_not_link_inputs(self):
        script = self.build / "extra.ld"
        script.write_text("INPUT(hidden.o)\n")
        self.targets["extra.ld"] = "CUSTOM_COMMAND"
        self.link_inputs = "object.o extra.ld"
        self.check(1, "which is a linker script or unknown file, not an object file or archive")

    def test_landing_pad_table_is_exception_support(self):
        side = self.build / "side.o"
        side.write_bytes(self.ELF_OBJECT)
        self.targets["side.o"] = "CUSTOM_COMMAND"
        self.link_inputs = "object.o side.o"
        self.sections = f"File: {side}\n  [ 4] .gcc_except_table PROGBITS\n"
        self.check(1, "it has .gcc_except_table")

    def test_shared_library_link_rejected_whatever_its_name(self):
        (self.build / "libown.a").write_bytes(b"\x7fELF\x02\x01\x01" + bytes(9) + (3).to_bytes(2, "little"))
        self.targets["libown.a"] = "CXX_SHARED_LIBRARY_LINKER__own"
        self.link_lines = "  LINK_LIBRARIES = libown.a\n"
        self.check(1, "a link uses the shared library libown.a")

    RUNTIME = "src/runtime/llmp-runtime"

    def ship(self, *built_from):
        """A test-only component the test executable uses, and the shipped runtime built from built_from."""
        tree = self.prepared / ("testonly-" + "b" * 16)
        tree.mkdir()
        self.receipt["components"][0]["use"] = "product"
        self.receipt["components"].append({"id": "testonly", "source_tree": "b" * 64, "source": str(tree),
                                           "use": "test"})
        self.targets["test_object.o"] = "CXX_COMPILER__example"
        self.deps["test_object.o"] = [str(tree / "kernel.cc"), str(tree / "kernel.h")]
        self.link_inputs = "object.o test_object.o"
        self.deps["app"].append("test_object.o")
        self.targets[self.RUNTIME] = "CXX_EXECUTABLE_LINKER__runtime"
        self.extra_edges += f"build {self.RUNTIME}: CXX_EXECUTABLE_LINKER__runtime {' '.join(built_from)}\n"
        self.closures[self.RUNTIME] = list(built_from)
        return tree

    def test_shipped_executable_from_product_components_accepted(self):
        self.ship("object.o")
        self.check()

    def test_shipped_executable_linking_a_test_only_component_rejected(self):
        # Its notices are not in the package (tools/llmp_package.py ships only `use: product` ones).
        self.ship("object.o", "test_object.o")
        self.check(1, f"{self.RUNTIME}, which the package ships, is built from testonly, a test-only component")

    def test_shipped_executable_including_a_test_only_header_rejected(self):
        tree = self.ship("object.o", "runtime.o")
        self.targets["runtime.o"] = "CXX_COMPILER__example"
        self.deps["runtime.o"] = [str(self.source / "runtime.cc"), str(tree / "kernel.h")]
        self.check(1, f"{self.RUNTIME}, which the package ships, is built from testonly, a test-only component")

    def test_shipped_executable_linking_a_test_only_component_output_rejected(self):
        self.ship("object.o", f"third_party/testonly-{'b' * 16}/libkernels.a")
        self.check(1, f"{self.RUNTIME}, which the package ships, is built from testonly, a test-only component")

    def test_shipped_executable_linking_a_test_only_archive_through_linker_options_rejected(self):
        # A path in LINK_FLAGS (target_link_options(... "LINKER:...")) is no Ninja input of the edge,
        # but the linker reads it all the same, alone or inside a -Wl, or -Xlinker list.
        archive = f"third_party/testonly-{'b' * 16}/libkernels.a"
        for flags in (f"-Wl,--whole-archive,{archive},--no-whole-archive", f"-Xlinker {archive}",
                      f"-Wl,--push-state -Wl,{archive} -Wl,--pop-state"):
            with self.subTest(flags=flags):
                self.setUp()
                self.ship("object.o")
                self.extra_edges += f"  LINK_FLAGS = {flags}\n"
                self.check(1, f"{self.RUNTIME}, which the package ships, is built from testonly, a test-only "
                              "component")

    def test_shipped_executable_linking_a_test_only_linker_script_rejected(self):
        # A linker script can name inputs of its own (INPUT, GROUP); one passed as an option's
        # value is read as surely as a bare path.
        script = f"third_party/testonly-{'b' * 16}/kernels.ld"
        for flags in (f"-Wl,--script={script}", f"-Wl,-T{script}", f"-Xlinker -T{script}",
                      f"-Xlinker=--script={script}"):
            with self.subTest(flags=flags):
                self.setUp()
                self.ship("object.o")
                self.extra_edges += f"  LINK_FLAGS = {flags}\n"
                self.check(1, f"{self.RUNTIME}, which the package ships, is built from testonly, a test-only "
                              "component")

    def test_shipped_executables_are_the_packaged_ones(self):
        sys.path.insert(0, str(SCRIPT.parents[2] / "tools"))
        import llmp_package  # noqa: PLC0415 (the package tool needs the tools directory on the path)

        self.assertEqual(closure.SHIPPED_EXECUTABLES, tuple(built for built, _ in llmp_package.EXECUTABLES))

    def test_libstdcxx_assertions_all_or_none(self):
        # D-083: as libstdc++'s c++config.h decides, explicitly or for an unoptimized compile.
        on, off = "does not define _GLIBCXX_ASSERTIONS", "defines _GLIBCXX_ASSERTIONS"
        for command, want, problem in (
                ("clang++ -O2 -c a.cc", False, None),
                ("clang++ -O2 -D_GLIBCXX_ASSERTIONS -c a.cc", True, None),
                ("clang++ -O2 -D _GLIBCXX_ASSERTIONS -c a.cc", True, None),
                ("clang++ -O2 -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_NO_ASSERTIONS -c a.cc", True, None),
                ("clang++ -O2 -c a.cc", True, on),
                ("clang++ -O2 -D_GLIBCXX_ASSERTIONS -U_GLIBCXX_ASSERTIONS -c a.cc", True, on),
                ("clang++ -O2 -D_GLIBCXX_ASSERTIONS -c a.cc", False, off),
                ("clang++ -O2 -D_GLIBCXX_DEBUG -c a.cc", False, off),
                ("clang++ -c a.cc", False, off),
                ("clang++ -O2 -O0 -c a.cc", False, off),
                ("clang++ -O0 -D_GLIBCXX_NO_ASSERTIONS -c a.cc", False, None),
                ("clang++ -O0 -c a.cc", True, None),
                ("nvcc -O3 -Xcompiler=-U_GLIBCXX_ASSERTIONS -D_GLIBCXX_ASSERTIONS -c a.cu", True, None),
                ("nvcc -O3 -D_GLIBCXX_ASSERTIONS -Xcompiler=-fPIC,-U_GLIBCXX_ASSERTIONS -c a.cu", True, on),
                ("nvcc -O3 --compiler-options -D_GLIBCXX_ASSERTIONS -c a.cu", False, off),
                ("nvcc -O3 --define-macro=_GLIBCXX_ASSERTIONS -c a.cu", False, off),
                ("nvcc -O3 -Xcompiler=-O0 -c a.cu", False, off),
                ("nvcc -c a.cu", False, off),
                ("nvcc --optimize 0 -c a.cu", False, off),
                ("nvcc -O3 -Xptxas -O0 -c a.cu", False, None)):
            with self.subTest(command=command, want=want):
                found = closure.assertions_problem(command.split(), want)
                if problem is None:
                    self.assertIsNone(found)
                else:
                    self.assertIsNotNone(found)
                    self.assertTrue(found.startswith(problem), found)

    def test_unoptimized_compile_rejected_where_the_build_has_no_assertions(self):
        # libstdc++ turns its assertions on for a compile without -O, whatever the preset says.
        self.commands[0]["command"] = "clang++ -fno-exceptions -c source.cc -o object.o"
        self.check(1, "defines _GLIBCXX_ASSERTIONS (unoptimized, or through _GLIBCXX_DEBUG), which this build "
                      "defines nowhere")

    def test_static_archive_external_object_rejected(self):
        self.link_rule = self.targets["app"] = "CXX_STATIC_LIBRARY_LINKER__app"
        del self.deps["app"]
        self.link_inputs = str(self.root / "external.o")
        self.check(1, "outside the source tree")


if __name__ == "__main__":
    unittest.main()
