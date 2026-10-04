# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test of D-057's source mechanism on a synthetic lock.

    python3 mechanism_test.py --repo REPO --work DIR --cmake CMAKE --ninja NINJA
                              --sdk SDK --toolchain FILE

Builds fixture components into archives, writes a lock for them, and drives
the real tools/prepare-sources and cmake/JitllmSources.cmake through the gates
in docs/source-dependencies.md: clean preparation from empty caches;
exclusion of an optional module (with generated output) that is never
fetched in the core profile; missing, changed or modified inputs rejected
before third-party code runs; a lock change giving a new tree and outputs;
switching a build from the optional profile back to core without reusing
its payloads; an archive cut down to its `archive.keep` paths, whose
discarded names could not be prepared; and undeclared downloads, package lookups, system libraries,
source overrides and dependency providers failing the build or its checks.
Every fixture is jitLLM-authored; the optional module's "copyleft" license
is a label in the fixture lock, not a real dependency.
"""

import sys

sys.dont_write_bytecode = True  # it imports tools/jitllm_sources.py from the checkout

import argparse
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import tarfile
import textwrap

ARGS = None
TOOLS = None
srclib = None

CORE_V1 = {
    # An old policy level, as many upstream projects declare: the lock's
    # options must still hold for its option() calls (CMP0077).
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 3.10)
        project(core_lib LANGUAGES CXX)
        file(TOUCH "${FIXTURE_MARKERS}/core-lib")
        option(FIXTURE_CORE_TOGGLE "a locked option" OFF)
        if(NOT FIXTURE_CORE_OPTION STREQUAL "locked-value" OR NOT FIXTURE_CORE_TOGGLE)
          message(FATAL_ERROR "the lock's cmake.options did not reach the subproject")
        endif()
        add_library(fixture_core STATIC core.cc)
        target_include_directories(fixture_core PUBLIC include)
        """,
    "include/core.h": """
        #pragma once
        #define FIXTURE_CORE_VALUE 1
        int fixture_core_value();
        """,
    "core.cc": """
        #include "core.h"
        int fixture_core_value() { return FIXTURE_CORE_VALUE; }
        """,
}
CORE_PATCH = """\
diff --git a/include/core.h b/include/core.h
--- a/include/core.h
+++ b/include/core.h
@@ -1,3 +1,3 @@
 #pragma once
-#define FIXTURE_CORE_VALUE 1
+#define FIXTURE_CORE_VALUE 2
 int fixture_core_value();
"""
CORE_V2 = dict(CORE_V1, **{"core.cc": """
        #include "core.h"
        int fixture_core_value() { return FIXTURE_CORE_VALUE + 1; }
        """})
OPTIONAL = {
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 4.4.3)
        project(opt_lib LANGUAGES CXX)
        file(TOUCH "${FIXTURE_MARKERS}/opt-lib")
        set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated/opt_generated.h")
        add_custom_command(OUTPUT "${generated}"
          COMMAND "${CMAKE_COMMAND}" "-DOUT=${generated}" -P "${CMAKE_CURRENT_SOURCE_DIR}/generate.cmake"
          DEPENDS generate.cmake VERBATIM)
        # A payload outside the component's binary directory, which only a
        # new build directory is guaranteed not to see again.
        file(WRITE "${CMAKE_BINARY_DIR}/opt-outside-payload.h" "#define FIXTURE_OUTSIDE 1\n")
        add_library(fixture_opt STATIC opt.cc "${generated}")
        target_include_directories(fixture_opt PUBLIC include PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
        target_link_libraries(fixture_opt PUBLIC fixture_core)
        """,
    "generate.cmake": """
        file(WRITE "${OUT}" "#define FIXTURE_OPT_MARKER \\"FIXTURE-OPTIONAL-PAYLOAD\\"\\n")
        """,
    "include/opt.h": """
        #pragma once
        const char* fixture_opt_marker();
        """,
    "opt.cc": """
        #include "opt.h"
        #include "opt_generated.h"
        const char* fixture_opt_marker() { return FIXTURE_OPT_MARKER; }
        """,
}
BAD_FETCH = {
    # googletest's policy level: FetchContent must fail here too (CMP0170).
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 3.16)
        project(bad_fetch LANGUAGES CXX)
        include(FetchContent)
        FetchContent_Declare(sneaky URL https://invalid.invalid/sneaky.tar.gz
          URL_HASH SHA256=0000000000000000000000000000000000000000000000000000000000000000)
        FetchContent_MakeAvailable(sneaky)
        add_library(fixture_bad_fetch INTERFACE)
        """,
}
BAD_FIND = {
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 4.4.3)
        project(bad_find LANGUAGES CXX)
        find_package(FixtureSys CONFIG)
        add_library(fixture_bad_find INTERFACE)
        """,
}
BAD_LINK = {
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 4.4.3)
        project(bad_link LANGUAGES CXX)
        find_library(FIXTURE_SYS_LIBRARY fixturesys PATHS "${FIXTURE_SYS_DIR}" NO_DEFAULT_PATH REQUIRED)
        add_library(fixture_bad_link INTERFACE)
        target_link_libraries(fixture_bad_link INTERFACE "${FIXTURE_SYS_LIBRARY}")
        """,
}
BAD_EXCEPTIONS = {
    # Turns C++ exceptions back on two ways: a driver flag after every target
    # flag, and a frontend flag that a later driver flag hides from the flag
    # check (-Xclang); the objects show both.
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 4.4.3)
        project(bad_exceptions LANGUAGES CXX)
        add_library(fixture_bad_exceptions STATIC throws.cc frontend.cc)
        set_source_files_properties(throws.cc PROPERTIES COMPILE_OPTIONS -fcxx-exceptions)
        set_source_files_properties(frontend.cc PROPERTIES
          COMPILE_OPTIONS "-Xclang;-fcxx-exceptions;-fno-cxx-exceptions")
        # An object no compile record describes, linked in directly under a
        # name that does not look like one.
        set(side "${CMAKE_CURRENT_BINARY_DIR}/side-object")
        add_custom_command(OUTPUT "${side}"
          COMMAND "${CMAKE_CXX_COMPILER}" -fexceptions -c "${CMAKE_CURRENT_SOURCE_DIR}/side.cc" -o "${side}"
          DEPENDS side.cc VERBATIM)
        add_custom_target(fixture_side_object DEPENDS "${side}")
        add_dependencies(fixture_bad_exceptions fixture_side_object)
        target_link_libraries(fixture_bad_exceptions PUBLIC "${side}")
        """,
    "side.cc": """
        int fixture_side(int v) {
          try { throw v; } catch (int caught) { return caught; }
        }
        """,
    "throws.cc": """
        int fixture_throws(int v) {
          try { throw v; } catch (int caught) { return caught; }
        }
        """,
    "frontend.cc": """
        int fixture_frontend(int v) {
          if (v != 0) throw v;
          return 0;
        }
        """,
}
BAD_ARCHIVE = {
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 4.4.3)
        project(bad_archive LANGUAGES CXX)
        add_library(fixture_bad_archive INTERFACE)
        """,
}
KEEP = {
    # archive.keep holds the build and license; the rest, with names CMake's
    # lists cannot carry and a path sharing a kept one's prefix, is dropped.
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 4.4.3)
        project(keep_lib LANGUAGES CXX)
        add_library(fixture_keep STATIC src/keep.cc)
        """,
    "LICENSE": """
        fixture license text
        """,
    "src/keep.cc": """
        #include "added.h"
        int fixture_keep() { return FIXTURE_KEEP_ADDED; }
        """,
    "src-extra/dropped.cc": """
        #error not kept
        """,
    "web/(group)/[id]/x.txt": """
        dropped
        """,
}
KEEP_PATHS = ["CMakeLists.txt", "LICENSE", "src"]
KEEP_PATCH = """\
diff --git a/src/added.h b/src/added.h
new file mode 100644
--- /dev/null
+++ b/src/added.h
@@ -0,0 +1 @@
+#define FIXTURE_KEEP_ADDED 4
"""
PROJECT = {
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 4.4.3)
        project(fixture LANGUAGES CXX)
        add_compile_options(-fno-exceptions)  # as jitLLM's root CMakeLists.txt (D-066)
        include("${JITLLM_ROOT}/cmake/JitllmSources.cmake")
        jitllm_sources_add(LOCK "${FIXTURE_LOCK}")
        add_executable(fixture_app app.cc)
        # Like a generated config header's directory: whatever the build tree
        # holds is visible to the app.
        target_include_directories(fixture_app PRIVATE "${CMAKE_BINARY_DIR}")
        target_link_libraries(fixture_app PRIVATE fixture_core)
        if(TARGET fixture_cuda AND NOT FIXTURE_SKIP_CUDA_LINK)
          target_link_libraries(fixture_app PRIVATE fixture_cuda)
          target_compile_definitions(fixture_app PRIVATE FIXTURE_WITH_CUDA)
        endif()
        if(FIXTURE_CUDA_HEADER)
          target_include_directories(fixture_app PRIVATE "${FIXTURE_CUDA_HEADER}")
          target_compile_definitions(fixture_app PRIVATE FIXTURE_IMPORT_CUDA_HEADER)
        endif()
        if("fixture-optional" IN_LIST JITLLM_MODULES)
          target_link_libraries(fixture_app PRIVATE fixture_opt)
          target_compile_definitions(fixture_app PRIVATE FIXTURE_WITH_OPTIONAL)
        endif()
        if("bad-link" IN_LIST JITLLM_MODULES)
          target_link_libraries(fixture_app PRIVATE fixture_bad_link)
        endif()
        if("bad-exceptions" IN_LIST JITLLM_MODULES)
          target_link_libraries(fixture_app PRIVATE fixture_bad_exceptions)
        endif()
        if("keep-subset" IN_LIST JITLLM_MODULES)
          target_link_libraries(fixture_app PRIVATE fixture_keep)
        endif()
        jitllm_sources_finalize()
        """,
    "app.cc": """
        #include <cstdio>
        #include "core.h"
        #if defined(FIXTURE_WITH_CUDA) || defined(FIXTURE_IMPORT_CUDA_HEADER)
        #include "cuda.h"
        static_assert(FIXTURE_CUDA_VALUE == 7);
        #endif
        #ifdef FIXTURE_WITH_OPTIONAL
        #include "opt.h"
        #elif __has_include("opt-outside-payload.h")
        #include "opt-outside-payload.h"  // a leftover no core build may use
        #endif
        int main() {
          std::printf("%d\\n", fixture_core_value());
        #ifdef FIXTURE_WITH_CUDA
          std::printf("%d\\n", fixture_cuda_value());
        #endif
        #ifdef FIXTURE_WITH_OPTIONAL
          std::printf("%s\\n", fixture_opt_marker());
        #endif
          return 0;
        }
        """,
}

CUDA_ONLY = {
    "CMakeLists.txt": """
        cmake_minimum_required(VERSION 4.4.3)
        project(cuda_only_fixture LANGUAGES CXX)
        file(TOUCH "${FIXTURE_MARKERS}/cuda-lib")
        add_library(fixture_cuda STATIC cuda.cc)
        target_include_directories(fixture_cuda PUBLIC include)
        """,
    "include/cuda.h": """
        #pragma once
        #define FIXTURE_CUDA_VALUE 7
        int fixture_cuda_value();
        """,
    "cuda.cc": """
        #include "cuda.h"
        int fixture_cuda_value() { return FIXTURE_CUDA_VALUE; }
        """,
}


CORE_OPTIONS = {"FIXTURE_CORE_OPTION": "locked-value", "FIXTURE_CORE_TOGGLE": "ON"}


class Failure(Exception):
    pass


def write_tree(root: pathlib.Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"))


def make_archive(path: pathlib.Path, top: str, files: dict[str, str], links: dict[str, str] | None = None) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.type, info.linkname, info.mtime = tarfile.SYMTYPE, target, 0
            tar.addfile(info)
        for name in sorted(files):
            data = textwrap.dedent(files[name]).lstrip("\n").encode()
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))


def expected_tree(work: pathlib.Path, files: dict[str, str], patches: list[str]) -> str:
    """The digest preparation must reach: the files, patched."""
    scratch = work / "digest-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    write_tree(scratch, files)
    for patch in patches:
        srclib.apply_patch(scratch, patch)
    digest = srclib.tree_digest(scratch)
    shutil.rmtree(scratch)
    return digest


class Fixture:
    def __init__(self, work: pathlib.Path):
        self.work = work
        self.archives = work / "archives"
        self.markers = work / "markers"
        self.project = work / "project"
        self.lock_dir = work / "lock"
        self.lock = self.lock_dir / "sources.lock.json"
        for d in (self.archives, self.markers, self.lock_dir / "patches"):
            d.mkdir(parents=True)
        write_tree(self.project, PROJECT)
        self.data = {"schema": 1, "modules": {
            "fixture-optional": {"description": "a synthetic optional module with generated output"},
            "bad-fetch": {"description": "tries an undeclared FetchContent download"},
            "bad-find": {"description": "looks up an undeclared package"},
            "bad-link": {"description": "links a library from outside the closure"},
            "bad-exceptions": {"description": "turns C++ exceptions back on for one file"},
            "bad-archive": {"description": "an archive with a symbolic link member"},
            "keep-subset": {"description": "an archive of which archive.keep prepares a part"},
        }, "components": {}}
        self.add("core-lib", "1", CORE_V1, tier="core", options=CORE_OPTIONS, patches=[CORE_PATCH])
        self.add("opt-lib", "1", OPTIONAL, module="fixture-optional", depends=["core-lib"])
        self.add("bad-fetch", "1", BAD_FETCH, module="bad-fetch")
        self.add("bad-find", "1", BAD_FIND, module="bad-find")
        self.add("bad-link", "1", BAD_LINK, module="bad-link")
        self.add("bad-exceptions", "1", BAD_EXCEPTIONS, module="bad-exceptions")
        self.add("bad-archive", "1", BAD_ARCHIVE, module="bad-archive",
                 links={"escape": "../../../../outside", "CMakeLists-link.txt": "/etc/hostname"})
        self.add("keep-lib", "1", KEEP, module="keep-subset", patches=[KEEP_PATCH], keep=KEEP_PATHS)
        self.save()

    def add(self, cid, version, files, *, tier="optional", module=None, depends=(), options=None,
            patches=(), links=None, keep=None) -> None:
        """patches holds at most one patch, recorded as patches/<cid>.patch."""
        archive = self.archives / f"{cid}-{version}.tar.gz"
        make_archive(archive, f"{cid}-{version}", files, links)
        data = archive.read_bytes()
        for patch in patches:
            (self.lock_dir / "patches" / f"{cid}.patch").write_text(patch)
        kept = {name: text for name, text in files.items()
                if keep is None or any(name == k or name.startswith(k + "/") for k in keep)}
        self.data["components"][cid] = {
            "version": version, "kind": "archive", "category": "implementation", "tier": tier,
            **({"module": module} if module else {}),
            "use": "product", "machine": "target",
            "upstream": {"repository": "https://example.invalid/fixture", "commit": "0" * 40},
            "archive": {"file": archive.name, "urls": [archive.as_uri()],
                        "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                        **({"keep": keep} if keep else {})},
            "patches": [{"path": f"patches/{cid}.patch",
                         "sha256": hashlib.sha256(patch.encode()).hexdigest()} for patch in patches],
            "tree_sha256": expected_tree(self.work, kept, list(patches)),
            "depends": list(depends),
            "cmake": {"subdirectory": "", "options": options or {}, "platform_packages": [],
                      "targets": []},
            "license": {"expression": "MIT" if tier == "core" else "LicenseRef-fixture-copyleft",
                        "files": ["CMakeLists.txt"], "scope": "fixture", "evidence": "fixture"},
            "verification": "generated by mechanism_test.py",
        }
        targets = {"core-lib": "fixture_core", "opt-lib": "fixture_opt", "bad-fetch": "fixture_bad_fetch",
                   "bad-find": "fixture_bad_find", "bad-link": "fixture_bad_link",
                   "bad-exceptions": "fixture_bad_exceptions", "bad-archive": "fixture_bad_archive",
                   "keep-lib": "fixture_keep", "cuda-lib": "fixture_cuda"}
        self.data["components"][cid]["cmake"]["targets"] = [targets[cid]]

    def save(self, data: dict | None = None, path: pathlib.Path | None = None) -> pathlib.Path:
        path = path or self.lock
        path.write_text(json.dumps(data or self.data, indent=2) + "\n")
        return path


def run(cmd: list, *, ok: bool, expect: str | None = None, env: dict | None = None) -> str:
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr
    if (result.returncode == 0) != ok:
        raise Failure(f"expected {'success' if ok else 'failure'} (exit {result.returncode}) from\n  "
                      + " ".join(map(str, cmd)) + "\n" + output)
    if expect and expect not in " ".join(output.split()):
        raise Failure(f"expected {expect!r} in the output of\n  " + " ".join(map(str, cmd)) + "\n" + output)
    return output


def prepare(fx: Fixture, dest: pathlib.Path, cache: pathlib.Path, *, modules: str = "", lock=None, ok=True,
            expect=None) -> str:
    cmd = [sys.executable, TOOLS / "prepare-sources", "--lock", lock or fx.lock, "--dest", dest,
           "--cache", cache]
    if modules:
        cmd += ["--modules", modules]
    return run(cmd, ok=ok, expect=expect)


def configure(fx: Fixture, build: pathlib.Path, sources: pathlib.Path, *, modules: str = "", extra=(),
              lock=None, ok=True, expect=None) -> str:
    cmd = [ARGS.cmake, "-S", fx.project, "-B", build, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ARGS.ninja}",
           f"-DCMAKE_TOOLCHAIN_FILE={ARGS.toolchain}", f"-DJITLLM_SDK={ARGS.sdk}", "-DJITLLM_CUDA=OFF",
           "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", f"-DJITLLM_ROOT={ARGS.repo}", f"-DFIXTURE_LOCK={lock or fx.lock}",
           f"-DFIXTURE_MARKERS={fx.markers}", f"-DJITLLM_SOURCES_DIR={sources}", f"-DJITLLM_MODULES={modules}",
           *extra]
    return run(cmd, ok=ok, expect=expect)


def build(build_dir: pathlib.Path, *, ok=True, expect=None) -> str:
    return run([ARGS.cmake, "--build", build_dir], ok=ok, expect=expect)


def markers(fx: Fixture) -> set[str]:
    return {p.name for p in fx.markers.iterdir()}


def reset_markers(fx: Fixture) -> None:
    for p in fx.markers.iterdir():
        p.unlink()


def receipt(build_dir: pathlib.Path) -> dict:
    return json.loads((build_dir / "jitllm-receipt.json").read_text())


def check(condition: bool, text: str) -> None:
    if not condition:
        raise Failure(text)


def cached(cache: pathlib.Path) -> set[str]:
    downloads = cache / "downloads"
    return {p.name for p in downloads.iterdir()} if downloads.is_dir() else set()


def files_named(root: pathlib.Path, name: str) -> list[pathlib.Path]:
    return list(root.rglob(name))


def check_closure(fx: Fixture, build_dir: pathlib.Path, *, ok=True, expect=None, extra=()) -> str:
    return run([sys.executable, pathlib.Path(__file__).with_name("check_closure.py"), "--build-dir", build_dir,
                "--source-dir", fx.project, "--sdk", ARGS.sdk, "--ninja", ARGS.ninja, *extra], ok=ok, expect=expect)


def check_receipt(fx: Fixture, build_dir: pathlib.Path, *, ok=True, expect=None) -> str:
    return run([ARGS.cmake, f"-DRECEIPT={build_dir / 'jitllm-receipt.json'}", f"-DLOCK={fx.lock}",
                f"-DSDK_IDENTITY={receipt(build_dir)['sdk']}", "-P",
                pathlib.Path(__file__).with_name("check_receipt.cmake")], ok=ok, expect=expect)


def check_cuda_selection(work: pathlib.Path) -> None:
    fx = Fixture(work)
    fx.add("cuda-lib", "1", CUDA_ONLY, tier="core")
    fx.data["components"]["cuda-lib"]["requires_cuda"] = True
    fx.save()
    sources, cache = work / "sources", work / "cache"
    prepare(fx, sources, cache)
    check(cached(cache) == {fx.data["components"][c]["archive"]["sha256"] for c in ("core-lib", "cuda-lib")},
          "default preparation did not retain the CPU/CUDA superset")
    cuda_source = srclib.prepared_dir(sources, "cuda-lib", fx.data["components"]["cuda-lib"])
    reset_markers(fx)
    cpu = work / "build-cpu"
    configure(fx, cpu, sources)
    check(markers(fx) == {"core-lib"}, "CPU configure executed CUDA-only component CMake")
    build(cpu)
    check_receipt(fx, cpu)
    check_closure(fx, cpu)
    cpu_receipt = receipt(cpu)
    check(cpu_receipt["cuda"] is False and [c["id"] for c in cpu_receipt["components"]] == ["core-lib"],
          f"CPU receipt {cpu_receipt}")

    # This synthetic component uses ordinary C++ so both selections can be
    # exercised without a CUDA compiler; real CUDA inventory is checked by
    # sources.closure in each CUDA-enabled production build.
    cuda = work / "build-cuda"
    configure(fx, cuda, sources, extra=["-DJITLLM_CUDA=ON"])
    build(cuda)
    check(run([cuda / "fixture_app"], ok=True).split() == ["2", "7"], "CUDA-only component was not consumed")
    check_receipt(fx, cuda)
    check_closure(fx, cuda)
    cuda_receipt = receipt(cuda)
    check(cuda_receipt["cuda"] is True and {c["id"] for c in cuda_receipt["components"]} == {"core-lib", "cuda-lib"},
          f"CUDA receipt {cuda_receipt}")

    path = cpu / "jitllm-receipt.json"
    saved = path.read_text()
    forged = json.loads(saved)
    forged["components"] = cuda_receipt["components"]
    path.write_text(json.dumps(forged))
    check_receipt(fx, cpu, ok=False, expect="the receipt's closure")
    check_closure(fx, cpu, ok=False, expect="which nothing compiles or links")
    path.write_text(saved)
    path = cuda / "jitllm-receipt.json"
    saved = path.read_text()
    forged = json.loads(saved)
    forged["components"] = cpu_receipt["components"]
    path.write_text(json.dumps(forged))
    check_receipt(fx, cuda, ok=False, expect="the receipt's closure")
    check_closure(fx, cuda, ok=False, expect="from no component in the receipt")
    path.write_text(saved)

    # Selecting a component never exempts it from the strict usage check.
    unused = work / "build-unused"
    configure(fx, unused, sources, extra=["-DJITLLM_CUDA=ON", "-DFIXTURE_SKIP_CUDA_LINK=ON"])
    build(unused)
    check_closure(fx, unused, ok=False, expect="which nothing compiles or links")
    imported = work / "build-imported"
    configure(fx, imported, sources, extra=[f"-DFIXTURE_CUDA_HEADER={cuda_source / 'include'}"])
    build(imported)
    check_closure(fx, imported, ok=False, expect="a prepared tree the receipt does not select")

    # Disabling CUDA cleans the old component's outputs and never executes
    # its CMake again. The new receipt and actual inventory agree.
    reset_markers(fx)
    configure(fx, cuda, sources)
    check(markers(fx) == {"core-lib"}, "CUDA-only CMake ran after switching to CPU")
    check(not files_named(cuda / "third_party", "*cuda-lib*"), "CUDA-only build outputs survived selection")
    build(cuda)
    check_receipt(fx, cuda)
    check_closure(fx, cuda)

    for value in (None, 0, 1, "ON", "OFF", [], {}):
        data = json.loads(json.dumps(fx.data))
        data["components"]["cuda-lib"]["requires_cuda"] = value
        fx.save(data)
        reset_markers(fx)
        configure(fx, work / "build-invalid", sources, ok=False, expect="requires_cuda must be a boolean")
        check(not markers(fx), "a malformed CUDA condition executed component CMake")
    data = json.loads(json.dumps(fx.data))
    data["components"]["core-lib"]["depends"] = ["cuda-lib"]
    fx.save(data)
    reset_markers(fx)
    configure(fx, work / "build-disabled-dependency", sources, ok=False, expect="does not select")
    check(not markers(fx), "a dependency on a disabled component executed component CMake")
    fx.save()


def main() -> int:
    global ARGS, TOOLS, srclib
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("repo", "work", "cmake", "ninja", "sdk", "toolchain"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    ARGS = parser.parse_args()
    TOOLS = ARGS.repo / "tools"
    sys.path.insert(0, str(TOOLS))
    import jitllm_sources  # noqa: PLC0415
    srclib = jitllm_sources

    work = ARGS.work
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    fx = Fixture(work)
    comps = fx.data["components"]
    sha = {cid: c["archive"]["sha256"] for cid, c in comps.items()}
    sources, cache = work / "sources", work / "cache"
    core_dir = sources / f"core-lib-{comps['core-lib']['tree_sha256'][:16]}"

    def step(text: str) -> None:
        print(f"-- {text}", flush=True)

    step("core profile from empty caches: only the closure is fetched, prepared and built")
    prepare(fx, sources, cache)
    check(cached(cache) == {sha["core-lib"]}, f"the core profile fetched {cached(cache)}")
    check({p.name for p in sources.iterdir() if not p.name.startswith(".")} == {core_dir.name},
          f"prepared {list(sources.iterdir())}")
    b = work / "build-core"
    configure(fx, b, sources)
    build(b)
    check(run([b / "fixture_app"], ok=True).split() == ["2"], "the recorded patch was not applied")
    r = receipt(b)
    check([c["id"] for c in r["components"]] == ["core-lib"] and r["license_profile"] == "core"
          and r["official"] is True and len(r["components"][0]["patches"]) == 1, f"receipt {r}")
    check("opt-lib" not in markers(fx), "the optional subproject ran in the core profile")
    for name in ("opt_generated.h", "libfixture_opt.a", "*opt-lib*"):
        check(not files_named(b, name), f"the core build has {name}")
    for name in ("build.ninja", "compile_commands.json", "CMakeCache.txt", "jitllm-receipt.json"):
        text = (b / name).read_text()
        check("opt-lib" not in text and "fixture_opt" not in text, f"{name} mentions the optional module")
    check_receipt(fx, b)
    check_closure(fx, b)
    step("CUDA-only source selection, receipt authentication and strict inventory")
    check_cuda_selection(work / "cuda-selection")
    # This build defines _GLIBCXX_ASSERTIONS nowhere, so one that must define
    # it everywhere fails (D-083).
    check_closure(fx, b, ok=False, expect="does not define _GLIBCXX_ASSERTIONS", extra=["--libstdcxx-assertions"])

    step("missing inputs fail before any third-party CMake code runs")
    reset_markers(fx)
    configure(fx, work / "build-missing", work / "empty-sources", ok=False, expect="Run `mise run prepare`")
    configure(fx, work / "build-missing-opt", sources, modules="fixture-optional", ok=False,
              expect="mise run prepare -- --modules fixture-optional")
    check(not markers(fx), f"third-party CMake code ran: {markers(fx)}")
    check(not (work / "build-missing" / "jitllm-receipt.json").exists(), "a failed configure left a receipt")

    step("a changed archive is rejected; a damaged cached copy is fetched again")
    origin = fx.archives / "core-lib-1.tar.gz"
    good = origin.read_bytes()
    origin.write_bytes(good + b"\0")
    prepare(fx, work / "sources-tampered", work / "cache-tampered", ok=False, expect="more than the locked")
    check(not any((work / "sources-tampered").glob("core-lib-*")), "a tampered archive was prepared")
    origin.write_bytes(good)
    damaged = cache / "downloads" / sha["core-lib"] / origin.name
    damaged.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    prepare(fx, work / "sources-refetch", cache)
    check(damaged.read_bytes() == good, "the damaged cache entry was not replaced")

    step("a changed patch is rejected before anything is fetched")
    patch = fx.lock_dir / "patches" / "core-lib.patch"
    patch.write_text(CORE_PATCH.replace("VALUE 2", "VALUE 5"))
    prepare(fx, work / "sources-patch", work / "cache-patch", ok=False, expect="does not match its recorded sha256")
    check(not cached(work / "cache-patch"), "a lock with a changed patch fetched archives")
    reset_markers(fx)
    configure(fx, b, sources, ok=False, expect="does not match its recorded sha256")
    check(not markers(fx), "third-party code ran with an unverified patch")
    patch.write_text(CORE_PATCH)
    configure(fx, b, sources)
    build(b)
    patch.write_text(CORE_PATCH.replace("VALUE 2", "VALUE 5"))
    build(b, ok=False, expect="does not match its recorded sha256")
    patch.write_text(CORE_PATCH)
    build(b)

    step("a modified prepared tree fails the build, configure and preparation")
    edited = core_dir / "core.cc"
    original = edited.read_text()
    edited.write_text(original + "// local edit\n")
    build(b, ok=False, expect="changed after configure")
    configure(fx, b, sources, ok=False, expect="no longer matches the lock")
    prepare(fx, sources, cache, ok=False, expect="was modified")
    edited.write_text(original)
    configure(fx, b, sources)
    build(b)

    step("new files, mode changes and timestamp-preserving edits fail even on an incremental build")
    added = core_dir / "include" / "added.h"
    added.write_text("#define UNRECORDED_INPUT 1\n")
    build(b, ok=False, expect="changed after configure")
    added.unlink()
    mode = edited.stat().st_mode
    edited.chmod(mode | 0o100)
    build(b, ok=False, expect="changed after configure")
    edited.chmod(mode)
    times = edited.stat()
    edited.write_text(original.replace("return FIXTURE_CORE_VALUE", "return 99"))
    os.utime(edited, ns=(times.st_atime_ns, times.st_mtime_ns))
    build(b, ok=False, expect="changed after configure")
    edited.write_text(original)
    build(b)

    step("configure validates the full lock before running prepared third-party code")
    for name, change, message in (
            ("bad-category", lambda d: d["components"]["core-lib"].update(category="build-tool"),
             "category must be one of implementation"),
            ("path-escape", lambda d: d["components"]["core-lib"]["cmake"].update(subdirectory="../.."),
             "relative path"),
            ("core-needs-enabled-optional", lambda d: d["components"]["core-lib"]["depends"].append("opt-lib"),
             "cannot depend on the optional opt-lib")):
        data = json.loads(json.dumps(fx.data))
        change(data)
        variant = fx.save(data, fx.lock_dir / f"{name}.lock.json")
        reset_markers(fx)
        configure(fx, b, sources, modules="fixture-optional", lock=variant, ok=False, expect=message)
        check(not markers(fx), f"third-party code ran for the {name} lock")
    configure(fx, b, sources)
    build(b)

    step("a lock change gives a new tree and new outputs")
    old_outputs = b / "third_party" / core_dir.name
    check(old_outputs.is_dir(), "no component outputs before the lock change")
    v1 = json.loads(json.dumps(fx.data))
    fx.add("core-lib", "2", CORE_V2, tier="core", options=CORE_OPTIONS,
           patches=[CORE_PATCH])
    fx.save()
    configure(fx, b, sources, ok=False, expect="Run `mise run prepare`")
    prepare(fx, sources, cache)
    configure(fx, b, sources)
    build(b)
    check(run([b / "fixture_app"], ok=True).split() == ["3"], "the build did not use the new tree")
    check(not old_outputs.exists(), "the old tree's outputs were kept")
    check_closure(fx, b)
    fx.data = v1
    fx.save()
    configure(fx, b, sources)
    build(b)

    step("the optional profile builds its module; its build directory never builds core again")
    prepare(fx, sources, cache, modules="fixture-optional")
    bad = {sha[m] for m in ("bad-fetch", "bad-find", "bad-link", "bad-exceptions", "bad-archive")}
    check(sha["opt-lib"] in cached(cache) and not bad & cached(cache),
          f"the fixture-optional profile fetched {cached(cache)}")
    configure(fx, b, sources, modules="fixture-optional")
    build(b)
    check("FIXTURE-OPTIONAL-PAYLOAD" in run([b / "fixture_app"], ok=True), "the optional module is not linked")
    r = receipt(b)
    check(r["license_profile"] == "core+fixture-optional"
          and [c["id"] for c in r["components"]] == ["core-lib", "opt-lib"], f"receipt {r}")
    check(files_named(b, "opt_generated.h") and (b / "opt-outside-payload.h").exists(),
          "the optional module did not generate its outputs")
    check_receipt(fx, b)
    check_closure(fx, b)
    configure(fx, b, sources, modules="", ok=False, expect="has built optional module(s) fixture-optional")
    configure(fx, b, sources, modules="", extra=["--fresh"], ok=False,
              expect="has built optional module(s) fixture-optional")
    record = b / "jitllm-modules-built.txt"
    forged = record.read_text().replace("fixture-optional", "")
    record.unlink()
    configure(fx, b, sources, modules="", ok=False, expect="the optional modules it has built are unknown")
    # A deliberately forged record is not proof: the inventory check still
    # rejects the core build that picks up the module's leftover.
    record.write_text(forged)
    configure(fx, b, sources, modules="")
    build(b)
    check_closure(fx, b, ok=False, expect="a file in the build tree that no build rule or selected component produced")
    bc = work / "build-core-again"
    configure(fx, bc, sources, modules="")
    build(bc)
    check(b"FIXTURE-OPTIONAL-PAYLOAD" not in (bc / "fixture_app").read_bytes(),
          "the core build links the optional payload")
    for name in ("opt_generated.h", "libfixture_opt.a", "*opt-lib*", "opt-outside-payload.h"):
        check(not files_named(bc, name), f"the core build has {name}")
    check(receipt(bc)["license_profile"] == "core", "the receipt names the optional profile")
    check_closure(fx, bc)

    step("an undeclared download, package lookup or library fails the build or its checks")
    prepare(fx, sources, cache, modules="bad-fetch,bad-find,bad-link")
    configure(fx, work / "build-bad-fetch", sources, modules="bad-fetch", ok=False,
              expect="FETCHCONTENT_FULLY_DISCONNECTED")
    prefix = work / "sysprefix" / "lib" / "cmake" / "FixtureSys"
    prefix.mkdir(parents=True)
    (prefix / "FixtureSysConfig.cmake").write_text("set(FixtureSys_FOUND TRUE)\n")
    configure(fx, work / "build-bad-find", sources, modules="bad-find",
              extra=[f"-DCMAKE_PREFIX_PATH={work / 'sysprefix'}"], ok=False, expect="looked for FixtureSys")
    syslib = work / "syslib"
    syslib.mkdir()
    (syslib / "sys.cc").write_text("int fixture_sys() { return 7; }\n")
    run([ARGS.sdk / "bin" / "clang++", "-c", syslib / "sys.cc", "-o", syslib / "sys.o"], ok=True)
    run([ARGS.sdk / "bin" / "llvm-ar", "rcs", syslib / "libfixturesys.a", syslib / "sys.o"], ok=True)
    bl = work / "build-bad-link"
    configure(fx, bl, sources, modules="bad-link", extra=[f"-DFIXTURE_SYS_DIR={syslib}"])
    build(bl)
    check_closure(fx, bl, ok=False, expect="outside the source tree")

    step("a component that turns exceptions back on fails the inventory check")
    prepare(fx, sources, cache, modules="bad-exceptions")
    be = work / "build-bad-exceptions"
    configure(fx, be, sources, modules="bad-exceptions")
    build(be)
    out = check_closure(fx, be, ok=False, expect="does not end its exception flags with -fno-exceptions")
    for name in ("throws.cc.o", "frontend.cc.o", "side-object"):
        check(f"{name} was compiled with exceptions" in out, f"the object check missed {name}:\n{out}")
    check("passes -fcxx-exceptions straight to the compiler frontend" in out, f"-Xclang went unnoticed:\n{out}")

    step("an archive with a link member is refused before anything is extracted")
    outside = work / "outside"
    prepare(fx, work / "sources-links", cache, modules="bad-archive", ok=False, expect="is not a file or directory")
    check(not outside.exists() and not any((work / "sources-links").glob("*archive*")),
          "extraction wrote through a link member")

    step("archive.keep prepares only the kept paths; names it discards never reach the tree")
    keep_sources = work / "sources-keep"
    keep_dir = keep_sources / f"keep-lib-{comps['keep-lib']['tree_sha256'][:16]}"
    data = json.loads(json.dumps(fx.data))
    del data["components"]["keep-lib"]["archive"]["keep"]
    variant = fx.save(data, fx.lock_dir / "no-keep.lock.json")
    prepare(fx, keep_sources, cache, modules="keep-subset", lock=variant, ok=False,
            expect="has an unsafe or unsupported path")
    check(not keep_dir.exists(), "the whole archive was prepared without archive.keep")
    prepare(fx, keep_sources, cache, modules="keep-subset")
    check(sorted(p.relative_to(keep_dir).as_posix() for p in keep_dir.rglob("*") if not p.is_dir())
          == ["CMakeLists.txt", "LICENSE", "src/added.h", "src/keep.cc"], f"prepared {list(keep_dir.rglob('*'))}")
    check(srclib.tree_digest(keep_dir) == comps["keep-lib"]["tree_sha256"], "the kept tree has the wrong digest")
    bk = work / "build-keep"
    configure(fx, bk, keep_sources, modules="keep-subset")
    build(bk)
    check(files_named(bk, "libfixture_keep.a"), "the kept component was not built")

    step("unrecorded source overrides and dependency providers are rejected; recorded ones are unofficial")
    configure(fx, work / "build-fc", sources, extra=[f"-DFETCHCONTENT_SOURCE_DIR_CORE-LIB={core_dir}"],
              ok=False, expect="would replace a locked source without a record")
    provider = work / "provider.cmake"
    provider.write_text("message(STATUS \"provider included\")\n")
    configure(fx, work / "build-provider", sources, extra=[f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={provider}"],
              ok=False, expect="injects CMake code or a dependency provider")
    copy = work / "core-lib-edited"
    shutil.copytree(core_dir, copy)
    (copy / "include" / "core.h").write_text((copy / "include" / "core.h").read_text().replace("VALUE 2", "VALUE 9"))
    bo = work / "build-override"
    configure(fx, bo, sources, extra=[f"-DJITLLM_SOURCE_OVERRIDE_CORE_LIB={copy}"], expect="local override")
    build(bo)
    check(run([bo / "fixture_app"], ok=True).split() == ["9"], "the override was not built")
    c = receipt(bo)["components"][0]
    check(receipt(bo)["official"] is False and c["override"] is True and c["modified"] is True,
          f"receipt {receipt(bo)}")
    check_receipt(fx, bo)
    configure(fx, bo, sources, extra=["-DJITLLM_REQUIRE_LOCKED_SOURCES=ON"], ok=False,
              expect="which JITLLM_REQUIRE_LOCKED_SOURCES forbids")
    configure(fx, bo, sources, extra=["-DJITLLM_REQUIRE_LOCKED_SOURCES=OFF", "-DJITLLM_SOURCE_OVERRIDE_CORE_LIB="])
    check(receipt(bo)["official"] is True, "clearing the override did not restore an official receipt")

    step("the lock is checked before anything is fetched")
    for name, change, message in (
            ("core-needs-optional", lambda d: d["components"]["core-lib"]["depends"].append("opt-lib"),
             "cannot depend on the optional opt-lib"),
            ("unclassified", lambda d: d["components"]["core-lib"].pop("license"), "license must be an object"),
            ("copyleft-core", lambda d: d["components"]["opt-lib"].update(tier="core", module=None),
             "outside D-017's core allowlist"),
            ("cmake-hook", lambda d: d["components"]["core-lib"]["cmake"]["options"].update(
                CMAKE_PROJECT_INCLUDE="evil"), "may not set CMAKE_*"),
            ("private-name", lambda d: d["components"]["core-lib"]["cmake"]["options"].update(
                _JITLLM_SOURCES_SCRIPTS="hooks"), "cmake.options must map"),
            ("path-value", lambda d: d["components"]["core-lib"]["cmake"]["options"].update(
                FIXTURE_CORE_OPTION="/tmp/evil.cmake"), "plain string values")):
        data = json.loads(json.dumps(fx.data))
        change(data)
        variant = fx.save(data, fx.lock_dir / f"{name}.lock.json")
        prepare(fx, work / f"sources-{name}", work / f"cache-{name}", lock=variant, ok=False, expect=message)
        check(not cached(work / f"cache-{name}"), f"the {name} lock fetched archives")

    print("all source mechanism gates passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failure as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
