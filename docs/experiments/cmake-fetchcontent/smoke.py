#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Check the pinned CMake's FetchContent semantics using synthetic local input."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", type=Path, required=True)
    args = parser.parse_args()
    cmake = args.cmake.resolve(strict=True)
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    version = pins["version"]
    env = os.environ.copy()
    # Do not inherit generator/project policy or toolchain injection.
    for key in list(env):
        if key.startswith(("CMAKE_", "FETCHCONTENT_")):
            del env[key]
    make = shutil.which("make")
    require(make is not None, "The configure-only cases need system make")

    def run(argv, cwd, expected=None, pathless=False):
        child_env = env.copy()
        if pathless:
            child_env["PATH"] = ""
        result = subprocess.run(
            [str(cmake), *argv], cwd=cwd, env=child_env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=60,
        )
        if expected is None:
            require(result.returncode == 0, result.stdout)
        else:
            require(result.returncode != 0 and expected in result.stdout,
                    f"Expected failure containing {expected!r}:\n{result.stdout}")
        return result.stdout

    passed = []
    with tempfile.TemporaryDirectory(prefix="llmp-cmake-fetchcontent-") as tmp:
        root = Path(tmp)
        require(run(["--version"], root).splitlines()[0] == f"cmake version {version}",
                f"Expected the pinned CMake {version}")
        fixture = root / "fixture"
        fixture.mkdir()
        (fixture / "payload.txt").write_text("original\n")
        (fixture / "CMakeLists.txt").write_text(
            'message(FATAL_ERROR "Acquisition executed upstream CMakeLists")\n'
        )
        archive = root / "fixture.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(fixture, arcname="fixture")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        prefix = f"cmake_minimum_required(VERSION {version})\ninclude(FetchContent)\n"

        def populate(name, sha, extra="", expected=None):
            work = root / name
            work.mkdir()
            script = work / "populate.cmake"
            script.write_text(prefix + extra + f'''
FetchContent_Populate(fixture
  URL [[{archive}]] URL_HASH SHA256={sha} TLS_VERIFY ON
  SOURCE_DIR [[{work / 'source'}]] BINARY_DIR [[{work / 'binary'}]])
''')
            run(["-P", str(script)], work, expected, pathless=True)
            return work / "source"

        source = populate("script", digest)
        require((source / "payload.txt").read_text() == "original\n", "Wrong payload")
        passed.append("script_population_without_build_tool_or_upstream_configure")

        source_flags = populate(
            "script-flags", digest,
            "set(FETCHCONTENT_FULLY_DISCONNECTED ON)\n"
            "set(FETCHCONTENT_UPDATES_DISCONNECTED ON)\n",
        )
        require((source_flags / "payload.txt").read_text() == "original\n",
                "Long-form population unexpectedly honored disconnected flags")
        passed.append("long_form_ignores_disconnected_flags")

        rejected = populate("bad-hash", "0" * 64,
                            expected="does not match expected value")
        require(not (rejected / "payload.txt").exists(),
                "Hash-mismatched archive was extracted")
        passed.append("incorrect_archive_hash_rejected")

        deprecated = root / "deprecated-project"
        deprecated.mkdir()
        (deprecated / "CMakeLists.txt").write_text(
            f"cmake_minimum_required(VERSION {version})\n"
            "project(deprecated_population NONE)\ninclude(FetchContent)\n" + f'''
FetchContent_Declare(fixture URL [[{archive}]] URL_HASH SHA256={digest})
FetchContent_Populate(fixture)
''')
        run(["-S", str(deprecated), "-B", str(root / "deprecated-build"),
             "-G", "Unix Makefiles", f"-DCMAKE_MAKE_PROGRAM={make}"],
            root, expected="CMP0169")
        passed.append("deprecated_single_argument_population_rejected")

        project = root / "project"
        project.mkdir()
        (project / "CMakeLists.txt").write_text(
            f"cmake_minimum_required(VERSION {version})\n"
            "project(fetchcontent_policy NONE)\ninclude(FetchContent)\n" + f'''
set(FETCHCONTENT_FULLY_DISCONNECTED ON)
FetchContent_Declare(fixture URL [[{archive}]] URL_HASH SHA256={digest}
  SOURCE_SUBDIR no-cmake-project)
FetchContent_MakeAvailable(fixture)
file(WRITE "${{CMAKE_BINARY_DIR}}/population-returned.txt" "returned")
if(EXISTS "${{fixture_SOURCE_DIR}}/payload.txt")
  file(READ "${{fixture_SOURCE_DIR}}/payload.txt" payload)
  file(WRITE "${{CMAKE_BINARY_DIR}}/observed.txt" "${{payload}}")
endif()
''')

        def configure(name, seed=None, expected=None):
            build = root / name
            if seed is not None:
                shutil.copytree(seed, build / "_deps" / "fixture-src")
            run(["-S", str(project), "-B", str(build), "-G", "Unix Makefiles",
                 f"-DCMAKE_MAKE_PROGRAM={make}"], root, expected)
            return build

        missing = configure("missing", expected="Policy CMP0170 controls enforcement")
        require(not (missing / "population-returned.txt").exists(),
                "Missing-source failure occurred after population returned")
        passed.append("declared_disconnected_missing_source_rejected")
        prepared = configure("prepared", source)
        require((prepared / "observed.txt").read_text() == "original\n",
                "Prepared source not reused")
        passed.append("declared_disconnected_prepared_source_accepted")

        (source / "payload.txt").write_text("modified\n")
        modified = configure("modified", source)
        require((modified / "observed.txt").read_text() == "modified\n",
                "Disconnected mode unexpectedly revalidated source bytes")
        passed.append("declared_disconnected_does_not_verify_prepared_bytes")

    print(json.dumps({"cmake": version, "host": platform.node(),
                      "machine": platform.machine(), "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
