# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""What the jitLLM package says about itself (D-017, D-029, D-063, D-071, D-074).

From a build's receipt, the source lock, toolchains/provenance.toml and the
SDK, this writes the package's documentation and control facts:

    control.json          the Debian version, architecture and dependencies
    copyright             /usr/share/doc/jitllm/copyright (Debian's machine-readable format)
    THIRD-PARTY-NOTICES   the notices of everything a packaged binary carries
    jitllm.spdx.json      the SBOM, SPDX 2.3

The dependencies follow the binaries: libc6 at the highest GLIBC_ symbol
version they import, and in CUDA builds the driver's libcuda.so.1 at
NVIDIA's minimum for the toolkit's major version (CUDA_DRIVER_FLOOR) and,
for the cuBLAS the package ships in /usr/lib/jitllm (D-076), libgcc-s1.
Notices are included for every platform unit that ships and every product
component, whether or not this build's code reaches the part a notice
covers: an extra notice costs nothing, a missing one is a defect. Third-party
data in jitLLM's own files (IN_TREE_UNITS: the tokenizer's Unicode tables)
is listed, with its license and notice, whenever Ninja's record shows a
packaged executable built from it; a src/ file declaring a license outside
the core's (D-017 as D-091 widens it), one no source-lock component or
shipped platform unit declares, or one only IN_TREE_UNITS brings, that
IN_TREE_UNITS does not list, a listed file that is gone, or a failed Ninja
query stops the package.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import pathlib
import re
import subprocess
import tarfile
import tempfile
import tomllib
import uuid

import jitllm_headers as headers
import jitllm_sdk as sdklib
import jitllm_sources as srclib

REPO = sdklib.REPO
PROVENANCE = REPO / "toolchains" / "provenance.toml"
PACKAGE = "jitllm"
HOMEPAGE = "https://github.com/pmeenan/jitLLM"
# The executables the package installs: (build-tree path, installed path).
EXECUTABLES = (("src/cli/jitllm", "usr/bin/jitllm"),
               ("src/runtime/jitllm-runtime", "usr/libexec/jitllm/jitllm-runtime"))
# NVIDIA's minimum driver for CUDA 13.x minor-version compatibility (CUDA
# Toolkit release notes, table 3, ">= 580", checked 2026-09-24); the build
# carries SASS only, so no PTX JIT needs a newer one.
CUDA_DRIVER_FLOOR = "580"
# The shared libraries a packaged binary may need, and where they come from:
# the package itself for cuBLAS (PRIVATE_LIBRARIES).
ALLOWED_NEEDED = {"libc.so.6": "libc6", "libm.so.6": "libc6", "ld-linux-aarch64.so.1": "libc6",
                  "ld-linux-x86-64.so.2": "libc6", "libcuda.so.1": "libcuda.so.1",
                  "libcublas.so.13": PACKAGE, "libcublasLt.so.13": PACKAGE}
# cuBLAS (D-076): the SDK's two shared libraries, which the package ships
# unmodified and unstripped (the EULA's Attachment A and section 2.3) in a
# private directory off the system linker's path; the one executable that
# needs them, the runtime, finds them through its run path, the only run path
# a packaged executable may have. libcublas.so.13 itself needs libgcc_s.so.1.
PRIVATE_LIBRARIES = {"usr/lib/jitllm/libcublas.so.13": "lib/jitllm/libcublas.so.13",
                     "usr/lib/jitllm/libcublasLt.so.13": "lib/jitllm/libcublasLt.so.13"}
PRIVATE_RUNPATH = {"usr/libexec/jitllm/jitllm-runtime": "$ORIGIN/../../lib/jitllm"}
CUBLAS_DEPENDS = "libgcc-s1"
# What those libraries need beyond ALLOWED_NEEDED: glibc's (libc6) and libgcc_s (CUBLAS_DEPENDS).
PRIVATE_LIBRARY_NEEDS = {"librt.so.1", "libpthread.so.0", "libdl.so.2", "libgcc_s.so.1"}
DEBIAN_ARCH = {"aarch64-linux-gnu": "arm64", "x86_64-linux-gnu": "amd64"}
# Provenance units whose code reaches only CUDA builds.
CUDA_UNITS = ("cuda-runtime", "cccl", "cublas")
# jitLLM's own files that hold third-party data under a license beyond
# Apache-2.0 and belong to no source-lock component (docs/licensing.md). A
# unit ships when a packaged executable is built from any of its `files`, as
# Ninja records the executables' inputs; the package then names its license
# in the copyright file, lists it in the SBOM and carries its `notice` (a
# provenance.toml [notices] record) under its own heading. So that no such
# file is missed, every file under src/ whose SPDX header declares a license
# outside the core's (srclib.CORE_LICENSES: D-017's allowlist as D-091 widens
# it), one no source-lock component or shipped provenance unit declares, or a
# license a unit here records, must be listed here, and every
# listed file must be a translation unit that exists: Ninja's inputs name
# sources, not the headers they include.
TRANSLATION_UNITS = (".c", ".cc", ".cpp", ".cu")
IN_TREE_UNITS = {
    "unicode-data": {
        "files": ("src/tokenizer/unicode_data.cc",),
        "name": "Unicode Character Database",
        "version": "15.1.0",
        "license": "Unicode-3.0",
        "copyright": "Copyright (c) 1991-2023 Unicode, Inc.",
        "download": "https://www.unicode.org/Public/15.1.0/ucd/",
        "notice": "unicode",
        "enters": "The tokenizer's tables, src/tokenizer/unicode_data.cc, which tools/gen-unicode-tables generates "
                  "from UnicodeData.txt, PropList.txt, DerivedNormalizationProps.txt, SpecialCasing.txt and "
                  "DerivedCoreProperties.txt (D-088)",
    },
}


class PackageError(Exception):
    pass


def extract(spec: dict, sdk_root: pathlib.Path) -> str:
    """A notice's text as provenance.toml's `extract` locates it."""
    kind, _, rel = spec["file"].partition(":")
    base = {"sdk": sdk_root, "repo": REPO}.get(kind)
    if base is None:
        raise PackageError(f"notice file {spec['file']!r} is neither sdk: nor repo:")
    path = base / rel
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise PackageError(f"cannot read the notice source {path}: {e}") from None
    if "from" not in spec:
        return "\n".join(lines).strip("\n") + "\n"
    start = next((i for i, line in enumerate(lines) if spec["from"] in line), None)
    if start is None:
        raise PackageError(f"{path} has no line containing {spec['from']!r}")
    end = next((i for i in range(start, len(lines)) if spec.get("to", spec["from"]) in lines[i]), None)
    if end is None:
        raise PackageError(f"{path} has no line containing {spec['to']!r} after {spec['from']!r}")
    end += spec.get("plus", 0)
    if end >= len(lines):
        raise PackageError(f"{path} ends before the notice does")
    return "\n".join(_uncomment(lines[start:end + 1])) + "\n"


def _uncomment(lines: list[str]) -> list[str]:
    """Lines without a comment prefix common to all of them (" * ", "// ", "#")."""
    for prefix in (" * ", "// ", "# ", " *", "//", "#"):
        if all(line.startswith(prefix) or line.strip() in ("", prefix.strip()) for line in lines):
            return [line[len(prefix):] if line.startswith(prefix) else "" for line in lines]
    return lines


def component_notice(source: pathlib.Path, notice: str) -> str:
    """A lock component's notice: a file in its prepared tree, whole or a line range."""
    rel, lines = srclib.notice_range(notice)
    try:
        text = (source / rel).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise PackageError(f"cannot read {source / rel}: {e}") from None
    if lines:
        if lines[1] > len(text):
            raise PackageError(f"{source / rel} has fewer than {lines[1]} lines")
        text = [line.strip().removeprefix("//").strip() for line in text[lines[0] - 1:lines[1]]]
    return "\n".join(text).strip("\n") + "\n"


def binary_facts(readelf: pathlib.Path, binary: pathlib.Path) -> dict:
    """The shared libraries a binary needs, the highest GLIBC_ version it imports, and its run paths (RPATH
    and RUNPATH entries, as written)."""
    dynamic = subprocess.run([readelf, "--dynamic", "--wide", binary], capture_output=True, text=True)
    versions = subprocess.run([readelf, "--version-info", "--wide", binary], capture_output=True, text=True)
    if dynamic.returncode or versions.returncode:
        raise PackageError(f"llvm-readelf cannot read {binary}: {dynamic.stderr or versions.stderr}")
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dynamic.stdout)
    glibc = [tuple(int(p) for p in v.split(".")) for v in re.findall(r"GLIBC_([0-9]+(?:\.[0-9]+)+)", versions.stdout)]
    runpaths = re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library (?:rpath|runpath): \[([^\]]*)\]", dynamic.stdout)
    return {"needed": needed, "glibc": max(glibc) if glibc else None, "runpath": runpaths}


def binary_problem(installed: str, facts: dict) -> str | None:
    """Why a packaged executable's needs or run path are not allowed, or None: every needed library has a
    source (ALLOWED_NEEDED), and the only run path is the runtime's, exactly PRIVATE_RUNPATH's, when it needs
    cuBLAS (D-060, D-076)."""
    unknown = sorted(set(facts["needed"]) - set(ALLOWED_NEEDED))
    if unknown:
        return f"{installed} needs {', '.join(unknown)}, which no dependency provides"
    cublas = any(ALLOWED_NEEDED[n] == PACKAGE for n in facts["needed"])
    if cublas and installed not in PRIVATE_RUNPATH:
        return f"{installed} needs cuBLAS, which only the runtime's run path finds"
    if facts["runpath"] and (not cublas or facts["runpath"] != [PRIVATE_RUNPATH[installed]]):
        return f"{installed} has a run path ({', '.join(facts['runpath'])}) other than its private cuBLAS directory"
    if cublas and not facts["runpath"]:
        return f"{installed} needs cuBLAS but has no run path to find it"
    return None


def expected_files(cuda: bool) -> dict[str, int]:
    """Every file the package installs, with its mode: EXPECTED_FILES, and in CUDA builds cuBLAS (D-076)."""
    files = dict(EXPECTED_FILES)
    if cuda:
        files.update({name: 0o644 for name in PRIVATE_LIBRARIES})
    return files


def _unit_version(sdk: sdklib.Sdk, unit: dict, arch: str) -> str:
    versions = set()
    for key in unit.get("artifacts", []):
        artifact = sdk.lock["artifacts"].get(f"{key}/{arch}") or sdk.lock["artifacts"].get(key)
        if artifact:
            versions.add(artifact["version"])
    return ", ".join(sorted(versions)) or "NOASSERTION"


# Where Debian keeps the full text of a license the notices only name.
COMMON_LICENSES = {"GPL-3.0-or-later": "GPL-3", "GPL-2.0-only": "GPL-2", "LGPL-2.1-or-later": "LGPL-2.1",
                   "Apache-2.0": "Apache-2.0"}


def license_pointer(expression: str) -> str:
    """A Debian copyright License paragraph's body: where the terms of expression are."""
    lines = []
    for base, name in COMMON_LICENSES.items():
        if re.search(rf"(^|[ (]){re.escape(base)}([ )]|$)", expression):
            lines.append(f" On Debian systems, the full text of {base} is in /usr/share/common-licenses/{name}.")
    for exception in re.findall(r"WITH ([A-Za-z0-9.-]+)", expression):
        lines.append(f" {exception}: https://spdx.org/licenses/{exception}.html; it lets object code built with the "
                     "component be distributed on the program's own terms.")
    if "LicenseRef-NVIDIA-CUDA-EULA" in expression:
        lines.append(" The NVIDIA CUDA Toolkit End User License Agreement, reproduced in full in")
        lines.append(" /usr/share/doc/jitllm/THIRD-PARTY-NOTICES.")
    if re.search(r"(^|[ (])MIT([ )]|$)", expression):
        lines.append(" The MIT texts, with their copyright notices, are in /usr/share/doc/jitllm/THIRD-PARTY-NOTICES.")
    if re.search(r"(^|[ (])Unicode-3\.0([ )]|$)", expression):
        lines.append(" The Unicode License V3 text, with Unicode, Inc.'s copyright notice, is in")
        lines.append(" /usr/share/doc/jitllm/THIRD-PARTY-NOTICES.")
    return "\n".join(lines)


def shipped_units(provenance: dict, cuda: bool) -> list[tuple[str, dict]]:
    return [(name, unit) for name, unit in provenance["units"].items()
            if unit["ships"] and (cuda or name not in CUDA_UNITS)]


def built_from(build: pathlib.Path, ninja: pathlib.Path) -> set[pathlib.Path]:
    """Every file the packaged executables are built from, recursively, as Ninja records their inputs, resolved."""
    build = build.resolve()
    targets = [built for built, _ in EXECUTABLES]
    result = subprocess.run([ninja, "-C", build, "-t", "inputs", "-0", "-E", *targets], capture_output=True, text=True)
    if result.returncode:
        raise PackageError(f"ninja cannot list the inputs of {', '.join(targets)} in {build}: "
                           f"{(result.stderr or result.stdout).strip()}")
    return {(build / path).resolve() for path in result.stdout.split("\0") if path}


# A header's license tag, split so that REUSE does not read this line as one.
_LICENSE_TAG = re.compile("SPDX-" + r"License-Identifier:\s*(.*?)\s*(?:\*/|-->)?\s*$")


def license_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        raise PackageError(f"cannot read {path}: {e}") from None


def header_license(path: pathlib.Path, text: str) -> str | None:
    expressions = {match[1] for line in text.splitlines()[:headers.HEADER_LINES]
                   if (match := _LICENSE_TAG.search(line))}
    if len(expressions) > 1:
        raise PackageError(f"{path} has conflicting license identifiers")
    return next(iter(expressions), None)


def sidecar_license(path: pathlib.Path, text: str) -> str | None:
    expression = header_license(path, text)
    if expression is None:
        return None
    # Late conflicting tags must not replace the license in the read window.
    if any(match[1] != expression for line in text.splitlines() if (match := _LICENSE_TAG.search(line))):
        raise PackageError(f"{path} has conflicting license identifiers")
    return expression


def declared_license(path: pathlib.Path) -> str | None:
    """The embedded license, or a noncommentable file's adjacent sidecar (D-029, D-071)."""
    text = license_text(path)
    if path.name.endswith(headers.SIDECAR):
        return sidecar_license(path, text)
    try:
        style = headers.style_of(path.as_posix(), text.split("\n", 1)[0].encode())
    except ValueError as e:
        raise PackageError(str(e)) from None
    sidecar = pathlib.Path(str(path) + headers.SIDECAR)
    embedded = header_license(path, text)
    if style is not None:
        if sidecar.exists():
            raise PackageError(f"{path} can hold a comment; embed its license and remove {sidecar}")
        return embedded
    if not sidecar.is_file():
        return None
    expression = sidecar_license(sidecar, license_text(sidecar))
    if embedded is not None and embedded != expression:
        raise PackageError(f"{path} and {sidecar} have conflicting license identifiers")
    return expression


def check_in_tree_units(root: pathlib.Path = REPO) -> None:
    """Fails unless IN_TREE_UNITS lists every file under root/src that declares a license outside the core's
    (srclib.CORE_LICENSES: D-017's allowlist as D-091 widens it to every recognized permissive license), a
    core license that no source-lock component or shipped provenance unit declares, a license an IN_TREE_UNITS
    entry records, or no license, and lists only translation units that exist. The core's licenses beside
    Apache-2.0 mark code adapted from a source-lock component or platform unit, whose notices come with it, so
    that component's record must name the license; a license only an in-tree unit brings (Unicode-3.0 for the
    tokenizer's tables) has no such component, so its files always need their record."""
    listed = {path for unit in IN_TREE_UNITS.values() for path in unit["files"]}
    # A core license admits a file; a record carries its notice. So beside Apache-2.0 an unlisted file may
    # declare only licenses that a source-lock component or a shipped platform unit declares too (a CUB-derived
    # file waits for provenance.toml's cccl unit to name BSD-3-Clause and BSL-1.0 and carry their notices).
    try:
        recorded = {i for comp in srclib.load_lock(modules=[])["components"].values()
                    for i in comp["license"]["expression"].split(" AND ")}
    except srclib.SourceError as e:
        raise PackageError(str(e)) from None
    recorded |= {i for unit in tomllib.loads(PROVENANCE.read_text())["units"].values() if unit["ships"]
                 for i in re.findall(r"[A-Za-z0-9.+-]+", unit["license"])}
    unrecorded = ((srclib.CORE_LICENSES & recorded) | {"Apache-2.0"}) - {unit["license"]
                                                                           for unit in IN_TREE_UNITS.values()}
    for path in sorted(listed):
        if pathlib.PurePosixPath(path).suffix not in TRANSLATION_UNITS or not (root / path).is_file():
            raise PackageError(f"IN_TREE_UNITS lists {path}, which is not a translation unit in the repository, "
                               "so Ninja's inputs cannot show whether an executable is built from it")
    for path in sorted(p for p in (root / "src").rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        declared = declared_license(path)
        ids = set(re.findall(r"[A-Za-z0-9.+-]+", declared or "")) - {"AND", "OR", "WITH"}
        if (not ids or not ids <= unrecorded) and rel not in listed:
            raise PackageError(f"{rel} declares {declared or 'no license'}, beyond the core's licenses (D-017, "
                               "D-091), recorded only in IN_TREE_UNITS or declared by no source-lock component "
                               "or shipped platform unit, and no IN_TREE_UNITS entry lists it (its license and "
                               "notice would not reach the package)")


def in_tree_units(inputs: set[pathlib.Path]) -> list[tuple[str, dict]]:
    """The IN_TREE_UNITS any of whose files is among inputs (resolved paths, from built_from), once
    check_in_tree_units passes."""
    check_in_tree_units()
    return [(name, unit) for name, unit in IN_TREE_UNITS.items()
            if any((REPO / path).resolve() in inputs for path in unit["files"])]


def data_notice(unit: dict, provenance: dict, sdk_root: pathlib.Path) -> str:
    """An in-tree unit's section of THIRD-PARTY-NOTICES, after its separator."""
    return "\n".join([f"{unit['name']} {unit['version']} ({unit['license']}), in jitLLM: {unit['enters']}",
                      unit["download"], unit["copyright"], "",
                      extract(provenance["notices"][unit["notice"]]["extract"], sdk_root)])


def generate(build: pathlib.Path, sdk: sdklib.Sdk, out: pathlib.Path) -> dict:
    """Writes the package's documents for the build in `build` into `out`; returns control.json's content."""
    try:
        receipt = json.loads((build / "jitllm-receipt.json").read_text())
    except (OSError, ValueError) as e:
        raise PackageError(f"cannot read {build / 'jitllm-receipt.json'}: {e}; build first") from None
    arch = DEBIAN_ARCH.get(receipt["target"])
    if arch is None:
        raise PackageError(f"no Debian architecture for {receipt['target']}")
    provenance = tomllib.loads(PROVENANCE.read_text())
    lock = srclib.load_lock(srclib.LOCK, modules=receipt["modules"])
    readelf = sdk.root / "bin" / "llvm-readelf"
    cuda = bool(receipt["cuda"])

    # Dependencies, from what the binaries import.
    glibc, needs_cuda, needs_cublas = (0,), False, False
    for built, installed in EXECUTABLES:
        facts = binary_facts(readelf, build / built)
        if problem := binary_problem(installed, facts):
            raise PackageError(problem)
        glibc = max(glibc, facts["glibc"] or (0,))
        needs_cuda = needs_cuda or "libcuda.so.1" in facts["needed"]
        needs_cublas = needs_cublas or any(ALLOWED_NEEDED[n] == PACKAGE for n in facts["needed"])
    if needs_cublas != cuda:
        raise PackageError("the runtime needs cuBLAS exactly in CUDA builds, whose package ships it")
    if needs_cublas:
        # The libraries the package ships beside the runtime set the glibc floor too, and may need only
        # glibc's libraries, libgcc_s (CUBLAS_DEPENDS) and each other.
        for installed, built in PRIVATE_LIBRARIES.items():
            facts = binary_facts(readelf, build / built)
            glibc = max(glibc, facts["glibc"] or (0,))
            unknown = sorted(set(facts["needed"]) - set(ALLOWED_NEEDED) - PRIVATE_LIBRARY_NEEDS)
            if unknown:
                raise PackageError(f"{installed} needs {', '.join(unknown)}, which no dependency provides")
    depends = [f"libc6 (>= {'.'.join(map(str, glibc))})"]
    if needs_cuda:
        depends.append(f"libcuda.so.1 (>= {CUDA_DRIVER_FLOOR})")
    if needs_cublas:
        depends.append(CUBLAS_DEPENDS)
    # systemd-sysusers and systemd-tmpfiles run from the maintainer scripts.
    depends.append("systemd")
    # runuser drops purge's deletion of owner-controlled spill files to the service user.
    depends.append("util-linux")

    products = [c for c in receipt["components"] if c["use"] == "product"]
    units = shipped_units(provenance, cuda)
    data_units = in_tree_units(built_from(build, sdk.root / "bin" / "ninja"))
    out.mkdir(parents=True, exist_ok=True)

    # THIRD-PARTY-NOTICES.
    parts = [
        "jitLLM third-party notices",
        "==========================",
        "",
        f"For jitllm {receipt['version']['product']}, {receipt['target']}, license profile "
        f"{receipt['license_profile']}. jitLLM's own code is under the Apache License 2.0 (LICENSE, NOTICE).",
        "The binaries also carry the code and data below, under the terms that follow. A notice is included",
        "whenever its component ships, whether or not this build uses the part it covers.",
    ]
    if cuda:
        parts += ["",
                  "The NVIDIA CUDA runtime object code (libcudart_static) and the code NVCC generates in these",
                  "binaries are under the NVIDIA CUDA Toolkit End User License Agreement, reproduced below, not",
                  "under the Apache License. The NVIDIA driver (libcuda.so.1) is not part of this package."]
    seen: set[str] = set()
    for component in products:
        entry = lock["components"][component["id"]]
        source = pathlib.Path(component["source"])
        parts += ["", "-" * 78, f"{component['id']} {component['version']} ({entry['license']['expression']})",
                  entry["upstream"]["repository"], ""]
        for notice in entry["license"]["notices"]:
            parts.append(component_notice(source, notice))
    # The GCC runtime's `unicode` notice below is the same text, for its own
    # tables; each work keeps its notice under its own heading.
    for name, unit in data_units:
        parts += ["", "-" * 78, data_notice(unit, provenance, sdk.root)]
    for name, unit in units:
        for notice in unit["notices"]:
            if notice in seen:
                continue
            seen.add(notice)
            record = provenance["notices"][notice]
            parts += ["", "-" * 78, f"{record['text']} ({name})", f"Applies: {record['when']}.", "",
                      extract(record["extract"], sdk.root)]
    (out / "THIRD-PARTY-NOTICES").write_text("\n".join(parts).rstrip("\n") + "\n")

    # copyright (https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/): everything is
    # jitLLM's, and the two executables also contain the components the notices cover.
    licenses = ["Apache-2.0"]
    contains = []
    for component in products:
        entry = lock["components"][component["id"]]
        contains.append(f" {component['id']} {component['version']}: {entry['license']['expression']}")
        licenses.append(entry["license"]["expression"])
    for name, unit in data_units:
        contains.append(f" {unit['name']} {unit['version']} (data, {', '.join(unit['files'])}): {unit['license']}")
        licenses.append(unit["license"])
    for name, unit in units:
        contains.append(f" {name} (build toolchain, {unit['category']}): {unit['license']}")
        licenses.append(unit["license"])
    distinct = list(dict.fromkeys(licenses))
    executables = " ".join(installed for _, installed in EXECUTABLES)
    stanzas = [
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
        f"Upstream-Name: jitLLM\nSource: {HOMEPAGE}",
        "Files: *\nCopyright: 2026 jitLLM contributors\nLicense: Apache-2.0",
        f"Files: {executables}\nCopyright: 2026 jitLLM contributors, and the holders named in THIRD-PARTY-NOTICES\n"
        f"License: {' AND '.join(f'({x})' if ' ' in x else x for x in distinct)}\n"
        "Comment: These executables also contain code or data from the following, whose notices are in\n"
        " /usr/share/doc/jitllm/THIRD-PARTY-NOTICES; jitllm.spdx.json lists them with their versions.\n"
        + "\n".join(contains),
        "License: Apache-2.0\n On Debian systems, the full text of the Apache License 2.0 is in\n"
        " /usr/share/common-licenses/Apache-2.0, and in /usr/share/doc/jitllm/LICENSE.",
    ]
    for expression in distinct[1:]:
        stanzas.append(f"License: {expression}\n" + license_pointer(expression))
    (out / "copyright").write_text("\n\n".join(stanzas) + "\n")

    # The SBOM (SPDX 2.3).
    version = receipt["version"]
    created = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    root_id = "SPDXRef-Package-jitllm"
    packages = [{
        "SPDXID": root_id, "name": PACKAGE, "versionInfo": version["product"],
        "downloadLocation": f"git+{HOMEPAGE}.git@{version['commit']}" if version.get("commit") else "NOASSERTION",
        "homepage": HOMEPAGE, "licenseConcluded": "NOASSERTION", "licenseDeclared": "Apache-2.0",
        "copyrightText": "2026 jitLLM contributors", "supplier": "Organization: jitLLM contributors",
        "primaryPackagePurpose": "APPLICATION", "filesAnalyzed": False,
        "comment": f"License profile {receipt['license_profile']}; target {receipt['target']}; SDK {receipt['sdk']}; "
                   f"source lock sha256 {receipt['source_lock']['sha256']}.",
    }]
    relationships = [{"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES",
                      "relatedSpdxElement": root_id}]
    for component in products:
        entry = lock["components"][component["id"]]
        ident = f"SPDXRef-Source-{component['id']}"
        packages.append({
            "SPDXID": ident, "name": component["id"], "versionInfo": component["version"],
            "downloadLocation": entry["archive"]["urls"][0],
            "checksums": [{"algorithm": "SHA256", "checksumValue": entry["archive"]["sha256"]}],
            "licenseConcluded": entry["license"]["expression"], "licenseDeclared": entry["license"]["expression"],
            "copyrightText": "NOASSERTION", "filesAnalyzed": False, "primaryPackagePurpose": "LIBRARY",
            "comment": f"Commit {entry['upstream']['commit']}; incorporated implementation, {entry['tier']} tier (D-017).",
        })
        relationships.append({"spdxElementId": root_id, "relationshipType": "CONTAINS", "relatedSpdxElement": ident})
    for name, unit in data_units:
        ident = f"SPDXRef-Data-{name}"
        packages.append({
            "SPDXID": ident, "name": unit["name"], "versionInfo": unit["version"],
            "downloadLocation": unit["download"], "licenseConcluded": unit["license"],
            "licenseDeclared": unit["license"], "copyrightText": unit["copyright"], "filesAnalyzed": False,
            "primaryPackagePurpose": "OTHER",
            "comment": f"Incorporated data, core tier (D-017, D-088): {unit['enters']}.",
        })
        relationships.append({"spdxElementId": root_id, "relationshipType": "CONTAINS", "relatedSpdxElement": ident})
    for name, unit in units:
        ident = f"SPDXRef-Platform-{name}"
        packages.append({
            "SPDXID": ident, "name": name, "versionInfo": _unit_version(sdk, unit, arch),
            "downloadLocation": "NOASSERTION", "licenseConcluded": "NOASSERTION",
            "licenseDeclared": unit["license"],
            "copyrightText": "NOASSERTION", "filesAnalyzed": False, "primaryPackagePurpose": "LIBRARY",
            "comment": f"D-017 {unit['category']} ({unit['license']}): {unit['enters']}",
        })
        # Everything listed contributes code to the executables (glibc: its
        # start files and header code; the shared C library is a dependency
        # the package declares).
        relationships.append({"spdxElementId": root_id, "relationshipType": "CONTAINS", "relatedSpdxElement": ident})
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"{HOMEPAGE}/sbom/{version['product']}/{receipt['target']}")
    sbom = {
        "spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{PACKAGE}-{version['product']}-{arch}",
        "documentNamespace": f"{HOMEPAGE}/spdx/{namespace}",
        "creationInfo": {"created": created, "creators": ["Tool: jitLLM tools/jitllm_package.py"]},
        "packages": packages, "relationships": relationships,
    }
    refs = sorted({ref for unit in (u for _, u in units) for ref in re.findall(r"LicenseRef-[A-Za-z0-9.-]+",
                                                                                 unit["license"])})
    if refs:
        sbom["hasExtractedLicensingInfos"] = [
            {"licenseId": ref, "name": ref.removeprefix("LicenseRef-"),
             "extractedText": "See /usr/share/doc/jitllm/THIRD-PARTY-NOTICES, which reproduces its terms."}
            for ref in refs]
    (out / "jitllm.spdx.json").write_text(json.dumps(sbom, indent=2) + "\n")

    control = {"package": PACKAGE, "version": version["debian"], "architecture": arch,
               "depends": ", ".join(depends), "homepage": HOMEPAGE,
               "license_profile": receipt["license_profile"], "official": receipt["official"]}
    (out / "control.json").write_text(json.dumps(control, indent=2) + "\n")
    return control


# The package's inventory ---------------------------------------------------------

# Every path the package installs (under ./), with its type and mode; the
# owner is always root.
EXPECTED_FILES = {
    "usr/bin/jitllm": 0o755,
    "usr/libexec/jitllm/jitllm-runtime": 0o755,
    "usr/lib/systemd/system/jitllm.service": 0o644,
    "usr/lib/sysusers.d/jitllm.conf": 0o644,
    "usr/lib/tmpfiles.d/jitllm.conf": 0o644,
    "usr/share/doc/jitllm/LICENSE": 0o644,
    "usr/share/doc/jitllm/NOTICE": 0o644,
    "usr/share/doc/jitllm/CHANGELOG.md": 0o644,
    "usr/share/doc/jitllm/copyright": 0o644,
    "usr/share/doc/jitllm/THIRD-PARTY-NOTICES": 0o644,
    "usr/share/doc/jitllm/jitllm.spdx.json": 0o644,
    "usr/share/doc/jitllm/examples/jitllm.toml": 0o644,
}
# Files installed unchanged from the repository.
VERBATIM = {"usr/share/doc/jitllm/LICENSE": "LICENSE", "usr/share/doc/jitllm/NOTICE": "NOTICE",
            "usr/share/doc/jitllm/CHANGELOG.md": "CHANGELOG.md",
            "usr/lib/systemd/system/jitllm.service": "packaging/jitllm.service",
            "usr/lib/sysusers.d/jitllm.conf": "packaging/jitllm.sysusers",
            "usr/lib/tmpfiles.d/jitllm.conf": "packaging/jitllm.tmpfiles",
            "usr/share/doc/jitllm/examples/jitllm.toml": "packaging/jitllm.example.toml"}
MAINTAINER_SCRIPTS = ("postinst", "prerm", "postrm")


def read_deb(path: pathlib.Path) -> dict[str, dict[str, tuple[tarfile.TarInfo, bytes | None]]]:
    """A .deb's members: {"control": {...}, "data": {...}}, each name to (entry, content of a regular file)."""
    data = path.read_bytes()
    if not data.startswith(b"!<arch>\n"):
        raise PackageError(f"{path} is not an ar archive")
    members, at = {}, 8
    while at < len(data):
        header = data[at:at + 60]
        name = header[:16].decode().strip().rstrip("/")
        size = int(header[48:58].decode().strip())
        members[name] = data[at + 60:at + 60 + size]
        at += 60 + size + (size & 1)
    if members.get("debian-binary") != b"2.0\n":
        raise PackageError(f"{path} is not a version 2.0 Debian package")
    out = {}
    for part in ("control", "data"):
        name = next((m for m in members if m.startswith(f"{part}.tar")), None)
        if name is None:
            raise PackageError(f"{path} has no {part} archive")
        entries = {}
        with tarfile.open(fileobj=io.BytesIO(members[name])) as tar:
            for info in tar.getmembers():
                content = tar.extractfile(info).read() if info.isfile() else None
                entries[info.name.removeprefix("./").rstrip("/")] = (info, content)
        out[part] = entries
    return out


def check_package(deb: pathlib.Path, build: pathlib.Path, sdk: sdklib.Sdk) -> list[str]:
    """Every way the package differs from what its build's receipt, the repository and its documents say."""
    problems = []
    receipt = json.loads((build / "jitllm-receipt.json").read_text())
    doc = build / "package" / "doc"
    control_facts = json.loads((doc / "control.json").read_text())
    parts = read_deb(deb)
    data, control = parts["data"], parts["control"]

    # The control file.
    fields = dict(re.findall(r"^([A-Za-z-]+): (.*)$", (control.get("control", (None, b""))[1] or b"").decode(),
                             re.MULTILINE))
    for field, want in (("Package", PACKAGE), ("Version", receipt["version"]["debian"]),
                        ("Architecture", DEBIAN_ARCH.get(receipt["target"])), ("Depends", control_facts["depends"])):
        if fields.get(field) != want:
            problems.append(f"control: {field} is {fields.get(field)!r}, not {want!r}")
    for script in MAINTAINER_SCRIPTS:
        entry = control.get(script)
        want = (REPO / "packaging" / "debian" / script).read_bytes()
        if entry is None or entry[1] != want or entry[0].mode & 0o777 != 0o755:
            problems.append(f"control: {script} is missing, differs from packaging/debian/{script} or is not 0755")
    md5sums = dict(line.split("  ", 1)[::-1] for line in (control.get("md5sums", (None, b""))[1] or b"")
                   .decode().splitlines() if "  " in line)

    # The files.
    expected = expected_files(bool(receipt["cuda"]))
    files = {name: entry for name, entry in data.items() if name and not entry[0].isdir()}
    for name in sorted(set(files) - set(expected)):
        problems.append(f"data: {name} is not part of the installed layout")
    for name, mode in expected.items():
        entry = files.get(name)
        if entry is None:
            problems.append(f"data: {name} is missing")
            continue
        info, content = entry
        if not info.isfile() or info.mode & 0o7777 != mode or info.uid != 0 or info.gid != 0:
            problems.append(f"data: {name} is not a regular file owned by root with mode {mode:04o} "
                            f"(mode {info.mode & 0o7777:04o}, uid {info.uid})")
        if md5sums.get(name) != hashlib.md5(content or b"").hexdigest():  # noqa: S324 (dpkg's own format)
            problems.append(f"control: md5sums does not match {name}")
        if name in VERBATIM and content != (REPO / VERBATIM[name]).read_bytes():
            problems.append(f"data: {name} differs from {VERBATIM[name]}")
        # cuBLAS as the SDK has it, unmodified and unstripped (D-076): the
        # build tree's lib/jitllm holds hard links to the SDK's files.
        if name in PRIVATE_LIBRARIES and content != (build / PRIVATE_LIBRARIES[name]).read_bytes():
            problems.append(f"data: {name} is not the SDK's {pathlib.PurePosixPath(name).name}, unmodified")
    for name, (info, _) in data.items():
        if info.isdir() and (info.mode & 0o7777 != 0o755 or info.uid != 0):
            problems.append(f"data: directory {name or '.'} is not root's with mode 0755")
    for name in ("copyright", "THIRD-PARTY-NOTICES", "jitllm.spdx.json"):
        entry = files.get(f"usr/share/doc/jitllm/{name}")
        if entry and entry[1] != (doc / name).read_bytes():
            problems.append(f"data: usr/share/doc/jitllm/{name} is not the one generated for this build")

    # The binaries: what they need, against the dependencies.
    with tempfile.TemporaryDirectory() as scratch:
        for _, installed in EXECUTABLES:
            entry = files.get(installed)
            if not entry or entry[1] is None:
                continue
            copy = pathlib.Path(scratch) / pathlib.PurePosixPath(installed).name
            copy.write_bytes(entry[1])
            facts = binary_facts(sdk.root / "bin" / "llvm-readelf", copy)
            if problem := binary_problem(installed, facts):
                problems.append(problem)
            if any(ALLOWED_NEEDED[n] == PACKAGE for n in facts["needed"] if n in ALLOWED_NEEDED) and \
                    CUBLAS_DEPENDS not in control_facts["depends"]:
                problems.append(f"{installed}: needs cuBLAS, but the package does not depend on {CUBLAS_DEPENDS}")
            floor = re.search(r"libc6 \(>= ([0-9.]+)\)", control_facts["depends"])
            if facts["glibc"] and (floor is None or facts["glibc"] > tuple(int(p) for p in floor[1].split("."))):
                problems.append(f"{installed}: imports GLIBC_{'.'.join(map(str, facts['glibc']))}, "
                                "above the libc6 dependency")
            if "libcuda.so.1" in facts["needed"] and "libcuda.so.1" not in control_facts["depends"]:
                problems.append(f"{installed}: needs libcuda.so.1, which the package does not depend on")

    # The SBOM and notices, against the receipt.
    sbom = json.loads((doc / "jitllm.spdx.json").read_text())
    listed = {p["name"]: p for p in sbom["packages"]}
    root = listed.get(PACKAGE, {})
    if root.get("versionInfo") != receipt["version"]["product"]:
        problems.append(f"SBOM: jitllm is {root.get('versionInfo')!r}, not {receipt['version']['product']}")
    notices = (doc / "THIRD-PARTY-NOTICES").read_text()
    lock = srclib.load_lock(srclib.LOCK, modules=receipt["modules"])
    for component in receipt["components"]:
        entry = listed.get(component["id"])
        if component["use"] != "product":
            if entry:
                problems.append(f"SBOM: {component['id']} is test-only and never ships")
            continue
        if not entry or entry["versionInfo"] != component["version"] or \
                entry.get("checksums", [{}])[0].get("checksumValue") != component["archive_sha256"]:
            problems.append(f"SBOM: {component['id']} {component['version']} is missing or differs from the receipt")
        for notice in lock["components"][component["id"]]["license"]["notices"]:
            text = component_notice(pathlib.Path(component["source"]), notice)
            if text.strip() not in notices:
                problems.append(f"notices: {component['id']}'s {notice} is missing")
    provenance = tomllib.loads(PROVENANCE.read_text())
    identities = {p["SPDXID"]: p for p in sbom["packages"]}
    copyright_text = (doc / "copyright").read_text()
    executables = " ".join(installed for _, installed in EXECUTABLES)
    license_line = re.search(rf"^Files: {re.escape(executables)}\n.*\nLicense: (.*)$", copyright_text, re.MULTILINE)
    for name, unit in in_tree_units(built_from(build, sdk.root / "bin" / "ninja")):
        entry = identities.get(f"SPDXRef-Data-{name}")
        if not entry or entry["versionInfo"] != unit["version"] or entry["licenseDeclared"] != unit["license"]:
            problems.append(f"SBOM: {name} {unit['version']} ({unit['license']}), which the executables are built "
                            "from, is missing or differs")
        if not license_line or unit["license"] not in re.split(r"[ ()]+", license_line[1]):
            problems.append(f"copyright: the executables' license does not name {unit['license']} ({name})")
        # The whole section, not the text alone, which the GCC runtime's
        # `unicode` notice already puts in every package.
        if data_notice(unit, provenance, sdk.root).strip() not in notices:
            problems.append(f"notices: {name}'s section, with its {unit['notice']} notice, is missing")
    for name, unit in shipped_units(provenance, bool(receipt["cuda"])):
        if name not in listed:
            problems.append(f"SBOM: the shipped platform unit {name} is missing")
        for notice in unit["notices"]:
            if extract(provenance["notices"][notice]["extract"], sdk.root).strip() not in notices:
                problems.append(f"notices: {notice} ({name}) is missing")
    return problems
