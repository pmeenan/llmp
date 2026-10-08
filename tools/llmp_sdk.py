# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared SDK definitions for tools/setup-toolchain and tools/check-toolchain.

The SDK for a build host lives at <home>/<identity>, where the identity is the
host architecture plus a digest of everything that determines the SDK's
contents: the manifest, the artifact lock and the setup program itself.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOLCHAINS = REPO / "toolchains"
MANIFEST = TOOLCHAINS / "manifest.toml"
LOCK = TOOLCHAINS / "artifacts.lock.json"
# Every file whose bytes determine what setup produces.
IDENTITY_INPUTS = (MANIFEST, LOCK, REPO / "tools" / "setup-toolchain", REPO / "tools" / "llmp_sdk.py")
RECEIPT = "sdk.json"
# Runs a module from one directory, searched after the standard library, as
# `python -m` would: argv is [-c, directory, module, tool arguments...].
_RUN_MODULE = ("import runpy, sys; sys.path.append(sys.argv[1]); module = sys.argv[2]; del sys.argv[1:3]; "
               "runpy.run_module(module, run_name='__main__', alter_sys=True)")


class SdkError(Exception):
    """A problem the user must fix; the message says how."""


def host_arch() -> str:
    machine = platform.machine()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(machine)
    if arch is None:
        raise SdkError(f"unsupported build host architecture {machine!r}")
    return arch


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_root(spec: dict) -> pathlib.Path:
    """Resolves a manifest root: an override variable, else XDG, else the default."""
    override = os.environ.get(spec["env"])
    if override:
        path = pathlib.Path(override).expanduser()
    else:
        base = os.environ.get(spec["xdg"]) or spec["default"]
        path = pathlib.Path(base).expanduser() / spec["suffix"]
    if not path.is_absolute():
        raise SdkError(f"{spec['env']} must be an absolute path, got {path}")
    return path


@dataclasses.dataclass(frozen=True)
class Sdk:
    manifest: dict
    lock: dict
    arch: str
    identity: str
    inputs: dict[str, str]
    home: pathlib.Path
    cache: pathlib.Path

    @property
    def root(self) -> pathlib.Path:
        return self.home / self.identity

    @property
    def host(self) -> dict:
        return self.manifest["hosts"][self.arch]

    @property
    def triple(self) -> str:
        return self.host["triple"]

    def components(self) -> list[tuple[str, dict]]:
        return [(name, self.manifest["components"][name]) for name in self.host["components"]]

    def python_tool(self, name: str) -> list[str]:
        """The command that runs a `wheels` component's tool with this Python.

        -I ignores PYTHON* variables, the working directory and user
        site-packages and -S skips site-packages, so only the standard library
        and the component's unpacked wheels are importable, never a different
        copy of a dependency. -B keeps bytecode out of the SDK, whose tree
        digest `doctor --deep` checks (-I also ignores PYTHONDONTWRITEBYTECODE).
        """
        comp = self.manifest["components"][name]
        return [sys.executable, "-B", "-I", "-S", "-c", _RUN_MODULE, str(self.root / comp["dest"]),
                comp["module"]]

    def artifact(self, key: str) -> dict:
        try:
            return self.lock["artifacts"][key]
        except KeyError:
            raise SdkError(f"{key} is not in {LOCK.relative_to(REPO)}") from None


def load(arch: str | None = None) -> Sdk:
    arch = arch or host_arch()
    manifest = tomllib.loads(MANIFEST.read_text())
    lock = json.loads(LOCK.read_text())
    if manifest.get("schema") != 1 or lock.get("schema") != 1:
        raise SdkError("unsupported manifest or lock schema")
    if arch not in manifest["hosts"]:
        raise SdkError(f"the manifest declares no SDK for {arch} build hosts")
    inputs = {str(p.relative_to(REPO)): sha256_file(p) for p in IDENTITY_INPUTS}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return Sdk(manifest=manifest, lock=lock, arch=arch, identity=f"{arch}-{digest[:16]}",
               inputs=inputs, home=resolve_root(manifest["sdk"]["home"]),
               cache=resolve_root(manifest["sdk"]["cache"]))


def tree_digest(root: pathlib.Path, *, exclude: tuple[str, ...] = (RECEIPT,)) -> str:
    """Digest of files and symlinks under root, except the named receipt files."""
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames + [d for d in dirnames if (pathlib.Path(dirpath) / d).is_symlink()]):
            path = pathlib.Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if rel in exclude:
                continue
            if path.is_symlink():
                entry = f"l {rel} {os.readlink(path)}"
            else:
                mode = path.stat().st_mode & 0o777
                entry = f"f {rel} {mode:o} {sha256_file(path)}"
            digest.update(entry.encode() + b"\n")
    return digest.hexdigest()


def read_receipt(root: pathlib.Path) -> dict | None:
    """The SDK's receipt, or None when it is missing or unreadable."""
    try:
        receipt = json.loads((root / RECEIPT).read_text())
        return receipt if isinstance(receipt, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


# Host prerequisites -------------------------------------------------------------

_REQUIREMENT = re.compile(r"^([a-z0-9][a-z0-9+.-]*)(?:\s*(>=|=)\s*(\S+))?$")


def prerequisites(sdk: Sdk) -> list[tuple[str, str | None, str | None]]:
    """Parses the host's prerequisite list: `package [>=|= version]`, `#` comments."""
    path = TOOLCHAINS / sdk.host["prerequisites"]
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REQUIREMENT.match(line)
        if not match:
            raise SdkError(f"{path.relative_to(REPO)}:{number}: cannot parse {line!r}")
        result.append(match.groups())
    return result


def installed_versions(packages: list[str]) -> dict[str, str]:
    """Versions of installed Debian packages; missing packages are absent."""
    if shutil.which("dpkg-query") is None:
        raise SdkError("dpkg-query not found; the SDK supports Ubuntu 24.04 build hosts")
    arch = subprocess.run(["dpkg", "--print-architecture"], capture_output=True, text=True,
                          check=True).stdout.strip()
    proc = subprocess.run(["dpkg-query", "-W", "-f", "${Package}\t${Architecture}\t${db:Status-Abbrev}\t${Version}\n",
                           *packages], capture_output=True, text=True)
    versions = {}
    for line in proc.stdout.splitlines():
        name, pkg_arch, status, version = line.split("\t")
        # The second status letter is the state: "ii" installed, "hi" held and
        # installed, "rc" removed. Multi-arch hosts can list foreign-architecture
        # copies; the native one counts.
        if status[1:2] == "i" and pkg_arch in (arch, "all"):
            versions[name] = version
    return versions


def version_at_least(installed: str, wanted: str) -> bool:
    return subprocess.run(["dpkg", "--compare-versions", installed, "ge", wanted]).returncode == 0


def check_prerequisites(sdk: Sdk) -> list[str]:
    """Returns one problem per unmet prerequisite (empty when all are met)."""
    reqs = prerequisites(sdk)
    versions = installed_versions([name for name, _, _ in reqs])
    problems = []
    for name, op, wanted in reqs:
        have = versions.get(name)
        if have is None:
            problems.append(f"{name}: not installed")
        elif op == ">=" and not version_at_least(have, wanted):
            problems.append(f"{name}: {have} installed, need >= {wanted}")
        elif op == "=" and have != wanted:
            problems.append(f"{name}: {have} installed, need exactly {wanted}")
    return problems


def install_hint(sdk: Sdk, problems: list[str]) -> str:
    names = sorted({p.split(":", 1)[0] for p in problems})
    return "sudo apt-get install " + " ".join(names)
