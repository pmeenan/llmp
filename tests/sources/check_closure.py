# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Checks a build's actual compile and link inventory against its receipt (D-017, D-057, D-066, D-083).

    python3 check_closure.py --build-dir DIR --source-dir REPO --sdk SDK --ninja NINJA [--cross]
                             [--libstdcxx-assertions] [--target OUTPUT ...]

Without --target, reads every input Ninja recorded for the current outputs (each compile's
headers and, where supported, each link's objects and libraries, from the
compiler's and linker's dependency files) and every link edge's inputs and
flags in build.ninja (CUDA and Spark GNU links may record no dependency file),
and checks:

- third-party sources come only from the receipt's components, and every
  component in the receipt is compiled into something (the closure is the
  inventory);
- third-party outputs come only from those components' build directories;
- everything else is jitLLM's own source, this build tree or the SDK, never
  another build directory; a native build may also use the host's glibc and
  kernel headers (the declared platform packages), a cross build nothing
  from the host at all;
- a build-tree file is an output of a current build rule or lies in a
  selected component's build directory, so no leftover of another profile
  and no file a script wrote at configure is used;
- libraries named with -l are the declared platform's: glibc and the
  static CUDA runtime (D-017, D-060);
- every C++ and CUDA compile, third-party code included, ends its exception
  flags (-f[no-][cxx-]exceptions, nvcc's -Xcompiler lists included) with
  -fno-exceptions, forwards none to the frontend with -Xclang, and uses no
  response file that could hide flags; and no object or archive compiled or
  linked from outside the SDK has exception support (throw, catch or
  personality symbols, or a landing-pad table), whatever the flags said;
- with --libstdcxx-assertions every C++ and CUDA compile, third-party code
  included, defines _GLIBCXX_ASSERTIONS, and without it none does, not even
  as libstdc++ does itself for an unoptimized compile, so no inline function
  is compiled both ways (D-083);
- every link input from outside the SDK and platform is, by its content, an
  object file or archive (audited as above): never a shared library (D-064),
  a linker script or anything else; and the only shared libraries linked
  from the SDK are the sysroot's glibc, NVIDIA's driver stub, libcuda.so,
  which binaries resolve to the driver's libcuda.so.1 at run time (D-072),
  and the pinned cuBLAS, libcublas.so.13 and libcublasLt.so.13 (D-076);
- nothing the package ships (its executables, with every object, archive and
  header they are built from) comes from a test-only component (the lock's
  `use: test`), whose notices the package does not carry.
"""

import argparse
import json
import os
import pathlib
import re
import re
import shlex
import subprocess
import sys

# The host's declared platform packages (D-017, D-070): glibc and the kernel
# UAPI headers. A native build may read these files and nothing else from the host.
HOST_PLATFORM_PACKAGES = ("libc6", "libc6-dev", "linux-libc-dev")
# Libraries a link may name with -l: glibc's and the static CUDA runtime's.
PLATFORM_LIBRARIES = ("c", "m", "dl", "rt", "pthread", "cudart_static", "cudadevrt")
# The executables the package ships, as tools/jitllm_package.py's EXECUTABLES
# names them (a tooling test keeps the two equal).
SHIPPED_EXECUTABLES = ("src/cli/jitllm", "src/runtime/jitllm-runtime")
# The SDK's cuBLAS, linked dynamically (D-076): the SDK-relative path of the
# file its libcublas.so or libcublasLt.so resolves to, inside the unpacked libcublas package
# (toolchains/manifest.toml, component cublas or cublas-sbsa-target).
CUBLAS = re.compile(r"^/pkgs/cuda/usr/local/cuda-[0-9.]+/targets/[^/]+/lib/libcublas(Lt)?\.so\.13\.[0-9.]+$")


def within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def host_platform_files() -> set[str]:
    # Qualified by the native architecture: a multiarch host may also have
    # foreign copies of these Multi-Arch: same packages.
    arch = subprocess.run(["dpkg", "--print-architecture"], capture_output=True, text=True,
                          check=True).stdout.strip()
    out = subprocess.run(["dpkg-query", "-L", *(f"{p}:{arch}" for p in HOST_PLATFORM_PACKAGES)],
                         capture_output=True, text=True, check=True).stdout
    return {line.strip() for line in out.splitlines() if line.startswith("/")}


def usr_merged(path: str) -> str:
    """The /usr path a merged-/usr host keeps a /lib, /lib64 or /bin file at."""
    for top in ("/lib/", "/lib64/", "/lib32/", "/bin/", "/sbin/"):
        if path.startswith(top):
            return "/usr" + path
    return path


# The flags that decide whether C++ exceptions are on: clang takes the last
# of these (D-066 requires it to be off).
EXCEPTION_FLAGS = ("-fexceptions", "-fno-exceptions", "-fcxx-exceptions", "-fno-cxx-exceptions")
CXX_SOURCES = (".cc", ".cpp", ".cxx", ".c++", ".C", ".cu")


# What an object compiled with exceptions contains: throw and catch support,
# the personality routine and unwinding through cleanups, and the landing-pad
# tables. None of it appears in an object compiled without exceptions.
EXCEPTION_SYMBOLS = frozenset({"__cxa_throw", "__cxa_rethrow", "__cxa_allocate_exception", "__cxa_free_exception",
                               "__cxa_begin_catch", "__cxa_end_catch", "__gxx_personality_v0", "_Unwind_Resume"})


def exception_problem(tokens: list[str]) -> str | None:
    """Why a C++ or CUDA compile could have exceptions on, or None when it cannot."""
    flags = []
    for previous, token in zip([""] + tokens, tokens):
        if token.startswith("@"):
            return f"uses an unexpanded response file {token}"
        forwarded = token[len("-Xclang="):] if token.startswith("-Xclang=") else token if previous == "-Xclang" else None
        if forwarded is not None and forwarded.endswith("exceptions") and not forwarded.startswith("-fno-"):
            # The frontend flag stays on whatever driver flags follow it.
            return f"passes {forwarded} straight to the compiler frontend"
        if token.startswith(("-Xcompiler=", "--compiler-options=")):
            flags += token.split("=", 1)[1].split(",")
        else:
            flags.append(token)
    decisive = [f for f in flags if f in EXCEPTION_FLAGS]
    if "-fno-exceptions" not in decisive or decisive[-1] not in ("-fno-exceptions", "-fno-cxx-exceptions"):
        return f"does not end its exception flags with -fno-exceptions ({' '.join(decisive) or 'none'})"
    return None


def assertions_problem(tokens: list[str], want: bool) -> str | None:
    """Why a C++ or CUDA compile's libstdc++ assertions differ from the build's (D-083), or None.

    As libstdc++'s c++config.h decides: the macro is on when the last -D/-U of
    it defines it, when _GLIBCXX_DEBUG is defined, or when the compile is
    unoptimized (no -O, or a last -O0) without _GLIBCXX_NO_ASSERTIONS. NVCC's
    -Xcompiler flags reach its host pass, so they count.
    """
    host = ("-Xcompiler", "--compiler-options")
    # Options whose value is for another tool (the assembler, the device
    # linker, the frontend...), never the host compile's -D or -O.
    elsewhere = ("-Xptxas", "--ptxas-options", "-Xnvlink", "--nvlink-options", "-Xlinker", "--linker-options",
                 "-Xfatbin", "--fatbin-options", "-Xarchive", "--archive-options", "-Xcudafe", "-Xclang")
    flags = []
    for previous, token in zip([""] + tokens, tokens):
        if previous in host:
            flags += token.split(",")
        elif token.startswith(tuple(f"{option}=" for option in host)):
            flags += token.split("=", 1)[1].split(",")
        elif previous not in elsewhere and token not in host and not token.startswith(
                tuple(f"{option}=" for option in elsewhere)):
            flags.append(token)
    macros: dict[str, bool] = {}
    optimized = False
    for previous, flag in zip([""] + flags, flags):
        if previous in ("-D", "-U", "--define-macro", "--undefine-macro"):
            flag = ("-U" if previous in ("-U", "--undefine-macro") else "-D") + flag
        elif flag.startswith(("--define-macro=", "--undefine-macro=")):
            flag = ("-U" if flag.startswith("--undefine-macro=") else "-D") + flag.split("=", 1)[1]
        if previous == "--optimize":
            optimized = flag != "0"
        elif flag.startswith(("-D", "-U")) and len(flag) > 2:
            macros[flag[2:].split("=", 1)[0]] = flag.startswith("-D")
        elif re.fullmatch(r"-O[0-9sgz]?|-Ofast|--optimize=[0-9]+", flag):
            optimized = flag not in ("-O0", "--optimize=0")
    on = (macros.get("_GLIBCXX_ASSERTIONS", False) or macros.get("_GLIBCXX_DEBUG", False)
          or (not optimized and not macros.get("_GLIBCXX_NO_ASSERTIONS", False)))
    if want and not on:
        return "does not define _GLIBCXX_ASSERTIONS, which this build defines everywhere"
    if not want and on:
        how = "" if macros.get("_GLIBCXX_ASSERTIONS") else " (unoptimized, or through _GLIBCXX_DEBUG)"
        return f"defines _GLIBCXX_ASSERTIONS{how}, which this build defines nowhere"
    return None


def exception_machinery(sdk: pathlib.Path, objects: list[str]) -> dict[str, list[str]]:
    """The exception support each object file contains (symbols it needs, or its landing-pad table)."""
    found: dict[str, list[str]] = {}
    for start in range(0, len(objects), 200):
        batch = objects[start:start + 200]
        nm = subprocess.run([str(sdk / "bin" / "llvm-nm"), "-A", "-u", *batch], capture_output=True, text=True,
                            check=True).stdout
        for line in nm.splitlines():
            match = re.match(r"^(.*):\s+U\s+(\S+)$", line)
            if match and match[2] in EXCEPTION_SYMBOLS:
                found.setdefault(match[1], []).append(match[2])
        sections = subprocess.run([str(sdk / "bin" / "llvm-readelf"), "-S", *batch], capture_output=True, text=True,
                                  check=True).stdout
        current = None
        for line in sections.splitlines():
            if line.startswith("File: "):
                current = line[len("File: "):]
            elif current and ".gcc_except_table" in line:
                found.setdefault(current, []).append(".gcc_except_table")
    return found


def link_input_kind(path: str) -> str:
    """"object" for anything the audit must inspect (ELF relocatable, archive,
    LLVM bitcode), "shared" for a shared library, else a description."""
    with open(path, "rb") as f:
        head = f.read(18)
    if head.startswith((b"!<arch>\n", b"!<thin>\n", b"BC\xc0\xde", b"\xde\xc0\x17\x0b")):
        return "object"
    if head.startswith(b"\x7fELF") and len(head) == 18:
        elf_type = int.from_bytes(head[16:18], "big" if head[5] == 2 else "little")
        return {1: "object", 3: "shared"}.get(elf_type, f"an ELF file of type {elf_type}")
    return "a linker script or unknown file" if head[:1].isascii() else "an unknown binary file"


def focused_outputs(ninja: str, build_dir: pathlib.Path, targets: list[str],
                    problems: list[str]) -> set[str]:
    """Actual recursive inputs of named Ninja outputs; no inferred source or object list."""
    text = subprocess.run([ninja, "-C", str(build_dir), "-t", "targets", "all"],
                          capture_output=True, text=True, check=True).stdout
    known = {line.rsplit(": ", 1)[0] for line in text.splitlines() if ": " in line}
    if any(target not in known for target in targets):
        problems.append("unknown focused target: " + ", ".join(t for t in targets if t not in known))
        return set()
    text = subprocess.run([ninja, "-C", str(build_dir), "-t", "inputs", "-0", "-E", *targets],
                          capture_output=True, text=True, check=True).stdout
    return set(targets) | {path for path in text.split("\0") if path}


def recorded_inputs(ninja: str, build_dir: pathlib.Path,
                    problems: list[str], scope: set[str] | None = None) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Each current output's recorded inputs (outputs no longer in build.ninja are skipped),
    and the rule of every current output."""
    targets = subprocess.run([ninja, "-C", str(build_dir), "-t", "targets", "all"], capture_output=True,
                             text=True, check=True).stdout
    rules = dict(line.rsplit(": ", 1) for line in targets.splitlines() if ": " in line)
    recording_rules = set()
    current_rule = None
    for line in (build_dir / "CMakeFiles/rules.ninja").read_text().splitlines():
        if line.startswith("rule "):
            current_rule = line[len("rule "):].strip()
        elif current_rule and line.strip().startswith("deps = "):
            recording_rules.add(current_rule)
    outputs = [output for output in rules if scope is None or output in scope]
    required = scope if scope is not None else set(subprocess.run(
        [ninja, "-C", str(build_dir), "-t", "inputs", "-0", "-E", "all"],
        capture_output=True, text=True, check=True).stdout.split("\0"))
    inputs: dict[str, list[str]] = {}
    for start in range(0, len(outputs), 200):
        text = subprocess.run([ninja, "-C", str(build_dir), "-t", "deps", *outputs[start:start + 200]],
                              capture_output=True, text=True, check=True).stdout
        current = None
        for line in text.splitlines():
            if not line.strip():
                continue
            if not line.startswith(" "):
                name, _, rest = line.partition(": ")
                # A STALE entry describes an earlier build of the output, not this one.
                current = name if "#deps" in rest and "(STALE)" not in rest else None
                if current is not None:
                    inputs[current] = []
            elif current is not None:
                inputs[current].append(line.strip())
    for output, rule in rules.items():
        if scope is not None:
            if output not in scope:
                continue
            if ("_COMPILER__" in rule or "_LINKER__" in rule) and not (build_dir / output).is_file():
                problems.append(f"focused output {output} is absent; build it before running this check")
        # Configured EXCLUDE_FROM_ALL targets need not have been built. But
        # each compile/link rule configured to record dependencies must have
        # current evidence in the default build (or when built explicitly).
        # Spark's GNU linker and some CUDA links have no dependency-file
        # support; their explicit link inputs and flags are still audited.
        records_deps = rule in recording_rules and ("_COMPILER__" in rule or "_LINKER__" in rule)
        if records_deps and (output in required or (build_dir / output).exists()) and not inputs.get(output):
            problems.append(f"{output} has no current dependency record; rebuild before running this check")
    return inputs, rules


def link_arguments(build_dir: pathlib.Path, file_inputs: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Each link edge's file inputs, LINK_FLAGS, LINK_PATH and LINK_LIBRARIES."""
    def unescape(text: str) -> str:
        return re.sub(r"\$([ $:])", r"\1", text)

    def paths(text: str) -> list[str]:
        # Ninja path escaping is not shell escaping: quotes are literal and
        # a dollar escapes spaces, colons and dollar signs.
        return [unescape(token) for token in re.findall(r"(?:\$[ $:]|[^\s])+", text)]

    links: dict[str, list[str]] = {}
    current = None
    contents = re.sub(r"\$\n[ \t]*", "", (build_dir / "build.ninja").read_text())
    for line in contents.splitlines():
        if line.startswith("build "):
            outputs, _, rule = line[len("build "):].partition(": ")
            fields = paths(rule)
            current = paths(outputs)[0] if fields and "_LINKER__" in fields[0] else None
            if current:
                links[current] = []
                if file_inputs is not None:
                    file_inputs[current] = []
                for path in fields[1:]:
                    if path in ("||", "|@"):
                        break  # order-only and validation dependencies are not link inputs
                    if path != "|":
                        links[current].append(path)
                        if file_inputs is not None:
                            file_inputs[current].append(path)
        elif current and line.startswith(("  LINK_LIBRARIES = ", "  LINK_PATH = ", "  LINK_FLAGS = ")):
            links[current] += shlex.split(unescape(line.split(" = ", 1)[1]))
        elif not line.startswith(" "):
            current = None
    return links


def expand_linker_options(arguments: list[str]) -> list[str]:
    """A link's arguments with CMake's LINKER: options (-Wl,... or -Xlinker)
    split into the linker arguments they carry."""
    expanded = []
    forwarded = iter(arguments)
    for arg in forwarded:
        if arg.startswith("-Wl,"):
            expanded.extend(arg[4:].split(","))
        elif arg.startswith(("-Xlinker=", "--linker-options=")):
            expanded.extend(arg.split("=", 1)[1].split(","))
        elif arg in ("-Xlinker", "--linker-options"):
            expanded.extend(next(forwarded, "").split(","))
        else:
            expanded.append(arg)
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sdk", type=pathlib.Path, required=True)
    parser.add_argument("--ninja", required=True)
    parser.add_argument("--cross", action="store_true", help="a cross build: nothing from the host")
    parser.add_argument("--libstdcxx-assertions", action="store_true",
                        help="every C++ and CUDA compile defines _GLIBCXX_ASSERTIONS (otherwise none does)")
    parser.add_argument("--target", action="append", default=[],
                        help="audit only this built Ninja output and its recursive inputs; repeatable")
    args = parser.parse_args()
    build = os.path.realpath(args.build_dir)
    source = os.path.realpath(args.source_dir)
    builds_root = os.path.join(source, "build")  # CMakePresets.json's binaryDir parent
    sdk = os.path.realpath(args.sdk)
    receipt = json.loads((args.build_dir / "jitllm-receipt.json").read_text())
    components = {c["id"]: c for c in receipt["components"]}
    trees = {cid: os.path.realpath(c["source"]) for cid, c in components.items()}
    outputs = {f"third_party/{cid}-{c['source_tree'][:16]}": cid for cid, c in components.items()}
    cache = next((line.split("=", 1)[1] for line in (args.build_dir / "CMakeCache.txt").read_text().splitlines()
                  if line.startswith("JITLLM_SOURCES_DIR:PATH=")), "")
    prepared_root = os.path.realpath(cache) if cache else None
    platform = None if args.cross else host_platform_files()

    problems, used = [], set()
    component_outputs = [os.path.join(build, top) for top in outputs]
    rules: dict[str, str] = {}  # every current Ninja output, filled in below

    def classify(path: str, what: str, *, count_used: bool = True) -> None:
        if not os.path.isabs(path):
            path = os.path.join(build, path)
        path = os.path.normpath(path)
        real = os.path.realpath(path)
        for location in {path, real}:
            if within(location, os.path.join(build, "third_party")):
                top = "/".join(os.path.relpath(location, build).split("/")[:2])
                if top not in outputs:
                    problems.append(f"{what} uses {path}, from no component in the receipt")
                    return
                if count_used:
                    used.add(outputs[top])
        for cid, tree in trees.items():
            if within(real, tree):
                if count_used:
                    used.add(cid)
                return
        if prepared_root and within(real, prepared_root):
            problems.append(f"{what} uses {path}, a prepared tree the receipt does not select")
        elif within(real, build):
            # A build-tree file is a selected component's output or a current
            # build rule's; anything else (a module's leftover, a file a
            # script wrote at configure) has no audited origin.
            if (os.path.isdir(real) or any(within(real, d) for d in component_outputs)
                    or os.path.relpath(real, build) in rules or os.path.relpath(path, build) in rules):
                return
            problems.append(f"{what} uses {path}, a file in the build tree that no build rule or selected "
                            "component produced")
        elif within(real, sdk):
            return
        elif within(real, builds_root):
            problems.append(f"{what} uses {path}, from another build directory")
        elif within(real, source):
            return
        elif platform is not None and ({path, real, usr_merged(path)} & platform):
            return
        else:
            problems.append(f"{what} uses {path}, outside the source tree, build tree, SDK"
                            + ("" if args.cross else " and declared host platform packages"))

    scope = focused_outputs(args.ninja, args.build_dir, args.target, problems) if args.target else None
    inputs, found_rules = recorded_inputs(args.ninja, args.build_dir, problems, scope)
    rules.update({os.path.normpath(o): r for o, r in found_rules.items()})
    if not inputs:
        problems.append("Ninja recorded no dependencies; build before running this check")
    for output, paths in inputs.items():
        for path in paths:
            classify(path, output)

    # What the package ships is never built from a test-only component: the
    # package carries only the notices of `use: product` components. Each
    # shipped executable's recursive inputs (sources, objects, archives,
    # component stamps) and the headers its objects were compiled from.
    test_only = {cid for cid, c in components.items() if c.get("use") == "test"}
    material_link_inputs: dict[str, list[str]] = {}
    link_edges = link_arguments(args.build_dir, material_link_inputs)
    if scope is not None:
        link_edges = {output: arguments for output, arguments in link_edges.items() if output in scope}
        for output, paths in material_link_inputs.items():
            if output in scope:
                for path in paths:
                    if not (args.build_dir / path).is_file():
                        problems.append(f"focused link {output} input {path} is absent; rebuild before running this check")

    def component_of(path: str) -> str | None:
        full = os.path.normpath(path if os.path.isabs(path) else os.path.join(build, path))
        for location in {full, os.path.realpath(full)}:
            if within(location, os.path.join(build, "third_party")):
                top = "/".join(os.path.relpath(location, build).split("/")[:2])
                if top in outputs:
                    return outputs[top]
            for cid, tree in trees.items():
                if within(location, tree):
                    return cid
        return None

    for shipped in SHIPPED_EXECUTABLES:
        if shipped not in rules or (scope is not None and shipped not in scope):
            continue
        closure = [p for p in subprocess.run([args.ninja, "-C", str(args.build_dir), "-t", "inputs", "-0", "-E",
                                              shipped], capture_output=True, text=True,
                                             check=True).stdout.split("\0") if p]
        # A path in the link's flags (a LINKER: option), bare or as a linker
        # script's value, is read by the linker though it is no input of the
        # edge.
        for arg in expand_linker_options(link_edges.get(shipped, [])):
            if arg.startswith(("--script=", "--version-script=")):
                closure.append(arg.split("=", 1)[1])
            elif arg.startswith("-T") and len(arg) > 2:
                closure.append(arg[2:])
            elif arg and not arg.startswith("-"):
                closure.append(arg)
        for path in [shipped, *closure]:
            for used_path in [path, *inputs.get(os.path.normpath(path), inputs.get(path, []))]:
                cid = component_of(used_path)
                if cid in test_only:
                    problems.append(f"{shipped}, which the package ships, is built from {cid}, a test-only "
                                    f"component whose notices it does not carry ({used_path})")

    # What links consume, from their dependency records and their edges.
    link_inputs = {path for output, paths in inputs.items() if "_LINKER__" in rules.get(os.path.normpath(output), "")
                   for path in paths}
    for output, arguments in link_edges.items():
        link_inputs.update(a for a in arguments if not a.startswith("-"))
        linked = (pathlib.Path(build) / output).exists()
        # CMake's LINKER: options arrive as -Wl,... or -Xlinker. Inspect
        # their payload as well as direct driver flags (especially for CUDA,
        # whose link rules do not produce a linker dependency file).
        arguments = iter(expand_linker_options(arguments))
        for arg in arguments:
            if arg in ("-rpath", "--rpath", "-rpath-link", "--rpath-link", "-soname", "--soname",
                       "-z", "-u", "--undefined", "-e", "--entry", "--defsym", "-Map", "--dependency-file"):
                # These take non-input values. Dependency files cover the
                # libraries actually resolved through a search directory.
                next(arguments, "")
                continue
            if arg.startswith(("--library=", "--library-path=")):
                option, value = arg.split("=", 1)
                arg = ("-l" if option == "--library" else "-L") + value
            if arg in ("-l", "-L", "--library", "--library-path"):
                flag = "-l" if arg in ("-l", "--library") else "-L"
                value = next(arguments, "")
                arg = flag + value
            if arg.startswith("-l"):
                if arg[2:] not in PLATFORM_LIBRARIES:
                    problems.append(f"{output} links {arg}, which is not a declared platform library")
            elif arg.startswith("-L"):
                classify(arg[2:], output, count_used=False)
            elif arg.startswith("@"):
                problems.append(f"{output} uses an unexpanded response file {arg}; its inventory cannot be audited")
            elif arg.startswith(("--script=", "--version-script=")):
                classify(arg.split("=", 1)[1], output, count_used=linked)
            elif arg.startswith("-T") and len(arg) > 2:
                classify(arg[2:], output, count_used=linked)
            elif not arg.startswith("-"):
                classify(arg, output, count_used=linked)

    commands = json.loads((args.build_dir / "compile_commands.json").read_text())
    objects = []
    for entry in commands:
        if scope is not None:
            output = entry.get("output")
            if not output:
                problems.append("focused audit requires compile_commands output identities")
                continue
            full = os.path.normpath(os.path.join(entry.get("directory", build), output))
            if os.path.relpath(full, build) not in scope:
                continue
        # A compile database includes EXCLUDE_FROM_ALL targets. Only actual
        # dependencies above establish that a receipt component was used.
        classify(entry["file"], entry.get("output", entry["file"]), count_used=False)
        if not entry["file"].endswith(CXX_SOURCES):
            continue  # C has no exceptions to turn off
        if "output" in entry:
            output = os.path.join(entry.get("directory", str(args.build_dir)), entry["output"])
            if os.path.isfile(output):  # EXCLUDE_FROM_ALL objects nothing needed are never built
                objects.append(output)
        command = entry.get("command", "")
        tokens = shlex.split(command) if command else entry.get("arguments", [])
        # nvcc takes -Xcompiler with its value as the next argument too.
        joined = [f"-Xcompiler={b}" if a == "-Xcompiler" else b for a, b in zip([""] + tokens, tokens)]
        problem = exception_problem(joined)
        if problem:
            problems.append(f"{entry['file']} {problem} (D-066)")
        problem = assertions_problem(tokens, args.libstdcxx_assertions)
        if problem:
            problems.append(f"{entry['file']} {problem} (D-083)")

    # Everything else a link consumes (a custom command's output, a prebuilt
    # file in a component) is audited the same way, chosen by content, not
    # name: the linker reads any object or archive, whatever it is called.
    # The SDK's runtimes and the host's glibc are the declared platform
    # (D-017, D-060). No build here makes a shared library (D-064), and the
    # only ones linked are the CUDA driver, through its stub in the SDK
    # (D-072), and the SDK's cuBLAS (D-076).
    for path in link_inputs:
        full = os.path.realpath(path if os.path.isabs(path) else os.path.join(build, path))
        if within(full, sdk):
            # The cross build's glibc lives in the SDK's sysroot.
            platform_shared = (within(full, os.path.join(sdk, "sysroot"))
                               or full.endswith("/lib/stubs/libcuda.so")
                               or CUBLAS.search(full[len(sdk):]) is not None)
            if not platform_shared and os.path.isfile(full) and link_input_kind(full) == "shared":
                problems.append(f"a link uses the SDK's shared library {path}; only glibc, the "
                                "CUDA driver stub and cuBLAS may be linked shared (D-060, D-072, D-076)")
            continue
        if platform is not None and ({path, full, usr_merged(path)} & platform):
            continue
        if not os.path.isfile(full):
            continue  # an input of a target that was never built; nothing linked it
        kind = link_input_kind(full)
        if kind == "object":
            objects.append(full)
        elif kind == "shared":
            problems.append(f"a link uses the shared library {path}; jitLLM links only static code (D-060, D-064)")
        else:
            problems.append(f"a link uses {path}, which is {kind}, not an object file or archive")
    objects = sorted(set(objects))

    # The flags are a model of the compiler; the objects are what it made.
    if objects:
        for obj, what in sorted(exception_machinery(pathlib.Path(sdk), objects).items()):
            problems.append(f"{obj} was compiled with exceptions: it has {', '.join(sorted(set(what)))} (D-066)")

    unused = sorted(set(components) - used)
    if unused and scope is None:
        problems.append(f"the receipt lists {', '.join(unused)}, which nothing compiles or links")
    if problems:
        print(f"{args.build_dir}: the build's inventory does not match its receipt:", file=sys.stderr)
        for problem in sorted(set(problems))[:50]:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"{args.build_dir}: {sum(map(len, inputs.values()))} recorded inputs of {len(inputs)} outputs; "
          f"components used: {', '.join(sorted(used)) or 'none'}")
    if scope is not None:
        print("Focused closure only: " + ", ".join(args.target) + "; whole build/package audit deferred")
        if unused:
            print("Receipt components outside focused inputs: " + ", ".join(unused))
    return 0


if __name__ == "__main__":
    sys.exit(main())
