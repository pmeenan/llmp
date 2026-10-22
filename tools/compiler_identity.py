#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compiler-owned implementation provenance for the pinned Ninja build.

The launcher captures the compiler's own depfile before Ninja consumes it.
The digest checks those receipts against current inputs and objects. No C++
text, include search rules, or Ninja dependency database is interpreted here.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time


class Refusal(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def absolute(value, cwd):
    return Path(os.path.abspath(os.path.join(cwd, value)))


def metadata(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def snapshot(path):
    before = metadata(path)
    value = sha(path.read_bytes())
    require(metadata(path) == before, f"input changed while reading: {path}")
    return {"sha256": value, "stat": before}


def normalized_path(value, roots, cwd=None):
    if not os.path.isabs(value):
        if cwd is None:
            return value
        value = os.path.join(cwd, value)
    value = os.path.normpath(value)
    for root, token in sorted(roots, key=lambda item: -len(item[0])):
        if value == root or value.startswith(root + os.sep):
            return token + value[len(root):]
    return value


def normalized_arguments(args, roots, source):
    """Only known filesystem operands are mapped; -D and unknown flags are literal."""
    operands = {"-o", "-I", "-isystem", "-iquote", "-idirafter", "-include",
                "-imacros", "-isysroot", "-MF", "-MT", "-ccbin"}
    joined = ("--gcc-install-dir=", "--sysroot=", "-resource-dir=", "-ccbin=", "-I")
    result = []
    pending = False
    for index, arg in enumerate(args):
        if pending:
            result.append(normalized_path(arg, roots))
            pending = False
        elif arg in operands:
            result.append(arg)
            pending = True
        elif arg.startswith(("-ffile-prefix-map=", "-Xcompiler=-ffile-prefix-map=")):
            head, body = arg.split("-ffile-prefix-map=", 1)
            require("=" in body, "malformed file prefix map")
            old, new = body.split("=", 1)
            result.append(head + "-ffile-prefix-map=" + normalized_path(old, roots) + "=" + new)
        elif index == 0 or arg == source:
            result.append(normalized_path(arg, roots))
        else:
            for option in joined:
                if arg.startswith(option) and len(arg) > len(option):
                    result.append(option + normalized_path(arg[len(option):], roots))
                    break
            else:
                result.append(arg)
    require(not pending, "missing compiler option operand")
    return result


def depfile_paths(text):
    """Read the single Make dependency rule emitted by Clang/NVCC -MD.

    Make escaping is a build-tool format, not C++ syntax. Additional rules,
    variables or malformed escapes refuse rather than dropping dependencies.
    """
    text = text.replace("\\\r\n", "").replace("\\\n", "")
    tokens, word = [], []
    colon = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 1
            require(index < len(text), "truncated depfile escape")
            require(text[index] in " \\#$:\t", "unsupported compiler depfile escape")
            word.append(text[index])
        elif char == "$":
            index += 1
            require(index < len(text) and text[index] == "$", "depfile variable is unsupported")
            word.append("$")
        elif char == ":":
            require(not colon, "multiple dependency rules or unescaped colon")
            if word:
                tokens.append("".join(word))
                word = []
            require(len(tokens) == 1, "expected one dependency target")
            tokens = []
            colon = True
        elif char.isspace():
            if word:
                tokens.append("".join(word))
                word = []
        elif char == "#":
            raise Refusal("unescaped depfile comment")
        else:
            word.append(char)
        index += 1
    if word:
        tokens.append("".join(word))
    require(colon and tokens, "missing compiler dependencies")
    return tokens


def dependency_rule(obj, inputs):
    """Expose launcher inputs to Ninja alongside compiler-discovered headers."""
    def quote(path):
        value = str(path)
        # Ninja's GCC-depfile grammar preserves ordinary backslashes. Before
        # a space it consumes pairs plus the escaping backslash; before #/:
        # it consumes only the final escape. Tabs and these ambiguous endings
        # have no supported spelling in our pinned single-rule contract.
        require(value and not any(ord(c) < 32 for c in value) and
                not value.endswith('\\') and '\\$' not in value,
                "unsupported Ninja dependency path")
        result = []
        backslashes = 0
        for char in value:
            if char == ' ':
                result.append('\\' * (backslashes + 1) + char)
            elif char in '#:':
                result.append('\\' + char)
            elif char == '$':
                result.append('$$')
            else:
                result.append(char)
            backslashes = backslashes + 1 if char == '\\' else 0
        return ''.join(result)
    require(inputs, "empty dependency inventory")
    return quote(obj) + ': ' + ' '.join(quote(p) for p in sorted(inputs)) + '\n'


def check_dependency_aliases(paths):
    """Refuse the pinned compiler's lossy POSIX-backslash filename spelling.

    Clang's MD output substitutes '/' for literal filename backslashes. An
    existing slash-spelled collision therefore cannot authenticate the input.
    Inspect only ancestor directory entries of the raw lexical dependencies,
    retaining '..' and repeated separators until this check. Never walk a
    provider tree or add unrelated directory metadata to portable identity.
    """
    directories = {}
    for path in paths:
        require(os.path.isabs(path) and '\\' not in path,
                f"unsupported compiler dependency spelling: {path}")
        target = os.path.normpath(path)
        directory = '/'
        components = path.split('/')[1:]
        for offset, component in enumerate(components):
            if directory not in directories:
                try:
                    def alias_names():
                        with os.scandir(directory) as entries:
                            return {entry.name for entry in entries if '\\' in entry.name}
                    aliases = alias_names()
                    require(alias_names() == aliases,
                            f"dependency aliases changed during inspection: {directory}")
                except (FileNotFoundError, NotADirectoryError):
                    aliases = []
                directories[directory] = aliases
            # A literal directory name such as 'a\\b' remains significant in
            # a/b/../header.h: cancelling '..' first would inspect the wrong
            # physical parent. Ignore only redundant separator/dot spellings.
            remaining = [part for part in components[offset:] if part not in ('', '.')]
            for name in directories[directory]:
                expanded = [part for part in name.replace('\\', '/').split('/') if part not in ('', '.')]
                require(expanded and remaining[:len(expanded)] != expanded,
                        f"ambiguous compiler dependency path: {path}; literal backslash entry: {directory}/{name}")
                candidate = os.path.normpath(os.path.join(directory, name.replace('\\', '/')))
                require(target != candidate and not target.startswith(candidate.rstrip('/') + '/'),
                        f"ambiguous compiler dependency path: {path}; literal backslash entry: {directory}/{name}")
            directory = os.path.join(directory, component)


def load_config(path):
    config = json.loads(path.read_text())
    require(config["schema"] == 1, "unsupported compiler identity configuration")
    config["roots"] = [(os.path.normpath(config[name]), token) for name, token in
                       (("build", "$BUILD"), ("source", "$SOURCE"),
                        ("sdk", "$SDK"), ("prepared", "$PREPARED"))]
    return config


def build_outputs(config):
    """Ninja owns output discovery, including custom outputs outside build/.

    These paths classify generated inputs, not implementation identity. The
    graph bytes are observed before/after capture as receipt-only provenance.
    """
    if "ninja" not in config:  # small launcher unit fixture
        return set(), {}
    build = Path(config["build"])
    graph = {p: snapshot(p) for p in (build / "build.ninja", build / "CMakeFiles/rules.ninja")}
    result = subprocess.run([config["ninja"], "-C", str(build), "-t", "targets", "all"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    outputs = set()
    for line in result.stdout.splitlines():
        require(": " in line, "unrecognized Ninja output inventory")
        name, rule = line.rsplit(": ", 1)
        require(name and rule, "malformed Ninja output inventory")
        # CMake emits phony missing-input guards for its authored listfiles.
        # Phony edges produce no bytes and therefore are not generators.
        if rule != "phony":
            outputs.add(str(absolute(name, build)))
    require(all(snapshot(p) == before for p, before in graph.items()), "build graph changed during output inventory")
    return outputs, graph


def effective_environment(config):
    forbidden = {"CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
                 "CUDA_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX",
                 "CUDAHOSTCXX", "CUDA_PATH", "CUDA_HOME", "CCC_OVERRIDE_OPTIONS",
                 "SOURCE_DATE_EPOCH", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET"}
    for name, value in os.environ.items():
        if value and (name in forbidden or name.startswith(("LD_", "DYLD_", "NVCC_", "CLANG_"))):
            raise Refusal(f"unsupported ambient compiler input {name}; express it in the configured build")
    # The compiler receives exactly this environment, not arbitrary parent
    # variables. --no-default-config also prevents compiler-side user config.
    build, sdk = Path(config["build"]), Path(config["sdk"])
    home, temporary = build / "identity-home", build / "identity-tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    require(not any(home.iterdir()), "compiler identity HOME must remain empty")
    return {"PATH": f"{sdk}/bin:{sdk}/cuda/bin:/usr/bin:/bin", "HOME": str(home),
            "TMPDIR": str(temporary), "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}


def environment_identity(env, roots):
    result = dict(env)
    result["PATH"] = [normalized_path(p, roots) for p in env["PATH"].split(os.pathsep)]
    for name in ("HOME", "TMPDIR"):
        result[name] = normalized_path(env[name], roots)
    return result


def expand_responses(args, cwd):
    expanded, inputs = [], {}
    for arg in args:
        if arg.startswith("@"):
            path = absolute(arg[1:], cwd)
            require(path not in inputs, f"duplicate response file: {path}")
            record = snapshot(path)
            values = shlex.split(path.read_text())
            require(not any(value.startswith("@") for value in values), "nested response files are unsupported")
            inputs[path] = record
            expanded.extend(values)
        else:
            require(not arg.startswith(("--options-file", "-optf")), "NVCC options files need explicit provenance support")
            expanded.append(arg)
    return expanded, inputs


def operand(args, name):
    require(args.count(name) == 1, f"expected exactly one {name} compiler operand")
    index = args.index(name)
    require(index + 1 < len(args), f"missing {name} compiler operand")
    return args[index + 1]


def without_dependency_options(args):
    result = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-MF", "-MT"):
            index += 2
        elif arg == "-MD":
            index += 1
        else:
            result.append(arg)
            index += 1
    return result


def receipt_path(obj):
    return Path(str(obj) + ".llmp-compile.json")


def write_if_changed(path, data):
    if path.exists() and path.read_bytes() == data:
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)


def source_record(path, config):
    """Generated inputs require a declared recipe, never a filename exemption."""
    roots = config["roots"]
    name = normalized_path(str(path), roots)
    if path == Path(config["sdk"]) / "sdk.json":
        receipt = json.loads(path.read_text())
        receipt.pop("created", None)  # installation time is receipt-only
        return {"sdk_receipt": receipt}
    if name.startswith("$SOURCE/"):
        require(str(path) in config.get("owned_inputs", [str(path)]),
                f"input is outside the frozen authored-source inventory: {path}; register its generation recipe")
    generated = config["generated"]
    is_generated = (name.startswith("$BUILD/") or str(path) in config.get("declared_generated", []) or
                    str(path) in config.get("_build_outputs", set()))
    if is_generated:
        require(str(path) in generated,
                f"generated input has no declared recipe: {path}; register its generator and complete inputs")
        recipe = generated[str(path)]
        require(recipe["inputs"], f"empty generated-input provenance: {path}")
        require(snapshot(path)["sha256"] == recipe["sha256"], f"generated input differs from its configured recipe: {path}")
        return {"recipe": {normalized_path(p, roots): snapshot(Path(p))["sha256"]
                           for p in recipe["inputs"]}, "parameters": config["identity"]["host_wrapper"]}
    require(name.startswith(("$SOURCE/", "$SDK/", "$PREPARED/", "/usr/include/")) or
            path == Path(__file__).resolve() or str(path) in config.get("declarations", []),
            f"unregistered external compiler input: {path}; add an explicit source/platform or generated-input contract")
    return {"sha256": snapshot(path)["sha256"]}


def constant_bytes(identity):
    require(re.fullmatch(r"[0-9a-f]{64}", identity), "malformed generated implementation identity")
    return ("// Generated by tools/compiler_identity.py; do not edit.\n"
            "namespace llmp::kernels::ggml {\n"
            f'const char* ModuleSourcesDigest() {{ return "{identity}"; }}\n'
            "}  // namespace llmp::kernels::ggml\n").encode()


def constant_context(args, config, cwd, env):
    """Header-free literal TU: a deliberately narrow, acyclic compile contract.

    Allow ordinary target/optimization/diagnostic flags and inert uppercase
    definitions, never an implicit input, plugin, forwarded option or response
    file. The exact allowed command still participates in primary identity.
    """
    require(args[0] == config["constant_compiler"], "constant TU requires the configured C++ compiler")
    require(absolute(operand(args, "-c"), cwd) == Path(config["constant_source"]), "unexpected constant source")
    flags = without_dependency_options(args)
    index = 1
    exact = {"-g", "-g0", "-gline-tables-only", "-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Og",
             "-fno-exceptions", "-fno-rtti", "-fPIC", "-fPIE", "-fno-omit-frame-pointer",
             "-fvisibility=hidden", "-fvisibility-inlines-hidden", "--no-default-config",
             "-std=c++23", "-std=gnu++23", "-fno-sanitize-recover=all"}
    while index < len(flags):
        arg = flags[index]
        if arg in ("-o", "-c", "-I", "-isystem", "-iquote", "-idirafter"):
            require(index + 1 < len(flags), "missing constant-TU path operand")
            index += 2
            continue
        allowed = arg in exact or bool(re.fullmatch(r"-W(?:[A-Za-z0-9=-]+)", arg))
        allowed |= bool(re.fullmatch(r"(?:--target=|-march=)[A-Za-z0-9_+.-]+", arg))
        allowed |= bool(re.fullmatch(r"-fsanitize=(?:address|undefined|thread)(?:,(?:address|undefined|thread))*", arg))
        allowed |= arg.startswith(("-I", "--gcc-install-dir=", "--sysroot=", "-ffile-prefix-map="))
        if arg.startswith("-D"):
            allowed = bool(re.fullmatch(r'-D[A-Z_][A-Z0-9_]*(?:=(?:[0-9]+|"[^"\\\r\n]*"))?', arg))
        require(allowed, f"constant TU compile input is unsupported: {arg}")
        index += 1
    return {"arguments": normalized_arguments(flags, config["roots"], config["constant_source"]),
            "directory": normalized_path(str(cwd), config["roots"]),
            "environment": environment_identity(env, config["roots"])}


def supported_compile_options(args):
    """Pinned object-compilation contract, not a permissive compiler frontend.

    Depfiles do not report plugins, profiles, overlays or compiler configuration.
    Unknown options (including forwarding aliases) therefore need an explicit
    provenance extension before use. Definitions remain literal identity inputs.
    """
    exact = {"--no-default-config", "-g", "-g0", "-gline-tables-only", "-lineinfo",
             "-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Og", "-fno-exceptions",
             "-fno-rtti", "-fPIC", "-fPIE", "-fno-omit-frame-pointer", "-fvisibility=hidden",
             "-fvisibility-inlines-hidden", "-fno-sanitize-recover=all"}
    def host_flag(arg):
        return (arg in exact or
                bool(re.fullmatch(r"-W[A-Za-z0-9=-]+", arg)) and arg not in ("-Wp", "-Wl", "-Wa") or
                bool(re.fullmatch(r"(?:--target=|-march=)[A-Za-z0-9_+.-]+", arg)) or
                bool(re.fullmatch(r"-fsanitize=(?:address|undefined|thread)(?:,(?:address|undefined|thread))*", arg)) or
                arg.startswith("-ffile-prefix-map="))
    paths = {"-o", "-c", "-I", "-isystem", "-iquote", "-idirafter", "-include", "-imacros",
             "-isysroot", "-MF", "-MT"}
    index = 1
    while index < len(args):
        arg = args[index]
        if arg in paths or arg in ("-D", "-U", "--define-macro", "--undefine-macro"):
            require(index + 1 < len(args), f"missing compiler operand: {arg}")
            index += 2
            continue
        if arg in ("-x", "--target-directory"):
            require(index + 1 < len(args) and args[index+1] in
                    (("cu",) if arg == "-x" else ("sbsa-linux",)), "unsupported CUDA language/target directory")
            index += 2
            continue
        allowed = host_flag(arg) or arg == "-MD"
        allowed |= arg in {"-forward-unknown-to-host-compiler", "--expt-relaxed-constexpr",
                           "-extended-lambda", "--extended-lambda", "--use_fast_math", "-use_fast_math"}
        allowed |= bool(re.fullmatch(r"-std=(?:c|gnu)\+\+(?:20|23)", arg))
        allowed |= bool(re.fullmatch(r"--diag-suppress=[0-9]+(?:,[0-9]+)*", arg))
        allowed |= bool(re.fullmatch(r"--generate-code=arch=compute_[0-9]+[a-z]?,code=\[(?:sm_|compute_)[0-9]+[a-z]?(?:,(?:sm_|compute_)[0-9]+[a-z]?)*\]", arg))
        allowed |= arg.startswith(("-D", "-U", "--define-macro=", "--undefine-macro=", "-I",
                                   "--gcc-install-dir=", "--sysroot=", "-resource-dir=", "-ccbin="))
        if arg.startswith("-Xcompiler="):
            # NVCC splits forwarding at commas, including inside mapping paths.
            # Every resulting host token must be independently supported.
            value = arg.split("=", 1)[1]
            allowed = all(host_flag(part) for part in value.split(","))
        if arg == "-Xptxas=-v":
            allowed = True
        require(allowed, f"compiler input option needs explicit provenance support: {arg}")
        index += 1


def compile_with_receipt(config_path, argv):
    config = load_config(config_path)
    env = effective_environment(config)
    config["_build_outputs"], build_graph = build_outputs(config)
    cwd = Path.cwd()
    require(cwd == Path(config["build"]), "identity launcher requires the configured build working directory")
    require(argv and argv[0] in config["compilers"], "unsupported compiler or launcher chain")
    args, responses = expand_responses(argv, cwd)
    supported_compile_options(args)
    prefix = "-Xcompiler=" if Path(argv[0]).name == "nvcc" else ""
    require(prefix + "--no-default-config" in args and prefix + "-Werror=date-time" in args,
            "identity compilation requires disabled implicit compiler config and date/time macros")
    definitions = []
    pending = None
    for arg in args:
        if pending:
            definitions.append(pending + arg)
            pending = None
        elif arg in ("-D", "--define-macro", "-U", "--undefine-macro"):
            pending = "-U" if arg in ("-U", "--undefine-macro") else "-D"
        elif arg.startswith("--define-macro="):
            definitions.append("-D" + arg.split("=", 1)[1])
        elif arg.startswith("--undefine-macro="):
            definitions.append("-U" + arg.split("=", 1)[1])
        elif arg.startswith(("-D", "-U")):
            definitions.append(arg)
        elif arg.startswith("-Xcompiler="):
            # The supported build emits macro definitions as top-level -D.
            # Refuse host-only macro options, including split comma operands.
            require(not any(word.startswith(("-D", "-U", "--define-macro", "--undefine-macro"))
                            for word in arg.split("=", 1)[1].split(",")),
                    "forwarded macro definitions need an explicit host/device contract")
    for name, value in (("__DATE__", "Jan  1 1970"), ("__TIME__", "00:00:00"),
                        ("__TIMESTAMP__", "Thu Jan  1 00:00:00 1970")):
        expected = '-D' + name + '="' + value + '"'
        protected = [arg[:2]+part for arg in definitions for part in arg[2:].split(",")
                     if part.split("=", 1)[0] == name]
        require(protected == [expected],
                f"compiler identity requires exactly the fixed {name} definition")
    for name, mapped in (("source", "llmp/source"), ("build", "llmp/build"),
                         ("sdk", "llmp/sdk"), ("prepared", "llmp/prepared")):
        require(prefix + "-ffile-prefix-map=" + config[name] + "=" + mapped in args,
                f"missing validated compiler file-prefix mapping for {name}")
    source = absolute(operand(args, "-c"), cwd)
    obj = absolute(operand(args, "-o"), cwd)
    depfile = absolute(operand(args, "-MF"), cwd)
    require(absolute(operand(args, "-MT"), cwd) == obj, "compiler dependency target differs from its object")
    require(not any(arg in ("-M", "-MM", "-E", "-S", "--dryrun", "-dryrun", "-###", "--generate-dependencies", "--generate-nonsystem-dependencies", "--preprocess") for arg in args),
            "dependency-only/preprocess/dry-run modes cannot produce implementation objects")
    require("-MD" in args and "-MMD" not in args, "identity requires full -MD compiler dependencies")
    require(not any(arg in ("-include-pch", "-emit-pch", "-fmodules", "-fmodules-ts") or
                    arg.startswith(("-fmodule-file", "-fmodules-cache-path")) for arg in args),
            "PCH/modules require a declared compiler-artifact provenance recipe")
    # The pinned build currently has no response files. A source-owned response
    # is supported and byte-covered; generated responses need a declared recipe.
    for path in responses:
        source_record(path, config)
    inputs = {source, config_path, Path(__file__).resolve(), Path(config["sdk"]) / "sdk.json"}
    inputs.update(responses)
    for arg in args:
        if arg.startswith("-ccbin="):
            wrapper = absolute(arg[len("-ccbin="):], cwd)
            source_record(wrapper, config)
            inputs.add(wrapper)
            inputs.update(Path(p) for p in config["generated"][str(wrapper)]["inputs"])
    excluded = str(source) in config["excluded_sources"]
    constant = str(source) == config.get("constant_source")
    constant_control = {}
    before = {p: snapshot(p) for p in inputs}
    before.update(build_graph)
    if constant:
        context = constant_context(args, config, cwd, env)
        identity_receipt_path = Path(config["identity_receipt"])
        constant_control = {identity_receipt_path: snapshot(identity_receipt_path)}
        identity_receipt = json.loads(identity_receipt_path.read_text())
        require(context == identity_receipt["primary"]["constant"], "constant TU command differs from primary identity")
        require(source.read_bytes() == constant_bytes(identity_receipt["identity"]), "constant source failed exact literal integrity")
    elif excluded:
        require(str(source) in config.get("derived", {}), f"excluded source has no integrity recipe: {source}")
        recipe = config["derived"][str(source)]
        inputs.update(Path(p) for p in recipe["inputs"])
        before.update({Path(p): snapshot(Path(p)) for p in recipe["inputs"]})
        # Regenerate into private scratch and compare the entire output, not
        # merely a constant/function name. Exclusions cannot hide inline code.
        with tempfile.TemporaryDirectory(dir=env["TMPDIR"]) as directory:
            expected = Path(directory) / "expected.cc"
            command = [arg.replace("@OUTPUT@", str(expected)) for arg in recipe["command"]]
            subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            require(expected.read_bytes() == source.read_bytes(), f"excluded generated source fails integrity: {source}")
    else:
        source_record(source, config)
    old = receipt_path(obj)
    if old.exists() and not excluded:
        previous = json.loads(old.read_text())
        for name in previous["inputs"]:
            path = Path(name)
            if path.is_file():
                before[path] = snapshot(path)
    # Successful dependency-only/dry-run modes must never authenticate an old object.
    obj.unlink(missing_ok=True)
    depfile.unlink(missing_ok=True)
    old.unlink(missing_ok=True)
    started = time.time_ns()
    result = subprocess.run(argv, env=env, check=False)
    if result.returncode:
        old.unlink(missing_ok=True)
        return result.returncode
    try:
        require(all(snapshot(p) == record for p, record in build_graph.items()),
                "build graph changed during compilation")
        # Keep lexical '..' and double separators until the actual filename
        # ambiguity check; absolute() intentionally normalizes them afterward.
        dependency_paths = [os.path.join(str(cwd), p) for p in depfile_paths(depfile.read_text())]
        check_dependency_aliases(dependency_paths)
        dependencies = {absolute(p, cwd) for p in dependency_paths}
        require(source in dependencies, "compiler depfile omitted its source")
        if constant:
            require(dependencies == {source}, "header-free constant TU acquired implicit compiler inputs")
        inputs.update(dependencies)
        captured = {}
        for path in sorted(inputs):
            captured[str(path)] = snapshot(path)
            if path in before:
                require(captured[str(path)] == before[path], f"input changed during compilation: {path}")
            else:
                st = captured[str(path)]["stat"]
                require(max(st[3], st[4]) <= started, f"new dependency changed during compilation: {path}")
        values = {source, config_path}
        if excluded:
            values.update(Path(p) for p in recipe.get("value_inputs", []))
        else:
            values = {config_path}
        primary = {"constant": context, "value": identity_receipt["identity"]} if constant else {
                   "schema": 1, "source": normalized_path(str(source), config["roots"]),
                   "directory": normalized_path(str(cwd), config["roots"]),
                   "arguments": normalized_arguments(args, config["roots"], str(source)),
                   "environment": environment_identity(env, config["roots"]),
                   "inputs": {normalized_path(str(p), config["roots"]): source_record(p, config)
                              for p in sorted(inputs) if p not in values},
                   "configuration": config["identity"]}
        if excluded:
            primary["derivation"] = recipe["identity"]
        object_record = snapshot(obj)
        for name, record in captured.items():
            require(snapshot(Path(name)) == record, f"input changed during receipt generation: {name}")
        require(snapshot(obj) == object_record, f"object changed during receipt generation: {obj}")
        receipt = {"schema": 1, "primary": primary, "identity": sha(canonical(primary)),
                   "object": str(obj), "object_record": object_record, "inputs": captured,
                   "compiler_dependency_paths": dependency_paths,
                   "derived": excluded, "constant": constant}
        check_dependency_aliases(dependency_paths)
        require(all(snapshot(p) == record for p, record in constant_control.items()),
                "identity value changed while compiling its constant")
        # The compiler does not list its -ccbin wrapper, response files, or our
        # generator/configuration inputs. Ninja must rebuild on those changes
        # too; otherwise a normal incremental build merely encounters a stale
        # receipt at the downstream identity step. Keep the complete validated
        # compiler dependencies and add every authenticated launcher input.
        write_if_changed(depfile, dependency_rule(operand(args, "-MT"), inputs).encode())
        write_if_changed(old, canonical(receipt) + b"\n")
    except Exception:
        # A successful compiler without complete provenance is not a reusable
        # build output; force the next build to compile again after repair.
        obj.unlink(missing_ok=True)
        old.unlink(missing_ok=True)
        raise
    return 0


def validated_receipt(obj):
    path = receipt_path(obj)
    require(path.is_file(), f"missing compiler receipt: {obj}; rebuild this object")
    receipt = json.loads(path.read_text())
    require(receipt["schema"] == 1 and receipt["object"] == str(obj), f"ambiguous compiler receipt: {obj}")
    require(sha(canonical(receipt["primary"])) == receipt["identity"], f"corrupt compiler receipt: {obj}")
    require(snapshot(obj) == receipt["object_record"], f"stale object receipt: {obj}")
    require("compiler_dependency_paths" in receipt, f"missing compiler filename proof: {obj}; rebuild this object")
    check_dependency_aliases(receipt["compiler_dependency_paths"])
    for name, record in receipt["inputs"].items():
        require(snapshot(Path(name)) == record, f"stale compiler input: {name}; rebuild {obj}")
    return receipt


def digest(config_path, objects_path, output, receipt_output, check=False):
    config = load_config(config_path)
    commands_path = Path(config["build"]) / "compile_commands.json"
    control_inputs = {p: snapshot(p) for p in (config_path, objects_path, commands_path)}
    entries = json.loads((Path(config["build"]) / "compile_commands.json").read_text())
    commands = {}
    for entry in entries:
        path = absolute(entry["output"], entry["directory"])
        require(path not in commands, f"ambiguous compile command: {path}")
        commands[path] = entry
    archive_prefix = config["constant_object_prefix"]
    for obj, entry in commands.items():
        if str(obj).startswith(archive_prefix):
            require(entry["file"] == str(output),
                    "downstream identity archive gained an interface source; preserve the original consumer context explicitly")
    constant_entries = [(obj, entry) for obj, entry in commands.items() if entry["file"] == str(output)]
    require(len(constant_entries) == 1, "missing or ambiguous constant-TU compilation")
    constant_object, constant_entry = constant_entries[0]
    require("@" not in constant_entry["command"], "constant TU response files need an explicit contract")
    constant = constant_context(shlex.split(constant_entry["command"]), config,
                                Path(constant_entry["directory"]), effective_environment(config))
    objects = [Path(p) for p in objects_path.read_text().splitlines() if p]
    require(objects and len(objects) == len(set(objects)), "empty or duplicate object inventory")
    receipts = {}
    for obj in sorted(objects):
        require(obj in commands, f"missing effective compile command: {obj}")
        entry = commands[obj]
        receipt = validated_receipt(obj)
        exported, _ = expand_responses(shlex.split(entry["command"]), entry["directory"])
        exported = normalized_arguments(exported, config["roots"], entry["file"])
        require(exported == without_dependency_options(receipt["primary"]["arguments"]),
                f"effective compile command changed: {obj}; rebuild this object")
        require(receipt["primary"]["directory"] == normalized_path(entry["directory"], config["roots"]),
                f"compile working directory changed: {obj}")
        # Actual executed argv, environment, depfile and source bytes are in the
        # receipt. compile_commands identifies the object/source bijection.
        require(receipt["primary"]["source"] == normalized_path(entry["file"], config["roots"]),
                f"compile source changed: {obj}")
        if entry["file"] in config["excluded_sources"]:
            require(receipt.get("derived"), f"missing generated integrity proof: {obj}")
        else:
            require(not receipt.get("derived"), f"unexpected derived object exclusion: {obj}")
        receipts[str(obj)] = receipt
    inventory = {normalized_path(p, config["roots"]): snapshot(Path(p))["sha256"]
                 for p in config["declarations"]}
    primary = {"schema": 1, "configuration": config["identity"], "declarations": inventory,
               "constant": constant,
               "objects": {normalized_path(p, config["roots"]): r["identity"] for p, r in receipts.items()}}
    identity = sha(canonical(primary))
    generated = constant_bytes(identity)
    # Detect input/object changes over the entire inventory generation, not
    # merely whether Ninja once marked a dependency record VALID.
    for name, old in receipts.items():
        require(validated_receipt(Path(name)) == old, f"receipt changed during inventory: {name}")
    require(inventory == {normalized_path(p, config["roots"]): snapshot(Path(p))["sha256"]
                          for p in config["declarations"]}, "build declarations changed during inventory")
    require(all(snapshot(p) == record for p, record in control_inputs.items()),
            "configuration/object/command inventory changed during identity generation")
    if check:
        require(output.read_bytes() == generated, "generated implementation identity is stale")
        compiled_constant = validated_receipt(constant_object)
        require(compiled_constant.get("constant") and
                compiled_constant["primary"] == {"constant": constant, "value": identity},
                "compiled constant does not match its bound command/value")
    else:
        write_if_changed(output, generated)
        write_if_changed(receipt_output, canonical({"identity": identity, "primary": primary,
            "objects": {p: r["object_record"] for p, r in receipts.items()}}) + b"\n")
    print(identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "digest", "check"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stamp")
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    options, compiler = parser.parse_known_args()
    try:
        if options.mode == "compile":
            require(compiler and compiler[0] == "--", "missing compiler argument separator")
            return compile_with_receipt(options.config, compiler[1:])
        require(not compiler and options.objects and options.output and options.receipt,
                "digest/check needs objects, output and receipt")
        digest(options.config, options.objects, options.output, options.receipt, options.mode == "check")
    except (Refusal, OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        print(f"compiler identity refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
