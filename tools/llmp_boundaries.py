# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The portability boundary check (docs/portability.md; D-026, D-082): vendor and OS code stays where it belongs.

Every other GPU platform and operating system is to be a compile target of
the same tree, so what is specific to one lives behind the modules that
implement it. Over llmpalooza's own C, C++ and CUDA sources under src/, this
check fails when:

- a Linux-specific header or call appears outside src/platform/ and the
  Linux provider implementations (LINUX_HOMES): <linux/...> and the
  Linux-only <sys/...> headers, io_uring, epoll, eventfd, signalfd, timerfd,
  inotify, memfd, O_DIRECT, O_TMPFILE, prctl, the sched_ affinity calls,
  getrandom, accept4, SOCK_CLOEXEC and the other Linux-only spellings in
  LINUX_WORDS, raw system calls, a path under /proc or /sys, or an include
  of the io_uring ring or its provider;
- CUDA appears outside src/providers/cuda/, the kernel sources, their
  shared device code (src/kernels/device/) and the composition root
  (src/kernels/compose/) (CUDA_HOMES; never in src/kernels/common/):
  a CUDA, cuBLAS, cuFile, NVML, NCCL, CUB, Thrust or CUTLASS header, a CUDA
  runtime or driver API name or type (cuda..., cu...(, CU..., CU_..., CUDA_...,
  cublas...), a kernel qualifier or launch, a .cu or .cuh file, or an include
  of providers/cuda/;
- kernel library names (ggml, exl3, cublas, cutlass, in any case, alone or
  embedded in an identifier, and in string literals) appear outside
  KERNEL_HOMES: the kernel sources themselves, the composition root that
  registers them (src/kernels/compose/) and the format, import, artifact
  and tokenizer layers, whose vocabulary names containers and layouts. The
  vendor-free kernel interface (src/kernels/common/), the planner,
  components, specs and model layers may use the format layer's names for
  layouts, encodings and containers (FORMAT_WORDS, FORMAT_STRINGS, an
  explicit list); the engine, runtime and scheduler, and the neutral ir/
  and execution/ layers, may name only the containers in their data
  (FORMAT_STRINGS) and nothing in code. In ir/ and execution/ vendor names
  and device spellings are refused as well;
- a kernel source (src/kernels/<source>/) includes another's headers, or
  src/kernels/common/ includes any kernel source's: sources meet only
  through the common interface and the composition root.

Includes are compared as paths under src/ (a relative "../" or "./" path is
resolved against the including file, and a leading "src/" dropped); an
include through a macro is resolved through the file's own #define, and one
the check cannot resolve is a problem outside the kernel sources.

POSIX calls (open, poll, sockets, mmap) are not Linux-specific and are not
checked: macOS has them, and a Windows port gives the platform module its
own implementations. Comments are not code: the check reads each file with
its comments removed, and matches names outside string literals and paths
inside them. A file allowed an exception is named in EXCEPTIONS with the
reason, and an exception no file needs any more is itself a problem, so the
list stays exact. Tests and benchmarks drive the CUDA build directly and
are not checked.
"""

from __future__ import annotations

import pathlib
import posixpath
import re

SOURCE_SUFFIXES = (".cc", ".cpp", ".cxx", ".c", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".inc", ".cu", ".cuh")
SCOPE = "src/"

LINUX = "linux"
CUDA = "cuda"
KERNEL = "kernel"
SOURCE = "source"
LINUX_HOMES = ("src/platform/", "src/providers/uring_storage.")
# CUDA (below): its provider, the kernel sources, the device code they share
# (src/kernels/device/) and the composition root; never the common interface.
# The kernel sources: each a directory of src/kernels/ whose headers no other
# source includes.
KERNEL_SOURCES = ("ggml", "exl3", "decode", "paging", "reference")
KERNEL_SOURCE_HOMES = tuple(f"src/kernels/{source}/" for source in KERNEL_SOURCES)
CUDA_HOMES = ("src/providers/cuda/",) + KERNEL_SOURCE_HOMES + ("src/kernels/device/",
                                                              "src/kernels/compose/")
KERNEL_HOMES = KERNEL_SOURCE_HOMES + (
    "src/kernels/compose/",
    "src/artifact/",
    "src/format/",
    "src/import/",
    "src/tokenizer/",
)
# Code that names no kernel library at all, not even through the format
# layer's layout names: it reaches kernels only through the common interface.
STRICT_HOMES = ("src/engine/", "src/runtime/", "src/scheduler/", "src/ir/", "src/execution/")

# path -> {rule: why}. Kept empty when nothing needs one.
EXCEPTIONS: dict[str, dict[str, str]] = {}

LINUX_HEADERS = re.compile(
    r"^(linux/|asm/|asm-generic/|liburing|mntent\.h$|malloc\.h$|sys/(epoll|eventfd|signalfd|timerfd|inotify|"
    r"fanotify|prctl|sysinfo|statfs|vfs|sendfile|random|xattr|capability|personality|syscall|klog|fsuid|"
    r"auxv|pidfd|memfd|sysmacros)\.h$)")
LINUX_WORDS = re.compile(
    r"\b(io_uring(_\w+)?|epoll_\w+|EPOLL\w+|eventfd(_\w+)?|EFD_\w+|signalfd(_\w+)?|SFD_\w+|timerfd_\w+|TFD_\w+|"
    r"inotify_\w+|IN_CLOEXEC|memfd_create|MFD_\w+|O_DIRECT|O_TMPFILE|O_PATH|O_NOATIME|prctl|PR_SET_\w+|"
    r"sched_setaffinity|sched_getaffinity|getrandom|GRND_\w+|accept4|SOCK_CLOEXEC|SOCK_NONBLOCK|MSG_NOSIGNAL|"
    r"pipe2|ppoll|vmsplice|fallocate|syncfs|statfs|fstatfs|pidfd_\w+|clone3|unshare|setns|"
    r"syscall|SYS_\w+|FS_IOC_\w+|MAP_POPULATE|MAP_HUGETLB|MADV_HUGEPAGE|mremap|gettid|getifaddrs|"
    r"statx|STATX_\w+|AT_EMPTY_PATH|sync_file_range|copy_file_range|openat2|renameat2|close_range|dup3|"
    r"posix_fadvise|SO_PEERCRED|TCP_USER_TIMEOUT|CLOCK_BOOTTIME|pthread_getattr_np|pthread_setaffinity_np|"
    r"sched_getcpu|MADV_DONTDUMP|MAP_FIXED_NOREPLACE|mlock2|malloc_trim|getauxval|"
    r"landlock_\w+|seccomp\w*|sd_notify)\b")
LINUX_PATHS = re.compile(r"(^|[^\w.])/(proc|sys)(/|$)")
LINUX_INCLUDES = ("platform/io_uring.h", "providers/uring_storage.h")

CUDA_HEADERS = re.compile(
    r"^(cuda|cublas|cufile|nvml|nccl|curand|cusparse|cusolver|cufft|nvrtc|nvtx|nvToolsExt|cooperative_groups|"
    r"mma\.h$|cub/|cuda/|thrust/|cutlass/|cute/|nv/)")
CUDA_WORDS = re.compile(
    r"\b(cuda[A-Z]\w*|CU[a-z]\w*|CU_[A-Z]\w*|CUDA_[A-Z]\w*|cublas[A-Z]\w*|cublasLt\w*|nccl[A-Z]\w*|"
    r"cufile\w*|cuFile\w*|nvml[A-Z]\w*|__global__|__device__|__host__|__shared__|__constant__|"
    r"__launch_bounds__)\b|\bcu[A-Z]\w*\s*\(|<<<")
CUDA_INCLUDES = ("providers/cuda/",)
CUDA_SUFFIXES = (".cu", ".cuh")

KERNEL_HEADERS = re.compile(r"^(ggml|kernels/(ggml|exl3|cutlass)/|llmp_exl3)")
# Neutral graph and selection code cannot choose vendors by enum, embedded
# identifier, device spelling or diagnostic either. Provenance comments remain
# outside the check; format names are interpreted by the format layer.
NEUTRAL_HOMES = ("src/ir/", "src/execution/")
KERNEL_WORDS = re.compile(r"\b\w*(ggml|exl3|cublas|cutlass)\w*\b", re.IGNORECASE)
NEUTRAL_VENDOR_WORDS = re.compile(r"\b\w*(cuda|nvidia|gb10(?!\d)|sm_?\d+[a-z]?)\w*\b", re.IGNORECASE)
# The format layer's names that carry a kernel library's (format/layout.h,
# format/encoding.h): layouts and encodings that GGML's and ExLlamaV3's
# formats define. Code outside STRICT_HOMES may name them.
FORMAT_WORDS = frozenset({"kGgmlBlocks", "kGgmlNvfp4Blocks", "kNvfp4GgmlBlocks", "kExl3Trellis",
                          "kExl3Tiles", "kGgmlRowPadding"})
# Format vocabulary in data, anywhere: the EXL3 checkpoint container and
# encoding name, the format layer's layout names, and GGUF's tokenizer keys.
FORMAT_STRINGS = re.compile(r"^(exl3|ggml_blocks|ggml_nvfp4_blocks|exl3_tiles|exl3_trellis)$", re.IGNORECASE)
GGUF_KEY = re.compile(r"\btokenizer\.ggml\.\w+")

INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(?:([<"])([^>"]+)[>"]|([A-Za-z_]\w*))', re.MULTILINE)
DEFINE = re.compile(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+[<"]([^>"]+)[>"]', re.MULTILINE)
MACRO = "<macro>"
# Where split's plain code may end: a comment, a literal or a number.
NEXT_SPECIAL = re.compile(r"[/\"']|(?<!\w)\d")
RAW_PREFIXES = ("R", "u8R", "uR", "UR", "LR")


def _word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _starts_number(text: str, i: int) -> bool:
    return text[i].isdigit() and (i == 0 or not _word_char(text[i - 1]))


def _literal_end(text: str, i: int, quote: str) -> int:
    """The offset just past the literal opened by `quote` at i (at the line's end if it is not closed)."""
    j = i + 1
    while j < len(text) and text[j] != quote and text[j] != "\n":
        j += 2 if text[j] == "\\" else 1
    return min(j + 1, len(text))


def split(text: str) -> tuple[str, str, str]:
    """Three views of a C++ or CUDA source, each as long as the text and with its line breaks.

    code: comments and literals blanked; strings: only string literals'
    contents kept; plain: comments blanked, everything else kept.
    Character literals are code in no view but plain; numbers with digit
    separators (1'000) are code.
    """
    views: tuple[list[str], list[str], list[str]] = ([], [], [])
    i, n = 0, len(text)

    def keep(chunk: str, as_code: bool, as_string: bool, as_plain: bool) -> None:
        blank = re.sub(r"[^\n]", " ", chunk)
        for view, kept in zip(views, (as_code, as_string, as_plain)):
            view.append(chunk if kept else blank)

    def keep_string(start: int, body_end: int, end: int) -> None:
        keep(text[start:start + 1], False, False, True)
        keep(text[start + 1:body_end], False, True, True)
        keep(text[body_end:end], False, False, True)

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            keep(text[i:end], False, False, False)
            i = end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            keep(text[i:end], False, False, False)
            i = end
        elif c == '"':
            start = i
            while start > 0 and _word_char(text[start - 1]):
                start -= 1
            open_paren = text.find("(", i + 1)
            if text[start:i] in RAW_PREFIXES and open_paren >= 0:
                delimiter = text[i + 1:open_paren]
                close = text.find(")" + delimiter + '"', open_paren + 1)
                body_end = n if close < 0 else close + len(delimiter) + 1
                end = min(body_end + 1, n)
            else:
                end = _literal_end(text, i, '"')
                body_end = end - 1 if end > i + 1 and text[end - 1] == '"' else end
            keep_string(i, body_end, end)
            i = end
        elif c == "'":
            end = _literal_end(text, i, "'")
            keep(text[i:end], False, False, True)
            i = end
        elif _starts_number(text, i):
            # A preprocessing number: digits, letters, underscores, dots,
            # digit separators, and signs after an exponent.
            j = i + 1
            while j < n and (_word_char(text[j]) or text[j] in ".'" or
                             (text[j] in "+-" and text[j - 1] in "eEpP")):
                j += 1
            keep(text[i:j], True, False, True)
            i = j
        else:
            special = NEXT_SPECIAL.search(text, i + 1)
            j = special.start() if special else n
            keep(text[i:j], True, False, True)
            i = j
    code, strings, plain = ("".join(view) for view in views)
    return code, strings, plain


def includes(plain: str) -> list[tuple[int, str]]:
    """(offset, name) of each #include in `plain` (comments blanked, strings kept).

    An include through a macro is the path the file #defines it as, or MACRO
    followed by the macro's name when the file does not.
    """
    defined = {m.group(1): m.group(2) for m in DEFINE.finditer(plain)}
    found = []
    for m in INCLUDE.finditer(plain):
        if m.group(3) is not None:
            found.append((m.start(), defined.get(m.group(3), MACRO + m.group(3))))
        else:
            found.append((m.start(), m.group(2)))
    return found


def normalized(name: str, header: str) -> str:
    """An include as a path under src/: a relative one resolved against the including file `name`."""
    if header.startswith(MACRO):
        return header
    if header.startswith(("./", "../")):
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), header))
        return resolved[len(SCOPE):] if resolved.startswith(SCOPE) else resolved
    header = posixpath.normpath(header)
    return header[len(SCOPE):] if header.startswith(SCOPE) else header


def kernel_source(path: str) -> str | None:
    """The kernel source a path under src/ (or a src/-relative include) belongs to, if any."""
    path = path[len(SCOPE):] if path.startswith(SCOPE) else path
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "kernels" and parts[1] in KERNEL_SOURCES:
        return parts[1]
    return None


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def findings(name: str, text: str) -> list[tuple[str, str]]:
    """(rule, problem) for one source file, whatever its place."""
    code, strings, plain = split(text)
    found: list[tuple[str, str]] = []

    def add(rule: str, offset: int, what: str) -> None:
        found.append((rule, f"{name}:{_line(text, offset)}: {what}"))

    source = kernel_source(name)
    common = name.startswith("src/kernels/common/")
    for offset, raw in includes(plain):
        header = normalized(name, raw)
        if header.startswith(MACRO):
            if source is None:
                add(KERNEL, offset, f"includes {header[len(MACRO):]}, a macro the check cannot resolve")
            continue
        if LINUX_HEADERS.search(header) or header in LINUX_INCLUDES:
            add(LINUX, offset, f"includes {header}, which is Linux-specific")
        if CUDA_HEADERS.search(header) or header.startswith(CUDA_INCLUDES):
            add(CUDA, offset, f"includes {header}, which is CUDA's")
        if KERNEL_HEADERS.search(header):
            add(KERNEL, offset, f"includes {header}, which names a kernel library")
        included = kernel_source(header)
        if included is not None and (common or (source is not None and included != source)):
            found.append((SOURCE, f"{name}:{_line(text, offset)}: includes {header}, another kernel source's"))
    for m in LINUX_WORDS.finditer(code):
        add(LINUX, m.start(), f"uses {m.group(0)}, which is Linux-specific")
    for m in LINUX_PATHS.finditer(strings):
        add(LINUX, m.start(), f"names /{m.group(2)}/, which is Linux's")
    for m in CUDA_WORDS.finditer(code):
        add(CUDA, m.start(), f"uses {m.group(0).rstrip('( ')}, which is CUDA's")
    strict = name.startswith(STRICT_HOMES)
    for m in KERNEL_WORDS.finditer(code):
        if strict or m.group(0) not in FORMAT_WORDS:
            add(KERNEL, m.start(), f"uses {m.group(0)}, which names a kernel library")
    # Include paths and the macros that name them are checked as includes.
    skipped = [range(k.start(), k.end()) for k in GGUF_KEY.finditer(strings)]
    skipped += [range(k.start(), k.end()) for k in INCLUDE.finditer(plain)]
    skipped += [range(k.start(), k.end()) for k in DEFINE.finditer(plain)]
    for m in KERNEL_WORDS.finditer(strings):
        if FORMAT_STRINGS.match(m.group(0)) or any(m.start() in span for span in skipped):
            continue
        add(KERNEL, m.start(), f"names {m.group(0)}, which names a kernel library")
    if name.startswith(NEUTRAL_HOMES):
        for view in (code, strings):
            for m in NEUTRAL_VENDOR_WORDS.finditer(view):
                add(CUDA, m.start(), f"names {m.group(0)}, which is vendor-specific")
    if name.endswith(CUDA_SUFFIXES):
        found.append((CUDA, f"{name}: a CUDA source file"))
    return found


def home(rule: str, name: str) -> bool:
    if rule == KERNEL:
        return name.startswith(KERNEL_HOMES)
    if rule == SOURCE:
        return False
    return name.startswith(LINUX_HOMES if rule == LINUX else CUDA_HOMES)


def check(root: pathlib.Path, names: list[str]) -> tuple[int, list[str]]:
    """(files checked, problems) over `names` (paths relative to root)."""
    problems: list[str] = []
    needed: set[tuple[str, str]] = set()
    checked = 0
    for name in sorted(names):
        if not name.startswith(SCOPE) or not name.endswith(SOURCE_SUFFIXES):
            continue
        path = root / name
        if path.is_symlink() or not path.is_file():
            continue
        checked += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for rule, problem in findings(name, text):
            if home(rule, name):
                continue
            if rule in EXCEPTIONS.get(name, {}):
                needed.add((name, rule))
                continue
            problems.append(problem)
    for name, rules in sorted(EXCEPTIONS.items()):
        for rule in sorted(rules):
            if (name, rule) not in needed:
                problems.append(f"{name}: allowed an exception to the {rule} rule it no longer needs")
    return checked, problems


def scope_files(root: pathlib.Path) -> list[str]:
    """Every file under the checked scope, for a tree without Git (a Spark's copy)."""
    return sorted(p.relative_to(root).as_posix() for p in (root / SCOPE).rglob("*") if p.is_file())


def main() -> int:
    """Checks this checkout's src/ (tools/check runs check() over the files Git lists)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    checked, problems = check(root, scope_files(root))
    for problem in problems:
        print(problem)
    print(f"boundary check: {checked} files, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
