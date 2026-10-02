# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""The embedded-header check (D-029, D-071): files that can hold a comment carry their own license header.

REUSE lint accepts a file's copyright and license information from its
comments, from a `.license` sidecar, from REUSE.toml or from .reuse/dep5.
D-029 asks for more, and this check enforces it over every file Git tracks
or would add, against the report of `reuse lint --json` for the same tree:

- a file whose format has comments has an `SPDX-FileCopyrightText` and an
  `SPDX-License-Identifier` tag on comment lines within its first
  HEADER_LINES lines, and REUSE reads exactly that header's license, and at
  least its copyright holders, from the file itself: not from a sidecar,
  REUSE.toml or dep5, and not from other tags elsewhere in the file;
- a file that cannot hold a comment (JSON, a patch, plain text, a lock file
  a tool rewrites) has a sidecar, and REUSE reads it from that sidecar alone;
- any license or copyright tag anywhere in a file or its sidecar, as a
  reader would see it (after Unicode normalization, without invisible
  characters, in any case, and in Markdown and HTML also without escapes,
  character references, comments, tags or emphasis), repeats a license
  expression and a holder REUSE reads for that file. So no tag REUSE skips
  (after line 10, in a sidecar's file, past REUSE's read window) can show a
  reader another license, and no file uses REUSE's ignore markers;
- every file and sidecar is UTF-8 without control characters other than
  tab and line ends;
- every sidecar belongs to a file, and there is no REUSE.toml or .reuse;
- REUSE covers exactly the files checked here, so no file escapes lint by
  its name or place (REUSE skips LICENSES/ at any depth, for example);
- a file of a type this module does not classify fails, so each new type
  is classified deliberately.

What the tags say is REUSE lint's to validate. Like REUSE, the check skips
symbolic links, empty files, the root LICENSES/ and files named like license
texts; but LICENSES/ may hold only `<id>.txt`, and a file named like a
license text fails unless it is plain text or Markdown without a `#!` line.
This module and its tests spell tag names in parts, so that neither REUSE nor
the check reads them as tags.
"""

from __future__ import annotations

import fnmatch
import functools
import html
import pathlib
import re
import unicodedata

HEADER_LINES = 10
SIDECAR = ".license"

HASH, SLASH, HTML = "#", "//", "<!--"
STYLE_LABELS = {HASH: "#", SLASH: "// or /* */", HTML: "<!-- -->"}
# Comment style by path or file-name pattern (fnmatch), then by suffix.
STYLE_BY_PATH = {"toolchains/prerequisites/*.txt": HASH}
STYLE_BY_NAME = {
    "CMakeLists.txt": HASH, "requirements*.txt": HASH, "Dockerfile": HASH, "Dockerfile.*": HASH,
    ".gitignore": HASH, ".dockerignore": HASH, ".clang-format": HASH, ".clang-tidy": HASH, ".clangd": HASH,
}
STYLE_BY_SUFFIX = {
    **dict.fromkeys((".py", ".sh", ".cmake", ".toml", ".yml", ".yaml", ".service", ".sysusers", ".tmpfiles"), HASH),
    **dict.fromkeys((".cc", ".cpp", ".cxx", ".c", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".cu", ".cuh"),
                    SLASH),
    **dict.fromkeys((".md", ".html", ".xml", ".svg"), HTML),
}
# Files that take a sidecar: formats without comments, and files whose exact
# bytes a tool writes (mise rewrites mise.lock) or a test hashes (third-party
# chat templates, .jinja). Checked after the styles above, so CMakeLists.txt
# is still commentable.
SIDECAR_NAMES = ("NOTICE", "mise.lock")
SIDECAR_SUFFIXES = (".jinja", ".json", ".patch", ".tsv", ".txt")
# Names REUSE skips as license texts (reuse 6.2.0, covered_files.py). Such a
# file escapes lint, so only these suffixes may carry the name.
LICENSE_TEXT_NAMES = re.compile(r"^(LICEN[CS]E([-.].*)?|COPYING([-.].*)?)$")
LICENSE_TEXT_SUFFIXES = ("", ".txt", ".md", ".rst")
LICENSE_TEXTS = "LICENSES/"
LICENSE_TEXT_FILE = re.compile(r"^LICENSES/[A-Za-z0-9.+-]+\.txt$")
# Configuration that can override an embedded header.
OVERRIDE_NAME, OVERRIDE_DIR = "REUSE.toml", ".reuse"

# Spelled in parts: see the module docstring.
_TAGS = ("SPDX-" + "FileCopyrightText", "SPDX-" + "License-Identifier")
_IGNORE_MARKERS = ("REUSE-" + "IgnoreStart", "REUSE-" + "IgnoreEnd")
_LINE = {
    HASH: r"^\s*#\s*{tag}:\s*(\S.*?)\s*$",
    SLASH: r"^\s*(?://|/\*|\*)\s*{tag}:\s*(\S.*?)\s*(?:\*/\s*)?$",
    HTML: r"^\s*<!--\s*{tag}:\s*(\S.*?)\s*-->\s*$",
}
_PATTERNS = {style: [re.compile(line.format(tag=tag)) for tag in _TAGS] for style, line in _LINE.items()}
# Any tag, anywhere in a line, as REUSE's extract.py reads it and more
# loosely: any case, and space before the colon.
_ANY_TAG = re.compile("SPDX-" + r"(FileCopyrightText|SnippetCopyrightText|License-Identifier)\s*:[ \t]*([^\r\n]*)",
                      re.IGNORECASE)
_ANY_MARKER = re.compile("|".join(_IGNORE_MARKERS), re.IGNORECASE)
_CLOSERS = re.compile(r"(\s|-->|\*/)+$")
_NOT_A_HOLDER = {"spdx", "filecopyrighttext", "snippetcopyrighttext", "copyright", "c"}
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"), "-")
_MARKDOWN_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")
_MARKUP = re.compile(r"<!--.*?-->|<[^<>]*>|[*_`~]", re.DOTALL)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@functools.cache
def _invisible() -> dict[int, None]:
    """Unicode format characters (soft hyphen, zero-width space and the like), to delete."""
    return {cp: None for cp in range(0x110000) if unicodedata.category(chr(cp)) == "Cf"}


def visible(text: str, style: str | None = None) -> str:
    """The text as a reader may see it: NFKC-normalized (full-width colons become
    colons), without invisible characters, with one kind of dash, and in Markdown
    and HTML without escapes and character references. For Markdown and HTML,
    renderings() adds a copy without markup."""
    if style == HTML:
        text = html.unescape(_MARKDOWN_ESCAPE.sub(r"\1", text))
    return unicodedata.normalize("NFKC", text).translate(_invisible()).translate(_DASHES)


def renderings(text: str, style: str | None) -> list[str]:
    """What the scan reads: the visible text and, for Markdown and HTML, also that text without
    comments, tags and emphasis, which a renderer would hide inside a tag's name."""
    seen = visible(text, style)
    return [seen, _MARKUP.sub("", seen)] if style == HTML else [seen]


def _expression(value: str) -> str:
    """An SPDX expression compared as REUSE may respell it: case and spacing."""
    return re.sub(r"\s+", " ", value.casefold()).replace("( ", "(").replace(" )", ")").strip()


def _holder(value: str) -> tuple[str, ...]:
    """A copyright notice reduced to its words and years, as REUSE may respell its prefix, (c) and spaces."""
    return tuple(w for w in re.findall(r"\w+", value.casefold()) if w not in _NOT_A_HOLDER)


def style_of(name: str, first_line: bytes) -> str | None:
    """The comment style of a file (a repository-relative POSIX path), or None if it cannot hold a comment.

    Raises ValueError for a file of a type the check does not classify.
    """
    base = name.rsplit("/", 1)[-1]
    for table, key in ((STYLE_BY_PATH, name), (STYLE_BY_NAME, base)):
        for pattern, style in table.items():
            if fnmatch.fnmatchcase(key, pattern):
                return style
    suffix = pathlib.PurePosixPath(base).suffix
    if suffix in STYLE_BY_SUFFIX:
        return STYLE_BY_SUFFIX[suffix]
    if base in SIDECAR_NAMES or suffix in SIDECAR_SUFFIXES:
        return None
    if not suffix and first_line.startswith(b"#!"):
        return HASH
    raise ValueError(f"{name}: a file type the header check does not classify; add it to "
                     "tools/jitllm_headers.py as commentable or as taking a sidecar")


def overrides(name: str) -> bool:
    return (name.rsplit("/", 1)[-1] == OVERRIDE_NAME or name == OVERRIDE_DIR
            or name.startswith(OVERRIDE_DIR + "/"))


def header_tags(style: str, head: list[str]) -> tuple[list[str], list[str]]:
    """The copyright and license values on comment lines among a file's first HEADER_LINES lines."""
    copyright_line, license_line = _PATTERNS[style]
    return ([m.group(1) for line in head if (m := copyright_line.match(line))],
            [m.group(1) for line in head if (m := license_line.match(line))])


def reuse_problems(name: str, own: str, entry: dict | None,
                   header: tuple[list[str], list[str]] | None) -> list[str]:
    """Where REUSE's report for a file disagrees with the file's own metadata:
    its header, or (header None) its sidecar, `own`."""
    if entry is None:
        return [f"{name}: REUSE lint does not cover it (a name or place REUSE skips); rename or move it"]
    infos = [*entry.get("copyrights", []), *entry.get("spdx_expressions", [])]
    sources = sorted({info["source"] for info in infos} - {own})
    problems = [f"{name}: REUSE reads license information from {', '.join(sources)}, not {own}"] if sources else []
    if header is not None:
        copyrights, licenses = header
        reuse_licenses = {info["value"] for info in entry.get("spdx_expressions", [])}
        holders = {_holder(visible(info["value"])) for info in entry.get("copyrights", [])}
        if {_expression(visible(v)) for v in reuse_licenses} != {_expression(visible(v)) for v in licenses}:
            problems.append(f"{name}: REUSE reads the license {' AND '.join(sorted(reuse_licenses)) or 'none'}, "
                            f"but the header says {' AND '.join(licenses)}")
        missing = [c for c in copyrights if _holder(visible(c)) not in holders]
        if missing:
            problems.append(f"{name}: REUSE does not read the header's copyright {'; '.join(missing)}")
    return problems


def body_problems(name: str, text: str, entry: dict | None) -> list[str]:
    """Tags anywhere in a text (a file or its sidecar, as `visible` shows it)
    that name a license expression or holder REUSE does not read for the file."""
    problems = [f"{name}: {marker} hides text from REUSE; don't use it (D-071)"
                for marker in sorted({m.group(0) for m in _ANY_MARKER.finditer(text)})]
    if entry is None:
        return problems
    licenses = {_expression(visible(i["value"])) for i in entry.get("spdx_expressions", [])}
    holders = {_holder(visible(i["value"])) for i in entry.get("copyrights", [])}
    for match in _ANY_TAG.finditer(text):
        kind, value = match.group(1), _CLOSERS.sub("", match.group(2).strip())
        if kind.casefold() == "license-identifier":
            unread = not value or _expression(value) not in licenses
        else:
            unread = _holder(value) not in holders
        if unread:
            problems.append(f"{name}: a {kind} tag REUSE does not read for its file: {value or '(empty)'!r}")
    return problems


def readable(path: pathlib.Path) -> str | None:
    """The file's text, or None unless it is UTF-8 without control characters but tab, CR and LF."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None
    return None if _CONTROL.search(text) else text


def check(root: pathlib.Path, names: list[str], report: dict) -> tuple[int, list[str]]:
    """Checks the named files (repository-relative POSIX paths) under root against
    REUSE's report for the same tree (`reuse lint --json`); returns (files checked, problems)."""
    present = set(names)
    covered = {entry["path"]: entry for entry in report.get("files", [])}
    problems = [f"{name}: can override embedded headers; use .license sidecars instead (D-071)"
                for name in sorted(present) if overrides(name)]
    problems += [f"{name}: can override embedded headers even if Git ignores it; remove it (D-071)"
                 for name in (OVERRIDE_NAME, OVERRIDE_DIR)
                 if not any(overrides(n) and (n == name or n.startswith(name + "/")) for n in present)
                 and ((root / name).is_symlink() or (root / name).exists())]
    checked = set()
    for name in sorted(present):
        path = root / name
        if name.endswith(SIDECAR):
            if name[:-len(SIDECAR)] not in present:
                problems.append(f"{name}: a sidecar for a file that does not exist")
            continue
        if overrides(name) or path.is_symlink() or not path.is_file():
            continue
        if name.startswith(LICENSE_TEXTS):
            if not LICENSE_TEXT_FILE.match(name):
                problems.append(f"{name}: LICENSES/ holds only license texts named <SPDX identifier>.txt")
            continue
        data = path.read_bytes()
        base = name.rsplit("/", 1)[-1]
        if LICENSE_TEXT_NAMES.match(base):
            if pathlib.PurePosixPath(base).suffix not in LICENSE_TEXT_SUFFIXES or data.startswith(b"#!"):
                problems.append(f"{name}: REUSE skips files named like license texts; rename it")
            continue
        if not data:
            continue
        checked.add(name)
        try:
            style = style_of(name, data.split(b"\n", 1)[0])
        except ValueError as e:
            problems.append(str(e))
            continue
        text = readable(path)
        if text is None:
            problems.append(f"{name}: not UTF-8 text without control characters, so neither REUSE nor this "
                            "check can be trusted to read it as a reader sees it")
            continue
        entry = covered.get(name)
        for rendering in renderings(text, style):
            problems += body_problems(name, rendering, entry)
        has_sidecar = name + SIDECAR in present
        if style is None:
            if has_sidecar:
                sidecar = readable(root / (name + SIDECAR))
                if sidecar is None:
                    problems.append(f"{name}{SIDECAR}: not UTF-8 text without control characters")
                else:
                    problems += body_problems(name + SIDECAR, visible(sidecar), entry)
                problems += reuse_problems(name, name + SIDECAR, entry, None)
            else:
                problems.append(f"{name}: cannot hold a comment, so it needs a {name}{SIDECAR} sidecar")
            continue
        if has_sidecar:
            problems.append(f"{name}: can hold a comment; embed the header and remove {name}{SIDECAR}, "
                            "which REUSE would read instead")
            continue
        header = header_tags(style, text.removeprefix("\ufeff").splitlines()[:HEADER_LINES])
        missing = [tag for tag, values in zip(_TAGS, header) if not values]
        if missing:
            problems.append(f"{name}: no {' or '.join(missing)} tag on a {STYLE_LABELS[style]} comment line "
                            f"within the first {HEADER_LINES} lines")
            continue
        problems += reuse_problems(name, name, entry, header)
    problems += [f"{name}: REUSE lint covers it, but the header check does not"
                 for name in sorted(set(covered) - checked)]
    return len(checked), list(dict.fromkeys(problems))
