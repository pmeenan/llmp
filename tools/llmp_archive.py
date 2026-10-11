# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The model archive's format: layout, names, revision manifests, verified file transfer,
catalog events, packs, compaction, the fold and the rendered inventory.

docs/model-archive.md is the specification (D-114); tools/archive-model,
tools/archive-catalog and tools/stage-model are its interim writers and
readers. Everything written goes through a temporary name, a flush and a
rename; archive-model serializes checkpoint writers with an archive-wide
lock. Catalog publication and staging do not rely on locks or symbolic links.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import fnmatch
import hashlib
import http.client
import json
import os
import pathlib
import re
import secrets
import socket
import threading
import stat
import time

VERSION = 1
FORMAT_ROOT = "llmp-archive-root"
FORMAT_MANIFEST = "llmp-archive-checkpoint"
FORMAT_EVENT = "llmp-archive-event"
FORMAT_PACK = "llmp-archive-pack"
FORMAT_CATALOG = "llmp-archive-catalog"
ROOT_MARKER = "llmp-archive-root.json"
WRITER_GUARD = ".llmp-writer"  # atomic directory creation is the cross-host exclusion
WRITER_OWNER = "owner.json"
MANIFEST = "llmp-archive.json"
PARTIAL = ".llmp-partial."
TMP = ".llmp-tmp."
ORIGIN = "hf"
TOOL_VERSION = "2"

MANIFEST_CAP = 16 << 20
EVENT_CAP = 16 << 20
PACK_CAP = 512 << 20
STALE_SECONDS = 15 * 60
CHUNK = 4 << 20
IDENTITY_CHECK_BYTES = 64 << 20
COMMON_FILE_CAP = 64 << 20  # non-weight files up to this size always come along
PACK_LOOSE_THRESHOLD = 64
PACK_COUNT_THRESHOLD = 8

REPO_PART = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
REPO_RE = re.compile(rf"^{REPO_PART}/{REPO_PART}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
BAD_NAME_CHARS = re.compile(r'[\x00-\x1f\x7f-\x9f:*?"<>|\\]')
SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
WEIGHT_SUFFIXES = (".gguf", ".safetensors")
FOREIGN_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".onnx", ".onnx_data", ".h5",
                    ".msgpack", ".npz", ".ot", ".tflite", ".mlmodel", ".nemo")
RELATIONS = ("quantized", "finetune", "merge", "adapter")
cancel = threading.Event()  # set by the CLI on SIGINT/SIGTERM: transfers stop at their next chunk


def sleep(seconds: float) -> None:  # tests replace it; a signal cuts a backoff short
    cancel.wait(seconds)


class ArchiveError(Exception):
    pass


class Mismatch(ArchiveError):
    pass


class WriterBusy(ArchiveError):
    """The archive-wide checkpoint writer lock is held."""


class Lost(ArchiveError):
    """Another writer adopted this writer's partial file."""


class WriteFailed(ArchiveError):
    """The archive refused a write (disk full, share gone): never retried as a network error."""


# ---- names and paths ----------------------------------------------------------------------

def now() -> float:
    return time.time()


def utc(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now() if t is None else t))


def host_name() -> str:
    return re.sub(r"[^a-z0-9-]", "-", socket.gethostname().split(".")[0].lower())[:32] or "host"


def writer_id(host: str | None = None) -> str:
    return f"{host or host_name()}.{os.getpid()}.{secrets.token_hex(4)}"


def check_repo(repo: str) -> str:
    if not REPO_RE.match(repo) or ".." in repo:
        raise ArchiveError(f"not a repository ID: {repo!r}")
    return repo


def check_commit(commit: str) -> str:
    if not COMMIT_RE.match(commit):
        raise ArchiveError(f"not a full commit: {commit!r}")
    return commit


def path_problem(path: str) -> str | None:
    """Why a repository path cannot be archived, or None."""
    if not path or path.startswith("/"):
        return "absolute or empty"
    if BAD_NAME_CHARS.search(path):
        return "a control or Windows-reserved character"
    for part in path.split("/"):
        if part in ("", ".", ".."):
            return "an empty, . or .. component"
        if part.endswith((".", " ")):
            return "a component ending in a dot or space"
        if len(part.encode()) > 255:
            return "a component over 255 bytes"
    if path == MANIFEST or PARTIAL in path or TMP in path:
        return "a name the archive reserves"
    return None


def revision_dir(root: pathlib.Path, repo: str, commit: str) -> pathlib.Path:
    org, name = check_repo(repo).split("/")
    return root / ORIGIN / org / f"{name}@{check_commit(commit)}"


def relative_revision_path(repo: str, commit: str) -> str:
    org, name = repo.split("/")
    return f"{ORIGIN}/{org}/{name}@{commit}"


# ---- JSON ---------------------------------------------------------------------------------

def dumps(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name):
    raise ValueError(f"non-finite number {name}")


def loads_strict(data: bytes, what: str):
    try:
        return json.loads(data, object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    except (ValueError, UnicodeDecodeError) as error:
        raise ArchiveError(f"{what}: not valid JSON ({error})") from None


def read_capped(path: pathlib.Path, cap: int) -> bytes:
    with open(path, "rb") as f:
        data = f.read(cap + 1)
    if len(data) > cap:
        raise ArchiveError(f"{path}: over {cap} bytes")
    return data


def _retry_rename(source: pathlib.Path, target: pathlib.Path) -> None:
    delay = 0.05
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:  # SMB: the target is open on a client without delete sharing
            if attempt == 7:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def atomic_write(path: pathlib.Path, data: bytes, writer: str) -> None:
    tmp = path.with_name(f"{path.name}{TMP}{writer}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _retry_rename(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


# ---- the root -----------------------------------------------------------------------------

def open_root(root: pathlib.Path, create: bool, writer: str = "") -> pathlib.Path:
    marker = root / ROOT_MARKER
    if marker.is_file():
        doc = loads_strict(read_capped(marker, 4096), str(marker))
        if not isinstance(doc, dict) or doc.get("format") != FORMAT_ROOT:
            raise ArchiveError(f"{marker}: not an archive root marker")
        if doc.get("format_version") != VERSION:
            raise ArchiveError(f"{marker}: format_version {doc.get('format_version')!r} is not {VERSION}")
        return root
    if not create:
        raise ArchiveError(f"{root}: not a model archive (no {ROOT_MARKER}); the share may be unmounted. "
                           "archive-model --init-root makes a new archive")
    if root.exists() and any(root.iterdir()):
        raise ArchiveError(f"{root}: not empty and not a model archive; refusing to write there")
    try:
        root.mkdir(exist_ok=True)  # never its parents: a mount point that is gone must not be recreated
    except FileNotFoundError:
        raise ArchiveError(f"{root}: its parent directory does not exist (share not mounted?); "
                           "refusing to create it") from None
    atomic_write(marker, dumps({"format": FORMAT_ROOT, "format_version": VERSION}), writer or writer_id())
    return root


def ensure_root(root: pathlib.Path) -> None:
    """Refuse to write when the archive root is not there (an unmounted share would otherwise
    have its directories recreated on the local disk)."""
    if not (root / ROOT_MARKER).is_file():
        raise ArchiveError(f"{root}: the archive root is gone (share unmounted?); nothing was written")


def _process_start(pid: int) -> str | None:
    try:
        # comm may contain spaces and parentheses; fields after its last ')' start at field 3.
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def _writer_owner(fd: int | None = None) -> dict:
    mutex = os.fstat(fd) if fd is not None else None
    return {"format": "llmp-archive-writer", "format_version": 2,
            "machine": pathlib.Path("/etc/machine-id").read_text().strip(),
            "boot": pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": os.stat("/proc/self/ns/pid").st_ino,
            "mount_namespace": os.stat("/proc/self/ns/mnt").st_ino,
            "local_mutex": [mutex.st_dev, mutex.st_ino] if mutex else None,
            "uid": os.getuid(), "pid": os.getpid(), "start": _process_start(os.getpid())}


def _owner_dead(owner, mine) -> bool:
    # Only prove death in this same host, boot, user and PID namespace. Foreign,
    # rebooted or incomplete ownership stays locked until independently recovered.
    if not isinstance(owner, dict) or owner.get("format") != "llmp-archive-writer" or owner.get(
            "format_version") != 2:
        return False
    for key in ("machine", "boot", "pid_namespace", "mount_namespace", "uid", "local_mutex"):
        if owner.get(key) != mine[key]:
            return False
    if mine["local_mutex"] is None:
        return False
    pid, start = owner.get("pid"), owner.get("start")
    if type(pid) is not int or pid <= 0 or not isinstance(start, str) or not start.isdecimal():
        return False
    return _process_start(pid) != start


@contextlib.contextmanager
def writer_lock(root: pathlib.Path, *, no_wait: bool = False, log=None):
    """Server-atomic directory ownership across hosts; a local flock serializes recovery.

    SMB flock may fail open or stall in open, so it is never our NAS mutex.
    A per-user, host-local flock ensures two local recovery attempts cannot
    remove a newly acquired guard. Only demonstrably dead local owners are
    recovered; unknown/foreign ownership is never stolen by age.
    """
    ensure_root(root)
    # Fixed across launches, independent of TMPDIR and /run/user availability.
    # Matching the held inode is also required before recovering any dead owner.
    lock_path = pathlib.Path(f"/tmp/llmp-archive-writer-{os.getuid()}.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    guard, owned = root / WRITER_GUARD, False
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
            raise ArchiveError(f"{lock_path}: not this user's regular lock file")
        announced = False
        def wait():
            nonlocal announced
            if no_wait:
                raise WriterBusy("another checkpoint writer holds the archive writer lock")
            if not announced:
                if log:
                    log("waiting for the archive writer lock")
                announced = True
            sleep(1)
        while True:
            if cancel.is_set():
                raise ArchiveError("interrupted while waiting for the archive writer lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                wait()
        mine = _writer_owner(fd)
        while True:
            if cancel.is_set():
                raise ArchiveError("interrupted while waiting for the archive writer lock")
            ensure_root(root)
            try:
                guard.mkdir()  # one server operation: exactly one host can create it
                owned = True
                atomic_write(guard / WRITER_OWNER, dumps(mine), writer_id())
                break
            except FileExistsError:
                try:
                    owner = loads_strict(read_capped(guard / WRITER_OWNER, 4096), str(guard))
                except FileNotFoundError:
                    owner = None  # creation in progress or interrupted; never steal it
                if _owner_dead(owner, mine):
                    (guard / WRITER_OWNER).unlink()
                    guard.rmdir()
                    if log:
                        log("recovered a dead local archive writer's lock")
                    continue
                wait()
        yield
    finally:
        try:
            if owned:
                (guard / WRITER_OWNER).unlink(missing_ok=True)
                # An interrupted atomic_write may have left its own temporary name.
                for temp in guard.glob(WRITER_OWNER + TMP + "*"):
                    temp.unlink()
                guard.rmdir()
        finally:
            os.close(fd)  # host-local mutex stays held until NAS ownership is released


def make_dirs(base: pathlib.Path, path: pathlib.Path) -> None:
    """Create `path` and the directories between it and `base`, but never `base` or anything above
    it: when the share has gone, the directories are not recreated on the local disk."""
    current = base
    for part in path.relative_to(base).parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass


def prune_empty(directory: pathlib.Path) -> None:
    """Remove `directory` and the directories under it if nothing is in them (a failed first
    attempt leaves no revision directory that looks like a download in progress)."""
    for dirpath, _, _ in os.walk(directory, topdown=False):
        try:
            os.rmdir(dirpath)
        except OSError:
            pass


# ---- manifests ----------------------------------------------------------------------------

def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_manifest(m, what: str = MANIFEST) -> dict:
    def fail(why):
        raise ArchiveError(f"{what}: {why}")
    if not isinstance(m, dict):
        fail("not an object")
    if m.get("format") != FORMAT_MANIFEST:
        fail(f"format is {m.get('format')!r}, not {FORMAT_MANIFEST!r}")
    if m.get("format_version") != VERSION:
        fail(f"format_version {m.get('format_version')!r} is not one this tool reads ({VERSION})")
    if m.get("origin") != ORIGIN:
        fail(f"origin {m.get('origin')!r}")
    if not isinstance(m.get("repo"), str) or not REPO_RE.match(m["repo"]):
        fail("repo")
    if not isinstance(m.get("commit"), str) or not COMMIT_RE.match(m["commit"]):
        fail("commit")
    if not isinstance(m.get("hub"), dict):
        fail("hub")
    files = m.get("files")
    if not isinstance(files, list):
        fail("files")
    seen = set()
    for entry in files:
        if not isinstance(entry, dict):
            fail("a file entry is not an object")
        path = entry.get("path")
        if not isinstance(path, str) or path_problem(path):
            fail(f"file path {path!r}")
        if path.lower() in seen:
            fail(f"file path {path!r} twice (ignoring case)")
        seen.add(path.lower())
        if not _is_int(entry.get("size")) or entry["size"] < 0:
            fail(f"{path}: size")
        if not isinstance(entry.get("present"), bool) or not isinstance(entry.get("lfs"), bool):
            fail(f"{path}: present or lfs")
        sha = entry.get("sha256")
        if sha is not None and (not isinstance(sha, str) or not HEX64_RE.match(sha)):
            fail(f"{path}: sha256")
        if entry["present"] and sha is None:
            fail(f"{path}: present without a sha256")
        oid = entry.get("git_oid")
        if oid is not None and (not isinstance(oid, str) or not COMMIT_RE.match(oid)):
            fail(f"{path}: git_oid")
    if [e["path"] for e in files] != sorted(e["path"] for e in files):
        fail("files are not sorted by path")
    variants = m.get("variants")
    if not isinstance(variants, list) or not all(isinstance(v, dict) and isinstance(v.get("id"), str)
                                                 and isinstance(v.get("files"), list) for v in variants):
        fail("variants")
    for v in variants:
        for p in v["files"]:
            if p.lower() not in seen:
                fail(f"variant {v['id']!r} names {p!r}, which the manifest does not list")
    return m


def read_manifest(directory: pathlib.Path) -> dict | None:
    path = directory / MANIFEST
    try:
        data = read_capped(path, MANIFEST_CAP)
    except FileNotFoundError:
        return None
    return validate_manifest(loads_strict(data, str(path)), str(path))


def manifest_sha256(m: dict) -> str:
    return hashlib.sha256(dumps(m)).hexdigest()


def list_revisions(root: pathlib.Path, repo: str) -> list[tuple[str, dict]]:
    """Archived revisions of `repo` as (commit, manifest), newest manifest first."""
    org, name = check_repo(repo).split("/")
    base = root / ORIGIN / org
    out = []
    try:
        entries = list(base.iterdir())
    except FileNotFoundError:
        return out
    for entry in entries:
        stem, _, commit = entry.name.rpartition("@")
        if stem.lower() != name.lower() or not COMMIT_RE.match(commit):
            continue
        m = read_manifest(entry)
        if m is not None:
            out.append((commit, m))
    out.sort(key=lambda item: (item[1].get("updated", ""), item[0]), reverse=True)
    return out


def hub_files(info: dict) -> list[dict]:
    """The Hub's file list (revision API, ?blobs=true) as manifest entries, every one absent."""
    files = []
    for sibling in info.get("siblings") or []:
        if not isinstance(sibling, dict) or not isinstance(sibling.get("rfilename"), str):
            raise ArchiveError("the Hub's file list has an entry without a name")
        lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else None
        size = lfs.get("size") if lfs else sibling.get("size")
        if not _is_int(size) or size < 0:
            raise ArchiveError(f"{sibling['rfilename']}: the Hub gave no size")
        sha = lfs.get("sha256") if lfs else None
        if lfs and not (isinstance(sha, str) and HEX64_RE.match(sha)):
            raise ArchiveError(f"{sibling['rfilename']}: an LFS file without a SHA-256")
        oid = sibling.get("blobId")
        files.append({"path": sibling["rfilename"], "size": size, "lfs": lfs is not None, "sha256": sha,
                      "git_oid": oid if isinstance(oid, str) and COMMIT_RE.match(oid) else None,
                      "present": False})
    files.sort(key=lambda e: e["path"])
    return files


def _base_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return [v for v in value if isinstance(v, str) and REPO_RE.match(v)] if isinstance(value, list) else []


def hub_summary(info: dict, requested: str) -> dict:
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    tags = [t for t in info.get("tags") or [] if isinstance(t, str)]
    tag_relation = {}
    for tag in tags:
        parts = tag.split(":")
        if len(parts) == 3 and parts[0] == "base_model" and parts[1] in RELATIONS:
            tag_relation[parts[2]] = parts[1]
    card_relation = card.get("base_model_relation") if card.get("base_model_relation") in RELATIONS else None
    bases = []
    for repo in _base_list(card.get("base_model")):
        if card_relation:
            bases.append({"repo": repo, "relation": card_relation, "source": "card"})
        elif repo in tag_relation:
            bases.append({"repo": repo, "relation": tag_relation[repo], "source": "tags"})
        else:
            bases.append({"repo": repo, "relation": None, "source": "card"})
    for repo, relation in sorted(tag_relation.items()):
        if REPO_RE.match(repo) and repo not in {b["repo"] for b in bases}:
            bases.append({"repo": repo, "relation": relation, "source": "tags"})
    gated = info.get("gated")

    def text(value):
        return value if isinstance(value, str) else None
    return {"requested": requested, "last_modified": text(info.get("lastModified")),
            "gated": gated if gated in (False, "auto", "manual") else bool(gated),
            "private": bool(info.get("private")), "license": text(card.get("license")),
            "license_name": text(card.get("license_name")), "license_link": text(card.get("license_link")),
            "pipeline_tag": text(info.get("pipeline_tag")), "library_name": text(info.get("library_name")),
            "base_models": bases, "tags": tags}


# ---- selection and variants ---------------------------------------------------------------

def is_weight(path: str) -> bool:
    return path.lower().endswith(WEIGHT_SUFFIXES)


def is_foreign(path: str) -> bool:
    return path.lower().endswith(FOREIGN_SUFFIXES)


def group_variants(paths: list[str]) -> dict[str, dict]:
    """Variant ID -> {files, role, kind} over the repository's weight files."""
    variants: dict[str, dict] = {}
    diffusers = "model_index.json" in paths
    for path in sorted(paths):
        lower = path.lower()
        if lower.endswith(".gguf"):
            vid = SPLIT_RE.sub("", path) if SPLIT_RE.search(path) else path[:-5]
            role = "projector" if path.rsplit("/", 1)[-1].lower().startswith("mmproj") else "weights"
            variants.setdefault(vid, {"files": [], "role": role, "kind": "gguf"})["files"].append(path)
        elif lower.endswith(".safetensors"):
            vid = "*" if diffusers or "/" not in path else path.rsplit("/", 1)[0]
            variants.setdefault(vid, {"files": [], "role": "weights", "kind": "safetensors"})["files"].append(path)
    return variants


def select_files(files: list[dict], includes: list[str], excludes: list[str],
                 all_variants: bool) -> tuple[list[dict], list[str]]:
    """The files to archive and a list of problems (empty when the selection is acceptable)."""
    chosen, problems = [], []
    for entry in files:
        path = entry["path"]
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in excludes):
            continue
        named = any(fnmatch.fnmatchcase(path, pattern) for pattern in includes)
        if is_weight(path):
            take = named or not includes
        elif is_foreign(path):
            take = named
        else:
            take = named or entry["size"] <= COMMON_FILE_CAP
        if take:
            chosen.append(entry)
    for entry in chosen:
        why = path_problem(entry["path"])
        if why:
            problems.append(f"{entry['path']}: cannot be archived ({why})")
    folded = {}
    for entry in files:
        folded.setdefault(entry["path"].lower(), []).append(entry["path"])
    for names in folded.values():
        if len(names) > 1:
            problems.append(f"{' and '.join(names)} differ only in case, which the share cannot hold")
    for pattern in includes:  # a typo must not end as a quiet archive of the config files only
        if not any(fnmatch.fnmatchcase(entry["path"], pattern) for entry in files):
            problems.append(f"--include {pattern!r} matches no file of the revision")
    if not includes and not all_variants:
        variants = group_variants([e["path"] for e in chosen])
        weights = {k: v for k, v in variants.items() if v["kind"] == "gguf" and v["role"] == "weights"}
        if len(weights) > 1:
            sizes = {e["path"]: e["size"] for e in files}
            listing = "\n".join(f"  {vid}  ({sum(sizes[p] for p in v['files']) / 1e9:.1f} GB)"
                                for vid, v in sorted(weights.items()))
            problems.append("several GGUF variants; choose with --include GLOB or take all with "
                            f"--all-variants:\n{listing}")
    return chosen, problems


# ---- verified transfer --------------------------------------------------------------------

class Hashes:
    def __init__(self, entry: dict):
        self.sha256 = hashlib.sha256()
        self.git = None
        if not entry["lfs"] and entry.get("git_oid"):
            self.git = hashlib.sha1(f"blob {entry['size']}\0".encode())  # noqa: S324 (Git's own ID)

    def update(self, data) -> None:
        self.sha256.update(data)
        if self.git is not None:
            self.git.update(data)

    def check(self, entry: dict, path: str) -> str:
        digest = self.sha256.hexdigest()
        if not entry.get("sha256") and self.git is None:
            raise Mismatch(f"{path}: the Hub gave neither a SHA-256 nor a Git blob ID to verify it against")
        if entry.get("sha256") and digest != entry["sha256"]:
            raise Mismatch(f"{path}: SHA-256 {digest} is not the recorded {entry['sha256']}")
        if self.git is not None and self.git.hexdigest() != entry["git_oid"]:
            raise Mismatch(f"{path}: Git blob ID {self.git.hexdigest()} is not the recorded {entry['git_oid']}")
        return digest


def _partials(final: pathlib.Path) -> list[pathlib.Path]:
    prefix = final.name + PARTIAL
    try:
        return [p for p in final.parent.iterdir() if p.name.startswith(prefix)]
    except FileNotFoundError:
        return []


def _same_file(path: pathlib.Path, fd: int) -> bool:
    try:
        return os.stat(path).st_ino == os.fstat(fd).st_ino
    except FileNotFoundError:
        return False


def _write_all(fd: int, data, offset: int) -> None:
    """pwrite every byte of `data`; a short write is continued, never counted as done."""
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise OSError(errno.EIO, "a write wrote nothing")
        view = view[written:]
        offset += written


def _progress(callback, phase, received, checked=0):
    """Optional telemetry must never fail the transfer or perform storage queries."""
    if callback:
        try:
            callback(phase, received, checked)
        except Exception:
            pass


def transfer(directory: pathlib.Path, entry: dict, opener, writer: str, *, retries: int = 8,
             stale: float = STALE_SECONDS, log=None, writer_locked: bool = False,
             progress=None) -> tuple[str, str | None]:
    """Bring `entry` into `directory` verified. Returns (status, sha256): status is
    `present` (already final with the recorded size), `done`, or `busy` (another writer's
    live partial holds it). `opener(start)` returns a readable whose `start` attribute is
    where it begins. Raises Mismatch (the partial is deleted), Lost, WriteFailed or, when the
    source fails for good, the source's error; a partial left behind by a failure is released
    (its modification time set to the epoch, or the file removed if empty) so that the next
    run adopts it at once instead of waiting out another writer's claim.
    writer_locked requires the archive-wide writer lock for the entire transfer:
    partials can then be adopted immediately and periodic pathname checks omitted.
    Staging uses the default timed claims and identity checks on its local target."""
    log = log or (lambda message: None)
    final = directory / entry["path"]
    try:
        st = os.stat(final)
        if st.st_size == entry["size"]:
            _progress(progress, "present", entry["size"])
            return "present", None
        bad = final.with_name(f"{final.name}{PARTIAL}bad-{writer}")
        os.replace(final, bad)  # a final name with the wrong size: never trusted, never served
        bad.unlink()
        log(f"{entry['path']}: wrong size on the archive, fetching it again")
    except FileNotFoundError:
        pass
    make_dirs(directory, final.parent)
    mine = final.with_name(f"{final.name}{PARTIAL}{writer}")
    for other in _partials(final):
        if other == mine:
            continue
        try:
            age = now() - os.stat(other).st_mtime
        except FileNotFoundError:
            continue
        if not writer_locked and age < stale:
            return "busy", None
        try:
            os.rename(other, mine)  # adoption: only one adopter's rename finds it
        except FileNotFoundError:
            return "busy", None  # another adopter won; its partial is live now
        try:
            os.utime(mine)  # a rename keeps the old time: show the claim as live at once
        except OSError:
            pass
        log(f"{entry['path']}: resuming a partial download")
        break
    hashes = Hashes(entry)
    fd = os.open(mine, os.O_RDWR | os.O_CREAT, 0o666)
    release = False
    try:
        size = os.fstat(fd).st_size
        if size > entry["size"]:
            os.ftruncate(fd, 0)
            size = 0
        offset, last_touch = 0, now()
        _progress(progress, "checking resumed prefix", size)
        while offset < size:  # hash the whole prefix being resumed, keeping the claim fresh
            if cancel.is_set():
                raise ArchiveError(f"{entry['path']}: interrupted")
            data = os.pread(fd, min(CHUNK, size - offset), offset)
            if not data:
                raise ArchiveError(f"{mine}: shorter than it said")
            hashes.update(data)
            offset += len(data)
            _progress(progress, "checking resumed prefix", size, offset)
            if now() - last_touch > 60:
                os.utime(fd)
                last_touch = now()
        attempt, since_check = 0, 0
        while offset < entry["size"]:
            if cancel.is_set():
                raise ArchiveError(f"{entry['path']}: interrupted")
            before = offset
            try:
                _progress(progress, "connecting", offset)
                source = opener(offset)
            except Exception as error:  # noqa: BLE001 (each source has its own errors)
                if not getattr(error, "retryable", False) or attempt >= retries:
                    raise
                attempt += 1
                _progress(progress, "retry backoff", offset)
                sleep(min(2 ** attempt, 60))
                continue
            try:
                if source.start != offset:  # the source ignored the range: start over
                    os.ftruncate(fd, 0)
                    offset, hashes = 0, Hashes(entry)
                    _progress(progress, "receiving", 0)
                while offset < entry["size"]:
                    if cancel.is_set():
                        raise ArchiveError(f"{entry['path']}: interrupted")
                    _progress(progress, "receiving", offset)
                    data = source.read(CHUNK)
                    if not data:
                        break
                    if offset + len(data) > entry["size"]:
                        raise Mismatch(f"{entry['path']}: longer than the recorded {entry['size']} bytes")
                    try:
                        _progress(progress, "writing", offset)
                        _write_all(fd, data, offset)
                    except OSError as error:
                        raise WriteFailed(f"{entry['path']}: cannot write the archive: {error}") from None
                    hashes.update(data)
                    offset += len(data)
                    _progress(progress, "receiving", offset)
                    since_check += len(data)
                    if not writer_locked and since_check >= IDENTITY_CHECK_BYTES:
                        since_check = 0
                        if not _same_file(mine, fd):
                            raise Lost(f"{entry['path']}: another writer adopted this download")
            except (OSError, http.client.HTTPException) as error:
                if attempt >= retries:
                    raise ArchiveError(f"{entry['path']}: {error}") from None
            finally:
                source.close()
            if offset > before:
                attempt = 0  # progress: the budget is for failures in a row
            if offset < entry["size"]:  # cut off: resume from here
                if attempt >= retries:
                    raise ArchiveError(f"{entry['path']}: the source ended at {offset} of {entry['size']} bytes")
                attempt += 1
                _progress(progress, "retry backoff", offset)
                sleep(min(2 ** attempt, 60))
        _progress(progress, "flushing and verifying", offset)
        os.fsync(fd)
        held = os.fstat(fd).st_size
        if held != entry["size"]:
            raise Mismatch(f"{entry['path']}: the partial file holds {held} bytes, not the recorded {entry['size']}")
        digest = hashes.check(entry, entry["path"])
        if not _same_file(mine, fd):
            raise Lost(f"{entry['path']}: another writer adopted this download")
    except Lost:
        raise
    except Mismatch:  # the bytes are wrong: nothing of them is kept
        os.close(fd)
        fd = -1
        mine.unlink(missing_ok=True)
        _progress(progress, "hash mismatch; discarded", 0)
        raise
    except BaseException:
        release = True
        raise
    finally:
        if fd >= 0:
            try:
                empty = release and os.fstat(fd).st_size == 0
            except OSError:
                empty = False
            os.close(fd)
            if release:
                try:
                    if empty:
                        mine.unlink()
                    else:
                        os.utime(mine, (0, 0))
                except OSError:
                    pass
    _progress(progress, "publishing file", entry["size"])
    _retry_rename(mine, final)
    _progress(progress, "verified", entry["size"])
    return "done", digest


class _FileSource:
    def __init__(self, path: pathlib.Path, start: int):
        self._f = open(path, "rb")
        self._f.seek(start)
        self.start = start

    def read(self, n):
        return self._f.read(n)

    def close(self):
        self._f.close()


def file_opener(path: pathlib.Path):
    return lambda start: _FileSource(path, start)


def hash_file(path: pathlib.Path, entry: dict, *, progress=None) -> str:
    hashes = Hashes(entry)
    checked = 0
    _progress(progress, "checking existing file", 0)
    with open(path, "rb") as f:
        while data := f.read(CHUNK):
            hashes.update(data)
            checked += len(data)
            _progress(progress, "checking existing file", 0, checked)
    if os.stat(path).st_size != entry["size"]:
        raise Mismatch(f"{entry['path']}: {os.stat(path).st_size} bytes, not the recorded {entry['size']}")
    return hashes.check(entry, entry["path"])


# ---- the manifest's merge -----------------------------------------------------------------

def merge_manifest(existing: dict | None, repo: str, commit: str, hub: dict, files: list[dict],
                   done: dict[str, dict], directory: pathlib.Path, detect) -> dict:
    """The manifest after this writer's files (`done`: path -> {sha256, fetched}) are final.
    Files present on disk with the recorded size but not listed are listed too."""
    base = dict(existing) if existing else {}
    old = {e["path"]: e for e in (existing or {}).get("files", [])}
    merged = []
    for entry in files:
        e = dict(old.get(entry["path"], entry))
        for key in ("size", "lfs", "git_oid"):
            e[key] = entry[key]
        if entry.get("sha256") and e.get("sha256") != entry["sha256"]:
            e["sha256"] = entry["sha256"]
        if entry["path"] in done:
            e.update(done[entry["path"]], present=True)
        else:
            final = directory / entry["path"]
            try:
                on_disk = final.is_file() and os.stat(final).st_size == entry["size"]
            except OSError:
                on_disk = False
            if e.get("present") and not on_disk:  # gone or resized, and not fetched again (a failed refetch)
                e["present"] = False
                e.pop("fetched", None)
            elif not e.get("present") and on_disk and e.get("sha256"):
                e["present"] = True
                e.setdefault("fetched", {"time": utc(), "host": host_name(),
                                         "tool": f"archive-model/{TOOL_VERSION}", "via": "found"})
        e.setdefault("present", False)
        merged.append(e)
    merged.sort(key=lambda e: e["path"])
    stamp = utc()
    base.update({"format": FORMAT_MANIFEST, "format_version": VERSION, "origin": ORIGIN, "repo": repo,
                 "commit": commit, "hub": base.get("hub") or hub, "files": merged,
                 "archived": base.get("archived") or stamp, "updated": stamp})
    base["variants"] = detect(directory, merged, base.get("variants") or [], set(done))
    return validate_manifest(base)


def write_manifest(directory: pathlib.Path, build, writer: str, attempts: int = 5) -> dict:
    """Read, merge (`build(existing)`), write and read back until this writer's merge holds."""
    for _ in range(attempts):
        existing = read_manifest(directory)
        manifest = build(existing)
        atomic_write(directory / MANIFEST, dumps(manifest), writer)
        back = read_manifest(directory) or {}
        present = {e["path"] for e in back.get("files", []) if e.get("present")}
        if all(present >= {e["path"]} for e in manifest["files"] if e["present"]):
            return back
        time.sleep(secrets.randbelow(1000) / 1000)
    raise ArchiveError(f"{directory / MANIFEST}: concurrent writers kept replacing it; rerun")


# ---- events, packs and compaction ---------------------------------------------------------

def catalog_dir(root: pathlib.Path) -> pathlib.Path:
    return root / "catalog"


def event_id(t: float | None = None, host: str | None = None) -> str:
    t = now() if t is None else t
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(t))}-{host or host_name()}-{os.getpid()}-{secrets.token_hex(4)}"


def write_event(root: pathlib.Path, kind: str, body: dict, tool: str, host: str | None = None) -> dict:
    ensure_root(root)
    t = now()
    host = host or host_name()
    event = {"format": FORMAT_EVENT, "format_version": VERSION, "id": event_id(t, host), "time": utc(t),
             "host": host, "tool": tool, "kind": kind, "body": body}
    events = catalog_dir(root) / "events"
    make_dirs(root, events)
    atomic_write(events / f"{event['id']}.json", dumps(event), writer_id(host))
    return event


def validate_event(e) -> dict:
    if (not isinstance(e, dict) or e.get("format") != FORMAT_EVENT or e.get("format_version") != VERSION
            or not isinstance(e.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9.-]{1,128}", e["id"])
            or not isinstance(e.get("time"), str) or not isinstance(e.get("kind"), str)
            or not isinstance(e.get("body"), dict)):
        raise ArchiveError("not a valid event")
    return e


def _quarantine(root: pathlib.Path, path: pathlib.Path, why: str, report: list) -> None:
    target = catalog_dir(root) / "quarantine"
    make_dirs(root, target)
    try:
        os.replace(path, target / path.name)
        report.append(f"quarantined {path.name}: {why}")
    except FileNotFoundError:
        pass


def _json_files(directory: pathlib.Path) -> list[pathlib.Path]:
    try:
        return sorted(p for p in directory.iterdir() if p.name.endswith(".json") and TMP not in p.name)
    except FileNotFoundError:
        return []


def read_all_events(root: pathlib.Path, report: list) -> tuple[dict, list, list]:
    """(events by ID, loose files read, pack files read).

    The loose events are listed before the packs: a compactor deletes a loose event only after
    publishing a pack that holds it, so every event deleted before the first listing is in a pack
    the second listing shows (a pack is deleted only after a later pack holds it too)."""
    cat = catalog_dir(root)
    for _ in range(3):
        events, loose, packs, vanished = {}, [], [], False
        loose_paths = _json_files(cat / "events")
        pack_paths = _json_files(cat / "packs")
        for path in pack_paths:
            try:
                doc = loads_strict(read_capped(path, PACK_CAP), path.name)
                if (not isinstance(doc, dict) or doc.get("format") != FORMAT_PACK
                        or doc.get("format_version") != VERSION or not isinstance(doc.get("events"), list)):
                    raise ArchiveError("not a valid pack")
                for e in doc["events"]:
                    events[validate_event(e)["id"]] = e
                packs.append(path)
            except FileNotFoundError:
                vanished = True
                break
            except ArchiveError as error:
                _quarantine(root, path, str(error), report)
        if vanished:
            continue
        for path in loose_paths:
            try:
                e = validate_event(loads_strict(read_capped(path, EVENT_CAP), path.name))
                events.setdefault(e["id"], e)
                loose.append(path)
            except FileNotFoundError:
                vanished = True
                break
            except ArchiveError as error:
                _quarantine(root, path, str(error), report)
        if not vanished:
            return events, loose, packs
    return events, loose, packs


def compact(root: pathlib.Path, report: list | None = None, *, force_pack: bool = False) -> dict:
    report = [] if report is None else report
    ensure_root(root)
    events, loose, packs = read_all_events(root, report)
    cat = catalog_dir(root)
    writer = writer_id()
    if events and (force_pack or len(loose) > PACK_LOOSE_THRESHOLD or len(packs) > PACK_COUNT_THRESHOLD):
        doc = {"format": FORMAT_PACK, "format_version": VERSION,
               "events": [events[k] for k in sorted(events)]}
        data = dumps(doc)
        name = f"pack-{hashlib.sha256(data).hexdigest()}.json"
        make_dirs(root, cat / "packs")
        atomic_write(cat / "packs" / name, data, writer)
        for path in [*loose, *packs]:  # each read whole and contained in the pack just published
            if path.name != name:
                try:
                    path.unlink(missing_ok=True)
                except OSError as error:  # a reader on another client holds it open: a later compaction removes it
                    report.append(f"left {path.name} for a later compaction: {error}")
        report.append(f"packed {len(events)} events into {name}")
    snapshot = fold(sorted(events.values(), key=lambda e: (e["time"], e["id"])))
    make_dirs(root, cat)
    atomic_write(cat / "catalog.json", dumps(snapshot), writer)
    atomic_write(cat / "catalog.md", render(snapshot).encode(), writer)
    return snapshot


# ---- the fold -----------------------------------------------------------------------------

QUANTIZED_DTYPES = ("F32", "F16", "BF16")


def _quantized(variant: dict) -> bool:
    d = variant.get("detected") or {}
    scheme = d.get("scheme") or ""
    if scheme.startswith("gguf:"):
        return scheme[5:].upper() not in ("ALL_F32", "F32", "F16", "MOSTLY_F16", "BF16", "MOSTLY_BF16")
    return bool(scheme) and ":" in scheme


def fold(events: list[dict]) -> dict:
    revisions: dict[tuple, dict] = {}
    ever_present: dict[tuple, set] = {}
    imports: dict[tuple, dict] = {}
    installs: dict[tuple, dict] = {}
    stages: dict[tuple, dict] = {}
    relation_override: dict[str, dict] = {}
    family_label: dict[str, str] = {}
    for e in events:
        b, kind = e["body"], e["kind"]
        try:
            if kind == "revision":
                m = validate_manifest(b["manifest"], f"event {e['id']}")
                key = (m["repo"], m["commit"])
                present = ever_present.setdefault(key, set())
                present |= {f["path"] for f in m["files"] if f["present"]}
                revisions[key] = {"manifest": m, "sha256": b.get("manifest_sha256"), "path": b.get("path")}
            elif kind == "revision-removed":
                key = (b["repo"], b["commit"])
                if b.get("paths") is None:
                    revisions.pop(key, None)
                    ever_present.pop(key, None)
                else:
                    ever_present.get(key, set()).difference_update(b["paths"])
            elif kind == "import":
                imports[(b["artifact"]["id"], b["host"])] = {**b, "time": e["time"]}
            elif kind in ("install", "uninstall"):
                installs[(b["host"], b["artifact_id"])] = {**b, "state": kind, "time": e["time"]}
            elif kind in ("stage", "unstage"):
                stages[(b["host"], b["repo"], b["commit"])] = {**b, "state": kind, "time": e["time"]}
            elif kind == "relation":
                if b.get("base") is None:
                    relation_override.pop(b["repo"], None)
                else:
                    relation_override[b["repo"]] = {"repo": b["base"], "relation": b.get("relation"),
                                                    "source": "override"}
            elif kind == "family":
                family_label[b["repo"]] = b["label"]
        except (KeyError, TypeError, ArchiveError):
            continue  # a malformed body of a known kind is skipped; unknown kinds are kept, unfolded

    out_revisions = []
    repo_info: dict[str, dict] = {}
    for (repo, commit), r in sorted(revisions.items()):
        m = r["manifest"]
        present = ever_present.get((repo, commit), set())
        files = [{**f, "present": f["present"] or f["path"] in present} for f in m["files"]]
        variants = []
        for v in m["variants"]:
            vp = [f for f in files if f["path"] in set(v["files"])]
            n = sum(f["present"] for f in vp)
            variants.append({"id": v["id"], "role": v.get("role"), "files": v["files"],
                             "present": n == len(vp) and n > 0 or (n and "partial") or False,
                             "bytes": sum(f["size"] for f in vp), "detected": v.get("detected"),
                             "imports": sorted((dict(i) for i in imports.values()
                                                if i.get("repo") == repo and i.get("commit") == commit
                                                and i.get("variant") == v["id"]),
                                               key=lambda i: (i["time"], i["artifact"]["id"]))})
        staged = sorted(({"host": s["host"], "path": s.get("path"), "time": s["time"],
                          "variants": s.get("variants")} for s in stages.values()
                         if s["repo"] == repo and s["commit"] == commit and s["state"] == "stage"),
                        key=lambda s: s["host"])
        hub = m["hub"]
        out_revisions.append({
            "repo": repo, "commit": commit, "path": r["path"] or relative_revision_path(repo, commit),
            "requested": hub.get("requested"), "license": hub.get("license"), "gated": hub.get("gated"),
            "archived": m.get("archived"), "updated": m.get("updated"), "manifest_sha256": r["sha256"],
            "bytes_present": sum(f["size"] for f in files if f["present"]),
            "bytes_total": sum(f["size"] for f in files), "base_models": hub.get("base_models") or [],
            "variants": variants, "staged": staged})
        info = repo_info.setdefault(repo, {"revisions": [], "bases": hub.get("base_models") or [],
                                           "quantized": False, "updated": ""})
        info["revisions"].append((m.get("updated") or "", commit))
        if (m.get("updated") or "") >= info["updated"]:
            info["updated"] = m.get("updated") or ""
            info["bases"] = hub.get("base_models") or []
        info["quantized"] |= any(_quantized(v) for v in variants if v["role"] == "weights")

    # Relations: each repository's single parent edge, overrides first.
    parent: dict[str, dict] = {}
    for repo, info in repo_info.items():
        bases = [b for b in info["bases"] if b.get("repo") and b["repo"] != repo]
        if repo in relation_override:
            parent[repo] = relation_override[repo]
        elif len(bases) >= 1:
            chosen = dict(sorted(bases, key=lambda b: b["repo"])[0])
            if chosen.get("relation") is None and info["quantized"] and len(bases) == 1:
                chosen.update(relation="quantized", source="inferred")
            parent[repo] = chosen
    for repo, edge in relation_override.items():
        parent.setdefault(repo, edge)

    def model_of(repo: str) -> str:
        seen = set()
        while repo in parent and parent[repo].get("relation") == "quantized" and repo not in seen:
            seen.add(repo)
            repo = parent[repo]["repo"]
        return repo

    def root_of(repo: str) -> str:
        seen = set()
        while repo in parent and repo not in seen and repo not in family_label:
            seen.add(repo)
            repo = parent[repo]["repo"]
        return repo

    repositories, models, families = [], {}, {}
    for repo in sorted(repo_info):
        model = model_of(repo)
        family = root_of(model)
        edge = parent.get(repo)
        repositories.append({"repo": repo, "model": model, "family": family,
                             "relation": edge.get("relation") if edge else None,
                             "base": edge.get("repo") if edge else None,
                             "relation_source": edge.get("source") if edge else None,
                             "revisions": [c for _, c in sorted(repo_info[repo]["revisions"], reverse=True)]})
        mentry = models.setdefault(model, {"repo": model, "family": family, "variants": [],
                                           "derived_from": None})
        if model in parent and parent[model].get("relation") != "quantized":
            mentry["derived_from"] = {"repo": parent[model]["repo"], "relation": parent[model].get("relation")}
        families.setdefault(family, {"id": family, "label": family_label.get(family, family), "models": set()})
        families[family]["models"].add(model)
    for rev in out_revisions:
        model = model_of(rev["repo"])
        for v in rev["variants"]:
            models[model]["variants"].append({"repo": rev["repo"], "commit": rev["commit"], "variant": v["id"]})
    installs_out = sorted(({"host": i["host"], "artifact_id": i["artifact_id"], "store": i.get("store"),
                            "time": i["time"]} for i in installs.values() if i["state"] == "install"),
                          key=lambda i: (i["host"], i["artifact_id"]))
    return {"format": FORMAT_CATALOG, "format_version": VERSION, "generated": utc(), "events": len(events),
            "revisions": out_revisions, "repositories": repositories,
            "models": [models[k] for k in sorted(models)],
            "families": [{**f, "models": sorted(f["models"])} for _, f in sorted(families.items())],
            "installs": installs_out}


# ---- the readable inventory ---------------------------------------------------------------

def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


def variant_rows(snapshot: dict) -> list[dict]:
    """One row per variant of every revision, with its family and model."""
    repos = {r["repo"]: r for r in snapshot["repositories"]}
    rows = []
    for rev in snapshot["revisions"]:
        info = repos.get(rev["repo"], {})
        for v in rev["variants"]:
            d = v.get("detected") or {}
            rows.append({"family": info.get("family", rev["repo"]), "model": info.get("model", rev["repo"]),
                         "repo": rev["repo"], "commit": rev["commit"], "variant": v["id"], "role": v["role"],
                         "container": d.get("container") or "?", "scheme": d.get("scheme") or "?",
                         "bits": d.get("bits"), "quantizer": d.get("quantizer"), "bytes": v["bytes"],
                         "present": v["present"], "imports": len(v["imports"]),
                         "staged": ", ".join(s["host"] for s in rev["staged"]), "license": rev["license"],
                         "archived": rev["archived"]})
    return rows


def render(snapshot: dict) -> str:
    def cell(text) -> str:
        return str(text).replace("|", "\\|")
    lines = ["<!-- Generated by tools/archive-catalog; do not edit. -->", "", "# Model archive inventory", "",
             f"Generated {snapshot['generated']} from {snapshot['events']} events: "
             f"{len(snapshot['revisions'])} revisions of {len(snapshot['repositories'])} repositories, "
             f"{len(snapshot['models'])} models in {len(snapshot['families'])} families. "
             "Specification: docs/model-archive.md.", ""]
    rows = variant_rows(snapshot)
    labels = {f["id"]: f["label"] for f in snapshot["families"]}
    derived = {m["repo"]: m["derived_from"] for m in snapshot["models"]}
    for family in sorted({r["family"] for r in rows}):
        lines += [f"## {labels.get(family, family)}", ""]
        for model in sorted({r["model"] for r in rows if r["family"] == family}):
            d = derived.get(model)
            lines += [f"### {model}" + (f" ({d['relation'] or 'derived'} of {d['repo']})" if d else ""), "",
                      "| Repository | Variant | Container | Scheme | Bits | Size | Present | Commit | "
                      "Archived | Imports | Staged on | License |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in sorted((r for r in rows if r["model"] == model and r["family"] == family),
                            key=lambda r: (r["repo"], r["variant"], r["commit"])):
                bits = f"{r['bits']:.2f}" if isinstance(r["bits"], float) else (r["bits"] or "")
                present = {True: "yes", False: "no", "partial": "partial"}[r["present"]]
                lines.append("| " + " | ".join(cell(x) for x in (
                    r["repo"], r["variant"] + (f" ({r['role']})" if r["role"] != "weights" else ""),
                    r["container"], r["scheme"], bits, _gb(r["bytes"]), present, r["commit"][:8],
                    (r["archived"] or "")[:10], r["imports"], r["staged"], r["license"] or "")) + " |")
            lines.append("")
    if snapshot["installs"]:
        lines += ["## Installed artifacts", "", "| Host | Artifact | Since |", "| --- | --- | --- |"]
        lines += [f"| {cell(i['host'])} | {i['artifact_id'][:16]} | {i['time'][:10]} |" for i in snapshot["installs"]]
        lines.append("")
    return "\n".join(lines)
