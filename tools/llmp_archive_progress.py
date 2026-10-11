# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Best-effort local archive telemetry. This module never reads model storage.

Download workers report already-known counters over a nonblocking Unix datagram
socket. The controller pushes newline-delimited JSON over a Unix stream socket
only to connected viewers. Both channels are bounded; progress cannot delay I/O.
"""

import collections
import copy
import fcntl
import json
import math
import os
import pathlib
import secrets
import socket
import threading
import time

MAX_FRAME = 65536
MAX_VIEWERS = 8


def encode(value):
    data = json.dumps(value, separators=(",", ":")).encode() + b"\n"
    return data if len(data) <= MAX_FRAME else None


def summary(plan, state):
    """Summarize the controller's own plan/state, with no filesystem access."""
    tasks = state.get("tasks", {})
    counts = collections.Counter(tasks.get(t["id"], {}).get("status", "queued") for t in plan)
    active = next((t for t in plan if t["id"] == state.get("active")), None)
    return {"protocol": 1, "time": time.time(), "status": state.get("status", "unknown"),
            "counts": dict(counts), "revisions": len(plan),
            "total": sum(t["selected_bytes"] for t in plan),
            "verified": sum(t["selected_bytes"] for t in plan
                            if tasks.get(t["id"], {}).get("status") == "complete"),
            "active": {k: active[k] for k in ("id", "repo", "commit", "selected_bytes")}
            if active else None,
            "issues": [{"repo": t["repo"], "status": tasks[t["id"]]["status"]}
                       for t in plan if tasks.get(t["id"], {}).get("status")
                       in ("failed", "waiting-for-access")][:32]}


class Reporter:
    """Full counter snapshots survive lost datagrams; no incremental event backlog."""

    def __init__(self, endpoint=None, session=None):
        self.endpoint, self.session = endpoint, session
        self.lock = threading.Lock()
        self.files = {}
        self.current_phase = "starting"
        self.changed = time.time()
        self.stop = threading.Event()
        self.socket = None
        self.thread = None
        if endpoint and session:
            try:
                self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                self.socket.setblocking(False)
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            except (OSError, RuntimeError):
                # Thread.start() reports resource exhaustion as RuntimeError;
                # optional telemetry must not abort the download or join a
                # thread that never started.
                self.thread = None
                self.close()

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("LLMP_ARCHIVE_PROGRESS_SOCKET"),
                   os.environ.get("LLMP_ARCHIVE_PROGRESS_SESSION"))

    def phase(self, value):
        if not self.socket:
            return
        with self.lock:
            if self.current_phase != value:
                self.current_phase = value
                self.changed = time.time()

    def plan(self, chosen, present):
        if not self.socket:
            return
        with self.lock:
            self.files = {e["path"]: {"path": e["path"], "size": e["size"],
                          "received": e["size"] if e["path"] in present else 0,
                          "phase": "verified" if e["path"] in present else "queued",
                          "checked": 0} for e in chosen}
            self.changed = time.time()

    def file(self, entry, phase, received, checked=0):
        if not self.socket:
            return
        row = {"path": entry["path"], "size": entry["size"], "phase": phase,
               "received": min(entry["size"], max(0, received)), "checked": checked}
        with self.lock:
            if self.files.get(entry["path"]) != row:
                self.files[entry["path"]] = row
                self.changed = time.time()

    def snapshot(self):
        with self.lock:
            rows = list(self.files.values())
            return {"protocol": 1, "session": self.session, "phase": self.current_phase,
                    "time": time.time(), "changed": self.changed,
                    "received": sum(r["received"] for r in rows),
                    "total": sum(r["size"] for r in rows),
                    "verified_files": sum(r["phase"] == "verified" for r in rows),
                    "files": [dict(r, path=r["path"][:2048]) for r in rows
                              if r["phase"] not in ("verified", "queued")][:8]}

    def _run(self):
        while not self.stop.is_set():
            data = encode(self.snapshot())
            if data:
                try:
                    self.socket.sendto(data, self.endpoint)
                except OSError:
                    pass  # no controller, full receiver queue, or disconnected socket
            self.stop.wait(0.5)

    def close(self):
        self.stop.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(0.6)
        if self.socket:
            self.socket.close()
            self.socket = None


class Controller:
    """An independent local publisher stays responsive during main-thread NAS I/O."""

    def __init__(self, base, plan, state):
        self.base = pathlib.Path(base)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.plan, self.state = copy.deepcopy(plan), copy.deepcopy(state)
        self.session = None
        self.phase = "preparing revision"
        self.transfer = None
        self.updated = None
        self.mutex = (self.base / "controller.lock").open("a+b")
        try:
            fcntl.flock(self.mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.mutex.close()
            raise
        self.input_path = self.base / "transfer.sock"
        self.output_path = self.base / "status.sock"
        self.receiver = self.listener = None
        self.thread = None
        try:
            for path in (self.input_path, self.output_path):
                path.unlink(missing_ok=True)
            self.receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.receiver.bind(str(self.input_path))
            self.receiver.setblocking(False)
            self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.listener.bind(str(self.output_path))
            self.listener.setblocking(False)
            self.listener.listen(MAX_VIEWERS)
            for path in (self.input_path, self.output_path):
                path.chmod(0o600)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def update(self, *, plan=None, state=None):
        with self.lock:
            if plan is not None:
                self.plan = copy.deepcopy(plan)
            if state is not None:
                if state.get("active") != self.state.get("active"):
                    self.transfer = None
                    self.session = None
                    self.phase = "preparing revision"
                self.state = copy.deepcopy(state)

    def begin(self, phase):
        with self.lock:
            self.session = secrets.token_hex(16)
            self.phase, self.transfer, self.updated = phase, None, None
            return {"LLMP_ARCHIVE_PROGRESS_SOCKET": str(self.input_path),
                    "LLMP_ARCHIVE_PROGRESS_SESSION": self.session}

    def end(self):
        with self.lock:
            self.session = None
            self.phase = "recording result"
            if self.transfer is not None:
                self.transfer["phase"] = self.phase

    def snapshot(self):
        with self.lock:
            result = summary(self.plan, self.state)
            result.update(phase=self.phase, transfer=copy.deepcopy(self.transfer),
                          reported_at=self.updated, live=True)
            return result

    def _receive(self):
        # Bound each pass so a noisy producer cannot starve viewers.
        for _ in range(32):
            try:
                data = self.receiver.recv(MAX_FRAME + 1)
            except BlockingIOError:
                break
            if len(data) > MAX_FRAME:
                continue
            try:
                value = json.loads(data)
                if not isinstance(value, dict) or type(value.get("protocol")) is not int or value["protocol"] != 1:
                    continue
                with self.lock:
                    if not self.session or value.get("session") != self.session:
                        continue
                # Validate the values consumed by the viewer, not arbitrary JSON.
                if any(type(value.get(k)) is not int or not 0 <= value[k] < 2**63
                       for k in ("received", "total", "verified_files")):
                    continue
                if not isinstance(value.get("phase"), str) or not isinstance(value.get("files"), list):
                    continue
                if len(value["files"]) > 8 or not isinstance(value.get("changed"), (int, float)):
                    continue
                if (not math.isfinite(value["changed"]) or not 0 <= value["changed"] <= time.time() + 60
                        or value["received"] > value["total"]):
                    continue
                for row in value["files"]:
                    if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                            or not isinstance(row.get("phase"), str)
                            or any(type(row.get(k)) is not int or not 0 <= row[k] < 2**63
                                   for k in ("size", "received", "checked"))):
                        raise ValueError("invalid file counters")
                    if row["received"] > row["size"]:
                        raise ValueError("invalid received count")
                # Discard unknown fields/deep structures before deepcopy and encoding.
                files = value["files"]
                value = {k: value[k] for k in ("session", "phase", "changed", "received", "total", "verified_files")}
                value["phase"] = value["phase"][:256]
                value["files"] = [{k: row[k] for k in ("path", "phase", "size", "received", "checked")}
                                  for row in files]
                for row in value["files"]:
                    row["path"], row["phase"] = row["path"][:2048], row["phase"][:256]
            except (ValueError, TypeError, RecursionError, OverflowError):
                continue
            with self.lock:
                if self.session and value.get("session") == self.session:
                    value.pop("session", None)
                    self.transfer = value
                    self.updated = time.time()

    def _run(self):
        clients = {}  # socket -> [pending bytes, pending since, next snapshot deadline]
        try:
            while not self.stop.is_set():
                self._receive()
                for _ in range(MAX_VIEWERS):
                    try:
                        client, _ = self.listener.accept()
                    except BlockingIOError:
                        break
                    if len(clients) >= MAX_VIEWERS:
                        client.close()
                    else:
                        client.setblocking(False)
                        clients[client] = [b"", 0, 0]
                now = time.monotonic()
                data = None
                for client, pending in list(clients.items()):
                    try:
                        if not pending[0] and now >= pending[2]:
                            if data is None:
                                data = encode(self.snapshot())
                            if data is None:
                                raise OSError("status frame too large")
                            pending[:] = [data, now, now + 1]
                        if pending[0]:
                            if now - pending[1] > 5:
                                raise OSError("slow viewer")
                            try:
                                count = client.send(pending[0])
                                if not count:
                                    raise OSError("viewer disconnected")
                                pending[0] = pending[0][count:]
                            except BlockingIOError:
                                pass
                    except OSError:
                        client.close()
                        del clients[client]
                self.stop.wait(0.05)
        finally:
            for client in clients:
                client.close()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(1)
        for sock in (self.receiver, self.listener):
            if sock:
                sock.close()
        for path in (self.input_path, self.output_path):
            path.unlink(missing_ok=True)
        self.mutex.close()
