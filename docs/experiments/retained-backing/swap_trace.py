#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Cross-model swap trace for the retained-backing comparison (D-035, D-079).

Builds a library of synthetic models whose groups take D-056's measured sizes
(library.json), plays episodes shaped like the frozen A->B->A trace with routes
from the paging-feasibility captures, and writes, per budget, the access stream
with a fragmentation-free reference eviction/restore sequence. At each
outward switch (a B request) the pool must return a shrink probe's bytes to
the OS, and gets them back before the next A request. Stdlib only.

    python3 swap_trace.py generate PARAMS LIBRARY EVIDENCE CAPTURES SESSIONS ROLE OUT
    python3 swap_trace.py verify OUT

CAPTURES is the paging study's external capture root (holding aba-capture-1,
small-capture-2, large-capture-2); SESSIONS is its A->B->A sessions.json;
ROLE is `primary` or `confirmation`. OUT must not exist.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEASIBILITY = HERE.parent / "paging-feasibility"
FORMAT = "llmp-swap-trace"
VERSION = 1
GIB = 1 << 30
# Every event a replayer may see, and the record types, in file order.
EVENTS = ("request", "lease", "release", "use", "shrink", "grow", "evict", "restore")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def line(record):
    return (json.dumps(record, separators=(",", ":")) + "\n").encode()


class Draws:
    """SHA-256 counter stream: the same draws on any Python and in C++."""

    def __init__(self, seed, label):
        self.prefix, self.counter = f"{seed}/{label}/", 0

    def below(self, n):
        if n < 1:
            raise ValueError("empty draw")
        limit = (1 << 64) // n * n
        while True:
            x = int.from_bytes(hashlib.sha256((self.prefix + str(self.counter)).encode()).digest()[:8], "big")
            self.counter += 1
            if x < limit:
                return x % n

    def permutation(self, items):
        items = list(items)
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


# ------------------------------------------------------------------ inputs

def load_feasibility():
    sys.path.insert(0, str(FEASIBILITY))
    try:
        import switch_replay
    finally:
        sys.path.remove(str(FEASIBILITY))
    return switch_replay


def check_params(params, library):
    if params.get("format") != "llmp-swap-trace-params" or params.get("version") != 1:
        raise ValueError("unknown parameter format")
    seeds = params["seeds"]
    if set(seeds) != {"primary", "confirmation"} or seeds["primary"] == seeds["confirmation"]:
        raise ValueError("need distinct primary and confirmation seeds")
    for num, den in params["ratios"]:
        if not (type(num) is int and type(den) is int and num > den > 0):
            raise ValueError("a ratio is unique bytes over budget, above 1")
    t = params["episode"]
    if len(t["roles"]) != len(t["arrival_s"]) or set(t["roles"]) != {"A", "B"} or t["span_s"] <= max(t["arrival_s"]):
        raise ValueError("bad episode template")
    if params["shrink_bytes"] < 0 or params["shrink_bytes"] % params["budget_granule"]:
        raise ValueError("the shrink probe is whole budget granules")
    if params["rounds"] < 1 or params["budget_granule"] < library["chunk_bytes"] or params["budget_granule"] % library["chunk_bytes"]:
        raise ValueError("bad rounds or budget granule")
    profiles = {p["profile"]: p for p in library["profiles"]}
    names = [m["name"] for m in params["library"]]
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError("library names must be unique, at least two")
    for m in params["library"]:
        p = profiles[m["profile"]]
        routed = {e["layer"] for e in p["expert_layers"]}
        if "layers" in m and (not m["layers"] or sorted(set(m["layers"])) != m["layers"] or not set(m["layers"]) <= routed):
            raise ValueError(f"{m['name']}: layers must be sorted routed layers")
        if routed and m["profile"] not in params["pools"]:
            raise ValueError(f"{m['name']}: routed profile without a route pool")
    return profiles


def check_sessions(params, sessions, evidence):
    """The episode template is the frozen A->B->A request order and arrivals."""
    if sha256_file(sessions) != evidence["inputs"]["aba"]["sha256"]:
        raise ValueError("A->B->A sessions.json does not match its recorded identity")
    requests = json.loads(Path(sessions).read_text())["requests"]
    t = params["episode"]
    if [r["model"] for r in requests] != t["roles"] or [r["arrival_s"] for r in requests] != t["arrival_s"]:
        raise ValueError("episode template differs from the frozen A->B->A trace")


def load_pools(params, profiles, evidence, captures):
    """Route pools per profile: validated captured requests, as per-layer expert unions."""
    replay = load_feasibility()
    receipts, pools, inputs = {}, {}, {}
    for key, source in params["captures"].items():
        # The receipt full-evidence.json records (with every run's event hash)
        # is the authority; the external copy must be that file or the
        # capture's original one, whose hash that receipt also records.
        record = evidence["captures"][source]
        receipt = record["receipt"]
        path = Path(captures) / record["external_directory"] / "receipt.json"
        if sha256_file(path) not in (record["receipt_file_sha256"], receipt.get("original_receipt_sha256")):
            raise ValueError(f"{source}: receipt does not match full-evidence.json")
        receipts[key] = (path.parent, receipt)
    for profile, runs in params["pools"].items():
        pool = []
        for key, name in runs:
            root, receipt = receipts[key]
            run = next(r for r in receipt["runs"] if r["name"] == name)
            if not (run["trace"] and run["reuse"] and run["swa"]):
                raise ValueError(f"{name}: pools use traced, retained, full-window runs")
            spec = next(m for m in receipt["spec"]["models"] if m["id"] == run["model"])
            check_spec(profile, profiles[profile], spec)
            events = replay.load_events(root / (name + ".jsonl"), run, spec)
            inputs[f"{key}/{name}"] = run["events_sha256"]
            for turn, request in enumerate(events):
                steps = collections.OrderedDict()
                for g in request["groups"]:
                    union = sorted({e for token in g["routes"] for e in token})
                    steps.setdefault((g["phase"], g["step"]), []).append((g["layer"], union))
                pool.append(dict(source=f"{key}/{run['model']}/{turn}", prompt=request["prompt"],
                                 reused=request["reused"], decode=request["decode"], steps=list(steps.values())))
        pools[profile] = pool
    return pools, inputs


def check_spec(profile, measured, spec):
    """The capture's layers and experts are the measured plan's, class for class."""
    layers = {e["layer"]: e for e in measured["expert_layers"]}
    if spec["experts"] != measured["experts"] or any(l not in layers for l in range(spec["layers"])):
        raise ValueError(f"{profile}: capture and plan disagree on experts or layers")
    classes = {}
    for l, logical in enumerate(spec["expert_bytes"]):
        if logical > layers[l]["used"] or classes.setdefault(logical, layers[l]["used"]) != layers[l]["used"]:
            raise ValueError(f"{profile}: layer {l} closure class differs from the plan")
    if len(set(classes.values())) != len(classes):
        raise ValueError(f"{profile}: capture closure classes do not map one to one")


# ------------------------------------------------------------------ trace

def build_groups(params, profiles):
    groups, models = [], {}
    for m in params["library"]:
        p = profiles[m["profile"]]
        keep = set(m.get("layers", [e["layer"] for e in p["expert_layers"]]))
        first, dense, experts = len(groups), [], {}
        for g in p["dense"]:
            if g["layer"] is None or not p["expert_layers"] or g["layer"] in keep:
                dense.append(len(groups))
                groups.append(dict(record="group", id=len(groups), model=m["name"], kind=g["kind"],
                                   layer=g["layer"], expert=None, used=g["used"], stored=g["stored"]))
        for e in p["expert_layers"]:
            if e["layer"] not in keep:
                continue
            experts[e["layer"]] = len(groups)
            for x in range(e["count"]):
                groups.append(dict(record="group", id=len(groups), model=m["name"], kind="expert",
                                   layer=e["layer"], expert=x, used=e["used"], stored=e["stored"]))
        models[m["name"]] = dict(record="model", name=m["name"], profile=m["profile"],
                                 layers=sorted(keep) if p["expert_layers"] else None,
                                 groups=[first, len(groups)], dense=dense, experts=experts)
    return groups, models


def schedule(params, seed):
    """Episodes: A takes each library model once per round, in seeded order; B is a seeded other."""
    names = [m["name"] for m in params["library"]]
    order = Draws(seed, "A")
    other = Draws(seed, "B")
    episodes = []
    for _ in range(params["rounds"]):
        for a in order.permutation(names):
            rest = [n for n in names if n != a]
            episodes.append((a, rest[other.below(len(rest))]))
    return episodes


def build_access(params, models, pools, seed):
    """The budget-independent access stream: requests, leases and per-layer uses."""
    events, cursor, leases, shrunk = [], collections.Counter(), 0, False
    t, probe = params["episode"], params["shrink_bytes"]
    for number, (a, b) in enumerate(schedule(params, seed)):
        for role, offset in zip(t["roles"], t["arrival_s"]):
            name = a if role == "A" else b
            if probe and shrunk != (role == "B"):
                events.append(dict(ev="shrink" if role == "B" else "grow", bytes=probe))
                shrunk = role == "B"
            m = models[name]
            request = dict(ev="request", model=name, episode=number, role=role,
                           arrival_s=number * t["span_s"] + offset)
            if m["experts"]:
                pool = pools[m["profile"]]
                q = pool[cursor[name] % len(pool)]
                cursor[name] += 1
                request.update(source=q["source"], prompt=q["prompt"], reused=q["reused"], decode=q["decode"])
            else:
                request.update(source="dense")
                q = dict(steps=[])
            events.append(request)
            events.append(dict(ev="lease", lease=leases, groups=m["dense"]))
            for step in q["steps"]:
                for layer, union in step:
                    if layer in m["experts"]:
                        base = m["experts"][layer]
                        events.append(dict(ev="use", groups=[base + e for e in union]))
            events.append(dict(ev="release", lease=leases))
            leases += 1
    if shrunk:
        events.append(dict(ev="grow", bytes=probe))
    return events


class Reference:
    """Fragmentation-free byte budget with LRU over unprotected whole groups.

    A group's recency is its latest lease, use or release; within one event
    the lower id counts as older. The evict and restore events it adds are
    the victims every design evicts first and the ideal restore stream."""

    def __init__(self, groups, budget):
        self.size = [g["stored"] for g in groups]
        self.budget, self.used, self.peak, self.shrunk = budget, 0, 0, 0
        self.resident = collections.OrderedDict()
        self.leases, self.protected, self.protected_bytes = {}, collections.Counter(), 0

    def need(self, ids):
        limit = self.budget - self.shrunk
        if self.protected_bytes + sum(self.size[i] for i in ids if not self.protected[i]) > limit:
            raise ValueError("an access does not fit the budget")
        missing = [i for i in ids if i not in self.resident]
        incoming = sum(self.size[i] for i in missing)
        excess, evicted, wanted = self.used + incoming - limit, [], set(ids)
        for i in self.resident:
            if excess <= 0:
                break
            if i not in wanted and not self.protected[i]:
                evicted.append(i)
                excess -= self.size[i]
        for i in evicted:
            del self.resident[i]
            self.used -= self.size[i]
        for i in missing:
            self.resident[i] = True
            self.used += self.size[i]
        self.peak = max(self.peak, self.used)
        self.touch(ids)
        out = []
        if evicted:
            out.append(dict(ev="evict", groups=evicted))
        if missing:
            out.append(dict(ev="restore", groups=missing))
        return out

    def touch(self, ids):
        for i in ids:
            self.resident.move_to_end(i)

    def apply(self, event):
        ids = event.get("groups")
        if event["ev"] == "lease":
            out = self.need(ids)
            self.leases[event["lease"]] = ids
            for i in ids:
                self.protected_bytes += self.size[i] if not self.protected[i] else 0
                self.protected[i] += 1
            return out
        if event["ev"] == "release":
            ids = self.leases.pop(event["lease"])
            for i in ids:
                self.protected[i] -= 1
                if not self.protected[i]:
                    del self.protected[i]
                    self.protected_bytes -= self.size[i]
            self.touch(ids)
            return []
        if event["ev"] == "use":
            return self.need(ids)
        if event["ev"] == "shrink":
            self.shrunk += event["bytes"]
            return self.need([])
        if event["ev"] == "grow":
            self.shrunk -= event["bytes"]
        return []


def unique_bytes(groups, events):
    touched = {i for e in events if e["ev"] in ("lease", "use") for i in e["groups"]}
    return sum(groups[i]["stored"] for i in touched), len(touched)


def budget_for(unique, ratio, granule):
    num, den = ratio
    return unique * den // num // granule * granule


def generate(params_path, library_path, evidence_path, captures, sessions, role, out):
    """Validate every input, then write OUT; returns the manifest's SHA-256."""
    identity = dict(generator_sha256=sha256_file(__file__), params_sha256=sha256_file(params_path),
                    library_sha256=sha256_file(library_path))
    params = json.loads(Path(params_path).read_text())
    library = json.loads(Path(library_path).read_text())
    evidence = json.loads(Path(evidence_path).read_text())
    profiles = check_params(params, library)
    check_sessions(params, sessions, evidence)
    pools, inputs = load_pools(params, profiles, evidence, captures)
    return write(params, library, profiles, pools, inputs, role, out, identity)


def write(params, library, profiles, pools, inputs, role, out, identity):
    seed = params["seeds"][role]
    groups, models = build_groups(params, profiles)
    events = build_access(params, models, pools, seed)
    access = hashlib.sha256()
    for e in events:
        access.update(line(e))
    unique, touched = unique_bytes(groups, events)
    out = Path(out)
    out.mkdir(parents=True)
    files = {}
    for ratio in params["ratios"]:
        budget = budget_for(unique, ratio, params["budget_granule"])
        ref = Reference(groups, budget)
        name = f"trace-r{ratio[0]}-{ratio[1]}.jsonl"
        header = dict(record="header", format=FORMAT, version=VERSION, role=role, seed=seed,
                      ratio=ratio, budget_bytes=budget, unique_bytes=unique, touched_groups=touched,
                      chunk_bytes=library["chunk_bytes"], file_align=library["file_align"],
                      **identity, captured_events=inputs,
                      access_sha256=access.hexdigest(), groups=len(groups), models=len(models))
        with open(out / name, "xb") as f:
            f.write(line(header))
            for m in models.values():
                f.write(line({k: m[k] for k in ("record", "name", "profile", "layers", "groups")}))
            for g in groups:
                f.write(line(g))
            for e in events:
                for extra in ref.apply(e):
                    f.write(line(extra))
                f.write(line(e))
        files[name] = dict(sha256=sha256_file(out / name), budget_bytes=budget, reference_peak_bytes=ref.peak)
    manifest = dict(format=FORMAT + "-manifest", version=VERSION, role=role, seed=seed, **identity,
                    unique_bytes=unique, access_sha256=access.hexdigest(), files=files)
    (out / "manifest.json").write_bytes(json.dumps(manifest, indent=1).encode() + b"\n")
    return sha256_file(out / "manifest.json")


# ------------------------------------------------------------------ checks

def validate(path):
    """Schema and replay check of one trace file; returns its counters."""
    with open(path, "rb") as f:
        records = (json.loads(raw) for raw in f)
        header = next(records)
        keys = {"record", "format", "version", "role", "seed", "ratio", "budget_bytes", "unique_bytes",
                "touched_groups", "chunk_bytes", "file_align", "generator_sha256", "params_sha256",
                "library_sha256", "captured_events", "access_sha256", "groups", "models"}
        if header.get("record") != "header" or set(header) != keys or header["format"] != FORMAT or header["version"] != VERSION:
            raise ValueError("bad header")
        align, budget = header["file_align"], header["budget_bytes"]
        models, groups = [], []
        for _ in range(header["models"]):
            m = next(records)
            if m.get("record") != "model" or set(m) != {"record", "name", "profile", "layers", "groups"}:
                raise ValueError("bad model record")
            models.append(m)
        for i in range(header["groups"]):
            g = next(records)
            if g.get("record") != "group" or g.get("id") != i or set(g) != {"record", "id", "model", "kind", "layer", "expert", "used", "stored"}:
                raise ValueError(f"bad group record {i}")
            if not (0 < g["used"] <= g["stored"] and g["stored"] % align == 0 and g["stored"] - g["used"] < align):
                raise ValueError(f"group {i}: stored bytes are not used bytes aligned to {align}")
            if g["kind"] == "expert" and (g["layer"] is None or g["expert"] is None):
                raise ValueError(f"group {i}: expert without layer")
            groups.append(g)
        spans = [range(*m["groups"]) for m in models]
        if [s.start for s in spans] != [0] + [s.stop for s in spans[:-1]] or spans[-1].stop != len(groups):
            raise ValueError("model group ranges do not tile the groups")
        resident, used, leases, protected, shrunk = set(), 0, {}, collections.Counter(), 0
        pending_evict, pending_restore = None, None
        access, touched = hashlib.sha256(), set()
        counts = collections.Counter()
        for e in records:
            ev = e.get("ev")
            if ev not in EVENTS:
                raise ValueError(f"unknown event {ev!r}")
            counts[ev] += 1
            ids = e.get("groups", [])
            if any(type(i) is not int or not 0 <= i < len(groups) for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("bad group ids")
            if ev == "evict":
                if pending_evict is not None or pending_restore is not None:
                    raise ValueError("evict out of order")
                for i in ids:
                    if i not in resident or protected[i]:
                        raise ValueError("evicts an absent or leased group")
                    resident.discard(i)
                    used -= groups[i]["stored"]
                pending_evict = ids
                continue
            if ev == "restore":
                if pending_restore is not None:
                    raise ValueError("restore out of order")
                pending_restore = ids
                continue
            access.update(line(e))
            if ev == "shrink":
                shrunk += e["bytes"]
            if ev in ("lease", "use", "shrink"):
                missing = [i for i in ids if i not in resident]
                if missing != (pending_restore or []):
                    raise ValueError("restore is not exactly the access's missing groups")
                incoming = sum(groups[i]["stored"] for i in missing)
                if pending_evict and used + groups[pending_evict[-1]]["stored"] + incoming <= budget - shrunk:
                    raise ValueError("evicts more than the access needs")
                for i in missing:
                    resident.add(i)
                    used += groups[i]["stored"]
                if used > budget - shrunk:
                    raise ValueError("reference exceeds its budget")
                touched.update(ids)
                if ev == "lease":
                    if e["lease"] in leases:
                        raise ValueError("lease reused")
                    leases[e["lease"]] = ids
                    protected.update(ids)
            elif pending_evict is not None or pending_restore is not None:
                raise ValueError("evict/restore not followed by an access")
            elif ev == "release":
                protected.subtract(leases.pop(e["lease"]))
            elif ev == "grow":
                shrunk -= e["bytes"]
                if shrunk < 0:
                    raise ValueError("grows more than it shrank")
            pending_evict = pending_restore = None
        if leases or shrunk or pending_evict is not None or pending_restore is not None:
            raise ValueError("trace ends with open leases or pending events")
        if access.hexdigest() != header["access_sha256"]:
            raise ValueError("access stream differs from its recorded hash")
        if sum(groups[i]["stored"] for i in touched) != header["unique_bytes"] or len(touched) != header["touched_groups"]:
            raise ValueError("unique bytes differ from the header")
        return header, dict(counts)


def verify(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    summary = {}
    for name, record in manifest["files"].items():
        if sha256_file(out / name) != record["sha256"]:
            raise ValueError(f"{name}: hash mismatch")
        header, counts = validate(out / name)
        if header["access_sha256"] != manifest["access_sha256"] or header["budget_bytes"] != record["budget_bytes"]:
            raise ValueError(f"{name}: disagrees with the manifest")
        summary[name] = dict(budget_bytes=header["budget_bytes"], unique_bytes=header["unique_bytes"], **counts)
    return summary


def main(argv):
    if len(argv) == 3 and argv[1] == "verify":
        print(json.dumps(verify(argv[2]), indent=1))
    elif len(argv) == 9 and argv[1] == "generate":
        if argv[7] not in ("primary", "confirmation"):
            raise SystemExit("ROLE is primary or confirmation")
        digest = generate(*argv[2:])
        manifest = json.loads((Path(argv[8]) / "manifest.json").read_text())
        print(json.dumps(dict(manifest_sha256=digest, **{k: v for k, v in manifest.items() if k != "files"}), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
