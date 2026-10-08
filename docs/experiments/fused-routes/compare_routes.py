# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare capture outputs exactly; differences are counted, never tolerated."""

import json

ROUTE_KEYS = ("request", "step", "phase", "layer", "output_only")


def read_lines(path, fields):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != fields:
            raise ValueError(f"{path.name}:{number}: expected {fields} fields")
        rows.append(tuple(parts))
    if not rows:
        raise ValueError(f"{path.name}: empty")
    return rows


def compare_records(left, right):
    """Compare two record lists that must describe the same positions."""
    if len(left) != len(right):
        raise ValueError("record count mismatch")
    different = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    return {"records": len(left), "different_records": len(different),
            "first_different_record": different[0] if different else None}


def compare_outputs(left, right):
    """Compare two arm directories' predictions (request, argmax) and logit hashes
    (request, step, FNV-1a of every vocabulary logit)."""
    predictions = [read_lines(d / "predictions", 2) for d in (left, right)]
    hashes = [read_lines(d / "logits", 3) for d in (left, right)]
    for rows, keyed in ((predictions, 1), (hashes, 2)):
        if [r[:keyed] for r in rows[0]] != [r[:keyed] for r in rows[1]]:
            raise ValueError("records describe different positions")
    return {"predictions": compare_records(*predictions), "logit_hashes": compare_records(*hashes)}


def read_routes(path):
    events = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") != "routes":
            continue
        rows = record["routes"]
        if not rows or any(not row or len(row) != len(set(row)) for row in rows):
            raise ValueError("invalid route rows")
        events.append(record)
    if not events:
        raise ValueError(f"{path.name}: no route events")
    return events


def compare_routes(left, right):
    """Compare route events position by position; every identity key must agree."""
    if len(left) != len(right):
        raise ValueError("route event count mismatch")
    different_events = different_rows = different_sets = rows = 0
    for a, b in zip(left, right):
        # The study's earlier capture format omits output_only, which was then always false.
        if tuple(a.get(k, False) for k in ROUTE_KEYS) != tuple(b.get(k, False) for k in ROUTE_KEYS):
            raise ValueError("route events describe different positions")
        if len(a["routes"]) != len(b["routes"]):
            raise ValueError("route row count mismatch")
        changed = sum(x != y for x, y in zip(a["routes"], b["routes"]))
        rows += len(a["routes"])
        different_rows += changed
        different_sets += sum(set(x) != set(y) for x, y in zip(a["routes"], b["routes"]))
        different_events += bool(changed)
    # Ordered rows compare the exact top-k tensor; sets compare the selected experts.
    return {"events": len(left), "different_events": different_events, "rows": rows,
            "different_ordered_rows": different_rows, "different_expert_sets": different_sets}
