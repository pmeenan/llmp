#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Which EXL3 fixture an artifact, or an FP64 oracle, is (reference only; no numpy, no GPU).

Reference-side experiment tooling for the backend proof's P3; it shares no code with llmpalooza. A native
run's manifests carry the fixture only as the label its command line gave. The label is never trusted:
- an artifact is the fixture whose checkpoint it was prepared from, as op_tier_e.py reads it: the
  installed directory's manifest.json must hash to the directory's name (the artifact id), and its
  source "model.safetensors" SHA-256 must be exactly one fixture's probe weight_sha256 in the operation
  plan record (exl3-op-plan-g.json);
- an oracle is the fixture whose recorded output SHA-256 it has
  (../backend-proof-p0/oracle-envelope.json, oracle.output_sha256).
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "exl3-op-plan-g.json"
ENVELOPE = HERE.parent / "backend-proof-p0" / "oracle-envelope.json"


class Unidentified(ValueError):
    """The artifact or oracle is not exactly one fixture's."""


def artifact_fixture(root, record=RECORD):
    """(fixture, artifact id) of an installed artifact directory."""
    root = Path(root).resolve()
    manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != root.name:
        raise Unidentified(f"{root}: manifest.json does not hash to the artifact id")
    source = {s.get("name"): s.get("sha256") for s in json.loads(manifest).get("source", [])}
    checkpoint = source.get("model.safetensors")
    probes = json.loads(Path(record).read_text())["provenance"]["probes"]
    fixtures = [f for f, p in probes.items() if checkpoint is not None and p.get("weight_sha256") == checkpoint]
    if len(fixtures) != 1:
        raise Unidentified(f"{root.name}: prepared from checkpoint {checkpoint}, which is not exactly one "
                           f"fixture's ({sorted(probes)})")
    return fixtures[0], root.name


def oracle_fixture(path, envelope=ENVELOPE):
    """The fixture whose FP64 oracle the file at path is."""
    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    recorded = json.loads(Path(envelope).read_text())["oracle"]["output_sha256"]
    fixtures = [f for f, d in recorded.items() if d == digest]
    if len(fixtures) != 1:
        raise Unidentified(f"{path} ({digest}) is not exactly one fixture's recorded oracle")
    return fixtures[0]
