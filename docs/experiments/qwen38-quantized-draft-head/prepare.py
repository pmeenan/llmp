# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Repack an authenticated selected BF16 MTP head as pinned GGML Q4_1.

All other resource/expert slices must remain byte-identical. This offline
experimental adapter reuses the existing artifact planner/writer/verifier;
it never modifies the source artifact or publishes an unverified result.
"""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LAYOUT = HERE.parent / "artifact-layout"
sys.path.insert(0, str(LAYOUT))
import layout
import modelopt_qwen38 as modelopt


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def pieces(path, offset, length):
    with path.open("rb") as file:
        file.seek(offset)
        while length:
            data = file.read(min(length, 4 << 20))
            if not data:
                raise ValueError("truncated prepared source")
            length -= len(data)
            yield data


def digest_range(span):
    digest = hashlib.sha256()
    for part in pieces(*span):
        digest.update(part)
    return digest.hexdigest()


def spans(root, index):
    result = {}
    for resource in index["resources"]:
        group = index["groups"][resource["group"]]
        shard = index["shards"][group["shard"]]
        result[resource["name"]] = (root / shard["path"],
            shard["data_offset"] + group["offset"] + resource["offset"], resource["bytes"])
    for array in index["expert_arrays"]:
        for expert in range(array["count"]):
            group = index["groups"][array["first_group"] + expert]
            shard = index["shards"][group["shard"]]
            result[f'{array["name"]}#{expert}'] = (root / shard["path"],
                shard["data_offset"] + group["offset"] + array["group_offset"], array["slice_bytes"])
    return result


def reconstruct(index, source_spans, quantized):
    groups = [dict(kind=g["kind"], layer=g["layer"], expert=g["expert"], members=[])
              for g in index["groups"]]
    arrays = []

    def member(name, rep, roles, nbytes, readable, access=None):
        span = (quantized, 0, 75475200) if name == "draft_output.weight" else source_spans[name]
        if name == "draft_output.weight":
            rep = dict(family="ggml", type="Q4_1", ne=[2560, 47172])
            nbytes = layout.repr_bytes(rep, name)
            readable = layout.readable_for(rep, nbytes)
        value = dict(name=name, repr=copy.deepcopy(rep), roles=roles,
                     nbytes=nbytes, readable=readable,
                     produce=lambda _unused, span=span: pieces(*span))
        if access:
            value["access"] = copy.deepcopy(access)
        return value

    for r in index["resources"]:
        m = member(r["name"], r["repr"], r["roles"], r["bytes"], r["readable_bytes"], r.get("access"))
        m["original_offset"] = r["offset"]
        groups[r["group"]]["members"].append(m)
    for number, a in enumerate(index["expert_arrays"]):
        array = dict(name=a["name"], layer=a["layer"], count=a["count"],
                     first_group=a["first_group"], repr=copy.deepcopy(a["repr"]),
                     slice_bytes=a["slice_bytes"], readable=a["readable_bytes"], members=[])
        for expert in range(a["count"]):
            m = member(f'{a["name"]}#{expert}', a["repr"], [], a["slice_bytes"], a["readable_bytes"])
            m["array"] = number
            m["original_offset"] = a["group_offset"]
            groups[a["first_group"] + expert]["members"].append(m)
            array["members"].append(m)
        arrays.append(array)
    for g in groups:
        g["members"].sort(key=lambda m: m.pop("original_offset"))
    plan = modelopt._place(layout, groups, arrays, layout.SHARD_TARGET)
    plan["arch"] = "qwen4exp-mtp"
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--quantizer-sha256", required=True)
    parser.add_argument("--out-store", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise ValueError("receipt must not already exist")
    if args.source.is_symlink() or args.out_store.is_symlink():
        raise ValueError("source/store roots must not be symlinks")
    source = args.source.resolve()
    if args.out_store.resolve().is_relative_to(source) or args.receipt.resolve().is_relative_to(source):
        raise ValueError("preparation outputs must be outside the immutable source")
    manifest, index = layout.verify(source, deep=False, expected_id=args.source_id)
    if manifest["model"]["architecture"] != "qwen4exp-mtp":
        raise ValueError("source is not an MTP drafter")
    resources = {r["name"]: r for r in index["resources"]}
    head, ids = resources["draft_output.weight"], resources["draft_output.ids"]
    if head["repr"] != dict(family="ggml", type="BF16", ne=[2560, 47172]):
        raise ValueError("source head is not the qualified BF16 selection")
    if ids["repr"] != dict(family="ggml", type="I32", ne=[1, 47172]):
        raise ValueError("source token map differs")
    old_spans = spans(source, index)
    # Authenticate every copied range in the SAME reads as its manifest
    # shard hash, including the replaced head. A later source reread is
    # not evidence that those bytes belong to the pinned parent artifact.
    identities, expected_ranges = layout._identity_pass(
        [source / f["path"] for f in manifest["files"]], {"groups": []},
        [(path, offset, length, name) for name, (path, offset, length) in old_spans.items()])
    for f in manifest["files"]:
        if identities[Path(f["path"]).name] != (f["bytes"], f["sha256"]):
            raise ValueError("prepared source identity changed: " + f["path"])
    map_bytes = b"".join(pieces(*old_spans["draft_output.ids"]))
    if hashlib.sha256(map_bytes).hexdigest() != expected_ranges["draft_output.ids"]:
        raise ValueError("actual token map differs from authenticated parent")
    token_ids = struct.unpack("<47172i", map_bytes)
    if any(t < 0 or t >= 248320 or (i and token_ids[i-1] >= t)
           for i, t in enumerate(token_ids)):
        raise ValueError("invalid ascending token map")
    helper = args.quantizer.resolve()
    if sha(helper) != args.quantizer_sha256:
        raise ValueError("quantizer identity differs")
    sources = {Path(f["path"]).name: (f["bytes"], f["sha256"]) for f in manifest["files"]}
    sources["manifest.json"] = (len(layout.dumps(manifest)), args.source_id)
    if len(sources) != len(manifest["files"]) + 1:
        raise ValueError("source base filenames collide")
    args.out_store.mkdir(exist_ok=True, parents=True)
    if args.out_store.resolve() == source:
        raise ValueError("output store cannot be the source artifact")
    staging = layout._staging_dir(args.out_store)
    job = "q4-head-" + args.source_id + "-" + args.quantizer_sha256[:16]
    lock = layout._try_lock(staging / (job + ".lock"))
    if lock is None:
        raise ValueError("preparation is already active")
    try:
        work = staging / job
        work.mkdir()  # preserve a failed attempt; no silent retry or deletion
        quantized = work / "head.q4_1"
        path, offset, _ = old_spans["draft_output.weight"]
        result = subprocess.run([str(helper), str(path), str(offset), "2560", "47172", str(quantized)],
                                check=True, timeout=120, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        witness = re.fullmatch(r"QWEN_SELECTED_HEAD_QUANTIZED rows=47172 width=2560 bytes=75475200 input_sha256=([0-9a-f]{64})\n", result.stdout)
        if witness is None or witness[1] != expected_ranges["draft_output.weight"]:
            raise ValueError("quantizer's actual BF16 input differs from authenticated parent")
        if quantized.stat().st_size != 75475200 or sha(helper) != args.quantizer_sha256:
            raise ValueError("quantized payload/helper identity differs")
        provenance = dict(parent_artifact=args.source_id, ggml_commit="d81235049384534c167caea52b85a694f6103d14",
                          quantizer_sha256=args.quantizer_sha256, type="Q4_1", shape=[2560, 47172],
                          source_head_sha256=expected_ranges["draft_output.weight"],
                          quantized_head_sha256=sha(quantized), ids_sha256=hashlib.sha256(map_bytes).hexdigest())
        provenance_path = work / "q4-head-provenance.json"
        provenance_path.write_bytes(layout.dumps(provenance))
        sources[provenance_path.name] = (provenance_path.stat().st_size, sha(provenance_path))
        metas = [(f["path"], source / f["path"]) for f in manifest["files"]
                 if f["role"] == "source-metadata"]
        metas.append(("meta/q4-head-provenance.json", provenance_path))
        plan = reconstruct(index, old_spans, quantized)
        artifact = work / "artifact"
        converter = dict(name="qwen38-quantized-draft-head/prepare.py",
                         version="q4_1-ref-d8123504-" + args.quantizer_sha256[:16])
        mbytes = modelopt.write(layout, plan, None, artifact, converter, sources, metas, workers=1)
        identity = hashlib.sha256(mbytes).hexdigest()
        _, new_index = layout.verify(artifact, deep=True, expected_id=identity)
        new_spans = spans(artifact, new_index)
        if set(new_spans) != set(old_spans):
            raise ValueError("resource/slice set changed")
        unchanged = {}
        for name, old in old_spans.items():
            if name == "draft_output.weight":
                if digest_range(new_spans[name]) != provenance["quantized_head_sha256"]:
                    raise ValueError("quantized member differs")
                continue
            before = expected_ranges[name]
            if before != digest_range(new_spans[name]):
                raise ValueError("unchanged resource differs: " + name)
            unchanged[name] = before
        if layout.verify(source, deep=True, expected_id=args.source_id) != (manifest, index):
            raise ValueError("prepared source changed")
        for directory in [artifact / "data", artifact / "meta", artifact]:
            layout._fsync_dir(directory)
        final = args.out_store / identity
        if final.exists() or final.is_symlink():
            raise ValueError("destination already exists; verify explicitly rather than overwrite")
        os.rename(artifact, final)
        layout._fsync_dir(staging)
        layout._fsync_dir(args.out_store)
        # Keep temporary preparation evidence outside the published artifact.
        receipt = dict(artifact_id=identity, artifact=str(final), provenance=provenance,
                       unchanged_resource_sha256=unchanged, source_unchanged=True)
        with args.receipt.open("x") as file:
            json.dump(receipt, file, indent=2)
            file.write("\n")
        print(json.dumps(dict(artifact=str(final), id=identity)))
    finally:
        os.close(lock)


if __name__ == "__main__":
    main()
