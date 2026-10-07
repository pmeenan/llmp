#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""M3's importer: the M0 prototype (layout.py) run against pinned sources.

layout.py stays byte-for-byte the planner the retained-backing trace was
measured with (extract_library.py pins its SHA-256), so this file adds what
M3's imports need around it without copying or changing it:

- the sources must be the pinned download (D-054): each source file's name,
  size and SHA-256 must equal an entry of an M3 pins file
  (docs/experiments/fast-swap/pins.json), checked by layout.build in the same
  pass that hashes the bytes it copies;
- the converter identity names this file, its version and layout.py's
  SHA-256, so an artifact records the code that wrote it;
- layout.py is refused unless it is the pinned planner;
- a multi-component pipeline (D-089; docs/artifact-format.md, "Compositions")
  is one ordinary v0 artifact per component, imported from the component's
  safetensors folder, plus a composition: a small content-addressed document
  naming each component's artifact by ID, with the pipeline's own metadata
  (model_index.json, the scheduler and processor files) kept verbatim.

  python3 import_m3.py build OUT PINS MODEL_ID SOURCE...
  python3 import_m3.py drafter OUT PINS MODEL_ID SOURCE... [--draft-vocab-ids FILE]
  python3 import_m3.py component OUT PINS MODEL_ID CHECKPOINT ROLE [--meta REL]...
  python3 import_m3.py compose OUT PINS MODEL_ID CHECKPOINT ROLE=ID... [--meta REL]...
  python3 import_m3.py verify ARTIFACT
  python3 import_m3.py verify-composition COMPOSITION [--store OUT]

`build` takes GGUF sources (M3's DeepSeek V4 Flash) through layout.py's
plan and writer, exactly as m3-1 wrote them. Its safetensors sources are
Qwen3.8 Flash Next's ModelOpt checkpoint, whose bytes are repacked on the
way (modelopt_qwen38.py): its shards and the config.json beside them are
checked against the pins, and layout.py's container, index and verifier
write and check the artifact.

`drafter` imports Qwen3.8's MTP block as its own drafter artifact
(modelopt_qwen38.py plan_mtp; architecture qwen4exp-mtp) from the shards
that hold it (the checkpoint's last) and the config.json beside them, both
checked against the pins; the target artifact is not touched.

`component` imports CHECKPOINT/ROLE/*.safetensors, whose config.json
(CHECKPOINT/ROLE/config.json) decides the architecture:
`model_type` for a transformers model, else a diffusers `_class_name`; REL
paths are relative to CHECKPOINT and kept verbatim. Sources are matched to
their pins by path relative to CHECKPOINT, since a pipeline's components
share base names (config.json). Every other rule is layout.py's.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYOUT_SHA256 = "a0d1980a9eddd1adf60863cf7b825396700691fd17b3cd62ca063fda2e0b6809"
CONVERTER_NAME = "artifact-layout/import_m3.py"
# m3-1: GGUF sources. m3-2 adds safetensors components and compositions; a
# GGUF import writes exactly what m3-1 wrote, so it keeps m3-1 (and the
# DeepSeek artifact its ID).
IMPORTER_VERSION = {"gguf": "m3-1", "component": "m3-2", "composition": "m3-2"}

# Component import policy (grouping labels only; the runtime binds by role):
# each layer's tensors form one group, the token table is a row table, the
# rest pack into the head or a global group.
COMPONENT_POLICY = {
    # Qwen3-VL's text path; the vision tower and lm_head stay in the head
    # group, which the text-to-image path never reads.
    "text_encoder": {"layer": r"model\.language_model\.layers\.(\d+)\.",
                     "tables": ("model.language_model.embed_tokens.weight",)},
    "transformer": {"layer": r"transformer_blocks\.(\d+)\.", "tables": ()},
    # The decoder's up blocks; the encoder (image editing only) and the rest
    # of the decoder pack into one global group.
    "vae": {"layer": r"decoder\.up_blocks\.(\d+)\.", "tables": ()},
}
ROLE = re.compile(r"[a-z][a-z0-9_]{0,63}")
COMPOSITION_FORMAT = "jitllm-composition"
COMPOSITION_VERSION = 0
COMPOSITION_KEYS = {"format", "format_version", "experimental", "model", "components", "source",
                    "converter", "files"}
MAX_COMPONENTS = 16


def load_layout():
    path = HERE / "layout.py"
    code = path.read_bytes()
    if hashlib.sha256(code).hexdigest() != LAYOUT_SHA256:
        raise SystemExit("layout.py is not the pinned M0 planner")
    spec = importlib.util.spec_from_file_location("layout", path)
    module = importlib.util.module_from_spec(spec)
    # Run the bytes that were hashed, not the file read again.
    exec(compile(code, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def load_modelopt():
    """modelopt_qwen38.py, from the bytes its digest (in the converter
    version) is taken over."""
    path = HERE / "modelopt_qwen38.py"
    code = path.read_bytes()
    digest = hashlib.sha256(code).hexdigest()
    loaded = sys.modules.get("modelopt_qwen38")
    if loaded is not None and getattr(loaded, "_import_m3_digest", None) == digest:
        return loaded, digest  # one module object, so its functions pickle by name
    spec = importlib.util.spec_from_file_location("modelopt_qwen38", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modelopt_qwen38"] = module  # worker processes import it by name
    exec(compile(code, str(path), "exec"), module.__dict__)  # noqa: S102
    module._import_m3_digest = digest
    return module, digest


def converter(kind="gguf"):
    # IMPORTER_VERSION changes with anything here that changes what is
    # written; layout.py's identity is its pinned digest.
    return {"name": CONVERTER_NAME, "version": f"{IMPORTER_VERSION[kind]}+layout-{LAYOUT_SHA256[:16]}"}


def _model_pins(pins_path, model_id):
    pins = json.loads(Path(pins_path).read_text())
    models = [m for m in pins.get("models", []) if m.get("id") == model_id]
    if len(models) != 1:
        raise SystemExit(f"{model_id}: not exactly one model of that id in {pins_path}")
    return models[0].get("files", [])


def pinned_sources(pins_path, model_id, source_paths):
    """The recorded identity (name -> SHA-256) of each source, from the pins.
    Refused if the model is unknown, a source is not one of its files (by
    base name), two sources share a name, or a size differs from the pin."""
    files = {}
    for f in _model_pins(pins_path, model_id):
        files.setdefault(Path(f["path"]).name, []).append(f)
    expected = {}
    for p in source_paths:
        name = Path(p).name
        if len(files.get(name, [])) != 1:
            raise SystemExit(f"{p}: not exactly one pinned file of {model_id} has this name")
        if name in expected:
            raise SystemExit(f"{name}: given twice")
        pin = files[name][0]
        size = Path(p).stat().st_size
        if size != pin["bytes"]:
            raise SystemExit(f"{p}: {size} bytes, pinned {pin['bytes']}")
        expected[name] = pin["sha256"]
    return expected


def _check_legacy_source(src, expected, architecture, filename, fixture, family, pin_label):
    """Closed preflight for the approved legacy checkpoint, before planning.

    The existing authenticated factual fixture is the one tensor/profile
    authority. The writer still hashes the whole source and reparses it before
    publication; this header check does not authenticate weight payloads.
    Other GGUF architectures retain their existing generic path.
    """
    if src["meta"].get("general.architecture") != architecture and filename not in expected:
        return
    fixture = HERE.parents[2] / fixture
    contract = json.loads(fixture.read_text())
    identity = contract["source"]
    name = identity["file"]
    if expected != {name: identity["full_sha256"]} or len(src["parts"]) != 1:
        raise ValueError(f"{family} import needs the single approved {pin_label} source pin")
    part = src["parts"][0]
    if (part["kind"] != "gguf" or Path(part["path"]).name != name or
            Path(part["path"]).stat().st_size != identity["full_bytes"]):
        raise ValueError(f"{family} import source identity differs")
    for key, value in contract["metadata"].items():
        if key == "vocab_count":
            tokens = src["meta"].get("tokenizer.ggml.tokens")
            actual = (tokens["array_len"] if type(tokens) is dict and
                      set(tokens) == {"array_len"} else None)
        else:
            actual = src["meta"].get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"{family} import metadata differs: {key}")
    tensors = src["tensors"]
    if len(tensors) != len(contract["tensors"]):
        raise ValueError(f"{family} import tensor count differs")
    by_name = {t["name"]: t for t in tensors}
    if len(by_name) != len(tensors):
        raise ValueError(f"{family} import duplicate tensor")
    for wanted in contract["tensors"]:
        tensor = by_name.get(wanted["name"])
        if (tensor is None or tensor["family"] != "ggml" or
                tensor["dtype"] != wanted["type"] or tensor["ne"] != wanted["ne"] or
                tensor["nbytes"] != wanted["bytes"] or
                tensor["offset"] != contract["data_offset"] + wanted["relative_offset"] or
                tensor["path"] != part["path"]):
            raise ValueError(f"{family} import tensor differs: {wanted['name']}")
    if (part["header_len"] != contract["complete_header_bytes"] or
            part["header_sha256"] != contract["complete_header_sha256"]):
        raise ValueError(f"{family} import authenticated header differs")


def check_gemma3_source(src, expected):
    _check_legacy_source(src, expected, "gemma3", "gemma-3-4b-it-qat-Q4_0.gguf",
                         "tests/unit/data/gemma3/gemma3_4b_qat.json", "Gemma3", "QAT")


def check_gemma2_source(src, expected):
    _check_legacy_source(src, expected, "gemma2", "gemma-2-2b-it-Q8_0.gguf",
                         "tests/unit/data/gemma2/gemma2_2b_q8.json", "Gemma2", "Q8_0")


def pinned_by_path(pins_path, model_id, checkpoint, rels):
    """name -> SHA-256 for files given by path relative to CHECKPOINT, each
    matched to the pin of that exact path. Refused if a path is not pinned,
    escapes the checkpoint, is given twice, two share a base name (an
    artifact records base names), or a size differs from the pin."""
    pins = {f["path"]: f for f in _model_pins(pins_path, model_id)}
    expected, seen = {}, set()
    for rel in rels:
        rel = str(rel)
        if rel.startswith("/") or ".." in Path(rel).parts or rel not in pins:
            raise SystemExit(f"{rel}: not a pinned file of {model_id}")
        if rel in seen:
            raise SystemExit(f"{rel}: given twice")
        seen.add(rel)
        name = Path(rel).name
        if name in expected:
            raise SystemExit(f"{name}: two files of one import share this base name")
        path = Path(checkpoint) / rel
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"{path}: not a regular file")
        if path.stat().st_size != pins[rel]["bytes"]:
            raise SystemExit(f"{path}: {path.stat().st_size} bytes, pinned {pins[rel]['bytes']}")
        expected[name] = pins[rel]["sha256"]
    return expected


def component_layout(layout, role):
    """layout.py with the component's grouping policy and its safetensors
    architecture rule: the config's `model_type`, else a diffusers
    `_class_name` (layout.py itself reads only `model_type`). The same
    module parses the plan and re-parses the hashed sources, so both see
    the same rule."""
    if role not in COMPONENT_POLICY:
        raise SystemExit(f"{role}: no component policy (one of {sorted(COMPONENT_POLICY)})")
    policy = COMPONENT_POLICY[role]
    layout.LAYER = re.compile(policy["layer"])
    layout.ROW_TABLES = set(policy["tables"])
    original = layout.read_safetensors

    def read_safetensors(path):
        part = original(path)
        if "general.architecture" not in part["meta"] and part["config"]:
            doc = json.loads(Path(part["config"]).read_bytes())
            name = doc.get("_class_name") if type(doc) is dict else None
            if isinstance(name, str):
                part["meta"]["general.architecture"] = name
        return part

    layout.read_safetensors = read_safetensors
    return layout


def import_component(layout, out, pins, model_id, checkpoint, role, meta_rels):
    layout = component_layout(layout, role)
    checkpoint = Path(checkpoint)
    folder = checkpoint / role
    if folder.is_symlink() or not folder.is_dir():
        raise SystemExit(f"{folder}: not a component folder")
    shards = sorted(p for p in folder.iterdir() if p.suffix == ".safetensors")
    if not shards:
        raise SystemExit(f"{folder}: no safetensors")
    rels = [f"{role}/{p.name}" for p in shards] + [f"{role}/config.json"] + list(meta_rels)
    expected = pinned_by_path(pins, model_id, checkpoint, rels)
    paths = [str(p) for p in shards]
    src = layout.load_sources(paths)
    p = layout.plan(src, tie_check=True)
    print(json.dumps(layout.stats(p)), flush=True)
    metas = [(f"meta/{Path(r).name}", str(checkpoint / r)) for r in meta_rels]
    return layout.build(p, src, out, paths, metas, converter=converter("component"),
                        expected_sources=expected)


# ---------------------------------------------------------------- compositions

def composition_doc(layout, components, sources, files, architecture):
    return {
        "format": COMPOSITION_FORMAT, "format_version": COMPOSITION_VERSION, "experimental": True,
        "model": {"architecture": architecture},
        "components": sorted(components, key=lambda c: c["role"]),
        "source": sorted(sources, key=lambda s: s["name"]),
        "converter": converter("composition"),
        "files": sorted(files, key=lambda f: f["path"]),
    }


def compose(layout, out, pins, model_id, checkpoint, bindings, meta_rels):
    """Write a composition naming each component's artifact (already
    published in OUT, verified here) and keeping the pipeline's metadata.
    The architecture is model_index.json's `_class_name`; every component
    role must be one of its entries."""
    out, checkpoint = Path(out), Path(checkpoint)
    rels = ["model_index.json"] + [r for r in meta_rels if r != "model_index.json"]
    expected = pinned_by_path(pins, model_id, checkpoint, rels)
    index_bytes = (checkpoint / "model_index.json").read_bytes()
    model_index = layout.strict_json(index_bytes, "model_index.json") if index_bytes.isascii() else None
    if type(model_index) is not dict or not isinstance(model_index.get("_class_name"), str):
        raise SystemExit("model_index.json: no pipeline _class_name")
    architecture = layout._str(model_index["_class_name"], "model_index._class_name", layout.NAME)
    components = []
    for binding in bindings:
        role, _, artifact_id = binding.partition("=")
        if not ROLE.fullmatch(role) or role not in model_index or not layout.HEX.fullmatch(artifact_id):
            raise SystemExit(f"{binding}: not ROLE=ARTIFACT_ID for an entry of model_index.json")
        if any(c["role"] == role for c in components):
            raise SystemExit(f"{role}: bound twice")
        manifest, _ = layout.verify(out / artifact_id, deep=False)
        components.append({"role": role, "artifact": artifact_id,
                           "architecture": manifest["model"]["architecture"]})
    if not components or len(components) > MAX_COMPONENTS:
        raise SystemExit(f"a composition binds 1 to {MAX_COMPONENTS} components")
    blobs, sources, files = {}, [], []
    for rel in rels:
        name = Path(rel).name
        data = (checkpoint / rel).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected[name]:
            raise SystemExit(f"{rel}: differs from its pin")
        blobs[name] = data
        sources.append({"name": name, "bytes": len(data), "sha256": digest})
        files.append({"path": f"meta/{name}", "role": "source-metadata", "bytes": len(data), "sha256": digest})
    doc = composition_doc(layout, components, sources, files, architecture)
    mbytes = layout.dumps(doc)
    comp_id = hashlib.sha256(mbytes).hexdigest()
    staging = layout._staging_dir(out)
    # As an import job: private staging under an exclusive lock, so two
    # writers of one composition never share a directory.
    lock = layout._try_lock(staging / f"compose-{comp_id}.lock")
    if lock is None:
        raise layout.ArtifactError("busy", f"compose {comp_id} is already running")
    try:
        work = staging / f"compose-{comp_id}"
        if work.is_symlink():
            work.unlink()
        elif work.exists():
            shutil.rmtree(work)
        (work / "meta").mkdir(parents=True)
        for name, data in blobs.items():
            layout._fsync_write(work / "meta" / name, data)
        layout._fsync_write(work / "manifest.json", mbytes)
        verify_composition(layout, work, expected_id=comp_id)  # never publish what we would reject
        for d in (work / "meta", work):
            layout._fsync_dir(d)
        final = out / comp_id
        if final.exists() or final.is_symlink():
            verify_composition(layout, final)  # raises if damaged; never silently reused
            shutil.rmtree(work)
        else:
            os.rename(work, final)
        layout._fsync_dir(staging)
        layout._fsync_dir(out)
        return final
    finally:
        os.close(lock)


def verify_composition(layout, root, expected_id=None, store=None):
    """Verify a composition; returns its manifest. Fails closed with
    layout.ArtifactError naming the rule, as layout.verify does: the
    directory (manifest.json and meta/ only, regular singly linked files),
    strict canonical JSON, its identity, the schema, the kept metadata as
    verbatim recorded sources, and with `store`, each component's artifact
    (shallow) with the architecture recorded for it."""
    E = layout.ArtifactError
    root = Path(root)
    try:
        if root.is_symlink() or not root.is_dir():
            raise E("file-type", "composition root must be a real directory")
        present = layout._walk(root)
        mbytes = layout._read_doc(root, "manifest.json", limit=layout.MAX_MANIFEST)
        doc = layout.strict_json(mbytes, "manifest.json")
        if type(doc) is not dict or doc.get("format") != COMPOSITION_FORMAT:
            raise E("format", "not a jitLLM composition")
        if (root / "data").exists() or any(p.startswith("data/") for p in present):
            raise E("file-set", "a composition holds no data shards")
        if not layout._exact(doc.get("format_version"), COMPOSITION_VERSION):
            raise E("unsupported-version", f"format_version {doc.get('format_version')!r}")
        if mbytes != layout.dumps(doc):
            raise E("canonical", "manifest.json is not in canonical form")
        if (expected_id or root.name) != hashlib.sha256(mbytes).hexdigest():
            raise E("identity", "composition name is not the manifest digest")
        layout._obj(doc, COMPOSITION_KEYS, "composition")
        if doc["experimental"] is not True:
            raise E("schema", "experimental must be true before D-018's gate")
        layout._obj(doc["model"], {"architecture"}, "model")
        layout._str(doc["model"]["architecture"], "model.architecture", layout.NAME)
        components = layout._list(doc["components"], "components", 1)
        if len(components) > MAX_COMPONENTS:
            raise E("schema", f"more than {MAX_COMPONENTS} components")
        for c in components:
            layout._obj(c, {"role", "artifact", "architecture"}, "component")
            layout._str(c["role"], "component.role", ROLE)
            layout._str(c["artifact"], "component.artifact", layout.HEX)
            layout._str(c["architecture"], "component.architecture", layout.NAME)
        layout._sorted_unique(components, lambda c: c["role"], "components")
        sources = layout._list(doc["source"], "source", 1)
        for s in sources:
            layout._obj(s, {"name", "bytes", "sha256"}, "source")
            layout._str(s["name"], "source.name", layout.NAME)
            layout._int(s["bytes"], "source.bytes")
            layout._str(s["sha256"], "source.sha256", layout.HEX)
        layout._sorted_unique(sources, lambda s: s["name"], "source")
        conv = layout._obj(doc["converter"], {"name", "version"}, "converter")
        layout._str(conv["name"], "converter.name", layout.TEXT)
        layout._str(conv["version"], "converter.version", layout.TEXT)
        listed = layout._list(doc["files"], "files", 1)
        files = {}
        for f in listed:
            layout._obj(f, {"path", "role", "bytes", "sha256"}, "file")
            path = layout._str(f["path"], "file.path", layout.META_PATH)
            if f["role"] != "source-metadata":
                raise E("path", f"{path!r} with role {f['role']!r}")
            layout._int(f["bytes"], "file.bytes", 0, layout.MAX_META)
            layout._str(f["sha256"], "file.sha256", layout.HEX)
            files[path] = f
        layout._sorted_unique(listed, lambda f: f["path"], "files")
        if "meta/model_index.json" not in files:
            raise E("file-set", "a composition keeps its model_index.json")
        if present != set(files) | {"manifest.json"}:
            raise E("file-set", "files differ from the manifest's list")
        by_name = {s["name"]: s for s in sources}
        if set(by_name) != {p[len("meta/"):] for p in files}:
            raise E("file-set", "every source is a kept file and every kept file a source")
        total = 0
        for path, f in files.items():
            src = by_name[path[len("meta/"):]]
            if (src["bytes"], src["sha256"]) != (f["bytes"], f["sha256"]):
                raise E("file-set", f"{path}: not a verbatim copy of a recorded source")
            total += f["bytes"]
            if total > layout.MAX_META_TOTAL:
                raise E("file-size", "kept metadata exceeds its total limit")
            data = layout._read_doc(root, path, f["bytes"], limit=layout.MAX_META)
            if hashlib.sha256(data).hexdigest() != f["sha256"]:
                raise E("hash", f"{path}: digest differs from the manifest")
        if store is not None:
            for c in components:
                manifest, _ = layout.verify(Path(store) / c["artifact"], deep=False)
                if manifest["model"]["architecture"] != c["architecture"]:
                    raise E("component", f"{c['role']}: artifact architecture differs")
        return doc
    except E:
        raise
    except (TypeError, KeyError, ValueError, IndexError, AttributeError, OverflowError, MemoryError,
            RecursionError) as e:
        raise E("malformed", f"{type(e).__name__}: {e}") from e
    except OSError as e:
        raise E("io", f"{type(e).__name__}: {e}") from e


def _split_meta(args):
    rest, metas, i = [], [], 0
    while i < len(args):
        if args[i] == "--meta" and i + 1 < len(args):
            metas.append(args[i + 1])
            i += 2
        else:
            rest.append(args[i])
            i += 1
    return rest, metas


def main(argv):
    layout = load_layout()
    cmd, args = (argv[1], argv[2:]) if len(argv) > 1 else ("", [])
    if cmd == "build" and len(args) >= 4:
        out, pins, model_id, paths = args[0], args[1], args[2], args[3:]
        if all(Path(p).suffix == ".safetensors" for p in paths):
            modelopt, digest = load_modelopt()
            expected = pinned_sources(pins, model_id, [*paths, str(Path(paths[0]).parent / "config.json")])
            conv = {"name": CONVERTER_NAME,
                    "version": f"{converter()['version']}+modelopt_qwen38-{digest[:16]}"}
            final, _ = modelopt.build(layout, out, paths, expected=expected, converter=conv)
            print(final)
            return
        if any(Path(p).suffix != ".gguf" for p in paths):
            raise SystemExit("build takes GGUF sources, or Qwen3.8's safetensors shards; "
                             "see component for a pipeline's")
        expected = pinned_sources(pins, model_id, paths)
        src = layout.load_sources(paths)
        check_gemma3_source(src, expected)
        check_gemma2_source(src, expected)
        p = layout.plan(src, tie_check=True)
        print(json.dumps(layout.stats(p)), flush=True)
        print(layout.build(p, src, out, paths, (), converter=converter(), expected_sources=expected))
    elif cmd == "drafter" and len(args) >= 4:
        out, pins, model_id, paths = args[0], args[1], args[2], args[3:]
        ids_path = None
        if "--draft-vocab-ids" in paths:
            i = paths.index("--draft-vocab-ids")
            if i + 2 != len(paths) or i == 0:
                raise SystemExit("--draft-vocab-ids FILE follows the drafter's shards")
            ids_path = paths[i + 1]
            paths = paths[:i]
        if any(Path(p).suffix != ".safetensors" for p in paths):
            raise SystemExit("drafter takes Qwen3.8's safetensors shards that hold the MTP block")
        modelopt, digest = load_modelopt()
        identities = [*paths, str(Path(paths[0]).parent / "config.json")]
        if ids_path is not None:
            identities.append(ids_path)
        expected = pinned_sources(pins, model_id, identities)
        conv = {"name": CONVERTER_NAME,
                "version": f"{converter()['version']}+modelopt_qwen38-{digest[:16]}+mtp"}
        final, _ = modelopt.build(layout, out, paths, expected=expected, converter=conv, mtp=True,
                                 draft_vocab_ids=ids_path)
        print(final)
    elif cmd == "component" and len(args) >= 5:
        rest, metas = _split_meta(args)
        if len(rest) != 5:
            raise SystemExit(__doc__)
        out, pins, model_id, checkpoint, role = rest
        print(import_component(layout, out, pins, model_id, checkpoint, role, metas))
    elif cmd == "compose" and len(args) >= 5:
        rest, metas = _split_meta(args)
        out, pins, model_id, checkpoint, bindings = rest[0], rest[1], rest[2], rest[3], rest[4:]
        print(compose(layout, out, pins, model_id, checkpoint, bindings, metas))
    elif cmd == "verify" and len(args) == 1:
        manifest, index = layout.verify(args[0])
        print(json.dumps({"ok": True, "chunks": len(index["chunk_sha256"]), "files": len(manifest["files"]),
                          "converter": manifest["converter"]}))
    elif cmd == "verify-composition" and len(args) in (1, 3) and (len(args) == 1 or args[1] == "--store"):
        doc = verify_composition(layout, args[0], store=args[2] if len(args) == 3 else None)
        print(json.dumps({"ok": True, "architecture": doc["model"]["architecture"],
                          "components": doc["components"]}))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
