# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

"""Normalize complete descriptor inventories backed by successful plan uses.

This is a completed-plan inventory, not a per-kernel execution trace. It does
not claim routing payloads, expert IDs, or launch counts from static recipes.
"""
import json
import pathlib
import re

MODEL_RE = re.compile(r'^llmp-runtime: debug plan model ("(?:[^"\\]|\\.)*"): (.*)$')


def require(record, *fields):
    for field in fields:
        if field not in record:
            raise ValueError(f"{record.get('kind', 'record')} lacks {field}")


def exact_ordinals(records, field, expected, label):
    if type(expected) is not int or expected < 0:
        raise ValueError(f"invalid {label} count")
    values = [r[field] for r in records]
    if (len(values) != expected or any(type(v) is not int for v in values) or
            sorted(values) != list(range(expected))):
        raise ValueError(f"missing, duplicate, or invalid {label} ordinal")


def parse_log(path: pathlib.Path) -> dict:
    plans = {}
    current = {}
    sequences = set()
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        match = MODEL_RE.match(raw)
        if not match:
            continue
        model, payload = json.loads(match.group(1)), json.loads(match.group(2))
        kind = payload.get("kind")
        if kind == "installed_plan":
            identity = (model, "installed", payload["id"])
            if not payload["id"] or identity in plans:
                raise ValueError(f"duplicate/zero installed identity at line {line_no}")
            current[model] = identity
            plans[identity] = {"model": model, "identity": identity[1:], "header": payload,
                               "report_line": line_no, "records": [], "completions": []}
        elif kind == "completed_plan":
            sequence = (model, payload["sequence"])
            if sequence in sequences or payload["sequence"] <= 0:
                raise ValueError(f"duplicate/invalid completed sequence at line {line_no}")
            sequences.add(sequence)
            if payload["transient"]:
                if payload["id"] != 0:
                    raise ValueError("transient plan unexpectedly has an installed identity")
                identity = (model, "transient", payload["sequence"])
                current[model] = identity
                plans[identity] = {"model": model, "identity": identity[1:], "header": payload,
                                   "report_line": line_no, "records": [], "completions": []}
            else:
                identity = (model, "installed", payload["id"])
                if identity not in plans:
                    raise ValueError(f"completed plan lacks installed inventory at line {line_no}")
                current.pop(model, None)
            plan = plans[identity]
            if any(plan["header"][k] != payload[k] for k in ("stage", "phase", "rows")):
                raise ValueError("completed plan shape differs from reported inventory")
            plan["completions"].append({**payload, "log_line": line_no})
        elif model in current:
            plans[current[model]]["records"].append(payload)
        elif kind in {"physical_plan", "physical_node", "launch_recipe", "routed_group",
                      "physical_root", "routed_arena"}:
            raise ValueError(f"unscoped physical inventory at line {line_no}")

    normalized = []
    for plan in plans.values():
        if not plan["completions"]:
            continue  # Installed lookahead is never execution evidence.
        records = plan.pop("records")
        logical = [r for r in records if r.get("kind") == "plan"]
        selections = [r for r in records if r.get("kind") == "selection"]
        if len(logical) != 1 or logical[0].get("version") != 1:
            raise ValueError("completed plan lacks its logical selection summary")
        require(logical[0], "steps", "primitive", "graph", "policy", "registry")
        for selection in selections:
            require(selection, "step", "implementation", "identity", "lane", "rung",
                    "expansion_rule", "group", "nodes")
            if not selection["nodes"]:
                raise ValueError("empty logical selection")
            for node in selection["nodes"]:
                require(node, "id", "op", "component", "sources", "outputs")
                if node["op"] in {"mul_mat", "mul_mat_id", "fused.grouped_mul_mat"}:
                    require(node, "product_arithmetic")
        exact_ordinals(selections, "step", logical[0]["steps"], "logical selection")
        summaries = [r for r in records if r.get("kind") == "physical_plan"]
        if len(summaries) != 1 or summaries[0].get("operand_schema") != 1:
            raise ValueError("completed plan lacks the complete physical operand schema")
        summary = summaries[0]
        require(summary, "descriptor_count", "graph_steps", "routed_groups", "physical_roots")
        nodes = {}
        for r in records:
            if r.get("kind") != "physical_node":
                continue
            if r["id"] in nodes:
                raise ValueError("duplicate descriptor identity")
            for key in ("op", "ir_node", "tensor", "root", "part", "dtype", "shape", "strides",
                        "ir_format", "sources", "storage",
                        "view_offset", "view_source"):
                if key not in r:
                    raise ValueError(f"descriptor lacks {key}")
            if not r["dtype"] or len(r["shape"]) != 4 or len(r["strides"]) != 4:
                raise ValueError("incomplete descriptor format")
            semantic = r["ir_format"]
            if semantic is not None and (not semantic.get("encoding") or
                    not semantic.get("layout") or len(semantic.get("shape", [])) != 4):
                raise ValueError("incomplete semantic format")
            nodes[r["id"]] = r
        if not nodes:
            raise ValueError("completed plan has no descriptor inventory")
        exact_ordinals(list(nodes.values()), "id", summary["descriptor_count"], "descriptor")
        for node in nodes.values():
            for edge in [node["storage"], node["view_source"], *node["sources"]]:
                if edge is not None and edge not in nodes:
                    raise ValueError("descriptor references an unreported source/storage/view")
            if node["storage"] is None:
                raise ValueError("descriptor has no backing identity")
        recipes = []
        for r in records:
            if r.get("kind") != "launch_recipe":
                continue
            require(r, "step", "nodes", "ordered_nodes", "reads", "writes", "implementation",
                    "identity", "op", "lane", "prologue")
            if r["ordered_nodes"] != len(r["nodes"]):
                raise ValueError("recipe node count differs from its ordered nodes")
            operated = []
            for descriptor_id in r["nodes"]:
                if descriptor_id not in nodes:
                    raise ValueError("launch recipe references an unreported descriptor")
                node = nodes[descriptor_id]
                operated.append({"output": node, "sources": [nodes[s] if s is not None else None
                                                              for s in node["sources"]]})
            if not operated:
                raise ValueError("empty launch recipe")
            expected_reads = [nodes[s]["storage"] for item in operated
                              for s in item["output"]["sources"] if s is not None]
            expected_writes = [item["output"]["storage"] for item in operated]
            if r["reads"] != expected_reads or r["writes"] != expected_writes:
                raise ValueError("recipe accesses differ from its concrete operand slots")
            recipes.append({**r, "operated_nodes": operated})
        if not recipes:
            raise ValueError("completed plan has no launch recipes")
        exact_ordinals(recipes, "step", summary["graph_steps"], "graph recipe")
        roots = [r for r in records if r.get("kind") == "physical_root"]
        if len(roots) != summary["physical_roots"] or len({r["root"] for r in roots}) != len(roots):
            raise ValueError("missing or duplicate physical root")
        for root in roots:
            require(root, "producer", "component", "backend_tensor", "op", "offset", "bytes",
                    "encoding", "shape", "strides")
            if root["backend_tensor"] not in nodes:
                raise ValueError("root references an unreported descriptor")
        groups = [r for r in records if r.get("kind") == "routed_group"]
        exact_ordinals(groups, "group", summary["routed_groups"], "routed group")
        arenas = [r for r in records if r.get("kind") == "routed_arena"]
        if len(arenas) != (1 if groups else 0):
            raise ValueError("missing or unexpected routed arena summary")
        if groups:
            arena = arenas[0]
            require(arena, "descriptors", "paid_descriptor_bound", "groups", "graph_steps")
            # This is the full physical inventory, not only routing operands.
            if (arena["descriptors"] != summary["descriptor_count"] or
                    arena["groups"] != len(groups) or arena["graph_steps"] != len(recipes) or
                    arena["paid_descriptor_bound"] < arena["descriptors"]):
                raise ValueError("routed arena summary differs from the complete inventory")
        previous_cut = 0
        for group in sorted(groups, key=lambda g: g["group"]):
            require(group, "segment", "continuation_segment", "begin", "cut", "experts", "used",
                    "tokens", "host_offset", "host_bytes", "activation_offset", "activation_bytes",
                    "preparation", "readback", "uploads", "selected_descriptors", "validation_scratch",
                    "readback_copies", "upload_copies", "selected_descriptor_count")
            if (group["begin"] != previous_cut or not previous_cut <= group["cut"] <= len(recipes) or
                    group["segment"] != group["group"] or
                    group["continuation_segment"] != group["group"] + 1):
                raise ValueError("inconsistent routed continuation boundaries")
            previous_cut = group["cut"]
            prep = group["preparation"]
            require(prep, "implementation", "identity", "descriptor", "ids_descriptor", "reads", "writes")
            if prep["descriptor"] not in nodes or prep["ids_descriptor"] not in nodes:
                raise ValueError("unreported routing preparation descriptor")
            if (prep["reads"] != [nodes[prep["ids_descriptor"]]["storage"]] or
                    prep["writes"] != [nodes[prep["descriptor"]]["storage"]]):
                raise ValueError("routing preparation accesses differ from descriptor backing")
            for field, count in (("readback", "readback_copies"), ("uploads", "upload_copies")):
                exact_ordinals(group[field], "ordinal", group[count], f"routed {field}")
                for copy in group[field]:
                    require(copy, "direction", "activation_offset", "host_offset", "bytes", "rows",
                            "device_stride")
                    if field == "readback":
                        require(copy, "after", "source_descriptor", "source_storage", "source_offset",
                                "host_stride")
                        if (copy["source_descriptor"] not in nodes or
                                copy["source_storage"] != nodes[copy["source_descriptor"]]["storage"]):
                            raise ValueError("unreported or inconsistent readback source")
                    else:
                        require(copy, "before")
            selected = group["selected_descriptors"]
            if (len(selected) != group["selected_descriptor_count"] or
                    len({d["selected_index"] for d in selected}) != len(selected)):
                raise ValueError("missing or duplicate selected routed descriptor")
            for descriptor in selected:
                require(descriptor, "descriptor", "host_offset", "bytes")
                if descriptor["descriptor"] not in nodes:
                    raise ValueError("unreported selected routed descriptor")
            require(group["validation_scratch"], "host_offset", "bytes")
        physical_kinds = {"physical_plan", "physical_node", "launch_recipe", "physical_root",
                          "routed_group", "routed_arena"}
        plan.update(logical_records=[r for r in records if r.get("kind") not in physical_kinds],
                    summary=summary, recipes=recipes,
                    descriptors=[nodes[i] for i in sorted(nodes)],
                    routed_groups=groups, physical_roots=roots)
        normalized.append(plan)
    if not normalized:
        raise ValueError(f"no successfully completed plans with complete operands in {path}")
    return {"format": "llmp-completed-plan-operands/1", "plans": normalized,
            "installed_without_success": sum(not p["completions"] for p in plans.values()),
            "scope": "selected plan inventories with successful-use receipts; not a kernel execution trace"}
