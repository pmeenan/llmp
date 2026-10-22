# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import pathlib
import tempfile
import unittest
from plan_reports import parse_log


def header(kind="installed_plan", **extra):
    return dict(kind=kind, id=7, stage=0, phase=1, rows=2, **extra)


def completed(**extra):
    return header("completed_plan", sequence=1, transient=False, **extra)


def inventory():
    return [dict(kind="physical_plan", operand_schema=1, descriptor_count=3, graph_steps=1,
                 routed_groups=0, physical_roots=0),
            dict(kind="launch_recipe", nodes=[2], ordered_nodes=1, implementation="test.add", step=0,
                 identity="identity", op="add", lane=0, prologue=False, reads=[1, 1], writes=[2]),
            dict(kind="physical_node", op="NONE", ir_node=None, tensor=None, root=None, part=None, id=1, dtype="f32", shape=[8, 2, 1, 1],
                 strides=[4, 64, 128, 128], ir_format=None, sources=[], storage=1,
                 view_source=None, view_offset=0),
            dict(kind="physical_node", op="NONE", ir_node=None, tensor=None, root=None, part=None, id=2, dtype="f32", shape=[8, 2, 1, 1],
                 strides=[4, 32, 64, 64], ir_format=None, sources=[1, None, 1],
                 storage=2, view_source=None, view_offset=0),
            dict(kind="plan", version=1, steps=1, primitive="primitive", graph="graph",
                 policy="policy", registry="registry"),
            dict(kind="selection", step=0, implementation="test.add", identity="identity",
                 lane=0, rung=0, expansion_rule=0, group="group",
                 nodes=[dict(id=0, op="mul_mat", component=0, sources=[1, 1], outputs=[2],
                             product_arithmetic="bf16")]),
            dict(kind="physical_node", op="NONE", ir_node=None, tensor=None, root=None, part=None, id=0, dtype="f32", shape=[1, 1, 1, 1],
                 strides=[4, 4, 4, 4], ir_format=None, sources=[], storage=0,
                 view_source=None, view_offset=0)]


def routed_inventory():
    records = inventory()
    records[0]["routed_groups"] = 1
    records.extend([
        dict(kind="routed_arena", descriptors=3, paid_descriptor_bound=4, groups=1, graph_steps=1),
        dict(kind="routed_group", group=0, segment=0, continuation_segment=1, begin=0, cut=0,
             experts=2, used=1, tokens=2, host_offset=0, host_bytes=64,
             activation_offset=0, activation_bytes=64, readback_copies=1, upload_copies=1,
             selected_descriptor_count=1,
             preparation=dict(implementation="test.route", identity="route-identity", descriptor=2,
                              ids_descriptor=1, reads=[1], writes=[2]),
             readback=[dict(ordinal=0, direction="device_to_host", after="prefix_completion",
                            source_descriptor=1, source_storage=1, source_offset=0,
                            activation_offset=0, host_offset=0, bytes=4, rows=2,
                            device_stride=64, host_stride=4)],
             uploads=[dict(ordinal=0, direction="host_to_device", before="continuation_graph",
                           activation_offset=0, host_offset=0, bytes=8, rows=1, device_stride=8)],
             selected_descriptors=[dict(selected_index=5, descriptor=1, host_offset=0, bytes=8)],
             validation_scratch=dict(host_offset=16, bytes=2))])
    return records


class PlanReportsTest(unittest.TestCase):
    def parse(self, records):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "runtime.log"
            path.write_text("\n".join("llmp-runtime: debug plan model " + json.dumps(model) +
                            ": " + json.dumps(record) for model, record in records))
            return parse_log(path)

    def test_completed_inventory_retains_source_slots(self):
        result = self.parse([(r'a"b\\c', r) for r in [header(), *inventory(), completed()]])
        self.assertEqual(result["plans"][0]["model"], r'a"b\\c')
        sources = result["plans"][0]["recipes"][0]["operated_nodes"][0]["sources"]
        self.assertEqual([s["id"] if s else None for s in sources], [1, None, 1])
        self.assertEqual(sources[0]["strides"], [4, 64, 128, 128])
        logical = result["plans"][0]["logical_records"]
        self.assertEqual(logical[1]["nodes"][0]["product_arithmetic"], "bf16")
        self.assertEqual(logical[1]["identity"], "identity")

    def test_install_only_is_not_success(self):
        with self.assertRaises(ValueError):
            self.parse([("model", r) for r in [header(), *inventory()]])

        for missing in ("plan", "selection"):
            records = [r for r in inventory() if r["kind"] != missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.parse([("model", r) for r in [header(), *records, completed()]])

        records = inventory()
        selection = next(r for r in records if r["kind"] == "selection")
        del selection["nodes"][0]["product_arithmetic"]
        with self.assertRaises(ValueError):
            self.parse([("model", r) for r in [header(), *records, completed()]])

    def test_missing_or_old_operand_schema_refused(self):
        for remove in ("dtype", "ir_format", "shape", "strides", "view_source", "part", "tensor"):
            records = inventory()
            del records[2][remove]
            with self.subTest(remove=remove), self.assertRaises(ValueError):
                self.parse([("model", r) for r in [header(), *records, completed()]])

    def test_missing_source_and_duplicate_node_refused(self):
        for records in (inventory()[:-1], inventory() + [copy.deepcopy(inventory()[2])]):
            with self.assertRaises(ValueError):
                self.parse([("model", r) for r in [header(), *records, completed()]])

    def test_whole_recipe_deletion_and_step_duplicates_refused(self):
        full = inventory()
        full[0]["graph_steps"] = 2
        full.append({**full[1], "step": 1})
        self.parse([("model", r) for r in [header(), *full, completed()]])
        bad_ordinal = copy.deepcopy(full)
        bad_ordinal[-1]["step"] = 0
        for records in (full[:-1], bad_ordinal, full + [copy.deepcopy(full[-1])]):
            with self.assertRaises(ValueError):
                self.parse([("model", r) for r in [header(), *records, completed()]])

    def test_unreferenced_descriptor_deletion_refused(self):
        full = inventory()
        self.assertEqual(full[-1]["id"], 0)
        with self.assertRaises(ValueError):
            self.parse([("model", r) for r in [header(), *full[:-1], completed()]])

    def test_recipe_access_multiplicity_and_metadata_required(self):
        for field in ("reads", "writes", "ordered_nodes", "identity"):
            records = inventory()
            del records[1][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse([("model", r) for r in [header(), *records, completed()]])
        records = inventory()
        records[1]["reads"] = [1]
        with self.assertRaises(ValueError):
            self.parse([("model", r) for r in [header(), *records, completed()]])

    def test_routed_inventory_requires_every_record_and_summary(self):
        full = routed_inventory()
        result = self.parse([("model", r) for r in [header(), *full, completed()]])
        self.assertEqual(len(result["plans"][0]["descriptors"]), 3)
        variants = [full[:-1], full[:-2] + [full[-1]], full + [copy.deepcopy(full[-1])]]
        for field in ("readback", "uploads", "selected_descriptors"):
            records = copy.deepcopy(full)
            records[-1][field] = []
            variants.append(records)
        records = copy.deepcopy(full)
        records[-2]["descriptors"] = 1  # Routing subset is not the arena's full descriptor count.
        variants.append(records)
        records = copy.deepcopy(full)
        records[-1]["cut"] = 2
        variants.append(records)
        for records in variants:
            with self.assertRaises(ValueError):
                self.parse([("model", r) for r in [header(), *records, completed()]])

    def test_private_success_is_scoped_by_sequence(self):
        first = completed()
        first.update(id=0, transient=True)
        second = {**first, "sequence": 2}
        result = self.parse([("model", r) for r in [first, *inventory(), second, *inventory()]])
        self.assertEqual([p["identity"] for p in result["plans"]],
                         [("transient", 1), ("transient", 2)])

    def test_models_do_not_share_serials_or_sequences(self):
        result = self.parse([(m, r) for m in ("a", "b")
                             for r in [header(), *inventory(), completed()]])
        self.assertEqual(len(result["plans"]), 2)

    def test_unknown_or_wrong_shape_completion_refused(self):
        with self.assertRaises(ValueError):
            self.parse([("model", completed())])
        bad = completed()
        bad["rows"] = 99
        with self.assertRaises(ValueError):
            self.parse([("model", r) for r in [header(), *inventory(), bad]])


if __name__ == "__main__":
    unittest.main()
