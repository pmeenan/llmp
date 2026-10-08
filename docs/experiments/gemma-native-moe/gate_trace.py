#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only gate logging over the exact committed stock controller.

No graph tensors/outputs, allocator hints, math launchers or policies change.
"""
import hashlib
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == 'fce9dd32edde4891ed180a4081d155ef921a9d1b50e619375caac58e2f6ba291'
s = source.decode()
needle = '''                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&'''
assert s.count(needle) == 1
s = s.replace(needle, '''                fprintf(stderr, "LLMP_ROUTE_GATE node=%s rows=%lld structural=%d shape=%d memory=%d\\n",
                    node->name, (long long) ggml_nrows(node),
                    (int) ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2),
                    (int) ggml_cuda_should_use_topk_moe(node, logits, weights, ids),
                    (int) ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, true));
''' + needle)
needle = '''                    if (!found) {
                        is_ok = false;'''
assert s.count(needle) == 1
s = s.replace(needle, '''                    if (!found) {
                        if (is_topk_moe) fprintf(stderr,
                            "LLMP_ROUTE_OVERLAP node=%s dst=%s src=%s dst_bytes=%llu src_bytes=%llu\\n",
                            cgraph->nodes[node_idx]->name, dst->name, src->name,
                            (unsigned long long) ggml_backend_buft_get_alloc_size(dst->buffer->buft, dst),
                            (unsigned long long) ggml_backend_buft_get_alloc_size(src->buffer->buft, src));
                        is_ok = false;''')
with pathlib.Path(sys.argv[2]).open('x') as f:
    f.write(s)
print('controller_gate_sha256=' + hashlib.sha256(s.encode()).hexdigest())
