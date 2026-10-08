#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-2026 The ggml authors
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Insert removable host observations into the original d812 CUDA controller."""
import hashlib
import pathlib
import re
import sys

PIN = '1315f06baff63da74e085b7ec01974474dac1f8ec1074af13eb70faba4751961'
BEGIN = '\n// LLMP_ROUTE_OBSERVER_BEGIN\n'
END = '// LLMP_ROUTE_OBSERVER_END\n'
HELPER = r'''
static thread_local unsigned long long llmp_route_graph = 0;
static thread_local unsigned long long llmp_route_records = 0;
static void llmp_route_prefix(const char * kind, const ggml_tensor * node) {
    if (++llmp_route_records > 20000) GGML_ABORT("route observer record bound exceeded");
    fprintf(stderr, "LLMP_ROUTE graph=%llu kind=%s node=%s rows=%lld",
        llmp_route_graph, kind, node ? node->name : "none",
        node ? (long long)ggml_nrows(node) : 0LL);
}
static void llmp_route_tensor(const char * role, const ggml_tensor * t) {
    if (t == nullptr) { fprintf(stderr, " %s=null", role); return; }
    fprintf(stderr, " %s=%s,data=%p,bytes=%llu,ne=%lld:%lld:%lld:%lld",
        role, t->name, t->data, (unsigned long long)ggml_nbytes(t),
        (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    if (t->buffer != nullptr) fprintf(stderr, ",alloc=%llu",
        (unsigned long long)ggml_backend_buft_get_alloc_size(t->buffer->buft, t));
    else fprintf(stderr, ",alloc=unknown");
}
'''


def insert(text, anchor, code, count=1):
    if text.count(anchor) != count:
        raise ValueError('unexpected original anchor count: ' + anchor[:100])
    return text.replace(anchor, BEGIN + code + '\n' + END + anchor)


def original(text):
    return re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', text, flags=re.S)


def transform(raw):
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError('wrong original d812 controller')
    text = raw.decode()
    text = insert(text, 'static int ggml_cuda_try_fuse(', HELPER)
    text = insert(text, '        if (can_fuse) {\n            const ggml_tensor * logits  = node->src[0];', r'''
        if (node->op == GGML_OP_SOFT_MAX && strncmp(node->name, "ffn_moe_probs-", 14) == 0) {
            llmp_route_prefix("candidate", node);
            fprintf(stderr, " recognized=%d\n", (int)can_fuse);
        }
''')
    gate = '''                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&'''
    text = insert(text, gate, r'''
                llmp_route_prefix("route_gate", node);
                fprintf(stderr, " structural=%d shape=%d memory=%d",
                    (int)ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2),
                    (int)ggml_cuda_should_use_topk_moe(node, logits, weights, ids),
                    (int)ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, true));
                llmp_route_tensor("logits", logits);
                llmp_route_tensor("weights", weights); llmp_route_tensor("ids", ids);
                fprintf(stderr, "\n");
''')
    text = insert(text, '                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);', r'''
                    llmp_route_prefix("route_selected", node); fprintf(stderr, "\n");
''', 2)
    text = insert(text, '\n        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {', r'''
        if (strncmp(node->name, "ffn_moe_down_scaled-", 20) == 0) {
            ggml_cuda_moe_weighted_reduction_match observed;
            const bool matched = ggml_cuda_match_moe_weighted_reduction(cgraph, i, observed);
            const int output_idx = matched ? i + observed.node_count - 1 : i;
            llmp_route_prefix("reduction_gate", node);
            fprintf(stderr, " matched=%d memory=%d\n", (int)matched,
                matched ? (int)ggml_cuda_check_fusion_memory_ranges(cgraph, i,
                    observed.node_count, &output_idx, 1) : -1);
        }
''')
    text = insert(text, '                ggml_cuda_op_moe_weighted_reduction(\n', r'''
                llmp_route_prefix("reduction_selected", node); fprintf(stderr, "\n");
''')
    text = insert(text, '    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);', r'''
    ++llmp_route_graph;
    llmp_route_prefix("graph", nullptr);
    fprintf(stderr, " nodes=%d capture=%d update=%d\n", cgraph->n_nodes,
        (int)use_cuda_graph, (int)cuda_graph_update_required);
''')
    if original(text).encode() != raw:
        raise ValueError('observer changed original controller bytes')
    return text.encode()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('ORIGINAL_CONTROLLER NEW_CONTROLLER')
    raw = pathlib.Path(sys.argv[1]).read_bytes()
    observed = transform(raw)
    with pathlib.Path(sys.argv[2]).open('xb') as file:
        file.write(observed)
    print('original_sha256=' + PIN)
