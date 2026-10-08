#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-2026 The ggml authors
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Insert removable host observations into the exact pinned CUDA controller.

No original byte is replaced. Removing all marked insertions must reconstruct
the pinned file byte for byte. Floating kernels stay in the original image.
"""
import hashlib
import pathlib
import re
import sys

PIN = '523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634'
BEGIN = '\n// LLMP_OBSERVER_BEGIN\n'
END = '// LLMP_OBSERVER_END\n'
HELPER = r'''
// External host observations only; no tensors are retained or mutated.
static thread_local unsigned long long llmp_observer_graph = 0;
static thread_local unsigned long long llmp_observer_records = 0;
static void llmp_observer_prefix(const char * kind) {
    if (++llmp_observer_records > 200000) GGML_ABORT("observer record bound exceeded");
    fprintf(stderr, "LLMP_OBSERVE graph=%llu kind=%s", llmp_observer_graph, kind);
}
static void llmp_observer_tensor(const char * role, const ggml_tensor * t) {
    if (t == nullptr) { fprintf(stderr, " %s=null", role); return; }
    fprintf(stderr, " %s=%s,%s,%s", role, t->name, ggml_op_name(t->op), ggml_type_name(t->type));
    fprintf(stderr, ",ne=%lld:%lld:%lld:%lld,nb=%llu:%llu:%llu:%llu,data=%p,bytes=%llu",
        (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
        (unsigned long long)t->nb[0], (unsigned long long)t->nb[1],
        (unsigned long long)t->nb[2], (unsigned long long)t->nb[3], t->data,
        (unsigned long long)ggml_nbytes(t));
    if (t->buffer != nullptr) fprintf(stderr, ",alloc=%llu",
        (unsigned long long)ggml_backend_buft_get_alloc_size(t->buffer->buft, t));
}
static void llmp_observer_product(const char * family, const ggml_tensor * a,
                                   const ggml_tensor * b, const ggml_tensor * ids,
                                   const ggml_tensor * dst) {
    llmp_observer_prefix("product");
    fprintf(stderr, " family=%s routed=%d", family, ids != nullptr);
    llmp_observer_tensor("a", a); llmp_observer_tensor("b", b);
    llmp_observer_tensor("ids", ids); llmp_observer_tensor("dst", dst);
    fprintf(stderr, "\n");
}
'''


def marked(code):
    return BEGIN + code + ('\n' if not code.endswith('\n') else '') + END


def insert(text, anchor, code, count=1):
    if text.count(anchor) != count:
        raise ValueError('unexpected upstream anchor count: ' + anchor[:100])
    return text.replace(anchor, marked(code) + anchor)


def original(text):
    return re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', text, flags=re.S)


def transform(raw):
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError('wrong pinned upstream controller')
    text = raw.decode()
    text = insert(text, 'static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,', HELPER)
    # Observe the actual branch immediately before its original launcher.
    start = text.index('static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,')
    end = text.index('// returns true when ggml_cuda_mul_mat_id', start)
    function = text[start:end]
    for family, call in [('CUBLAS', 'ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);'),
                         ('MMVF', 'ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);'),
                         ('MMVF_TRANSPOSE', 'ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);'),
                         ('MMF', 'ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);'),
                         ('MMVQ', 'ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);'),
                         ('MMQ', 'ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);')]:
        function = insert(function, call,
                          f'    llmp_observer_product("{family}", src0, src1, nullptr, dst);',
                          2 if family == 'CUBLAS' else 1)
    text = text[:start] + function + text[end:]
    start = text.index('static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx,')
    end = text.index('    // note: this path should not be reached', start)
    function = text[start:end]
    for family, call in [('MMVQ', 'ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);'),
                         ('MMVF', 'ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);'),
                         ('MMQ', 'ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);'),
                         ('MMF', 'ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);')]:
        function = insert(function, call,
                          f'    llmp_observer_product("{family}", src0, src1, ids, dst);')
    text = text[:start] + function + text[end:]
    text = insert(text, '    // note: this path should not be reached when recording CUDA graphs,',
                  '    llmp_observer_product("SORTED_FALLBACK", src0, src1, ids, dst);')
    # These host-selected fused product sites bypass the ordinary dispatcher.
    start = text.index('static int ggml_cuda_try_fuse(')
    end = text.index('static void ggml_cuda_graph_evaluate_and_capture(', start)
    function = text[start:end]
    calls = set(re.findall(r'ggml_cuda_mul_mat_vec_([qf])\(\*cuda_ctx, src0, src1, ids, ([^;]+)\);', function))
    for variant, call in sorted(calls):
        destination = call.split(', &fusion_data')[0]
        anchor = 'ggml_cuda_mul_mat_vec_' + variant + '(*cuda_ctx, src0, src1, ids, ' + call + ');'
        family = 'MMVQ_FUSED' if variant == 'q' else 'MMVF_FUSED'
        function = insert(function, anchor,
                          f'    llmp_observer_product("{family}", src0, src1, ids, {destination});',
                          function.count(anchor))
    text = text[:start] + function + text[end:]
    text = insert(text, '        if (can_fuse) {\n            const ggml_tensor * logits  = node->src[0];',
                  '        if (node->op == GGML_OP_SOFT_MAX && strncmp(node->name, "ffn_moe_probs-", 14) == 0) {\n'
                  '            llmp_observer_prefix("route_candidate");\n'
                  '            fprintf(stderr, " recognized=%d", (int)can_fuse);\n'
                  '            llmp_observer_tensor("node", node); fprintf(stderr, "\\n");\n'
                  '        }')
    route = '''                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&'''
    text = insert(text, route, r'''
                llmp_observer_prefix("route_gate");
                fprintf(stderr, " structural=%d shape=%d memory=%d count=%llu",
                    (int)ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2),
                    (int)ggml_cuda_should_use_topk_moe(node, logits, weights, ids),
                    (int)ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, true),
                    (unsigned long long)ops.size());
                llmp_observer_tensor("node", node); llmp_observer_tensor("logits", logits);
                llmp_observer_tensor("weights", weights); llmp_observer_tensor("ids", ids);
                fprintf(stderr, "\n");
''')
    text = insert(text, '                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);',
                  '                    llmp_observer_prefix("route_selected");\n'
                  '                    llmp_observer_tensor("node", node); fprintf(stderr, "\\n");', 2)
    anchor = '        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {'
    text = insert(text, anchor, r'''
        if (strncmp(node->name, "ffn_moe_down_scaled-", 20) == 0) {
            ggml_cuda_moe_weighted_reduction_match observed;
            const bool matched = ggml_cuda_match_moe_weighted_reduction(cgraph, i, observed);
            const int observed_output_idx = matched ? i + observed.node_count - 1 : i;
            llmp_observer_prefix("reduction_gate");
            fprintf(stderr, " matched=%d count=%d scale=%d memory=%d", (int)matched,
                matched ? observed.node_count : 0, matched && observed.expert_scale != nullptr,
                matched ? (int)ggml_cuda_check_fusion_memory_ranges(cgraph, i,
                    observed.node_count, &observed_output_idx, 1) : -1);
            llmp_observer_tensor("node", node);
            if (matched) {
                llmp_observer_tensor("experts", observed.experts);
                llmp_observer_tensor("scale", observed.expert_scale);
                llmp_observer_tensor("weights", observed.weights);
                llmp_observer_tensor("dst", observed.dst);
            }
            fprintf(stderr, "\n");
        }
''')
    text = insert(text, '                ggml_cuda_op_moe_weighted_reduction(\n',
                  '                llmp_observer_prefix("reduction_selected");\n'
                  '                llmp_observer_tensor("node", node); fprintf(stderr, "\\n");')
    text = insert(text, '    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);',
                  r'''
    ++llmp_observer_graph;
    llmp_observer_prefix("graph");
    fprintf(stderr, " nodes=%d cuda_graph=%d update=%d", cgraph->n_nodes,
        (int)use_cuda_graph, (int)cuda_graph_update_required);
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const auto * t = cgraph->nodes[j];
        if (t->op == GGML_OP_SOFT_MAX && strncmp(t->name, "ffn_moe_probs-", 14) == 0) {
            fprintf(stderr, " router_rows=%lld", (long long)ggml_nrows(t));
            break;
        }
    }
    fprintf(stderr, "\n");
''')
    if original(text).encode() != raw:
        raise ValueError('observer changed original bytes')
    return text.encode()


def main():
    if len(sys.argv) != 3:
        raise ValueError('SOURCE NEW_DESTINATION')
    raw = pathlib.Path(sys.argv[1]).read_bytes()
    output = transform(raw)
    with pathlib.Path(sys.argv[2]).open('xb') as file:
        file.write(output)
    print('original_sha256=' + PIN)
    print('observed_sha256=' + hashlib.sha256(output).hexdigest())
    print('reconstructed_sha256=' + hashlib.sha256(original(output.decode()).encode()).hexdigest())


if __name__ == '__main__':
    main()
