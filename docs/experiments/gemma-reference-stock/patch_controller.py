#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-2026 The ggml authors
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Remove one pinned stock fusion family, retaining every other stock gate."""
import hashlib
import pathlib
import sys

PINNED_SHA = '523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634'

HELPER = r'''
// External diagnosis only: all other original stock gates remain in place.
static int llmp_reference_policy() {
    static const int policy = [] {
        const char * p = getenv("LLMP_REFERENCE_FUSION_POLICY");
        if (p == nullptr || strcmp(p, "all") == 0) return 0;
        if (strcmp(p, "no_routing") == 0) return 1;
        if (strcmp(p, "no_reduction") == 0) return 2;
        if (strcmp(p, "no_norm_mul") == 0) return 3;
        if (strcmp(p, "no_softcap") == 0) return 4;
        GGML_ABORT("unknown external stock-minus fusion policy");
    }();
    return policy;
}
'''

BANNER = r'''
    static const bool announced = [] {
        fprintf(stderr, "LLMP_REF_CONTROLLER policy=%d\n", llmp_reference_policy());
        return true;
    }();
    (void) announced;
'''

TRACE = r'''
                static const bool trace = getenv("LLMP_REFERENCE_TRACE") != nullptr;
                if (trace && nodes_to_skip > 0) {
                    fprintf(stderr, "LLMP_REF_FUSION count=%d", nodes_to_skip + 1);
                    for (int f = i; f <= i + nodes_to_skip; ++f) {
                        const auto * t = cgraph->nodes[f];
                        fprintf(stderr, " | %s:%s:%s:[%lld,%lld,%lld,%lld]",
                                t->name, ggml_op_name(t->op), ggml_type_name(t->type),
                                (long long)t->ne[0], (long long)t->ne[1],
                                (long long)t->ne[2], (long long)t->ne[3]);
                    }
                    fprintf(stderr, "\n");
                }
'''

def main():
    assert len(sys.argv) == 3
    source, destination = map(pathlib.Path, sys.argv[1:])
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINNED_SHA, 'wrong upstream source identity'
    assert not destination.exists(), 'diagnostic destination must be new'
    text = raw.decode()
    anchor = '// try and fuse nodes and return the number of nodes to skip\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, HELPER + '\n' + anchor)
    start = text.index('static int ggml_cuda_try_fuse(')
    end = text.index('static void ggml_cuda_graph_evaluate_and_capture(', start)
    function = text[start:end]
    anchor = '    ggml_tensor * node = cgraph->nodes[i];\n'
    assert function.count(anchor) == 1
    function = function.replace(anchor, BANNER + '\n' + anchor)
    anchor = '    if (node->op == GGML_OP_MUL) {\n        ggml_cuda_moe_weighted_reduction_match match;'
    assert function.count(anchor) == 1
    function = function.replace(anchor, '    if (llmp_reference_policy() != 2 && node->op == GGML_OP_MUL) {\n        ggml_cuda_moe_weighted_reduction_match match;')
    anchor = '    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||\n            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {'
    assert function.count(anchor) == 1
    function = function.replace(anchor, '    if (llmp_reference_policy() != 1 && (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||\n            cgraph->nodes[i]->op == GGML_OP_ARGSORT)) {')
    for policy, pattern in [(3, '{ GGML_OP_RMS_NORM, GGML_OP_MUL }, {}'),
                            (4, '{ GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH }')]:
        anchor = '    if (ggml_cuda_can_fuse(cgraph, i, ' + pattern + ')) {'
        assert function.count(anchor) == 1
        function = function.replace(anchor, '    if (llmp_reference_policy() != ' + str(policy) + ' && ggml_cuda_can_fuse(cgraph, i, ' + pattern + ')) {')
    text = text[:start] + function + text[end:]
    anchor = '                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + TRACE)
    start = text.index('static void ggml_backend_cuda_graph_optimize(')
    anchor = '    if (!disable_fusion) {'
    offset = text.index(anchor, start)
    # Preserve all original weighted-reduction allocation hints except in
    # the one policy which disables that specialized reduction kernel.
    text = text[:offset] + text[offset:].replace(
        anchor, '    if (!disable_fusion && llmp_reference_policy() != 2) {', 1)
    destination.write_text(text)
    print('original_sha256=' + PINNED_SHA)
    print('diagnostic_sha256=' + hashlib.sha256(destination.read_bytes()).hexdigest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
