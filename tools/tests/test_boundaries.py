# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tools/llmp_boundaries.py, the portability boundary check, and the check over this tree."""

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import llmp_boundaries as boundaries  # noqa: E402


class Split(unittest.TestCase):
    def test_comments_and_literals_leave_code_with_the_same_lines(self):
        text = ('int a = 1\'000; // epoll_wait\n'
                '/* cudaMalloc\n */ f("/proc/x", \'"\');\n'
                'auto r = R"x(eventfd ")x" + 2;\n')
        code, strings, plain = boundaries.split(text)
        self.assertEqual(len(code), len(text))
        self.assertEqual(len(strings), len(text))
        self.assertEqual(len(plain), len(text))
        self.assertEqual([i for i, c in enumerate(code) if c == "\n"],
                         [i for i, c in enumerate(text) if c == "\n"])
        self.assertIn("1'000", code)
        self.assertNotIn("epoll_wait", code)
        self.assertNotIn("cudaMalloc", code)
        self.assertNotIn("/proc/x", code)
        self.assertIn("/proc/x", strings)
        self.assertNotIn("eventfd", code)
        self.assertIn("eventfd", strings)
        self.assertIn("+ 2;", code)
        self.assertIn('f("/proc/x"', plain)
        self.assertNotIn("epoll_wait", plain)

    def test_includes_are_read_without_comments(self):
        _, _, plain = boundaries.split('#include <sys/epoll.h>\n// #include <cuda.h>\n#  include "a/b.h"\n')
        self.assertEqual([name for _, name in boundaries.includes(plain)], ["sys/epoll.h", "a/b.h"])


class Check(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))

    def check(self, files: dict[str, str]) -> list[str]:
        for name, text in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return boundaries.check(self.root, sorted(files))[1]

    def test_linux_code_belongs_to_the_platform_and_the_linux_providers(self):
        linux = ('#include <sys/epoll.h>\n#include <linux/fs.h>\n'
                 'int f() { return ::epoll_create1(EPOLL_CLOEXEC) + O_DIRECT + ::getrandom(0, 0, 0); }\n'
                 'const char* m = "/proc/meminfo";\n')
        self.assertEqual(self.check({"src/platform/a.cc": linux, "src/providers/uring_storage.cc": linux}), [])
        problems = self.check({"src/runtime/a.cc": linux})
        self.assertEqual(problems, [
            "src/runtime/a.cc:1: includes sys/epoll.h, which is Linux-specific",
            "src/runtime/a.cc:2: includes linux/fs.h, which is Linux-specific",
            "src/runtime/a.cc:3: uses epoll_create1, which is Linux-specific",
            "src/runtime/a.cc:3: uses EPOLL_CLOEXEC, which is Linux-specific",
            "src/runtime/a.cc:3: uses O_DIRECT, which is Linux-specific",
            "src/runtime/a.cc:3: uses getrandom, which is Linux-specific",
            "src/runtime/a.cc:4: names /proc/, which is Linux's",
        ])

    def test_linux_only_file_and_thread_calls_are_found(self):
        problems = self.check({"src/runtime/a.cc": (
            "#include <sys/sysmacros.h>\n"
            "int f(int fd) { struct statx s; return ::statx(fd, \"\", AT_EMPTY_PATH, 0, &s) + "
            "::posix_fadvise(fd, 0, 0, 0) + ::pthread_setaffinity_np(0, 0, nullptr); }\n")})
        self.assertEqual(problems, [
            "src/runtime/a.cc:1: includes sys/sysmacros.h, which is Linux-specific",
            "src/runtime/a.cc:2: uses statx, which is Linux-specific",
            "src/runtime/a.cc:2: uses statx, which is Linux-specific",
            "src/runtime/a.cc:2: uses AT_EMPTY_PATH, which is Linux-specific",
            "src/runtime/a.cc:2: uses posix_fadvise, which is Linux-specific",
            "src/runtime/a.cc:2: uses pthread_setaffinity_np, which is Linux-specific",
        ])

    def test_the_ring_is_included_only_where_it_belongs(self):
        self.assertEqual(self.check({"src/engine/a.cc": '#include "providers/uring_storage.h"\n'}),
                         ["src/engine/a.cc:1: includes providers/uring_storage.h, which is Linux-specific"])

    def test_posix_comments_and_look_alikes_pass(self):
        self.assertEqual(self.check({"src/runtime/a.cc": (
            "#include <fcntl.h>\n#include <poll.h>\n#include <sys/socket.h>\n"
            "// epoll_wait, O_TMPFILE and cudaMalloc in a comment\n"
            "int g = ::open(\"x\", O_RDONLY | O_DIRECTORY | O_CLOEXEC);\n"
            "const char* url = \"https://example.com/sys/x\"; int io_uringish_count = 0;\n"
            "int cuda_count = 0; auto s = providers::cuda::Open(); // int cublas_bytes = 0;\n"
            "Log(\"cudaMalloc failed\"); Log(\"CUDA_VISIBLE_DEVICES\");\n")}), [])

    def test_cuda_belongs_to_its_provider_and_the_kernels(self):
        cuda = ('#include <cuda_runtime.h>\n#include "providers/cuda/cuda_facts.h"\n'
                "cudaStream_t s; CUevent_st* e; int r = cuMemcpyAsync (0, 0, 0, s);\n"
                "__global__ void K() {}\nvoid L() { K<<<1, 1>>>(); cudaStreamSynchronize(s); }\n")
        self.assertEqual(self.check({"src/providers/cuda/a.cc": cuda, "src/kernels/ggml/a.cu": cuda,
                                     "src/kernels/device/a.cuh": cuda}), [])
        # The common kernel interface is vendor-free.
        self.assertEqual(self.check({"src/kernels/common/a.cc": "cudaStream_t s;\n"}),
                         ["src/kernels/common/a.cc:1: uses cudaStream_t, which is CUDA's"])
        problems = self.check({"src/engine/a.cc": cuda, "src/engine/b.cu": "int x;\n"})
        self.assertEqual(problems, [
            "src/engine/a.cc:1: includes cuda_runtime.h, which is CUDA's",
            "src/engine/a.cc:2: includes providers/cuda/cuda_facts.h, which is CUDA's",
            "src/engine/a.cc:3: uses cudaStream_t, which is CUDA's",
            "src/engine/a.cc:3: uses CUevent_st, which is CUDA's",
            "src/engine/a.cc:3: uses cuMemcpyAsync, which is CUDA's",
            "src/engine/a.cc:4: uses __global__, which is CUDA's",
            "src/engine/a.cc:5: uses <<<, which is CUDA's",
            "src/engine/a.cc:5: uses cudaStreamSynchronize, which is CUDA's",
            "src/engine/b.cu: a CUDA source file",
        ])

    def test_kernel_libraries_stay_out_of_engine_and_runtime(self):
        sample = (
            '#include "kernels/ggml/cublas.h"\n'
            '#include "kernels/exl3/packed_tuning.h"\n'
            '#include "kernels/cutlass/gemm.h"\n'
            'int f() {\n'
            '  ggml_tensor* t = nullptr;\n'
            '  CublasHandle* h = nullptr;\n'
            '  int cublas_bytes = 0;\n'
            '  Exl3Source s;\n'
            '  cutlass_gemm g;\n'
            '  return 0;\n'
            '}\n'
        )
        self.assertEqual(self.check({"src/kernels/ggml/a.cc": sample.replace(
            '#include "kernels/exl3/packed_tuning.h"\n#include "kernels/cutlass/gemm.h"\n', "")}), [])
        problems_engine = self.check({"src/engine/a.cc": sample})
        self.assertEqual(problems_engine, [
            "src/engine/a.cc:1: includes kernels/ggml/cublas.h, which names a kernel library",
            "src/engine/a.cc:2: includes kernels/exl3/packed_tuning.h, which names a kernel library",
            "src/engine/a.cc:3: includes kernels/cutlass/gemm.h, which names a kernel library",
            "src/engine/a.cc:5: uses ggml_tensor, which names a kernel library",
            "src/engine/a.cc:6: uses CublasHandle, which names a kernel library",
            "src/engine/a.cc:7: uses cublas_bytes, which names a kernel library",
            "src/engine/a.cc:8: uses Exl3Source, which names a kernel library",
            "src/engine/a.cc:9: uses cutlass_gemm, which names a kernel library",
        ])
        problems_runtime = self.check({"src/runtime/a.cc": sample})
        self.assertEqual(problems_runtime, [
            "src/runtime/a.cc:1: includes kernels/ggml/cublas.h, which names a kernel library",
            "src/runtime/a.cc:2: includes kernels/exl3/packed_tuning.h, which names a kernel library",
            "src/runtime/a.cc:3: includes kernels/cutlass/gemm.h, which names a kernel library",
            "src/runtime/a.cc:5: uses ggml_tensor, which names a kernel library",
            "src/runtime/a.cc:6: uses CublasHandle, which names a kernel library",
            "src/runtime/a.cc:7: uses cublas_bytes, which names a kernel library",
            "src/runtime/a.cc:8: uses Exl3Source, which names a kernel library",
            "src/runtime/a.cc:9: uses cutlass_gemm, which names a kernel library",
        ])

    def test_embedded_names_strings_and_include_spellings_are_found(self):
        sample = (
            '#include "../kernels/ggml/launch.h"\n'
            '#include "src/kernels/exl3/linear.h"\n'
            '#define LLMP_HEADER "kernels/ggml/tensors.h"\n'
            '#include LLMP_HEADER\n'
            '#include LLMP_UNKNOWN\n'
            'bool UsesCublas(int);\n'
            'auto layout = format::Layout::kGgmlBlocks;\n'
            'auto implementation = "ggml.mul_mat";\n'
            'auto container = "exl3"; auto key = "tokenizer.ggml.tokens";\n'
        )
        for directory in ("engine", "runtime", "scheduler"):
            with self.subTest(directory=directory):
                name = f"src/{directory}/a.cc"
                self.assertEqual(self.check({name: sample}), [
                    f"{name}:1: includes kernels/ggml/launch.h, which names a kernel library",
                    f"{name}:2: includes kernels/exl3/linear.h, which names a kernel library",
                    f"{name}:4: includes kernels/ggml/tensors.h, which names a kernel library",
                    f"{name}:5: includes LLMP_UNKNOWN, a macro the check cannot resolve",
                    f"{name}:6: uses UsesCublas, which names a kernel library",
                    f"{name}:7: uses kGgmlBlocks, which names a kernel library",
                    f"{name}:8: names ggml, which names a kernel library",
                ])
        # The planner, components, specs, model and the common kernel
        # interface may name the format layer's layouts, nothing else.
        for directory in ("planner", "components", "spec", "model"):
            with self.subTest(directory=directory):
                name = f"src/{directory}/a.cc"
                self.assertEqual(self.check({name: sample}), [
                    f"{name}:1: includes kernels/ggml/launch.h, which names a kernel library",
                    f"{name}:2: includes kernels/exl3/linear.h, which names a kernel library",
                    f"{name}:4: includes kernels/ggml/tensors.h, which names a kernel library",
                    f"{name}:5: includes LLMP_UNKNOWN, a macro the check cannot resolve",
                    f"{name}:6: uses UsesCublas, which names a kernel library",
                    f"{name}:8: names ggml, which names a kernel library",
                ])

    def test_kernel_sources_meet_only_through_the_common_interface(self):
        self.assertEqual(self.check({
            "src/kernels/decode/a.cc": ('#include "kernels/common/selected_operation.h"\n'
                                        '#include "kernels/decode/products.h"\n'
                                        '#include "../ggml/tensors.h"\n'
                                        '#include "kernels/exl3/linear.h"\n'),
            "src/kernels/compose/a.cc": ('#include "kernels/ggml/step_program.h"\n'
                                         '#include "kernels/decode/selected_source.h"\n'
                                         '#include "kernels/exl3/selected_source.h"\n'),
            "src/kernels/common/a.cc": ('#include "kernels/common/tensor_ref.h"\n'
                                        '#include "../paging/paging.h"\n'
                                        'auto layout = format::Layout::kGgmlBlocks;\n'
                                        'bool UsesCublas(int);\n'),
        }), [
            "src/kernels/common/a.cc:2: includes kernels/paging/paging.h, another kernel source's",
            "src/kernels/common/a.cc:4: uses UsesCublas, which names a kernel library",
            "src/kernels/decode/a.cc:3: includes kernels/ggml/tensors.h, another kernel source's",
            "src/kernels/decode/a.cc:4: includes kernels/exl3/linear.h, another kernel source's",
        ])

    def test_ir_and_execution_reject_vendor_code_and_diagnostics(self):
        for directory in ("ir", "execution"):
            for suffix in (".cc", ".h", ".inc"):
                for snippet in (
                    '#include "kernels/ggml/launch.h"',
                    'auto bytes = info.ggml->block_bytes;',
                    'auto layout = format::Layout::kGgmlBlocks;',
                    'auto encoding = format::Encoding::kExl3Trellis;',
                    'int my_cutlass_choice = 0;',
                    'Fail("GGML requires whole blocks");',
                    'auto implementation = "ggml.mul_mat";',
                    'auto model = "NVIDIA GB10";',
                    'auto model = "GB10";',
                    'constexpr int kGb10Rank = 1;',
                    'auto architecture = "sm_121";',
                    'auto architecture = "sm_121a";',
                    'constexpr int kSm121Rank = 1;',
                    'constexpr int kSm_121aRank = 1;',
                    'auto provider = providers::cuda::Open();',
                ):
                    with self.subTest(directory=directory, suffix=suffix, snippet=snippet):
                        name = f"src/{directory}/vendor{suffix}"
                        self.assertTrue(self.check({name: snippet + "\n"}))

    def test_ir_and_execution_keep_neutral_facts_and_provenance(self):
        sample = ('// SPDX-FileCopyrightText: 2023-2026 The ggml authors\n'
                  '// Original CUDA behavior on GB10 (sm_121a) is provenance.\n'
                  'auto unit = format::AddressUnitOf(tensor.format);\n'
                  'auto encoding = format::Encoding::kNvfp4;\n'
                  'auto rank = DeviceRank(query, implementation, table);\n')
        self.assertEqual(self.check({"src/ir/a.inc": sample, "src/execution/a.h": sample}), [])

    def test_only_sources_under_src_are_checked(self):
        checked, problems = boundaries.check(self.root, [])
        self.assertEqual((checked, problems), (0, []))
        self.assertEqual(self.check({"tests/unit/a.cc": "cudaStream_t s;\n", "src/notes.md": "epoll_wait\n"}), [])

    def test_exceptions_are_named_and_kept_exact(self):
        with mock.patch.dict(boundaries.EXCEPTIONS, {"src/runtime/a.cc": {boundaries.LINUX: "why"}}):
            self.assertEqual(self.check({"src/runtime/a.cc": "int e = eventfd(0, 0);\n"}), [])
            self.assertEqual(self.check({"src/runtime/a.cc": "int e = 0;\n"}), [
                "src/runtime/a.cc: allowed an exception to the linux rule it no longer needs"])
            self.assertEqual(self.check({"src/runtime/a.cc": "cudaStream_t s;\n"}), [
                "src/runtime/a.cc:1: uses cudaStream_t, which is CUDA's",
                "src/runtime/a.cc: allowed an exception to the linux rule it no longer needs"])


class ThisTree(unittest.TestCase):
    def test_this_trees_sources_keep_to_their_boundaries(self):
        checked, problems = boundaries.check(REPO, boundaries.scope_files(REPO))
        self.assertGreater(checked, 100)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
