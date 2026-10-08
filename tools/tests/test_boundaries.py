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
            "int cuda_count = 0; auto s = providers::cuda::Open(); int cublas_bytes = 0;\n"
            "Log(\"cudaMalloc failed\"); Log(\"CUDA_VISIBLE_DEVICES\");\n")}), [])

    def test_cuda_belongs_to_its_provider_and_the_kernels(self):
        cuda = ('#include <cuda_runtime.h>\n#include "providers/cuda/cuda_facts.h"\n'
                "cudaStream_t s; CUevent_st* e; int r = cuMemcpyAsync (0, 0, 0, s);\n"
                "__global__ void K() {}\nvoid L() { K<<<1, 1>>>(); cublasCreate(nullptr); }\n")
        self.assertEqual(self.check({"src/providers/cuda/a.cc": cuda, "src/kernels/x/a.cu": cuda}), [])
        problems = self.check({"src/engine/a.cc": cuda, "src/engine/b.cu": "int x;\n"})
        self.assertEqual(problems, [
            "src/engine/a.cc:1: includes cuda_runtime.h, which is CUDA's",
            "src/engine/a.cc:2: includes providers/cuda/cuda_facts.h, which is CUDA's",
            "src/engine/a.cc:3: uses cudaStream_t, which is CUDA's",
            "src/engine/a.cc:3: uses CUevent_st, which is CUDA's",
            "src/engine/a.cc:3: uses cuMemcpyAsync, which is CUDA's",
            "src/engine/a.cc:4: uses __global__, which is CUDA's",
            "src/engine/a.cc:5: uses <<<, which is CUDA's",
            "src/engine/a.cc:5: uses cublasCreate, which is CUDA's",
            "src/engine/b.cu: a CUDA source file",
        ])

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
