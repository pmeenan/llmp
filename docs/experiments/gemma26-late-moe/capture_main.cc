// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
// External host-only wrapper around the unchanged original C API client source.
#include <dlfcn.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int joined_main(int argc, char** argv);
int main(int argc, char** argv) {
  umask(0077);
  if (argc != 6 || std::strcmp(argv[3], "4") || std::strcmp(argv[4], "128") ||
      std::strlen(argv[2]) > 3900 || std::getenv("GGML_CUDA_DISABLE_GRAPHS") ||
      std::getenv("GGML_CUDA_DISABLE_FUSION"))
    return 2;
  using Prepare = int (*)(const char*);
  using Finish = int (*)();
  auto prepare = reinterpret_cast<Prepare>(dlsym(RTLD_DEFAULT, "jitllm_late_prepare"));
  auto finish = reinterpret_cast<Finish>(dlsym(RTLD_DEFAULT, "jitllm_late_finish"));
  if (!prepare || !finish) return 2;
  const std::string captures = std::string(argv[2]) + "-operators";
  if (prepare(captures.c_str()) != 0) std::_Exit(1);
  // The original source performs complete getters and explicit batch/context/model/backend frees.
  const int result = joined_main(argc, argv);
  if (result != 0) std::_Exit(result);  // Never release snapshot owners after failed retirement.
  if (finish() != 0) std::_Exit(1);
  std::fprintf(stderr, "LATE_CLIENT_EXIT0 explicit_public_backend_teardown=source_verified\n");
  return 0;
}
