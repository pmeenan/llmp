<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GCC 16.2 runtime, statically linked

D-060 replaces D-059's GCC 14.2 library headers. Llmpalooza now uses the **GCC
16.2 C++ runtime**: libstdc++, libgcc and libatomic, built from signed source.
Executables link it statically, and link the CUDA runtime statically too.
Clang 22.1.8 still compiles all host code, including NVCC's host passes.
GCC's own compiler binaries are only used to build these libraries.

Everything below ran on 2026-09-23, on the workstation and on `spark`
(`spark-c4e2`, GB10, driver 580.178.04). All builds went to external
scratch; no host packages or defaults changed. This is an M0 toolchain proof.
Folding GCC into the persistent SDK, and the application build, are M1 work.

## Versions

Checked 2026-09-23 against upstream sources, on the principle of the newest
release compatible with the target runtime (glibc 2.39, driver R580) and
with NVCC:

| Component | Pin | Why this version |
| --- | --- | --- |
| GCC runtime | 16.2.0 (released 2026-08-07) | Newest GCC release. NVIDIA's CUDA 13.4 Update 1 guide supports GCC 6–16 and libstdc++ only |
| LLVM / Clang / tools | 22.1.8 | Newest Clang that NVCC 13.4 accepts as host compiler (7–22). LLVM 23.1.2 exists; the owner chose to stay on 22 rather than split compilers |
| CUDA | 13.4.92 packages, CCCL 13.3.4.3.1 | Newest in NVIDIA's Ubuntu 24.04 repositories for every pinned component, on both architectures. Runs on the R580 driver through CUDA minor-version compatibility (D-032) |
| CMake | 4.4.3 | Newest release (4.3.5 and 4.2.8 are later patch releases of older series) |
| Ninja | 1.13.2 | Newest release |
| GoogleTest | 1.18.0 | Newest release |
| glibc, NVIDIA driver | System: 2.39, 580.178.04 | Platform runtime; not changed |

NCCL and cuBLAS are not pinned yet. See the linking policy below.

## Building the runtime

[pins.json](pins.json) records the source identities:

- **Source.** `gcc-16.2.0.tar.xz` passes `gpgv` against the GNU keyring. The
  signature uses RSA subkey `7F74F97C103468EE5D750B583AB00996FC26A641` of
  primary key `13975A70E63C361C73AE69EF6EEB81F8981C74C7`, which belongs to
  Richard Biener (GCC release manager) and is listed under his former name,
  Richard Guenther.
- **Prerequisites.** GMP, MPFR, MPC, ISL and gettext came from
  `contrib/download_prerequisites`, which checks each one's SHA-512 against
  the list shipped in the signed tarball. They go into the compiler only.
- **Build.** [build-gcc.sh](build-gcc.sh) builds natively on each host into
  a new prefix: C and C++ only, no bootstrap, default PIE, and libsanitizer,
  libgomp, libitm, libssp, libvtv and libquadmath disabled. The system GCC
  13 builds the GCC 16 compiler, and that new compiler then builds the
  target runtime libraries. `make install-strip` leaves the runtime archives
  without debug info, so debuggers do not step into libstdc++ source.

Llmpalooza uses only these outputs:

- the `include/c++/16` headers;
- `libstdc++.a` and `libsupc++.a` (`libstdc++exp.a` is built but not linked
  in release builds; see below);
- `libatomic.a`;
- the `crt*.o` objects, `libgcc.a` and `libgcc_eh.a` in `lib/gcc/<triple>/16`.

Every compile selects them with an explicit
`--gcc-install-dir=<prefix>/lib/gcc/<triple>/16`.

**Finding: cross builds need the runtime inside the sysroot.** With
`--sysroot`, Clang searches the GCC install's parent library directory
(`<install>/../../../../lib64`) only if that directory sits inside the
sysroot. With the ARM runtime outside it, the cross link failed with
`unable to find library -lstdc++`. Placing the ARM install at
`<sysroot>/opt/gcc-16.2.0` fixed it. Native builds, which have no sysroot,
are unaffected.

## Results

Static linking used `-static-libstdc++ -static-libgcc`. CUDA programs link
`libcudart_static.a` with `-ldl -lrt -lpthread`.

| Check | Workstation x86-64 | `spark` AArch64 |
| --- | --- | --- |
| [probe.cc](probe.cc): `<flat_map>`, `<flat_set>`, `<mdspan>` with multidimensional `[]`, `<print>`, `<generator>`, monadic `std::expected`, `std::ranges::to`, exception unwinding | Passed (native) | Passed (native and cross-built) |
| Feature macros | `__GLIBCXX__=20260807`, `__cpp_lib_print=202406`, `__cpp_lib_flat_map=202511`, `__cpp_lib_mdspan=202207` | Same |
| Probe's dynamic dependencies | `libc`, `libm`, `ld-linux-x86-64`; highest symbol version `GLIBC_2.38` | `libc`, `libm`; highest `GLIBC_2.38` |
| [cuda_probe.cu](cuda_probe.cu): NVCC 13.4, Clang 22 host compiler, GCC 16 headers in host code, `-Werror all-warnings`, static cudart | Compiled (no GPU on this host) | Native and cross builds ran on the GB10; dependencies `libc`, `libm` only |
| D-032 CPU/CUDA smoke with the static runtimes | Not run (no GPU) | Passed: C++23 checks, `libstdc++=20260807`, 257 CUDA values; run without `LD_LIBRARY_PATH` |
| GoogleTest suite ([dev-tools](../dev-tools/README.md) harness, `STATIC_RUNTIME=1`) | 10/10; negative control exit 8 | Native 10/10. Cross build run from the workstation through CTest over SSH 10/10. Both negative controls exit 8 |
| ASan+UBSan suite with static runtimes; symbolizer probe | 10/10; frame resolves to `asan_probe.cc:12` | 10/10; same |
| clang-tidy (harness config), clangd `--check`, clang-format | Clean. clangd shows only its 22 known code-action failures in gMock macros | Same |

CUDA runs used `CUDA_DISABLE_PTX_JIT=1`. Static linking made the test binary
(`RelWithDebInfo`, with debug info) 7,547,952 bytes, against 5,868,160 bytes
when dynamically linked against GCC 14.

## Linking policy this supports

The runtime dependencies are glibc (`libc`, `libm`, the dynamic loader) and
the driver's `libcuda`. The static CUDA runtime loads `libcuda` at run time.
On `spark`, the libraries that must stay dynamic depend only on C
libraries:

- **`libcuda.so.1`** needs libc, libm, libdl, librt and libpthread.
- **`libnvidia-ml.so.1`** needs the same set.
- **`libibverbs.so.1`** needs libc and libnl, and loads its providers as
  plugins.

None of them brings in a second C++ runtime. NCCL, once pinned, must be
linked statically or built with `-static-libstdc++`. The interconnect
experiment's source-built NCCL 2.30.7 depends on `libstdc++.so.6`. cuBLAS
must be static too, subject to its license and binary-size review.

## Reproduction

1. Download the GCC tarball, its signature and the GNU keyring from
   [pins.json](pins.json), then run
   `gpgv --keyring ./gnu-keyring.gpg gcc-16.2.0.tar.xz.sig gcc-16.2.0.tar.xz`
   and compare the SHA-256.
2. Extract the tarball outside the repository and run
   `contrib/download_prerequisites` in the source tree.
3. On each host, run
   `bash build-gcc.sh gcc-16.2.0 BUILD_DIR /abs/new-prefix`. The build uses
   the host's system GCC and make. Measured scratch: 1.6 GB for the prepared
   source, 4.3–4.7 GB per build tree, and 232–245 MB per install prefix.
4. For cross builds, copy D-032's Spark sysroot snapshot and place the ARM
   prefix at `<sysroot>/opt/gcc-16.2.0`.
5. Build [probe.cc](probe.cc) with
   `clang++-22 -std=c++23 --gcc-install-dir=<prefix>/lib/gcc/<triple>/16 -static-libstdc++ -static-libgcc`,
   adding `--target`, `--sysroot` and `--ld-path` for cross builds.
6. Compile [cuda_probe.cu](cuda_probe.cu) with NVCC. Pass
   `-ccbin <wrapper>`, where the wrapper adds `--gcc-install-dir` and, for
   cross builds, the target, sysroot and LLD. Link with the same Clang
   wrapper against `libcudart_static.a`.
7. For the GoogleTest suite, run the [dev-tools](../dev-tools/README.md)
   instructions with `STATIC_RUNTIME=1` and `GCC_INSTALL_DIR` pointing at
   the GCC 16 `lib/gcc/<triple>/16` directory; for cross builds, use the
   copy inside the sysroot.
8. Inspect the results with `readelf -d` and `objdump -T`.

## Provenance and use

The primary terms of libstdc++, libgcc and libatomic are GPL-3.0-or-later
WITH GCC Runtime Library Exception 3.1, with embedded components under the
additional terms below. D-017's declared platform dependency family covers
the selected platform's libstdc++ and compiler support runtimes; D-060
selects this source-built runtime instead of Ubuntu's and links it
statically.

The [exception](https://www.gnu.org/licenses/gcc-exception-3.1.html)
permits the static combination only if all Target Code in it was generated
by Eligible Compilation Processes: ones using GCC alone or with
GPL-compatible software, or ones using no work based on GCC. Llmpalooza's code
takes the second route. Neither Clang nor NVCC is based on GCC; NVCC's
proprietary passes are not GPL-compatible, so compatibility is not the
argument. Statically linked third-party archives (`libcudart_static.a`, and
later NCCL or cuBLAS) need the same check. The owner confirmed this reading
on 2026-09-23 (D-060).

The static archives and installed headers also embed code under other
licenses. Whether a notice ships depends on what a binary incorporates:

- `libstdc++.a`: Ryu (Apache-2.0 OR BSL-1.0) and fast_float
  (MIT in the pinned `src/c++17/fast_float/fast_float.h`). A static test
  linked 19 `ryu::` symbols when printing a `double` with `std::println`,
  and no fast_float. A test using
  `std::from_chars` on a `double` linked 23 `fast_float::` symbols and no
  Ryu. Under BSL-1.0, Ryu needs no notice for object code; if fast_float is
  linked, its MIT notice ships;
- `libstdc++exp.a`: libbacktrace (BSD-3-Clause), used by `<stacktrace>`.
  In GCC 16.2 it is present only in this archive, not in `libstdc++.a`. None
  of the test, sanitizer or probe binaries contains libbacktrace symbols
  (`__glibcxx_backtrace_*`), so release builds that avoid `std::stacktrace`
  have no libbacktrace notice to ship;
- the `<execution>` headers include PSTL (Apache-2.0 WITH LLVM-exception);
- libstdc++ headers, including
  [`bits/stl_vector.h`](https://github.com/gcc-mirror/gcc/blob/releases/gcc-16.2.0/libstdc%2B%2B-v3/include/bits/stl_vector.h)
  and `bits/stl_algobase.h`, retain Hewlett-Packard and Silicon Graphics
  permission notices. Preserve their copyright and permission text in the
  supporting documentation shipped with code using these headers, including
  the probe's `std::vector`;
- the `bits/shared_ptr*.h` and `tr1/shared_ptr.h` headers include
  Boost-derived code under BSL-1.0. As with Ryu, object code needs no notice,
  but redistributed headers retain it. Other installed headers retain
  Jeremy Siek's permission notice (`bits/boost_concept_check.h`) and
  IBM-HRL permission notices (`ext/pb_ds/*`, `ext/typelist.h` and
  `ext/throw_allocator.h`); retain the applicable texts when used or shipped;
- `libgcc.a` on both architectures contains glibc soft-fp objects, including
  `addtf3.o`. Their
  [source terms](https://github.com/gcc-mirror/gcc/blob/releases/gcc-16.2.0/libgcc/soft-fp/addtf3.c)
  are LGPL-2.1-or-later with a separate permission to distribute compiled
  combinations without restrictions from these files. That permission does
  not waive the LGPL obligations for modifications or distribution outside
  a combined executable, such as shipping the runtime archive in an SDK.

M1's notices and SBOM must record the static runtimes, these embedded
components and the GCC source identity, and audit the exact used or shipped
header set; these examples are not an exhaustive header-license inventory.
*Corrected 2026-09-24 by M1's audit* ([licensing.md](../../licensing.md#what-builds-llmpalooza),
D-071): Ryu links with any `<format>` or `<print>` use, not only floating
point; `cp-demangle.o` (GPL-2.0-or-later WITH GCC-exception-2.0) is always
linked; the AArch64 soft-fp objects that link are the TF comparisons;
PSTL headers arrive through `<algorithm>`, `<memory>` and `<numeric>`; and
the tz database, `<format>`'s Unicode-derived tables, `<barrier>` and the
compiled HP/SGI objects (`tree.o`, `list.o`) belong on this list.
GCC's compiler binaries and the
in-tree GMP, MPFR, MPC, ISL and gettext are build tools only; nothing from
them links into llmpalooza.

No archive, build tree or binary is committed.
