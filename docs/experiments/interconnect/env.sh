# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# Source this file on either node; all dependencies stay in the experiment directory.
export LLMP_NET_ROOT=/home/pmeenan/.local/share/llmp/interconnect
export OPAL_PREFIX="$LLMP_NET_ROOT/mpi-root/usr"
export OPAL_LIBDIR="$OPAL_PREFIX/lib/aarch64-linux-gnu"
export OPAL_DATADIR="$OPAL_PREFIX/share"
export OPAL_SYSCONFDIR="$OPAL_PREFIX/etc"
export OMPI_MCA_mca_base_component_path="$OPAL_LIBDIR/openmpi/lib/openmpi3"
export PATH="$OPAL_PREFIX/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$LLMP_NET_ROOT/nccl-runtime/lib:$OPAL_LIBDIR:$OPAL_LIBDIR/openmpi/lib:/usr/local/cuda/lib64"
