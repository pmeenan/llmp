#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -eu
source /home/pmeenan/.local/share/llmp/interconnect/env.sh
exec "$OPAL_PREFIX/bin/orted" "$@"
