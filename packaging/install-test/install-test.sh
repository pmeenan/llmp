#!/bin/sh
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# The arm64 install test (D-074), run as root in the install-test image with
# no network and /in holding jitllm_<version>_arm64.deb, the version it must
# report and the CUDA driver stub (libcuda.so.1), and a volume at
# /var/lib/jitllm, since the container's own overlay filesystem is one the
# storage roles refuse. It installs the package
# over a stand-in for the driver's libcuda.so.1, checks the installed layout
# (D-063), runs both executables, reinstalls it as an upgrade, then removes
# and purges it and checks what stays. The unit is not started: there is no
# service manager here and no GPU; the runtime is run directly as `jitllm`
# instead, and refuses this host at its platform step (exit 75).
set -eu

fail() { echo "install-test: FAILED: $*" >&2; exit 1; }
check() { desc=$1; shift; "$@" || fail "$desc"; echo "install-test: ok: $desc"; }
stat_is() { [ "$(stat -c '%a %U:%G' "$1")" = "$2" ] || { echo "$1 is $(stat -c '%a %U:%G' "$1"), not $2" >&2; return 1; }; }

check "this is arm64" [ "$(uname -m)" = aarch64 ]
check "there is no network" [ "$(ls /sys/class/net)" = lo ]
check "/var/lib/jitllm is a volume, not the overlay" sh -c '! stat -f -c %T /var/lib/jitllm | grep -q overlay'
echo "install-test: systemd $(dpkg-query -W -f '${Version}' systemd)"
deb=$(ls /in/jitllm_*_arm64.deb)
version=$(cat /in/version)

# A stand-in for the NVIDIA driver's package: it provides libcuda.so.1 with
# NVIDIA's stub, which binaries load but whose cuInit fails.
mkdir -p /tmp/driver/DEBIAN /tmp/driver/usr/lib/aarch64-linux-gnu
cp /in/libcuda.so.1 /tmp/driver/usr/lib/aarch64-linux-gnu/libcuda.so.1
cat > /tmp/driver/DEBIAN/control <<CONTROL
Package: jitllm-test-driver
Version: 580.0-1
Architecture: arm64
Provides: libcuda.so.1 (= 580.0-1)
Maintainer: jitLLM install test <test@jitllm.invalid>
Description: stand-in for libcuda.so.1 in the jitLLM install test
CONTROL
dpkg-deb --root-owner-group --build /tmp/driver /tmp/driver.deb >/dev/null
check "the stand-in driver package installs" dpkg -i /tmp/driver.deb
ldconfig

check "the package installs" dpkg -i "$deb"
check "jitllm is a system user without a login shell, at home in /var/lib/jitllm" \
  sh -c 'getent passwd jitllm | grep -Eq "^jitllm:x:[0-9]+:[0-9]+:[^:]*:/var/lib/jitllm:/usr/sbin/nologin$"'
check "the jitllm group exists" getent group jitllm
check "/var/lib/jitllm is jitllm's, 0755" stat_is /var/lib/jitllm "755 jitllm:jitllm"
check "the checkpoint store is jitllm's, 1777" stat_is /var/lib/jitllm/checkpoints "1777 jitllm:jitllm"
check "the executables are root's, 0755" sh -c 'stat_is() { [ "$(stat -c "%a %U:%G" "$1")" = "$2" ]; };
  stat_is /usr/bin/jitllm "755 root:root" && stat_is /usr/libexec/jitllm/jitllm-runtime "755 root:root"'
check "no configuration file is shipped" sh -c '! ls /etc/jitllm/jitllm.toml /etc/jitllm/jitllm.d 2>/dev/null'
check "the unit is enabled" test -L /etc/systemd/system/multi-user.target.wants/jitllm.service
check "systemd accepts the unit" systemd-analyze verify --man=no /usr/lib/systemd/system/jitllm.service
check "jitllm --version reports $version" sh -c "jitllm --version | head -1 | grep -qxF 'jitllm $version'"

set +e
jitllm doctor > /tmp/doctor.txt 2>&1
doctor=$?
set -e
cat /tmp/doctor.txt
check "doctor fails without a GPU (exit 1), having reported the configuration" \
  sh -c "[ $doctor -eq 1 ] && grep -q '^  node: standalone$' /tmp/doctor.txt && grep -q '^storage$' /tmp/doctor.txt"

set +e
setpriv --reuid=jitllm --regid=jitllm --init-groups /usr/libexec/jitllm/jitllm-runtime > /tmp/runtime.txt 2>&1
runtime=$?
set -e
cat /tmp/runtime.txt
check "the runtime refuses this host at its platform step (exit 75)" \
  sh -c "[ $runtime -eq 75 ] && grep -q 'refusing to start: this host cannot run this build now' /tmp/runtime.txt"
check "the runtime made its roles" sh -c 'stat_is() { [ "$(stat -c "%a %U:%G" "$1")" = "$2" ]; };
  stat_is /var/lib/jitllm/models "755 jitllm:jitllm" && stat_is /var/lib/jitllm/spill "700 jitllm:jitllm" &&
  stat_is /var/lib/jitllm/state "700 jitllm:jitllm" && stat_is /var/lib/jitllm/enrollment.lock "600 jitllm:jitllm"'
check "doctor now finds the roles" sh -c 'jitllm doctor 2>&1 | grep -q "^  installed: /var/lib/jitllm/models, owner uid"'

check "the package reinstalls as an upgrade" dpkg -i "$deb"
check "the package removes" dpkg -r jitllm
check "its files are gone" sh -c '! test -e /usr/bin/jitllm && ! test -e /usr/libexec/jitllm/jitllm-runtime'
check "the data directory and user stay" sh -c 'test -d /var/lib/jitllm/models && getent passwd jitllm >/dev/null'
check "a kept conversation stands in" setpriv --reuid=jitllm --regid=jitllm --init-groups \
  sh -c 'umask 077 && mkdir -p /var/lib/jitllm/spill/conversations/a && : > /var/lib/jitllm/spill/conversations/a/slot-0.record'
check "the package purges" dpkg -P jitllm
check "the data directory and user still stay" sh -c 'test -d /var/lib/jitllm/models && getent passwd jitllm >/dev/null'
check "the kept conversations are gone (D-105)" sh -c '! test -e /var/lib/jitllm/spill/conversations && test -d /var/lib/jitllm/spill'
check "the unit is no longer enabled" sh -c '! test -e /etc/systemd/system/multi-user.target.wants/jitllm.service'
echo "install-test: passed"
