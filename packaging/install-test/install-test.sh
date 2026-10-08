#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# The arm64 install test (D-074), run as root in the install-test image with
# no network and /in holding llmp_<version>_arm64.deb, the version it must
# report and the CUDA driver stub (libcuda.so.1), and a volume at
# /var/lib/llmp, since the container's own overlay filesystem is one the
# storage roles refuse. It installs the package
# over a stand-in for the driver's libcuda.so.1, checks the installed layout
# (D-063), runs both executables, reinstalls it as an upgrade, then removes
# and purges it and checks what stays. The unit is not started: there is no
# service manager here and no GPU; the runtime is run directly as `llmp`
# instead, and refuses this host at its platform step (exit 75).
set -eu

fail() { echo "install-test: FAILED: $*" >&2; exit 1; }
check() { desc=$1; shift; "$@" || fail "$desc"; echo "install-test: ok: $desc"; }
stat_is() { [ "$(stat -c '%a %U:%G' "$1")" = "$2" ] || { echo "$1 is $(stat -c '%a %U:%G' "$1"), not $2" >&2; return 1; }; }

check "this is arm64" [ "$(uname -m)" = aarch64 ]
check "there is no network" [ "$(ls /sys/class/net)" = lo ]
check "/var/lib/llmp is a volume, not the overlay" sh -c '! stat -f -c %T /var/lib/llmp | grep -q overlay'
echo "install-test: systemd $(dpkg-query -W -f '${Version}' systemd)"
deb=$(ls /in/llmp_*_arm64.deb)
version=$(cat /in/version)

# A stand-in for the NVIDIA driver's package: it provides libcuda.so.1 with
# NVIDIA's stub, which binaries load but whose cuInit fails.
mkdir -p /tmp/driver/DEBIAN /tmp/driver/usr/lib/aarch64-linux-gnu
cp /in/libcuda.so.1 /tmp/driver/usr/lib/aarch64-linux-gnu/libcuda.so.1
cat > /tmp/driver/DEBIAN/control <<CONTROL
Package: llmp-test-driver
Version: 580.0-1
Architecture: arm64
Provides: libcuda.so.1 (= 580.0-1)
Maintainer: llmpalooza install test <test@llmp.invalid>
Description: stand-in for libcuda.so.1 in the llmpalooza install test
CONTROL
dpkg-deb --root-owner-group --build /tmp/driver /tmp/driver.deb >/dev/null
check "the stand-in driver package installs" dpkg -i /tmp/driver.deb
ldconfig

check "the package installs" dpkg -i "$deb"
check "llmp is a system user without a login shell, at home in /var/lib/llmp" \
  sh -c 'getent passwd llmp | grep -Eq "^llmp:x:[0-9]+:[0-9]+:[^:]*:/var/lib/llmp:/usr/sbin/nologin$"'
check "the llmp group exists" getent group llmp
check "/var/lib/llmp is llmp's, 0755" stat_is /var/lib/llmp "755 llmp:llmp"
check "the checkpoint store is llmp's, 1777" stat_is /var/lib/llmp/checkpoints "1777 llmp:llmp"
check "the executables are root's, 0755" sh -c 'stat_is() { [ "$(stat -c "%a %U:%G" "$1")" = "$2" ]; };
  stat_is /usr/bin/llmp "755 root:root" && stat_is /usr/libexec/llmp/llmp-runtime "755 root:root"'
check "no configuration file is shipped" sh -c '! ls /etc/llmp/llmp.toml /etc/llmp/llmp.d 2>/dev/null'
check "the unit is enabled" test -L /etc/systemd/system/multi-user.target.wants/llmp.service
check "systemd accepts the unit" systemd-analyze verify --man=no /usr/lib/systemd/system/llmp.service
check "llmp --version reports $version" sh -c "llmp --version | head -1 | grep -qxF 'llmp $version'"

set +e
llmp doctor > /tmp/doctor.txt 2>&1
doctor=$?
set -e
cat /tmp/doctor.txt
check "doctor fails without a GPU (exit 1), having reported the configuration" \
  sh -c "[ $doctor -eq 1 ] && grep -q '^  node: standalone$' /tmp/doctor.txt && grep -q '^storage$' /tmp/doctor.txt"

set +e
setpriv --reuid=llmp --regid=llmp --init-groups /usr/libexec/llmp/llmp-runtime > /tmp/runtime.txt 2>&1
runtime=$?
set -e
cat /tmp/runtime.txt
check "the runtime refuses this host at its platform step (exit 75)" \
  sh -c "[ $runtime -eq 75 ] && grep -q 'refusing to start: this host cannot run this build now' /tmp/runtime.txt"
check "the runtime made its roles" sh -c 'stat_is() { [ "$(stat -c "%a %U:%G" "$1")" = "$2" ]; };
  stat_is /var/lib/llmp/models "755 llmp:llmp" && stat_is /var/lib/llmp/spill "700 llmp:llmp" &&
  stat_is /var/lib/llmp/state "700 llmp:llmp" && stat_is /var/lib/llmp/enrollment.lock "600 llmp:llmp"'
check "doctor now finds the roles" sh -c 'llmp doctor 2>&1 | grep -q "^  installed: /var/lib/llmp/models, owner uid"'

check "the package reinstalls as an upgrade" dpkg -i "$deb"
check "the package removes" dpkg -r llmp
check "its files are gone" sh -c '! test -e /usr/bin/llmp && ! test -e /usr/libexec/llmp/llmp-runtime'
check "the data directory and user stay" sh -c 'test -d /var/lib/llmp/models && getent passwd llmp >/dev/null'
check "a kept conversation stands in" setpriv --reuid=llmp --regid=llmp --init-groups \
  sh -c 'umask 077 && mkdir -p /var/lib/llmp/spill/conversations/a && : > /var/lib/llmp/spill/conversations/a/slot-0.record'
check "the package purges" dpkg -P llmp
check "the data directory and user still stay" sh -c 'test -d /var/lib/llmp/models && getent passwd llmp >/dev/null'
check "the kept conversations are gone (D-105)" sh -c '! test -e /var/lib/llmp/spill/conversations && test -d /var/lib/llmp/spill'
check "the unit is no longer enabled" sh -c '! test -e /etc/systemd/system/multi-user.target.wants/llmp.service'
echo "install-test: passed"
