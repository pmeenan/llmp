#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run one pinned stable-diffusion.cpp image case in a container and record it.

Host side, standard library only. Model files are verified against pins.json,
mounted read-only and never executed. Outputs go to a new private directory.
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
PROMPT = "A red ceramic teapot on a plain wooden table, soft daylight, no text."
IMAGE = "llmp-image-gguf:20260922"
DENOISERS = {
    "bf16": ("control", "diffusion_models/qwen_image_2.1_bf16.safetensors"),
    "q8_0": ("gguf", "qwen-image-2.1-Q8_0.gguf"),
    "q4_k_m": ("gguf", "qwen-image-2.1-Q4_K_M.gguf"),
}
TEXT_ENCODER = ("gguf", "text_encoders/qwen3vl_8b_bf16.safetensors")
VAE = ("gguf", "vae/qwen_image_2.1_vae_bf16.safetensors")
# Params placements. "resident" pins every module's parameters to the GPU
# backend (auto-fit off); on Spark that is the same physical memory as RAM.
PLACEMENTS = {
    "resident": ["--params-backend", "cuda0", "--eager-load"],
    "phase-disk": ["--params-backend", "diffusion=cuda0,te=disk,vae=disk"],
    "all-disk": ["--params-backend", "disk"],
}
TIMING = {
    "load": re.compile(r"loading tensors completed, taking ([0-9.]+)s \(read: ([0-9.]+)s, memcpy: ([0-9.]+)s, "
                       r"convert: ([0-9.]+)s, copy_to_backend: ([0-9.]+)s\)"),
    "condition": re.compile(r"get_learned_condition completed, taking ([0-9.]+)s"),
    "sampling": re.compile(r"sampling completed, taking ([0-9.]+)s"),
    "decode": re.compile(r"decode_first_stage completed, taking ([0-9.]+)s"),
    "generate": re.compile(r"generate_image completed in ([0-9.]+)s"),
}
MEMORY_LINE = re.compile(r"(compute buffer size|params|budget|segment|resident|evict|prefetch|MiB|MB\()", re.I)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pinned(models, pins, section, relative):
    row = next(r for r in pins[section]["files"] if r["path"] == relative)
    path = models / ("bf16-control" if section == "control" else "gguf") / relative
    return path, row


def verify(path, row):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
        raise ValueError(f"unverified model file: {path}")


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def cached_bytes(path):
    """Page-cache residency of a file via mincore(2); no privileges needed."""
    size = path.stat().st_size
    page = mmap.PAGESIZE
    with path.open("rb") as stream:
        address = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, stream.fileno(), 0)
        if address in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            pages = (size + page - 1) // page
            vector = (ctypes.c_ubyte * pages)()
            if _libc.mincore(address, size, vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return sum(v & 1 for v in vector) * page
        finally:
            _libc.munmap(address, size)


def evict(path):
    with path.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.split()[0]) * 1024
    return values


def vmstat(keys=("pswpin", "pswpout", "oom_kill")):
    values = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {k: int(values.get(k, 0)) for k in keys}


def cgroup_dir(container_id):
    path = Path(f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope")
    return path if path.is_dir() else None


def io_read_bytes(directory):
    try:
        text = (directory / "io.stat").read_text()
    except OSError:
        return None
    return sum(int(m.group(1)) for m in re.finditer(r"rbytes=(\d+)", text))


class Sampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop = threading.Event()
        self.cgroup = None
        self.samples = 0
        self.min_available = None
        self.max_cgroup_read = None
        self.max_cgroup_peak = None

    def run(self):
        while not self.stop.is_set():
            available = meminfo()["MemAvailable"]
            self.min_available = available if self.min_available is None else min(self.min_available, available)
            if self.cgroup:
                read = io_read_bytes(self.cgroup)
                if read is not None:
                    self.max_cgroup_read = max(self.max_cgroup_read or 0, read)
                try:
                    peak = int((self.cgroup / "memory.peak").read_text())
                    self.max_cgroup_peak = max(self.max_cgroup_peak or 0, peak)
                except (OSError, ValueError):
                    pass
            self.samples += 1
            self.stop.wait(self.interval)


def docker(*args, check=True, capture=True):
    return subprocess.run(["sudo", "-n", "docker", *args], check=check, text=True,
                          stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True, help="directory holding gguf/ and bf16-control/")
    parser.add_argument("--output", type=Path, required=True, help="new private output directory")
    parser.add_argument("--denoiser", choices=sorted(DENOISERS), required=True)
    parser.add_argument("--placement", choices=sorted(PLACEMENTS), default="resident")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max-vram", help="sd.cpp managed budget, e.g. cuda0=3")
    parser.add_argument("--cold", action="store_true", help="drop model files from page cache first")
    parser.add_argument("--skip-verify", action="store_true", help="hashes already verified in this session")
    parser.add_argument("--extra", action="append", default=[], help="additional sd-cli argument")
    parser.add_argument("--image", default=IMAGE, help="runner image (default: patched-GGML build)")
    args = parser.parse_args()

    pins = json.loads((HERE / "pins.json").read_text())
    pins["control"] = pins["bf16_control"]
    models = args.models.resolve()
    selected = [DENOISERS[args.denoiser], TEXT_ENCODER, VAE]
    files = [pinned(models, pins, section, relative) for section, relative in selected]
    if not args.skip_verify:
        for path, row in files:
            verify(path, row)
    args.output.mkdir(mode=0o700)
    out = args.output.resolve()

    if args.cold:
        for path, _ in files:
            evict(path)
    residency = {path.name: cached_bytes(path) for path, _ in files}

    mounts = []
    container_paths = []
    for index, (path, _) in enumerate(files):
        target = f"/models/{index}-{path.name}"
        mounts += ["--mount", f"type=bind,src={path},dst={target},readonly"]
        container_paths.append(target)
    sd_args = ["--diffusion-model", container_paths[0], "--llm", container_paths[1], "--vae", container_paths[2],
               "-p", PROMPT, "-W", str(args.size), "-H", str(args.size), "--steps", str(args.steps),
               "--seed", "42", "--cfg-scale", "1.0", "--sampling-method", "euler", "--rng", "cuda",
               "--backend", "cuda0", *PLACEMENTS[args.placement], "--diffusion-fa",
               "--disable-image-metadata", "-v", "-o", "/out/image.png"]
    if args.max_vram:
        sd_args += ["--max-vram", args.max_vram]
    sd_args += args.extra

    name = f"llmp-image-gguf-{os.getpid()}-{int(time.time())}"
    (out / "container-name").write_text(name + "\n")
    before_mem, before_vm = meminfo(), vmstat()
    sampler = Sampler()
    sampler.start()
    started = time.monotonic()
    container_id = docker("run", "-d", "--name", name, "--gpus", "all", "--network", "none", "--read-only",
                          "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp",
                          *mounts, "--mount", f"type=bind,src={out},dst=/out",
                          args.image, *sd_args).stdout.strip()
    try:
        sampler.cgroup = cgroup_dir(container_id)
        exit_code = int(docker("wait", container_id).stdout.strip())
        elapsed = time.monotonic() - started
        final_read = io_read_bytes(sampler.cgroup) if sampler.cgroup else None
    finally:
        sampler.stop.set()
        sampler.join()
        log = docker("logs", container_id, check=False).stdout
        (out / "sd-cli.log").write_text(log)
        docker("rm", "-f", container_id, check=False)
    after_vm = vmstat()

    timings = {key: [[float(g) for g in m.groups()] for m in rx.finditer(log)] for key, rx in TIMING.items()}
    image = out / "image.png"
    result = {
        "case": {"denoiser": args.denoiser, "placement": args.placement, "size": args.size, "steps": args.steps,
                 "max_vram": args.max_vram, "cold": args.cold, "extra": args.extra, "image": args.image},
        "files": [{"path": str(path.relative_to(models)), "sha256": row["sha256"], "bytes": row["bytes"]}
                  for path, row in files],
        "sd_cli_args": sd_args,
        "exit_code": exit_code,
        "outward_seconds": round(elapsed, 3),
        "timings": timings,
        "memory_lines": [line for line in log.splitlines() if MEMORY_LINE.search(line)][:400],
        "page_cache_before": residency,
        "mem_available_before": before_mem["MemAvailable"],
        "mem_total": before_mem["MemTotal"],
        "min_mem_available": sampler.min_available,
        "max_host_used_delta": before_mem["MemAvailable"] - sampler.min_available,
        "cgroup_read_bytes": final_read if final_read is not None else sampler.max_cgroup_read,
        "cgroup_memory_peak": sampler.max_cgroup_peak,
        "samples": sampler.samples,
        "vmstat_delta": {k: after_vm[k] - before_vm[k] for k in after_vm},
        "image_sha256": sha256(image) if image.is_file() else None,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("exit_code", "outward_seconds", "timings", "max_host_used_delta",
                                            "cgroup_read_bytes", "image_sha256")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
