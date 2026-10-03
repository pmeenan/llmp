# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Hang recovery (D-102) and conversations kept across a restart (D-105),
checked end to end through the service on a Spark.

    restart_check.py SCENARIO --binary JITLLM_RUNTIME --out DIR [--port P]

The service serves DeepSeek V4 Flash (DSpark) and Qwen3.8 Flash Next (MTP)
from the artifacts installed under ~/.local/share/jitllm/m3-artifacts (their
IDs below) on loopback; requests are greedy and non-streaming. R1 and R2 are
two turns to Qwen3.8, R3 one to DeepSeek (its swap writes Qwen3.8's state
back), R4 Qwen3.8's third turn, R5 DeepSeek's second.

    control  one service: R1, R2, R3, R4, R5
    kept     service A: R1, R2, R3, SIGTERM (a graceful stop keeps them);
             service B: R4, R5
    crash    as kept, but A is killed (SIGKILL) once Qwen3.8's record exists
    tamper   as kept, with one byte of Qwen3.8's kept spill file flipped
             before B (the record is refused)
    foreign  as kept, with B serving Qwen3.8 at another context (another
             layout: refused)
    hang     one service with hang_seconds 60 and the read-holding test hook
             (JITLLM_TEST_HOLD_READS) in its cancellable form
             (JITLLM_TEST_HOLD_READS_CANCELLABLE=1, as reads still queued):
             a new conversation's state growth hangs, rung 1 cancels it,
             the cancellation drains, rung 2 resets the model, and the
             service goes on
    stuck    the service as a systemd user unit (Restart=on-failure,
             RestartSec=5s) with the hook in its default form, a drive's: a
             held read is not ended by its cancellation. R1, R2, R3 (Qwen3.8's
             conversation written back and recorded), then a new DeepSeek
             conversation whose state growth hangs: rung 1 cancels its
             wait, the read stays in flight, so the cancellation never
             drains and rung 3 exits; the hold is lifted once the process
             is gone; systemd restarts it, which adopts the kept
             conversation; R4 and R5 follow
    swapcost-on, swapcost-off
             one service, R1 then three alternations of DeepSeek and
             Qwen3.8, with conversations kept (records hashed in the
             background after each write-back) or not ([memory]
             keep_across_restart = false): the swaps' and requests' times

Each writes DIR/SCENARIO.json (the replies, their usage and the log lines
that matter) and prints it; compare kept/crash/stuck's R4 and R5 with
control's.
"""
import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

HOME = Path.home()
STORE = HOME / ".local/share/jitllm/m3-artifacts"
QWEN_M = HOME / ".local/share/jitllm/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6"
QWEN = "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93"
QWEN_D = "8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40"
DS = "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234"
DS_D = "dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5"


def config(d, port, qwen_context, client=(), memory=()):
    lines = ["schema_version = 2", "[storage]", f'data_dir = "{d}/data"', f'installed = "{STORE}"',
             "[client]", 'bind = ["loopback"]', f"port = {port}", *client,
             "[memory]", *memory,
             "[models.deepseek]", f'artifact = "{DS}"', f'drafter = "{DS_D}"', "context = 32768",
             '[models."qwen3.8"]', f'artifact = "{QWEN}"', f'drafter = "{QWEN_D}"',
             f"context = {qwen_context}", "prefill_chunk = 4096",
             f'tokenizer = "{QWEN_M}/tokenizer.json"', f'chat_template = "{QWEN_M}/chat_template.jinja"']
    (d / "config.toml").write_text("\n".join(lines) + "\n")


def http_json(port, method, path, body=None, timeout=1800):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, json.dumps(body) if body is not None else None,
                     {"Content-Type": "application/json"} if body is not None else {})
        r = conn.getresponse()
        return r.status, r.read().decode(errors="replace")
    finally:
        conn.close()


class Service:
    def __init__(self, binary, d, port, name, env=None):
        self.port = port
        self.log_path = d / f"{name}.log"
        self.log = open(self.log_path, "w")
        full_env = dict(os.environ)
        full_env.update(env or {})
        self.p = subprocess.Popen([binary, "--config", str(d / "config.toml"), "--anchor", str(d / "anchor")],
                                  stdout=self.log, stderr=subprocess.STDOUT, env=full_env)
        t0 = time.monotonic()
        while True:
            if self.p.poll() is not None:
                raise RuntimeError(f"{name} exited during startup")
            try:
                s, _ = http_json(port, "GET", "/v1/models", timeout=2)
                if s == 200:
                    break
            except (OSError, http.client.HTTPException):
                pass
            if time.monotonic() - t0 > 600:
                raise RuntimeError(f"{name} not ready")
            time.sleep(0.5)
        self.ready = time.monotonic() - t0

    def stop(self, sig=signal.SIGTERM):
        t0 = time.monotonic()
        if self.p.poll() is None:
            self.p.send_signal(sig)
        try:
            rc = self.p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self.p.kill()
            rc = self.p.wait(timeout=60)
        self.log.close()
        return rc, time.monotonic() - t0

    def lines(self, pattern):
        return [line for line in self.log_path.read_text(errors="replace").splitlines()
                if re.search(pattern, line)]


def chat(port, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    t0 = time.monotonic()
    s, text = http_json(port, "POST", "/v1/chat/completions", body)
    if s != 200:
        raise RuntimeError(f"HTTP {s}: {text[:400]}")
    v = json.loads(text)
    m = v["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "usage": v["usage"], "seconds": time.monotonic() - t0}


def assistant(r):
    m = {"role": "assistant", "content": r["content"]}
    if r["reasoning"]:
        m["reasoning_content"] = r["reasoning"]
    return m


U1 = {"role": "user", "content": "In two short sentences, what is the moon made of? Answer briefly."}
U2 = {"role": "user", "content": "And how far away is it? One sentence."}
U4 = {"role": "user", "content": "Name one mission that landed there. One sentence."}


def first_turns(port, out):
    out["r1"] = chat(port, "qwen3.8", [U1], 160)
    out["r2"] = chat(port, "qwen3.8", [U1, assistant(out["r1"]), U2], 160)
    out["r3"] = chat(port, "deepseek", [{"role": "user", "content": "Say hello in one word."}], 24)


def last_turn(port, out, key="r4"):
    msgs = [U1, assistant(out["r1"]), U2, assistant(out["r2"]), U4]
    out[key] = chat(port, "qwen3.8", msgs, 160)
    # DeepSeek's conversation goes on too (R5): its state was spilled by
    # the graceful stop, or written back by R4's swap.
    hello = {"role": "user", "content": "Say hello in one word."}
    out["r5"] = chat(port, "deepseek",
                     [hello, assistant(out["r3"]),
                      {"role": "user", "content": "Now say it in French, one word."}], 24)


def kept_dir(d):
    return d / "data" / "spill" / "conversations" / QWEN


def flip_first_used_byte(rec, d):
    states = [p for p in sorted(kept_dir(d).glob("slot-*.state"))
              if ".turn-" not in p.name and (kept_dir(d) / p.name.replace(".state", ".record")).exists()]
    target = states[0]
    with open(target, "r+b") as f:
        off = 0
        size = target.stat().st_size
        while off < size:
            f.seek(off)
            chunk = f.read(1 << 21)
            if any(chunk):
                f.seek(off + 4096)
                b = f.read(1)
                f.seek(off + 4096)
                f.write(bytes([b[0] ^ 0x5A]))
                rec["tampered"] = {"file": target.name, "offset": off + 4096}
                return
            off += 1 << 21


def hang(a, d, rec):
    """Rungs 1 and 2 end to end: a new conversation's state growth hangs on
    held reads (the runtime's test hook); the ladder cancels it after
    hang_seconds, its request fails with a 503, the model is reset in place,
    and once reads flow the service serves on (the process never exits)."""
    hold = d / "hold"
    config(d, a.port, 33792, ["stall_seconds = 1", "hang_seconds = 60"])
    s = Service(a.binary, d, a.port, "a", {"JITLLM_TEST_HOLD_READS": str(hold),
                                           "JITLLM_TEST_HOLD_READS_CANCELLABLE": "1"})
    rec["a_ready"] = s.ready
    rec["r1"] = chat(a.port, "qwen3.8", [U1], 160)
    hold.touch()
    t0 = time.monotonic()
    try:
        rec["hung"] = chat(a.port, "qwen3.8",
                           [{"role": "user", "content": "Another conversation: name three planets."}], 64)
    except RuntimeError as e:
        rec["hung_error"] = str(e)[:400]
    rec["hung_seconds"] = time.monotonic() - t0
    hold.unlink()
    rec["after"] = chat(a.port, "qwen3.8", [U1, assistant(rec["r1"]), U2], 160)
    rec["alive"] = s.p.poll() is None
    rec["a_stop"] = s.stop()
    rec["rungs"] = s.lines(r"hang recovery|rung|cancelled|test hook|made no progress")


def user_unit(unit, prop):
    out = subprocess.run(["systemctl", "--user", "show", "-p", prop, "--value", unit],
                         capture_output=True, text=True, check=False)
    return out.stdout.strip()


def wait_ready(port, limit=600):
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit:
        try:
            s, _ = http_json(port, "GET", "/v1/models", timeout=2)
            if s == 200:
                return time.monotonic() - t0
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.5)
    raise RuntimeError("not ready")


def stuck(a, d, rec):
    """Rung 3 end to end under systemd: a read the drive holds outlives its
    cancelled wait, so the cancellation never drains; the process exits for
    systemd to restart it, and the restart adopts what was kept."""
    hold = d / "hold"
    log = d / "a.log"
    config(d, a.port, 33792, ["stall_seconds = 1", "hang_seconds = 60"])
    unit = f"jitllm-stuck-{os.getpid()}"
    subprocess.run(["systemd-run", "--user", "--no-block", "--collect", f"--unit={unit}",
                    "-p", "Type=notify", "-p", "NotifyAccess=main", "-p", "Restart=on-failure",
                    "-p", "RestartSec=5s", "-p", "TimeoutStartSec=5min", "-p", "TimeoutStopSec=90s",
                    "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}",
                    f"--setenv=JITLLM_TEST_HOLD_READS={hold}",
                    a.binary, "--config", str(d / "config.toml"), "--anchor", str(d / "anchor")],
                   check=True)
    try:
        rec["a_ready"] = wait_ready(a.port)
        first_pid = user_unit(unit, "MainPID")
        first_turns(a.port, rec)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120 and not (
                kept_dir(d).exists() and list(kept_dir(d).glob("slot-*.record"))):
            time.sleep(0.2)
        rec["kept_before"] = sorted(p.name for p in kept_dir(d).iterdir()) if kept_dir(d).exists() else []
        hold.touch()
        t0 = time.monotonic()
        outcome = {}

        def hung_request():
            try:
                outcome["reply"] = chat(a.port, "deepseek",
                                        [{"role": "user", "content": "Another conversation: name three planets."}],
                                        24)
            except (RuntimeError, OSError, http.client.HTTPException) as e:
                outcome["error"] = repr(e)[:300]
            outcome["seconds"] = time.monotonic() - t0

        th = threading.Thread(target=hung_request)
        th.start()
        # The hold lifts once the process is gone (before systemd's restart).
        while time.monotonic() - t0 < 400:
            pid = user_unit(unit, "MainPID")
            if pid != first_pid:
                rec["exited_after"] = time.monotonic() - t0
                break
            time.sleep(0.1)
        hold.unlink()
        th.join(timeout=60)
        rec["hung"] = outcome
        wait_ready(a.port)
        rec["b_ready_after"] = time.monotonic() - t0
        rec["n_restarts"] = user_unit(unit, "NRestarts")
        rec["second_pid"] = user_unit(unit, "MainPID")
        rec["first_pid"] = first_pid
        last_turn(a.port, rec)
    except BaseException as e:  # noqa: BLE001
        rec["error"] = repr(e)
    finally:
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
        if hold.exists():
            hold.unlink()
    text = log.read_text(errors="replace").splitlines() if log.exists() else []
    rec["lines"] = [line for line in text if re.search(
        r"hang recovery|rung|test hook|made no progress|exiting|kept|adopt|refused|records queued|ready$",
        line)]


def swapcost(a, d, rec, keep):
    """Swaps away and back with records hashed in the background, or not."""
    config(d, a.port, 33792, memory=() if keep else ("keep_across_restart = false",))
    s = Service(a.binary, d, a.port, "a")
    rec["a_ready"] = s.ready
    rec["r1"] = chat(a.port, "qwen3.8", [U1], 160)
    hello = [{"role": "user", "content": "Say hello in one word."}]
    rec["rounds"] = []
    for _ in range(3):
        ds = chat(a.port, "deepseek", hello, 24)
        qw = chat(a.port, "qwen3.8", [U1], 160)
        rec["rounds"].append({"deepseek_s": ds["seconds"], "qwen_s": qw["seconds"],
                              "same": qw["content"] == rec["r1"]["content"]})
    rec["a_stop"] = s.stop()
    rec["swaps"] = s.lines(r"^.*swap .* -> .*: ready in")
    rec["kept"] = s.lines(r"kept for the next start|conversations are|hashed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["control", "kept", "crash", "tamper", "foreign", "hang",
                                         "stuck", "swapcost-on", "swapcost-off"])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18171)
    a = ap.parse_args()
    d = Path(a.out) / a.scenario
    if d.exists():
        subprocess.run(["rm", "-rf", str(d)], check=True)
    d.mkdir(parents=True)
    rec = {"scenario": a.scenario}
    if a.scenario in ("hang", "stuck", "swapcost-on", "swapcost-off"):
        try:
            if a.scenario == "hang":
                hang(a, d, rec)
            elif a.scenario == "stuck":
                stuck(a, d, rec)
            else:
                swapcost(a, d, rec, a.scenario == "swapcost-on")
        except BaseException as e:  # noqa: BLE001
            rec["error"] = repr(e)
        (Path(a.out) / f"{a.scenario}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k not in ("r1",)}, indent=1)[:8000])
        return
    try:
        config(d, a.port, 33792)
        s = Service(a.binary, d, a.port, "a")
        rec["a_ready"] = s.ready
        first_turns(a.port, rec)
        if a.scenario == "control":
            last_turn(a.port, rec)
            rec["a_stop"] = s.stop()
            rec["a_kept"] = s.lines(r"kept|record|conversations")
        else:
            if a.scenario == "crash":
                found_name = None
                t0 = time.monotonic()
                while time.monotonic() - t0 < 120:
                    found = sorted(kept_dir(d).glob("slot-*.record")) if kept_dir(d).exists() else []
                    if found:
                        found_name = found[0].name
                        break
                    time.sleep(0.2)
                rec["crash_record"] = found_name
                rec["a_stop"] = s.stop(signal.SIGKILL)
            else:
                rec["a_stop"] = s.stop()
            rec["a_kept"] = s.lines(r"kept|record|conversations")
            rec["files"] = sorted(p.name for p in kept_dir(d).iterdir()) if kept_dir(d).exists() else []
            if a.scenario == "tamper":
                flip_first_used_byte(rec, d)
            if a.scenario == "foreign":
                config(d, a.port, 32768)
            b = Service(a.binary, d, a.port, "b")
            rec["b_ready"] = b.ready
            last_turn(a.port, rec)
            rec["b_stop"] = b.stop()
            rec["b_kept"] = b.lines(r"kept|adopt|refused|record|conversations")
            rec["files_after"] = sorted(p.name for p in kept_dir(d).iterdir()) if kept_dir(d).exists() else []
    except BaseException as e:  # noqa: BLE001
        rec["error"] = repr(e)
    (Path(a.out) / f"{a.scenario}.json").write_text(json.dumps(rec, indent=1) + "\n")
    brief = {k: v for k, v in rec.items() if k not in ("r1", "r2", "r3")}
    print(json.dumps(brief, indent=1)[:8000])


if __name__ == "__main__":
    main()
