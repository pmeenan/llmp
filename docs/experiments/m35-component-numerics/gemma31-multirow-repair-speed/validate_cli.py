# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util, json, pathlib, subprocess, sys, tomllib

B = pathlib.Path(__file__).resolve().parent
P = json.loads((B / "speed-pins.json").read_text())
spec = importlib.util.spec_from_file_location("c2_speed_executor", B / "run_c2.py")
run = importlib.util.module_from_spec(spec); sys.modules[spec.name] = run; spec.loader.exec_module(run)
g = run.helper(); captured = []
def forbidden(*args, **kwargs):
    raise AssertionError("process/service/network calls are forbidden in parser validation")
def capture(args):
    captured.append({k: v for k, v in vars(args).items() if k != "func"})
    return 0
real_popen, real_port, real_run = subprocess.Popen, g.port_open, g.cmd_run
g.port_open = forbidden; subprocess.Popen = forbidden; g.cmd_run = capture
for pair, order in enumerate(P["orders"], 1):
    for arm in order:
        argv = run.argv(pair, arm)
        assert g.main(argv[3:]) == 0
        parsed = captured[-1]
        assert parsed["family"] == "gemma31" and parsed["cells"] == "2"
        assert parsed["label"] == {"a": "old", "r": "new"}[arm]
        assert parsed["block"] == pair and parsed["position"] == order.index(arm)
        assert parsed["prime"] is False and parsed["runtime"] == P["runtime"]["path"]
        assert run.sha(parsed["config_template"]) == P["config_sha256"]
        cfg = tomllib.loads(g.render_config(g.arm_template(g.FAMILIES["gemma31"],
            parsed["config_template"], parsed["label"]), pathlib.Path("/home/pmeenan/.local/share/llmp"),
            B / f"validation-work/{pair}-{arm}/data", 18114))
        assert cfg["models"]["gemma31"] == {"artifact": P["artifact_id"], "context": 8192,
            "prefill_chunk": 256, "max_slots": 2, "kv_cache_encoding": "f16"}
        assert g.parse_runtime_env(parsed["runtime_env"]) == P["runtime_env"][arm]
assert len(captured) == 6
subprocess.Popen = real_popen; g.port_open = real_port; g.cmd_run = real_run
print(json.dumps({"pass": True, "actual_parser_calls": len(captured),
                  "inference": False, "service_launched": False,
                  "network_calls": False, "timing": False}))
