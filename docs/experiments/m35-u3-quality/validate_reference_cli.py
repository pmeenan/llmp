#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Parser/help validation in the exact reference image; no model load."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import vllm
from vllm.entrypoints.openai import cli_args
from vllm.utils.argparse_utils import FlexibleArgumentParser
from reference import IMAGE, server_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
argv = server_args()[2:]
parsed = vars(cli_args.make_arg_parser(FlexibleArgumentParser()).parse_args(argv))
for key, expected in {"model": "/model", "served_model_name": ["llama33"],
                      "dtype": "bfloat16", "kv_cache_dtype": "float16",
                      "enable_prefix_caching": False, "max_model_len": 12288,
                      "max_num_seqs": 1, "max_num_batched_tokens": 512,
                      "max_logprobs": 1, "logprobs_mode": "raw_logprobs",
                      "load_format": "instanttensor", "generation_config": "vllm",
                      "enable_log_requests": False}.items():
    if parsed.get(key) != expected:
        raise ValueError(f"CLI meaning mismatch for {key}: {parsed.get(key)!r}")
help_run = subprocess.run([sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                           "--help=all"], capture_output=True, timeout=120, check=True)
(args.out / "help.txt").write_bytes(help_run.stdout + help_run.stderr)
source = Path(inspect.getfile(cli_args)).read_bytes()
receipt = {"format": "llmp-u3-reference-cli/1", "image": IMAGE,
           "vllm_version": vllm.__version__, "argv": argv,
           "parser_source_sha256": hashlib.sha256(source).hexdigest(),
           "parsed": parsed, "model_loaded": False}
(args.out / "cli.json").write_text(json.dumps(receipt, sort_keys=True, default=str, indent=2) + "\n")
print("Pinned-image reference CLI parser and help passed; no model loaded")
