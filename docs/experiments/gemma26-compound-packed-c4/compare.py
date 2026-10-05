#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse the reviewed full-128-head comparison, including root input admission."""
import pathlib
import runpy

runpy.run_path(str(pathlib.Path(__file__).resolve().parent.parent /
                  'gemma26-packed-attention-c4/compare.py'), run_name='__main__')
