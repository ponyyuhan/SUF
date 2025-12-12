#!/usr/bin/env python3
"""
Coordinator that runs Sigma and SUF benchmarks using a common config and
collates results. This script is intentionally lightweight and tolerates
missing binaries; it will skip runs that cannot be executed and log why.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return {}


def maybe_hardware_blob() -> Dict[str, Any]:
    host = socket.gethostname()
    path = Path("bench") / f"hardware_{host}.json"
    if path.exists():
        return load_json(path)
    return {}


def run_command(cmd: List[str], log_path: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print(f"[skip] binary not found: {cmd[0]}", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"[fail] command failed ({e.returncode}): {' '.join(cmd)}", file=sys.stderr)
        return False
    if not log_path.exists():
        print(f"[fail] expected log missing: {log_path}", file=sys.stderr)
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Sigma vs SUF benchmarks from config.")
    ap.add_argument("--config", default="bench/configs/sigma_vs_suf.json", help="Path to config JSON.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    models = cfg.get("models", [])
    seq_lens = cfg.get("seq_lens", [128])
    batch_sizes = cfg.get("batch_sizes", [1])
    backends = cfg.get("backends", ["cpu", "gpu"])
    n_iters = int(cfg.get("n_iters", 5))
    sigma_root = Path(cfg.get("sigma_root", "external/sigma_ezpc/GPU-MPC"))
    results_dir = Path(cfg.get("results_dir", "bench/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    notes = cfg.get("notes", "")
    suf_bin = cfg.get("suf_binary", "build_ninja/bench_suf_transformer")
    sigma_bin_rel = cfg.get("sigma_binary", "build_gpu/bin/sigma_run_model")

    hardware = maybe_hardware_blob()

    summary_rows: List[Dict[str, Any]] = []
    for model in models:
        for L in seq_lens:
            for bs in batch_sizes:
                for backend in backends:
                    # Sigma run
                    sigma_bin = sigma_root / sigma_bin_rel
                    sigma_log = results_dir / f"sigma_{model}_{backend}_L{L}_B{bs}.json"
                    sigma_cmd = [
                        str(sigma_bin),
                        "--model",
                        model,
                        "--backend",
                        backend,
                        "--seq-len",
                        str(L),
                        "--batch-size",
                        str(bs),
                        "--n-iters",
                        str(n_iters),
                        "--log-json",
                        str(sigma_log),
                    ]
                    if args.dry_run:
                        print(" ".join(sigma_cmd))
                    else:
                        run_command(sigma_cmd, sigma_log)

                    # SUF run
                    suf_log = results_dir / f"suf_{model}_{backend}_L{L}_B{bs}.json"
                    suf_cmd = [
                        str(Path(suf_bin)),
                        "--model",
                        model,
                        "--backend",
                        backend,
                        "--seq-len",
                        str(L),
                        "--batch-size",
                        str(bs),
                        "--n-iters",
                        str(n_iters),
                        "--log-json",
                        str(suf_log),
                    ]
                    if args.dry_run:
                        print(" ".join(suf_cmd))
                    else:
                        run_command(suf_cmd, suf_log)

                    # Consolidate logs if present
                    for system, log_path in [("sigma", sigma_log), ("suf", suf_log)]:
                        if not log_path.exists():
                            continue
                        data = load_json(log_path)
                        if not data:
                            continue
                        data.setdefault("system", system)
                        data.setdefault("backend", backend)
                        data.setdefault("model", model)
                        data.setdefault("seq_len", L)
                        data.setdefault("batch_size", bs)
                        if hardware:
                            data.setdefault("hardware", hardware)
                        if notes:
                            data.setdefault("notes", notes)
                        summary_rows.append(data)

    # Write summary CSV/JSONL
    if summary_rows:
        jsonl_path = results_dir / "summary.jsonl"
        with jsonl_path.open("w") as f:
            for row in summary_rows:
                f.write(json.dumps(row) + "\n")
        csv_path = results_dir / "summary.csv"
        # Flatten a conservative set of columns
        fieldnames = [
            "system",
            "backend",
            "model",
            "seq_len",
            "batch_size",
            "timing.online_time_s_mean",
            "communication.online_bytes",
        ]
        def get_field(obj: Dict[str, Any], dotted: str) -> Any:
            cur: Any = obj
            for part in dotted.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return ""
            return cur

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for row in summary_rows:
                writer.writerow([get_field(row, fn) for fn in fieldnames])
        print(f"[done] wrote {jsonl_path} and {csv_path}")
    else:
        print("[warn] no successful runs to summarize", file=sys.stderr)


if __name__ == "__main__":
    main()
