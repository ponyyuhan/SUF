#!/usr/bin/env python3
"""
Run an end-to-end comparison between:

- SIGMA (GPU-MPC implementation from ePrint 2023/1269)
- This repo's SUF/PFSS prototype benchmark

and collate timing + key size + communication into JSON logs and summary CSV/JSONL.

This runner executes SIGMA locally by spawning *two parties* (P0/P1) on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    return load_json(path) if path.exists() else {}


def _parse_first_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return int(m.group(1))


def parse_sigma_dealer(path: Path) -> Dict[str, Any]:
    text = path.read_text(errors="replace")
    total_us = _parse_first_int(r"^Total time=(\d+)\s+us", text)
    key_bytes = _parse_first_int(r"^Key size=(\d+)\s+B", text)
    out: Dict[str, Any] = {"paths": {"dealer_txt": str(path)}}
    if total_us is not None:
        out.setdefault("timing", {})["keygen_time_s"] = total_us / 1e6
    if key_bytes is not None:
        out.setdefault("preprocessing", {})["key_bytes"] = key_bytes
    return out


def parse_sigma_evaluator(path: Path) -> Dict[str, Any]:
    text = path.read_text(errors="replace")
    total_us = _parse_first_int(r"^Total time=(\d+)\s+us", text)
    comm_us = _parse_first_int(r"^Comm time=(\d+)\s+us", text)
    transfer_us = _parse_first_int(r"^Transfer time=(\d+)\s+us", text)
    total_comm_bytes = _parse_first_int(r"^Total Comm=(\d+)\s+B", text)
    out: Dict[str, Any] = {"paths": {"evaluator_txt": str(path)}}
    if total_us is not None:
        out.setdefault("timing", {})["online_time_s"] = total_us / 1e6
    if comm_us is not None:
        out.setdefault("timing", {})["comm_time_s"] = comm_us / 1e6
    if transfer_us is not None:
        out.setdefault("timing", {})["transfer_time_s"] = transfer_us / 1e6
    if total_comm_bytes is not None:
        out.setdefault("communication", {})["online_bytes"] = total_comm_bytes
    return out


def sigma_output_dir(sigma_root: Path, party: int, model: str, seq_len: int) -> Path:
    # SIGMA writes to output/P<party>/models/<model>-<seq_len>/
    return sigma_root / "output" / f"P{party}" / "models" / f"{model}-{seq_len}"

def sigma_model_name(cfg: Dict[str, Any], model: str) -> str:
    # Allow config overrides for mismatched naming between this repo and SIGMA's CLI.
    overrides = cfg.get("sigma_model_map", {})
    if isinstance(overrides, dict) and model in overrides:
        return str(overrides[model])
    default = {
        # SIGMA CLI names (see EzPC GPU-MPC/experiments/sigma/sigma.cu).
        "gpt-neo-1.3b": "gpt-neo",
        "gpt-neo": "gpt-neo",
        "llama-7b": "llama7b",
        "llama2-7b": "llama7b",
        "llama-13b": "llama13b",
        "llama2-13b": "llama13b",
    }
    return default.get(model, model)


def run_sigma_local(
    sigma_root: Path,
    sigma_bin: Path,
    model: str,
    model_label: Optional[str],
    seq_len: int,
    cpu_threads: int,
    timeout_s: int,
    keep_output: bool,
) -> Tuple[bool, Dict[str, Any]]:
    out0 = sigma_output_dir(sigma_root, 0, model, seq_len)
    out1 = sigma_output_dir(sigma_root, 1, model, seq_len)
    if not keep_output:
        shutil.rmtree(out0, ignore_errors=True)
        shutil.rmtree(out1, ignore_errors=True)

    cmd0 = [str(sigma_bin), model, str(seq_len), "0", "127.0.0.1", str(cpu_threads)]
    cmd1 = [str(sigma_bin), model, str(seq_len), "1", "127.0.0.1", str(cpu_threads)]

    p0_log = sigma_root / "output" / "P0" / "models" / f"{model}-{seq_len}" / "logs_p0.txt"
    p1_log = sigma_root / "output" / "P1" / "models" / f"{model}-{seq_len}" / "logs_p1.txt"
    p0_log.parent.mkdir(parents=True, exist_ok=True)
    p1_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    ok = True
    started = time.time()
    try:
        with p0_log.open("w") as f0, p1_log.open("w") as f1:
            p0 = subprocess.Popen(cmd0, cwd=str(sigma_root), stdout=f0, stderr=subprocess.STDOUT, env=env)
            # Give the server a moment to set up its listener.
            time.sleep(0.25)
            p1 = subprocess.Popen(cmd1, cwd=str(sigma_root), stdout=f1, stderr=subprocess.STDOUT, env=env)

            while True:
                if p0.poll() is not None and p1.poll() is not None:
                    break
                if time.time() - started > timeout_s:
                    p0.kill()
                    p1.kill()
                    ok = False
                    break
                time.sleep(0.2)

            rc0 = p0.wait(timeout=5) if p0.poll() is None else p0.returncode
            rc1 = p1.wait(timeout=5) if p1.poll() is None else p1.returncode
            if rc0 != 0 or rc1 != 0:
                ok = False
    except FileNotFoundError:
        return False, {"error": f"sigma binary not found: {sigma_bin}"}
    except Exception as e:
        return False, {"error": f"sigma run failed: {e}"}

    info: Dict[str, Any] = {
        "system": "sigma",
        "backend": "gpu",
        "model": model_label or model,
        "sigma_model": model,
        "seq_len": seq_len,
        "paths": {"p0_log": str(p0_log), "p1_log": str(p1_log)},
        "timing": {"wall_time_s": time.time() - started},
    }

    dealer = out0 / "dealer.txt"
    evaluator = out0 / "evaluator.txt"
    if dealer.exists():
        info = deep_merge(info, parse_sigma_dealer(dealer))
    if evaluator.exists():
        info = deep_merge(info, parse_sigma_evaluator(evaluator))
    if not dealer.exists() or not evaluator.exists():
        ok = False
        info.setdefault("error", "missing sigma output files (dealer.txt/evaluator.txt)")

    return ok, info


def run_suf(
    suf_bin: Path,
    model: str,
    backend: str,
    seq_len: int,
    batch_size: int,
    n_iters: int,
    log_json: Path,
) -> bool:
    log_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(suf_bin),
        "--model",
        model,
        "--backend",
        backend,
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--n-iters",
        str(n_iters),
        "--log-json",
        str(log_json),
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print(f"[skip] suf binary not found: {suf_bin}", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"[fail] suf failed ({e.returncode}): {' '.join(cmd)}", file=sys.stderr)
        return False
    return log_json.exists()


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def flatten_get(obj: Dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return ""
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SIGMA vs SUF and collate results.")
    ap.add_argument("--config", default="bench/configs/sigma_vs_suf.json", help="Path to config JSON.")
    ap.add_argument("--timeout-sigma-s", type=int, default=3600, help="Timeout for a SIGMA run (seconds).")
    ap.add_argument("--keep-sigma-output", action="store_true", help="Keep SIGMA output directory between runs.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands without executing.")
    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    models: List[str] = list(cfg.get("models", []))
    seq_lens: List[int] = list(cfg.get("seq_lens", [128]))
    batch_sizes: List[int] = list(cfg.get("batch_sizes", [1]))
    backends: List[str] = list(cfg.get("backends", ["cpu"]))
    n_iters: int = int(cfg.get("n_iters", 1))

    sigma_root = Path(cfg.get("sigma_root", "external/sigma_ezpc/GPU-MPC/experiments/sigma")).resolve()
    sigma_bin = (sigma_root / cfg.get("sigma_binary", "sigma")).resolve()
    sigma_cpu_threads = int(cfg.get("sigma_cpu_threads", 16))

    results_dir = Path(cfg.get("results_dir", "bench/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    suf_bin = Path(cfg.get("suf_binary", "build_ninja/bench_suf_transformer")).resolve()
    notes = cfg.get("notes", "")
    hardware = maybe_hardware_blob()

    rows: List[Dict[str, Any]] = []

    for model in models:
        for seq_len in seq_lens:
            # SIGMA: one run per (model, seq_len)
            sigma_cli_model = sigma_model_name(cfg, model)
            sigma_log = results_dir / f"sigma_{model}_L{seq_len}.json"
            if args.dry_run:
                print(f"[dry] sigma: {sigma_bin} {sigma_cli_model} {seq_len} 0/1 127.0.0.1 {sigma_cpu_threads}")
            else:
                ok, sigma_row = run_sigma_local(
                    sigma_root=sigma_root,
                    sigma_bin=sigma_bin,
                    model=sigma_cli_model,
                    model_label=model,
                    seq_len=seq_len,
                    cpu_threads=sigma_cpu_threads,
                    timeout_s=args.timeout_sigma_s,
                    keep_output=args.keep_sigma_output,
                )
                if notes:
                    sigma_row["notes"] = notes
                if hardware:
                    sigma_row["hardware"] = hardware
                sigma_log.write_text(json.dumps(sigma_row, indent=2) + "\n")
                if ok:
                    rows.append(sigma_row)
                else:
                    print(f"[fail] sigma {model} L{seq_len}: {sigma_row.get('error','unknown')}", file=sys.stderr)

            # SUF: run per backend + batch size
            for backend in backends:
                for bs in batch_sizes:
                    suf_log = results_dir / f"suf_{model}_{backend}_L{seq_len}_B{bs}.json"
                    if args.dry_run:
                        print(f"[dry] suf: {suf_bin} --model {model} --backend {backend} --seq-len {seq_len} --batch-size {bs}")
                        continue
                    ok = run_suf(
                        suf_bin=suf_bin,
                        model=model,
                        backend=backend,
                        seq_len=seq_len,
                        batch_size=bs,
                        n_iters=n_iters,
                        log_json=suf_log,
                    )
                    if not ok:
                        continue
                    suf_row = load_json(suf_log)
                    if not suf_row:
                        continue
                    if notes:
                        suf_row.setdefault("notes", notes)
                    if hardware:
                        suf_row.setdefault("hardware", hardware)
                    rows.append(suf_row)

    if not rows:
        print("[warn] no successful runs to summarize", file=sys.stderr)
        return

    jsonl_path = results_dir / "summary.jsonl"
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    csv_path = results_dir / "summary.csv"
    fields = [
        "system",
        "backend",
        "model",
        "seq_len",
        "batch_size",
        "preprocessing.key_bytes",
        "timing.keygen_time_s",
        "timing.online_time_s",
        "timing.wall_time_s",
        "communication.online_bytes",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for row in rows:
            w.writerow([flatten_get(row, fn) for fn in fields])

    print(f"[done] wrote {jsonl_path} and {csv_path}")


if __name__ == "__main__":
    main()
