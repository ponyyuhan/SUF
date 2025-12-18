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
import resource
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
    mha_us = _parse_first_int(r"^MHA time=(\d+)\s+us", text)
    matmul_us = _parse_first_int(r"^Matmul time=(\d+)\s+us", text)
    trunc_us = _parse_first_int(r"^Truncate time=(\d+)\s+us", text)
    gelu_us = _parse_first_int(r"^Gelu time=(\d+)\s+us", text)
    softmax_us = _parse_first_int(r"^Softmax time=(\d+)\s+us", text)
    layernorm_us = _parse_first_int(r"^Layernorm time=(\d+)\s+us", text)
    total_comm_bytes = _parse_first_int(r"^Total Comm=(\d+)\s+B", text)
    gelu_comm_bytes = _parse_first_int(r"^Gelu Comm=(\d+)\s+B", text)
    softmax_comm_bytes = _parse_first_int(r"^Softmax Comm=(\d+)\s+B", text)
    layernorm_comm_bytes = _parse_first_int(r"^Layernorm Comm=(\d+)\s+B", text)
    out: Dict[str, Any] = {"paths": {"evaluator_txt": str(path)}}
    if total_us is not None:
        out.setdefault("timing", {})["online_time_s"] = total_us / 1e6
    if comm_us is not None:
        out.setdefault("timing", {})["comm_time_s"] = comm_us / 1e6
    if transfer_us is not None:
        out.setdefault("timing", {})["transfer_time_s"] = transfer_us / 1e6
    if mha_us is not None:
        out.setdefault("timing", {})["mha_time_s"] = mha_us / 1e6
    if matmul_us is not None:
        out.setdefault("timing", {})["matmul_time_s"] = matmul_us / 1e6
    if trunc_us is not None:
        out.setdefault("timing", {})["trunc_time_s"] = trunc_us / 1e6
    if gelu_us is not None:
        out.setdefault("timing", {})["gelu_time_s"] = gelu_us / 1e6
    if softmax_us is not None:
        out.setdefault("timing", {})["softmax_time_s"] = softmax_us / 1e6
    if layernorm_us is not None:
        out.setdefault("timing", {})["layernorm_time_s"] = layernorm_us / 1e6
    if total_comm_bytes is not None:
        out.setdefault("communication", {})["online_bytes"] = total_comm_bytes
        # SIGMA's peer bytes are already wire bytes; treat as packed-bytes baseline.
        out.setdefault("communication", {})["online_packed_bytes"] = total_comm_bytes
    if gelu_comm_bytes is not None:
        out.setdefault("communication", {})["gelu_bytes"] = gelu_comm_bytes
    if softmax_comm_bytes is not None:
        out.setdefault("communication", {})["softmax_bytes"] = softmax_comm_bytes
    if layernorm_comm_bytes is not None:
        out.setdefault("communication", {})["layernorm_bytes"] = layernorm_comm_bytes
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
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
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
    heartbeat_s = 5.0
    next_heartbeat = started + heartbeat_s
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
                now = time.time()
                if now >= next_heartbeat:
                    e = now - started
                    print(
                        f"[sigma] {model_label or model} L{seq_len}: {e:.1f}s elapsed (p0={p0.poll()} p1={p1.poll()})",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_heartbeat = now + heartbeat_s
                time.sleep(0.2)

            rc0 = p0.wait(timeout=5) if p0.poll() is None else p0.returncode
            rc1 = p1.wait(timeout=5) if p1.poll() is None else p1.returncode
            if rc0 != 0 or rc1 != 0:
                ok = False
    except FileNotFoundError:
        return False, {"error": f"sigma binary not found: {sigma_bin}"}
    except Exception as e:
        return False, {"error": f"sigma run failed: {e}"}

    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_user_s = max(0.0, float(ru1.ru_utime - ru0.ru_utime))
    cpu_sys_s = max(0.0, float(ru1.ru_stime - ru0.ru_stime))
    max_rss_kb = int(getattr(ru1, "ru_maxrss", 0))

    info: Dict[str, Any] = {
        "system": "sigma",
        "backend": "gpu",
        "model": model_label or model,
        "sigma_model": model,
        "seq_len": seq_len,
        "paths": {"p0_log": str(p0_log), "p1_log": str(p1_log)},
        "timing": {"wall_time_s": time.time() - started},
        "resources": {
            "cpu_user_s": cpu_user_s,
            "cpu_sys_s": cpu_sys_s,
            "cpu_util_avg": (cpu_user_s + cpu_sys_s) / max(1e-9, (time.time() - started)),
            "max_rss_kb": max_rss_kb,
        },
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

    # Normalize schema with SUF so downstream CSV/JSON consumers can treat these as consistent fields.
    info.setdefault("preprocessing", {}).setdefault("key_bytes_composite", 0)
    info.setdefault("preprocessing", {}).setdefault("key_bytes_matmul_triples", 0)
    info.setdefault("preprocessing", {}).setdefault("key_bytes_beaver_triples", 0)
    info.setdefault("preprocessing", {}).setdefault("key_bytes_row_triples", 0)
    info.setdefault("preprocessing", {}).setdefault("key_bytes_other", 0)
    info.setdefault("communication", {}).setdefault("pfss_bytes", 0)
    # SIGMA reports wire bytes as "Total Comm". Map that to `net_bytes` so
    # downstream comparisons with SUF's `net_bytes` are meaningful.
    info.setdefault("communication", {}).setdefault(
        "net_bytes", info.get("communication", {}).get("online_bytes", 0)
    )
    info.setdefault("communication", {}).setdefault("open_bytes", 0)
    info.setdefault("communication", {}).setdefault("open_packed_bytes", 0)
    info.setdefault("communication", {}).setdefault("open_bytes_beaver", 0)
    info.setdefault("communication", {}).setdefault("open_bytes_mask", 0)
    info.setdefault("communication", {}).setdefault("open_bytes_other", 0)
    info.setdefault("communication", {}).setdefault("open_packed_bytes_beaver", 0)
    info.setdefault("communication", {}).setdefault("open_packed_bytes_mask", 0)
    info.setdefault("communication", {}).setdefault("open_packed_bytes_other", 0)
    info.setdefault("communication", {}).setdefault("pfss_related_bytes", 0)
    info.setdefault("communication", {}).setdefault("pfss_related_packed_bytes", 0)
    info.setdefault("communication", {}).setdefault("beaver_related_bytes", 0)
    info.setdefault("communication", {}).setdefault("beaver_related_packed_bytes", 0)
    info.setdefault("communication", {}).setdefault("online_packed_bytes", info["communication"].get("online_bytes", 0))
    info.setdefault("communication", {}).setdefault("gelu_bytes", 0)
    info.setdefault("communication", {}).setdefault("softmax_bytes", 0)
    info.setdefault("communication", {}).setdefault("layernorm_bytes", 0)
    info.setdefault("resources", {}).setdefault("cpu_user_s", 0)
    info.setdefault("resources", {}).setdefault("cpu_sys_s", 0)
    info.setdefault("resources", {}).setdefault("cpu_util_avg", 0)
    info.setdefault("resources", {}).setdefault("max_rss_kb", 0)
    info.setdefault("pfss", {}).setdefault("open_flushes", 0)
    info.setdefault("pfss", {}).setdefault("opened_words", 0)

    return ok, info


def run_suf(
    suf_bin: Path,
    model: str,
    backend: str,
    seq_len: int,
    batch_size: int,
    n_iters: int,
    log_json: Path,
    extra_args: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
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
    ] + (list(extra_args) if extra_args else []) + [
        "--log-json",
        str(log_json),
    ]
    try:
        env = os.environ.copy()
        if extra_env:
            for k, v in extra_env.items():
                env[str(k)] = str(v)
        subprocess.check_call(cmd, env=env)
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


def normalize_sigma_schema(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("system") != "sigma":
        return row
    row.setdefault("preprocessing", {}).setdefault("key_bytes_composite", 0)
    row.setdefault("preprocessing", {}).setdefault("key_bytes_matmul_triples", 0)
    row.setdefault("preprocessing", {}).setdefault("key_bytes_beaver_triples", 0)
    row.setdefault("preprocessing", {}).setdefault("key_bytes_row_triples", 0)
    row.setdefault("preprocessing", {}).setdefault("key_bytes_other", 0)
    comm = row.setdefault("communication", {})
    comm.setdefault("pfss_bytes", 0)
    comm.setdefault("online_packed_bytes", comm.get("online_bytes", 0))
    comm.setdefault("net_bytes", comm.get("online_bytes", 0))
    comm.setdefault("open_bytes", 0)
    comm.setdefault("open_packed_bytes", 0)
    comm.setdefault("open_bytes_beaver", 0)
    comm.setdefault("open_bytes_mask", 0)
    comm.setdefault("open_bytes_other", 0)
    comm.setdefault("open_packed_bytes_beaver", 0)
    comm.setdefault("open_packed_bytes_mask", 0)
    comm.setdefault("open_packed_bytes_other", 0)
    comm.setdefault("pfss_related_bytes", 0)
    comm.setdefault("pfss_related_packed_bytes", 0)
    comm.setdefault("beaver_related_bytes", 0)
    comm.setdefault("beaver_related_packed_bytes", 0)
    comm.setdefault("gelu_bytes", 0)
    comm.setdefault("softmax_bytes", 0)
    comm.setdefault("layernorm_bytes", 0)
    row.setdefault("resources", {}).setdefault("cpu_user_s", 0)
    row.setdefault("resources", {}).setdefault("cpu_sys_s", 0)
    row.setdefault("resources", {}).setdefault("cpu_util_avg", 0)
    row.setdefault("resources", {}).setdefault("max_rss_kb", 0)
    row.setdefault("pfss", {}).setdefault("open_flushes", 0)
    row.setdefault("pfss", {}).setdefault("opened_words", 0)
    row.setdefault("pfss", {}).setdefault("opened_words_beaver", 0)
    row.setdefault("pfss", {}).setdefault("opened_words_mask", 0)
    row.setdefault("pfss", {}).setdefault("opened_words_other", 0)
    return row


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
    ap.add_argument("--skip-sigma", action="store_true", help="Skip running SIGMA (reuse cached logs if present).")
    ap.add_argument("--skip-suf", action="store_true", help="Skip running SUF (reuse cached logs if present).")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands without executing.")
    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    models: List[str] = list(cfg.get("models", []))
    seq_lens: List[int] = list(cfg.get("seq_lens", [128]))
    batch_sizes: List[int] = list(cfg.get("batch_sizes", [1]))
    backends: List[str] = list(cfg.get("backends", ["cpu"]))
    n_iters: int = int(cfg.get("n_iters", 1))
    suf_extra_args: List[str] = [str(x) for x in cfg.get("suf_extra_args", [])]
    suf_extra_args_by_model: Dict[str, List[str]] = {}
    raw_by_model = cfg.get("suf_extra_args_by_model", {})
    if isinstance(raw_by_model, dict):
        for k, v in raw_by_model.items():
            if isinstance(v, list):
                suf_extra_args_by_model[str(k)] = [str(x) for x in v]

    suf_env: Dict[str, str] = {}
    raw_env = cfg.get("suf_env", {})
    if isinstance(raw_env, dict):
        suf_env = {str(k): str(v) for k, v in raw_env.items()}
    suf_env_by_model: Dict[str, Dict[str, str]] = {}
    raw_env_by_model = cfg.get("suf_env_by_model", {})
    if isinstance(raw_env_by_model, dict):
        for k, v in raw_env_by_model.items():
            if isinstance(v, dict):
                suf_env_by_model[str(k)] = {str(kk): str(vv) for kk, vv in v.items()}

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
                if args.skip_sigma:
                    if sigma_log.exists():
                        sigma_row = load_json(sigma_log)
                        if sigma_row:
                            if notes:
                                sigma_row["notes"] = notes
                            if hardware:
                                sigma_row["hardware"] = hardware
                            rows.append(normalize_sigma_schema(sigma_row))
                            print(f"[reuse] sigma {model} L{seq_len}: {sigma_log}")
                    else:
                        print(f"[skip] sigma {model} L{seq_len}: --skip-sigma and no cached log {sigma_log}",
                              file=sys.stderr)
                elif not sigma_bin.exists():
                    if sigma_log.exists():
                        sigma_row = load_json(sigma_log)
                        if sigma_row:
                            if notes:
                                sigma_row["notes"] = notes
                            if hardware:
                                sigma_row["hardware"] = hardware
                            rows.append(normalize_sigma_schema(sigma_row))
                            print(f"[reuse] sigma {model} L{seq_len}: {sigma_log}")
                        else:
                            print(f"[skip] sigma {model} L{seq_len}: missing sigma binary and unreadable cached log",
                                  file=sys.stderr)
                    else:
                        print(f"[skip] sigma {model} L{seq_len}: sigma binary missing ({sigma_bin})",
                              file=sys.stderr)
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
                    if args.skip_suf:
                        if not suf_log.exists():
                            print(f"[skip] suf {model} {backend} L{seq_len} B{bs}: --skip-suf and no cached log {suf_log}",
                                  file=sys.stderr)
                            continue
                    else:
                        extra_args_model = suf_extra_args + suf_extra_args_by_model.get(model, [])
                        env_model = dict(suf_env)
                        env_model.update(suf_env_by_model.get(model, {}))
                        ok = run_suf(
                            suf_bin=suf_bin,
                            model=model,
                            backend=backend,
                            seq_len=seq_len,
                            batch_size=bs,
                            n_iters=n_iters,
                            log_json=suf_log,
                            extra_args=extra_args_model,
                            extra_env=env_model if env_model else None,
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
        "preprocessing.key_bytes_scope",
        "preprocessing.key_bytes_composite",
        "preprocessing.key_bytes_matmul_triples",
        "preprocessing.key_bytes_beaver_triples",
        "preprocessing.key_bytes_row_triples",
        "preprocessing.key_bytes_other",
        "timing.keygen_time_s",
        "timing.online_time_s",
        "timing.wall_time_s",
        "communication.online_bytes",
        "communication.online_packed_bytes",
        "communication.pfss_bytes",
        "communication.net_bytes",
        "communication.open_bytes",
        "communication.open_packed_bytes",
        "communication.open_bytes_beaver",
        "communication.open_bytes_mask",
        "communication.open_bytes_other",
        "communication.open_packed_bytes_beaver",
        "communication.open_packed_bytes_mask",
        "communication.open_packed_bytes_other",
        "communication.pfss_related_bytes",
        "communication.pfss_related_packed_bytes",
        "communication.beaver_related_bytes",
        "communication.beaver_related_packed_bytes",
        "communication.gelu_bytes",
        "communication.softmax_bytes",
        "communication.layernorm_bytes",
        "pfss.open_flushes",
        "pfss.opened_words",
        "pfss.opened_words_beaver",
        "pfss.opened_words_mask",
        "pfss.opened_words_other",
        "resources.cpu_user_s",
        "resources.cpu_sys_s",
        "resources.cpu_util_avg",
        "resources.max_rss_kb",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for row in rows:
            w.writerow([flatten_get(row, fn) for fn in fields])

    print(f"[done] wrote {jsonl_path} and {csv_path}")


if __name__ == "__main__":
    main()
