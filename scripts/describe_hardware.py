#!/usr/bin/env python3
"""
Collects a lightweight hardware description (CPU, GPU, network) and stores it
as JSON under bench/hardware_<hostname>.json. Safe to run on hosts without
GPUs; any missing fields are filled with "unknown".
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model": platform.processor() or "unknown",
        "cores": os.cpu_count() or 0,
    }
    lscpu = _run(["lscpu"])
    if lscpu:
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):
                info["model"] = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):") and info.get("cores", 0) == 0:
                try:
                    info["cores"] = int(line.split(":", 1)[1])
                except ValueError:
                    pass
    meminfo = _run(["grep", "MemTotal", "/proc/meminfo"])
    if meminfo:
        try:
            kb = int(meminfo.split()[1])
            info["ram_gb"] = round(kb / (1024 ** 2), 2)
        except Exception:
            info["ram_gb"] = "unknown"
    else:
        info["ram_gb"] = "unknown"
    return info


def gpu_info() -> List[Dict[str, Any]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: List[Dict[str, Any]] = []
    if not out:
        return gpus
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            name, mem, driver = parts[:3]
            try:
                mem_gb = float(mem) / 1024.0
            except ValueError:
                mem_gb = "unknown"
            gpus.append(
                {
                    "name": name,
                    "memory_gb": mem_gb,
                    "cuda_driver": driver,
                }
            )
    return gpus


def network_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    ping_out = _run(["ping", "-c", "3", "127.0.0.1"])
    if ping_out:
        for line in ping_out.splitlines():
            if "min/avg/max" in line or "min/avg/max/mdev" in line:
                try:
                    stats = line.split("=")[1].split()[0]
                    _, avg, _, _ = stats.split("/")
                    info["ping_ms"] = float(avg)
                except Exception:
                    pass
                break
    return info


def main() -> None:
    host = socket.gethostname()
    data = {
        "hostname": host,
        "cpu": cpu_info(),
        "gpus": gpu_info(),
        "network": network_info(),
    }

    out_dir = Path("bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hardware_{host}.json"
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
