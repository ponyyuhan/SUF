#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from accuracy_bench import FixedPointSpec, eval_glue_classification, eval_lambada_next_token


def load_cfg(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _run_glue(entry: Dict[str, Any],
              device: str,
              max_examples: Optional[int],
              frac_bits: int) -> Dict[str, Any]:
    model_id = entry["model_id"]
    task = entry["task"]
    n_bits = int(entry["n_bits"])
    pt = eval_glue_classification(task, model_id, max_examples, device, None)
    suf = eval_glue_classification(task, model_id, max_examples, device, FixedPointSpec(n_bits, frac_bits))
    return {
        "model": entry.get("model", model_id),
        "task": task,
        "dataset": task,
        "train_size": entry.get("train_size", pt.get("train_size")),
        "val_size": entry.get("val_size", pt.get("val_size")),
        "pytorch_accuracy": pt["accuracy"],
        "sigma_accuracy": entry.get("sigma_accuracy"),
        "sigma_bits": n_bits,
        "suf_accuracy": suf["accuracy"],
        "suf_bits": n_bits,
        "frac_bits": frac_bits,
    }


def _run_lambada(entry: Dict[str, Any],
                 device: str,
                 max_examples: Optional[int],
                 frac_bits: int) -> Dict[str, Any]:
    model_id = entry["model_id"]
    n_bits = int(entry["n_bits"])
    pt = eval_lambada_next_token(model_id, max_examples, device, None)
    suf = eval_lambada_next_token(model_id, max_examples, device, FixedPointSpec(n_bits, frac_bits))
    return {
        "model": entry.get("model", model_id),
        "task": "lambada",
        "dataset": "lambada",
        "train_size": entry.get("train_size", pt.get("train_size")),
        "val_size": entry.get("val_size", pt.get("val_size")),
        "pytorch_accuracy": pt["accuracy"],
        "sigma_accuracy": entry.get("sigma_accuracy"),
        "sigma_bits": n_bits,
        "suf_accuracy": suf["accuracy"],
        "suf_bits": n_bits,
        "frac_bits": frac_bits,
    }


def _format_acc(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.2%}"
    except Exception:
        return str(v)


def _format_size(v: Any) -> str:
    if v is None:
        return "-"
    return str(v)


def _write_md(rows: List[Dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("| Model | Dataset | Train size | Val size | PyTorch acc | Sigma acc | Sigma bits | SUF acc | SUF bits |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        if r.get("skipped"):
            lines.append(
                f"| {r['model']} | {r['dataset']} | {r.get('train_size','-')} | {r.get('val_size','-')} | "
                f"{_format_acc(r.get('pytorch_accuracy'))} | {_format_acc(r.get('sigma_accuracy'))} | "
                f"{r.get('sigma_bits','-')} | skipped | {r.get('suf_bits','-')} |"
            )
            continue
        lines.append(
            f"| {r['model']} | {r['dataset']} | {_format_size(r.get('train_size'))} | {_format_size(r.get('val_size'))} | "
            f"{_format_acc(r.get('pytorch_accuracy'))} | {_format_acc(r.get('sigma_accuracy'))} | {r.get('sigma_bits','-')} | "
            f"{_format_acc(r.get('suf_accuracy'))} | {r.get('suf_bits','-')} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to accuracy_table4.json.")
    ap.add_argument("--out-json", required=True, help="Write results JSON here.")
    ap.add_argument("--out-md", required=True, help="Write Markdown table here.")
    ap.add_argument("--device", default="cpu", help="cpu|cuda")
    ap.add_argument("--max-examples", type=int, default=0, help="0 => full validation split.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    frac_bits = int(cfg.get("frac_bits", 12))
    max_examples: Optional[int] = None if args.max_examples == 0 else int(args.max_examples)

    rows: List[Dict[str, Any]] = []

    for entry in cfg.get("glue", []):
        if entry.get("skip"):
            rows.append({
                "model": entry.get("model", entry.get("model_id")),
                "dataset": entry.get("task"),
                "train_size": entry.get("train_size"),
                "val_size": entry.get("val_size"),
                "pytorch_accuracy": entry.get("pytorch_accuracy"),
                "sigma_accuracy": entry.get("sigma_accuracy"),
                "sigma_bits": entry.get("n_bits"),
                "suf_bits": entry.get("n_bits"),
                "skipped": True,
                "skip_reason": entry.get("skip_reason", "skipped"),
            })
            continue
        rows.append(_run_glue(entry, args.device, max_examples, frac_bits))

    for entry in cfg.get("lambada", []):
        if entry.get("skip"):
            rows.append({
                "model": entry.get("model", entry.get("model_id")),
                "dataset": "lambada",
                "train_size": entry.get("train_size"),
                "val_size": entry.get("val_size"),
                "pytorch_accuracy": entry.get("pytorch_accuracy"),
                "sigma_accuracy": entry.get("sigma_accuracy"),
                "sigma_bits": entry.get("n_bits"),
                "suf_bits": entry.get("n_bits"),
                "skipped": True,
                "skip_reason": entry.get("skip_reason", "skipped"),
            })
            continue
        rows.append(_run_lambada(entry, args.device, max_examples, frac_bits))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"config": cfg, "rows": rows}, indent=2) + "\n")

    _write_md(rows, Path(args.out_md))


if __name__ == "__main__":
    main()
