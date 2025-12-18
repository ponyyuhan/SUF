#!/usr/bin/env python3
"""
Accuracy benchmark scaffold aligned with ePrint 2023/1269 (SIGMA) Table 4.

This script computes:
  - PyTorch float32 accuracy (baseline)
  - Optional "SUF-style fixed-point emulation" accuracy by quantizing hidden states
    to (n_bits, frac_bits) after each Transformer block.

Notes / limitations:
  - This repo's C++ SUF prototype does not load real pretrained weights, so we
    cannot measure end-to-end MPC accuracy directly.
  - The "SUF emulation" mode here is a *cleartext* approximation intended to
    sanity-check whether fixed-point rounding at the chosen bitwidths is likely
    to preserve accuracy on the same datasets.
  - You must provide task-specific fine-tuned checkpoints for GLUE tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise SystemExit(
            f"Missing dependency `{pkg}` ({e}).\n"
            "Install:\n"
            "  python3 -m pip install --upgrade pip\n"
            "  python3 -m pip install torch transformers datasets\n"
        )


_require("torch")
_require("transformers")
_require("datasets")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@dataclass(frozen=True)
class FixedPointSpec:
    n_bits: int
    frac_bits: int


def quantize_like_fixed_point(x: torch.Tensor, spec: FixedPointSpec) -> torch.Tensor:
    if not torch.is_floating_point(x):
        return x
    if spec.n_bits <= 0 or spec.n_bits > 63:
        raise ValueError(f"n_bits must be 1..63 for this emulation (got {spec.n_bits})")
    if spec.frac_bits < 0 or spec.frac_bits >= spec.n_bits:
        raise ValueError(f"frac_bits must be 0..n_bits-1 (got {spec.frac_bits})")
    scale = float(1 << spec.frac_bits)
    y = torch.round(x * scale).to(torch.int64)
    lo = -(1 << (spec.n_bits - 1))
    hi = (1 << (spec.n_bits - 1)) - 1
    y = torch.clamp(y, lo, hi)
    return (y.to(torch.float32) / scale).to(x.dtype)


class BlockQuantHook:
    def __init__(self, spec: FixedPointSpec):
        self.spec = spec

    def __call__(self, _mod: torch.nn.Module, _inp: Tuple[Any, ...], out: Any) -> Any:
        # HF blocks commonly return Tensor or a tuple whose first element is hidden states.
        if isinstance(out, torch.Tensor):
            return quantize_like_fixed_point(out, self.spec)
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            out0 = quantize_like_fixed_point(out[0], self.spec)
            if isinstance(out, tuple):
                return (out0,) + tuple(out[1:])
            out = list(out)
            out[0] = out0
            return out
        return out


def attach_block_quant_hooks(model: torch.nn.Module, spec: FixedPointSpec) -> List[Any]:
    handles: List[Any] = []
    hook = BlockQuantHook(spec)

    # BERT-like encoder blocks.
    for m in model.modules():
        if m.__class__.__name__ in ("BertLayer", "RobertaLayer"):
            handles.append(m.register_forward_hook(hook))

    # GPT-like decoder blocks.
    for m in model.modules():
        if m.__class__.__name__ in ("GPT2Block", "GPTNeoBlock", "LlamaDecoderLayer"):
            handles.append(m.register_forward_hook(hook))

    return handles


def _accuracy(preds: Iterable[int], labels: Iterable[int]) -> float:
    total = 0
    correct = 0
    for p, y in zip(preds, labels):
        total += 1
        correct += int(p == y)
    return float(correct) / float(total) if total else float("nan")


def eval_glue_classification(
    task: str,
    model_id: str,
    max_examples: Optional[int],
    device: str,
    quant_spec: Optional[FixedPointSpec],
) -> Dict[str, Any]:
    ds = load_dataset("glue", task, split="validation")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(device)

    handles: List[Any] = []
    if quant_spec is not None:
        handles = attach_block_quant_hooks(model, quant_spec)

    preds: List[int] = []
    labels: List[int] = []

    # Task-specific field mapping.
    if task == "sst2":
        fields = ("sentence",)
    elif task == "mrpc":
        fields = ("sentence1", "sentence2")
    elif task == "qnli":
        fields = ("question", "sentence")
    else:
        raise ValueError(f"unsupported GLUE task: {task}")

    with torch.no_grad():
        for ex in ds:
            texts = [ex[f] for f in fields]
            enc = tok(
                *texts,
                return_tensors="pt",
                truncation=True,
                max_length=tok.model_max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pred = int(torch.argmax(out.logits, dim=-1).item())
            preds.append(pred)
            labels.append(int(ex["label"]))

    for h in handles:
        h.remove()

    return {
        "task": task,
        "model_id": model_id,
        "n": len(labels),
        "metric": "accuracy",
        "accuracy": _accuracy(preds, labels),
    }


def eval_lambada_next_token(
    model_id: str,
    max_examples: Optional[int],
    device: str,
    quant_spec: Optional[FixedPointSpec],
) -> Dict[str, Any]:
    ds = load_dataset("lambada", "plain_text", split="validation")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Ensure pad token exists for batchless use.
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model.to(device)

    handles: List[Any] = []
    if quant_spec is not None:
        handles = attach_block_quant_hooks(model, quant_spec)

    total = 0
    correct = 0

    with torch.no_grad():
        for ex in ds:
            text = ex["text"]
            ids = tok(text, return_tensors="pt", truncation=True, max_length=tok.model_max_length)["input_ids"][0]
            if ids.numel() < 2:
                continue
            inp = ids[:-1].unsqueeze(0).to(device)
            tgt = int(ids[-1].item())
            out = model(input_ids=inp)
            pred = int(torch.argmax(out.logits[0, -1, :]).item())
            total += 1
            correct += int(pred == tgt)

    for h in handles:
        h.remove()

    return {
        "task": "lambada",
        "model_id": model_id,
        "n": total,
        "metric": "next_token_acc",
        "accuracy": float(correct) / float(total) if total else float("nan"),
    }


def load_cfg(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config.")
    ap.add_argument("--out-json", required=True, help="Write results JSON here.")
    ap.add_argument("--device", default="cpu", help="cpu|cuda")
    ap.add_argument("--max-examples", type=int, default=0, help="0 => full validation split.")
    ap.add_argument("--suf-emulate", action="store_true", help="Enable fixed-point emulation hooks.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device=cuda but torch.cuda.is_available() is false")

    max_examples: Optional[int] = None if args.max_examples == 0 else int(args.max_examples)

    spec = FixedPointSpec(int(cfg["n_bits"]), int(cfg["frac_bits"]))
    quant_spec = spec if args.suf_emulate else None

    results: Dict[str, Any] = {
        "config": cfg,
        "device": device,
        "max_examples": max_examples,
        "suf_emulate": bool(args.suf_emulate),
        "runs": [],
    }

    for run in cfg.get("glue", []):
        results["runs"].append(
            eval_glue_classification(
                task=str(run["task"]),
                model_id=str(run["model_id"]),
                max_examples=max_examples,
                device=device,
                quant_spec=quant_spec,
            )
        )

    for run in cfg.get("lambada", []):
        results["runs"].append(
            eval_lambada_next_token(
                model_id=str(run["model_id"]),
                max_examples=max_examples,
                device=device,
                quant_spec=quant_spec,
            )
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()

