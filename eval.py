#!/usr/bin/env python3
"""Smoke-test evaluator: loads a checkpoint written by train.py and scores it on ONE 16 GB GPU.

Weights are loaded in fp16 (about 2 GB for the 1B preset), so this fits a T4 easily.
Reports loss, perplexity, next-token accuracy, throughput and peak GPU memory on
fresh batches (different seed from training). Self-contained: plain PyTorch.

Slurm (eval.sbatch runs it with the venv python):
    sbatch /opt/team7/jobs/eval.sbatch --ckpt /opt/dlami/nvme/team7/ckpt/<run>/step_00000030
    sbatch /opt/team7/jobs/eval.sbatch --random-init          # GPU load test, no checkpoint
Directly on the node:
    /opt/team7/venv/bin/python /opt/team7/src/eval.py --ckpt <dir>

--ckpt may be a step directory, a run directory, or a root directory: the newest
directory containing a COMPLETE marker is used. Without --ckpt it searches
[checkpoints].local_dir. Only model.pt and meta.json are needed (optim.pt may be absent).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- model
# Must match train.py (the checkpoint's meta.json carries the architecture).

PRESETS = {
    "tiny": dict(n_layer=2, n_head=2, d_model=64, vocab=256, seq=64),
    "125m": dict(n_layer=12, n_head=12, d_model=768, vocab=32000, seq=512),
    "350m": dict(n_layer=24, n_head=16, d_model=1024, vocab=32000, seq=512),
    "1b": dict(n_layer=18, n_head=16, d_model=2048, vocab=32000, seq=512),
}
DATA = dict(vocab=256, a=5, b=3)  # same synthetic chain as train.py


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        d = c["d_model"]
        self.wte = nn.Embedding(c["vocab"], d)
        self.wpe = nn.Embedding(c["seq"], d)
        self.blocks = nn.ModuleList(Block(d, c["n_head"]) for _ in range(c["n_layer"]))
        self.ln_f = nn.LayerNorm(d)
        for m in self.modules():  # same init as train.py, so --random-init gives sane logits
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return F.linear(self.ln_f(x), self.wte.weight)


def chain_table(seq: int, data: dict) -> torch.Tensor:
    t = torch.empty(data["vocab"], seq + 1, dtype=torch.long)
    t[:, 0] = torch.arange(data["vocab"])
    for i in range(seq):
        t[:, i + 1] = (data["a"] * t[:, i] + data["b"]) % data["vocab"]
    return t


# --------------------------------------------------------------------------- checkpoints


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def resolve_ckpt(path: str | None, default_root: str) -> Path:
    base = Path(path or default_root)
    if (base / "COMPLETE").exists():
        return base
    found = sorted(base.glob("**/COMPLETE"), key=lambda p: p.stat().st_mtime)
    if not found:
        sys.exit(f"[eval] no checkpoint with a COMPLETE marker under {base}")
    return found[-1].parent


def verify(d: Path) -> None:
    """Check model.pt and meta.json against the manifest (optim.pt is not needed)."""
    files = json.loads((d / "manifest.json").read_text())["files"]
    for n in ("model.pt", "meta.json"):
        if n not in files or not (d / n).exists():
            sys.exit(f"[eval] {d} is missing {n}")
        if sha256_file(d / n) != files[n]["sha256"]:
            sys.exit(f"[eval] {d / n} does not match its manifest checksum (corrupt or partial copy)")


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None, help="step dir, run dir or root dir (default: [checkpoints].local_dir)")
    ap.add_argument("--random-init", action="store_true", help="skip loading; random weights (GPU load test)")
    ap.add_argument("--preset", default="1b", choices=PRESETS, help="architecture for --random-init")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345, help="data seed (train.py uses 0)")
    ap.add_argument("--skip-verify", action="store_true", help="skip the sha256 check of model.pt")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[eval] ignoring unknown arguments: {unknown}", flush=True)

    root = "./ckpt"
    try:
        import team7config

        root = team7config.load().get("checkpoints.local_dir") or root
    except Exception as ex:
        print(f"[eval] team7config unavailable ({ex}); default checkpoint root {root}", flush=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32

    data, ckpt = dict(DATA), None
    if args.random_init:
        c, step = dict(PRESETS[args.preset]), None
    else:
        ckpt = resolve_ckpt(args.ckpt, root)
        if not args.skip_verify:
            t0 = time.time()
            verify(ckpt)
            print(f"[eval] checksums OK ({time.time() - t0:.1f}s)", flush=True)
        meta = json.loads((ckpt / "meta.json").read_text())
        c, step, data = meta["model"], meta["step"], meta.get("data", data)

    torch.set_default_dtype(dtype)
    with torch.device(device):
        model = GPT(c)
    torch.set_default_dtype(torch.float32)
    if ckpt:
        sd = torch.load(ckpt / "model.pt", map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(sd)  # copies and casts fp32 -> fp16 tensor by tensor
        del sd
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] device={device} ({torch.cuda.get_device_name(0) if use_cuda else 'cpu'}) params={n_params / 1e9:.3f}B "
          f"dtype={dtype} ckpt={ckpt or 'random-init'} step={step} seq={c['seq']} "
          f"batches={args.batches}x{args.batch_size}", flush=True)

    table, vocab = chain_table(c["seq"], data), c["vocab"]
    gen = torch.Generator()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.long)
    tokens, t_start = 0, None
    with torch.inference_mode():
        for i in range(args.batches):
            if i == 1:  # time everything after the first (warm-up) batch
                if use_cuda:
                    torch.cuda.synchronize()
                t_start, tokens = time.perf_counter(), 0
            gen.manual_seed(args.seed + i)
            seqs = table[torch.randint(0, table.size(0), (args.batch_size,), generator=gen)]
            x, y = seqs[:, :-1].to(device), seqs[:, 1:].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_cuda):
                logits = model(x)
            loss_sum += F.cross_entropy(logits.float().view(-1, vocab), y.reshape(-1), reduction="sum").double()
            correct += (logits.argmax(-1) == y).sum()
            tokens += y.numel()
            del logits
    if use_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t_start if t_start else float("nan")
    n_tok = args.batches * args.batch_size * c["seq"]
    loss = loss_sum.item() / n_tok
    result = {
        "ckpt": str(ckpt) if ckpt else "random-init",
        "step": step,
        "loss": round(loss, 4),
        "ppl": round(math.exp(min(loss, 50)), 3),
        "next_token_acc": round(correct.item() / n_tok, 4),
        "tok_per_s": round(tokens / wall) if t_start else None,
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if use_cuda else None,
    }
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
