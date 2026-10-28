#!/usr/bin/env python3
"""Smoke-test trainer: a GPT of about 1B parameters on ONE 16 GB GPU (T4).

Meant to run for a few steps so you can watch GPU utilization. Self-contained:
plain PyTorch, no extra packages, synthetic data.

Memory (about 1B params): fp32 weights 3.9 GB + fp32 grads 3.9 GB + fp16 autocast
weight copies 1.9 GB + Adafactor state (factored, small) + activations with
checkpointing. Expect roughly 11-12 GB peak. AdamW would need 16 bytes/param
(16 GB) and does not fit, hence Adafactor.

Slurm (train.sbatch runs it under the DeepSpeed launcher):
    sbatch /opt/team7/jobs/train.sbatch --preset 1b --steps 30
Directly on the node:
    /opt/team7/venv/bin/deepspeed --num_gpus 1 /opt/team7/src/train.py --preset 1b --steps 30

Follows the contract in cluster/README.md: PID file, SIGUSR1 -> save -> exit with
the requeue code, atomic checkpoints with a COMPLETE marker, resume of a requeued
job (same job ID) from its newest complete checkpoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import hashlib
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --------------------------------------------------------------------------- model

PRESETS = {
    "tiny": dict(n_layer=2, n_head=2, d_model=64, vocab=256, seq=64),  # CPU smoke test
    "125m": dict(n_layer=12, n_head=12, d_model=768, vocab=32000, seq=512),
    "350m": dict(n_layer=24, n_head=16, d_model=1024, vocab=32000, seq=512),
    "1b": dict(n_layer=18, n_head=16, d_model=2048, vocab=32000, seq=512),  # ~0.97B
}
T4_FP16_PEAK_TFLOPS = 65.0


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
    def __init__(self, c: dict, grad_ckpt: bool = False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        d = c["d_model"]
        self.wte = nn.Embedding(c["vocab"], d)
        self.wpe = nn.Embedding(c["seq"], d)
        self.blocks = nn.ModuleList(Block(d, c["n_head"]) for _ in range(c["n_layer"]))
        self.ln_f = nn.LayerNorm(d)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        for n, p in self.named_parameters():
            if n.endswith(("proj.weight", "out.weight")):
                nn.init.normal_(p, 0.0, 0.02 / (2 * c["n_layer"]) ** 0.5)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.grad_ckpt and self.training) else blk(x)
        return F.linear(self.ln_f(x), self.wte.weight)  # tied output head


# --------------------------------------------------------------------------- data
# Learnable synthetic data: token[t+1] = (A * token[t] + B) mod DATA_VOCAB, a
# full-period chain over a 256-token sub-alphabet. Each sequence is a random
# start token, so the loss can fall from ln(vocab) toward 0. How fast depends on
# --lr (the tiny preset reaches ~0 in 100-300 steps at lr 1e-2; not verified at 1B).

DATA = dict(vocab=256, a=5, b=3)


def chain_table(seq: int, data: dict) -> torch.Tensor:
    t = torch.empty(data["vocab"], seq + 1, dtype=torch.long)
    t[:, 0] = torch.arange(data["vocab"])
    for i in range(seq):
        t[:, i + 1] = (data["a"] * t[:, i] + data["b"]) % data["vocab"]
    return t


def get_batch(table, bs, gen, device):
    seqs = table[torch.randint(0, table.size(0), (bs,), generator=gen)]
    return seqs[:, :-1].to(device), seqs[:, 1:].to(device)


# --------------------------------------------------------------------------- checkpoints


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def fsync_path(p: Path) -> None:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_ckpt(d: Path) -> bool:
    try:
        files = json.loads((d / "manifest.json").read_text())["files"]
        return all(sha256_file(d / n) == m["sha256"] for n, m in files.items())
    except (OSError, KeyError, ValueError):
        return False


def latest_valid(run_dir: Path) -> Path | None:
    for marker in sorted(run_dir.glob("step_*/COMPLETE"), reverse=True):
        if verify_ckpt(marker.parent):
            return marker.parent
        print(f"[ckpt] skipping corrupt checkpoint {marker.parent}", flush=True)
    return None


def save_ckpt(run_dir: Path, step: int, model, opt, scaler, meta: dict, keep: int) -> Path:
    t0 = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp, final = run_dir / f"step_{step:08d}.tmp", run_dir / f"step_{step:08d}"
    for p in (tmp, final):
        if p.exists():
            shutil.rmtree(p)
    tmp.mkdir()
    torch.save(model.state_dict(), tmp / "model.pt")
    torch.save({"optimizer": opt.state_dict(), "scaler": scaler.state_dict()}, tmp / "optim.pt")
    (tmp / "meta.json").write_text(json.dumps({**meta, "step": step}, indent=2))
    files = {n: {"sha256": sha256_file(tmp / n), "bytes": (tmp / n).stat().st_size}
             for n in ("model.pt", "optim.pt", "meta.json")}
    (tmp / "manifest.json").write_text(json.dumps({"files": files}, indent=2))
    for n in (*files, "manifest.json"):
        fsync_path(tmp / n)
    os.rename(tmp, final)
    (final / "COMPLETE.tmp").write_text(str(step))
    os.replace(final / "COMPLETE.tmp", final / "COMPLETE")
    fsync_path(final)
    done = sorted(p.parent for p in run_dir.glob("step_*/COMPLETE"))
    for old in done[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"[ckpt] saved {final} in {time.time() - t0:.1f}s", flush=True)
    return final


# --------------------------------------------------------------------------- main


def load_team7():
    try:
        import team7config

        return team7config.load()
    except Exception as ex:  # not on a cluster node
        print(f"[train] team7config unavailable ({ex}); using built-in defaults", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="1b", choices=PRESETS)
    ap.add_argument("--steps", type=int, default=30, help="total optimizer steps")
    ap.add_argument("--micro-batch", type=int, default=4, help="sequences per forward pass")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq", type=int, default=None, help="override the preset sequence length")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default=None,
                    help="default: [training].precision from cluster.toml, else fp16")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable activation checkpointing")
    ap.add_argument("--ckpt-root", default=None, help="default: [checkpoints].local_dir")
    ap.add_argument("--run-name", default=None, help="default: $SLURM_JOB_ID, else a timestamp")
    ap.add_argument("--save-every", type=int, default=0, help="0 = only save at the end")
    ap.add_argument("--no-save-final", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    args, unknown = ap.parse_known_args()  # the launcher adds --local_rank, the template --deepspeed_config
    if unknown:
        print(f"[train] ignoring launcher/template arguments: {unknown}", flush=True)

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        sys.exit("train.py is single-GPU: run with --num_gpus 1 (or one GPU per job)")

    cfg = load_team7()
    requeue_exit = cfg.requeue_exit if cfg else 75
    precision = args.precision or (cfg.get("training.precision") if cfg else None) or "fp16"
    if precision not in ("fp16", "fp32"):
        sys.exit(f"precision {precision!r} not supported here (the T4 has no bf16); use fp16 or fp32")
    ckpt_root = Path(args.ckpt_root or (cfg.get("checkpoints.local_dir") if cfg else None) or "./ckpt")
    keep = int((cfg.get("checkpoints.keep_local") if cfg else None) or 2)
    run_name = args.run_name or os.environ.get("SLURM_JOB_ID") or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = ckpt_root / run_name

    # PID file so the batch script can forward SIGUSR1 to this process only.
    try:
        pid_dir = Path(os.environ.get("TEAM7_PID_DIR") or (cfg.pid_dir() if cfg else "/tmp/team7-local"))
        pid_dir.mkdir(parents=True, exist_ok=True)
        (pid_dir / f"rank-{os.environ.get('RANK', '0')}.pid").write_text(str(os.getpid()))
    except Exception as ex:
        print(f"[train] could not write PID file: {ex}", flush=True)
    stop = False

    def on_usr1(signum, frame):
        nonlocal stop
        stop = True  # only set a flag; the loop saves at the next step boundary

    signal.signal(signal.SIGUSR1, on_usr1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = use_cuda and precision == "fp16"
    c = dict(PRESETS[args.preset])
    if args.seq:
        c["seq"] = args.seq
    torch.manual_seed(args.seed)

    with torch.device(device):
        model = GPT(c, grad_ckpt=not args.no_grad_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    model.train()
    opt = torch.optim.Adafactor(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10, enabled=amp)
    table = chain_table(c["seq"], DATA)
    meta = {"model": c, "data": DATA, "preset": args.preset, "n_params": n_params, "precision": precision}

    start_step = 0
    resume = latest_valid(run_dir) if run_dir.exists() else None
    if resume:
        model.load_state_dict(torch.load(resume / "model.pt", map_location=device, weights_only=True))
        st = torch.load(resume / "optim.pt", map_location=device, weights_only=True)
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        start_step = json.loads((resume / "meta.json").read_text())["step"]
        print(f"[train] resumed from {resume} at step {start_step}", flush=True)

    tokens_per_step = args.micro_batch * args.grad_accum * c["seq"]
    flops_per_token = (6 if args.no_grad_ckpt else 8) * n_params  # 8N = fwd + recompute + bwd
    print(f"[train] job={os.environ.get('SLURM_JOB_ID', '-')} restart={os.environ.get('SLURM_RESTART_COUNT', '0')} "
          f"device={device} ({torch.cuda.get_device_name(0) if use_cuda else 'cpu'}) preset={args.preset} "
          f"params={n_params / 1e9:.3f}B precision={precision} grad_ckpt={not args.no_grad_ckpt}", flush=True)
    print(f"[train] micro_batch={args.micro_batch} grad_accum={args.grad_accum} seq={c['seq']} "
          f"tokens/step={tokens_per_step} steps={start_step}->{args.steps} ckpt_dir={run_dir}", flush=True)

    gen = torch.Generator()
    skipped_total = 0
    t_run = time.perf_counter()
    last_loss = float("nan")
    for step in range(start_step, args.steps):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        loss_sum = torch.zeros((), device=device)
        for micro in range(args.grad_accum):
            gen.manual_seed(args.seed * 1_000_003 + step * args.grad_accum + micro)  # data depends on step only
            x, y = get_batch(table, args.micro_batch, gen, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
            del logits
            scaler.scale(loss / args.grad_accum).backward()
            loss_sum += loss.detach() / args.grad_accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        last_loss = loss_sum.item()  # syncs the GPU
        dt = time.perf_counter() - t0
        skipped = amp and scaler.get_scale() < scale_before
        skipped_total += int(skipped)
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            tflops = flops_per_token * tokens_per_step / dt / 1e12
            mem = f" peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB" if use_cuda else ""
            util = f" ({100 * tflops / T4_FP16_PEAK_TFLOPS:.0f}% of T4 fp16 peak)" if use_cuda else ""
            print(f"step {step + 1:>4}/{args.steps} loss={last_loss:.4f} gnorm={float(gnorm):.2f} "
                  f"lr={opt.param_groups[0]['lr']:.1e} {dt:.2f}s/step {tokens_per_step / dt:,.0f} tok/s "
                  f"~{tflops:.1f} TFLOP/s{util}{mem}{' [fp16 overflow, step skipped]' if skipped else ''}",
                  flush=True)

        if stop:
            print(f"[train] SIGUSR1 received: saving at step {step + 1}", flush=True)
            save_ckpt(run_dir, step + 1, model, opt, scaler, meta, keep)
            print(f"[train] exiting {requeue_exit} (Slurm requeues the job)", flush=True)
            return requeue_exit
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < args.steps:
            save_ckpt(run_dir, step + 1, model, opt, scaler, meta, keep)

    if start_step >= args.steps:
        print(f"[train] nothing to do: checkpoint is already at step {start_step}", flush=True)
    elif not args.no_save_final:
        save_ckpt(run_dir, args.steps, model, opt, scaler, meta, keep)
    wall = time.perf_counter() - t_run
    peak = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
    print("SUMMARY " + json.dumps({"steps_run": max(0, args.steps - start_step), "final_loss": round(last_loss, 4),
                                   "skipped_fp16_steps": skipped_total, "wall_s": round(wall, 1),
                                   "peak_mem_gib": round(peak, 2), "ckpt_dir": str(run_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
