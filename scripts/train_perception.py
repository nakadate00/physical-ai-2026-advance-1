#!/usr/bin/env python3
"""草の株元セグメンテーション U-Net を学習する (CPU で完結)."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from agripioneer.perception.unet import GrassUNet, dice_loss, preprocess


def augment(rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """左右反転と測光ノイズによるデータ拡張."""
    if rng.random() < 0.5:
        rgb = rgb[:, :, ::-1, :]
        mask = mask[:, :, ::-1]
    rgb = rgb.astype(np.float32)
    # 明るさ・コントラスト・色かぶり
    b = rng.uniform(-22, 22, size=(rgb.shape[0], 1, 1, 1)).astype(np.float32)
    c = rng.uniform(0.85, 1.15, size=(rgb.shape[0], 1, 1, 1)).astype(np.float32)
    t = rng.uniform(-10, 10, size=(rgb.shape[0], 1, 1, 3)).astype(np.float32)
    rgb = (rgb - 128.0) * c + 128.0 + b + t
    rgb += rng.normal(0, 5.0, size=rgb.shape).astype(np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8), mask.copy()


def iou(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum().item()
    union = ((p + target) > 0).float().sum().item()
    return inter / union if union > 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/perception.npz")
    ap.add_argument("--out", default="models/grass_unet.pt")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    d = np.load(args.data)
    rgb, mask = d["rgb"], d["mask"]
    n = len(rgb)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"train={len(tr_idx)} val={len(val_idx)} positive_rate={mask.mean():.4f}", flush=True)

    model = GrassUNet(base=args.base)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_par / 1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(tr_idx) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    pos_w = torch.tensor([float((1 - mask.mean()) / max(mask.mean(), 1e-6))]).clamp(1, 12)
    print(f"BCE pos_weight={pos_w.item():.2f}")

    xv = preprocess(rgb[val_idx])
    yv = torch.from_numpy(mask[val_idx]).float().unsqueeze(1)

    best = -1.0
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_idx))
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(order), args.batch):
            sel = tr_idx[order[i : i + args.batch]]
            rb, mb = augment(rgb[sel], mask[sel], rng)
            x = preprocess(rb)
            y = torch.from_numpy(mb).float().unsqueeze(1)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w) \
                + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            vl, vi = 0.0, 0.0
            for i in range(0, len(xv), 32):
                lg = model(xv[i : i + 32])
                yb = yv[i : i + 32]
                vl += F.binary_cross_entropy_with_logits(lg, yb, pos_weight=pos_w).item() * len(yb)
                vi += iou(lg, yb) * len(yb)
            vl /= len(xv)
            vi /= len(xv)
        print(f"epoch {ep + 1:2d}/{args.epochs}  train_loss={tot / nb:.4f}  "
              f"val_bce={vl:.4f}  val_IoU={vi:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        if vi > best:
            best = vi
            torch.save({"state_dict": model.state_dict(), "base": args.base,
                        "val_iou": vi}, out)

    print(f"best val IoU = {best:.4f}  -> {out}")


if __name__ == "__main__":
    main()
