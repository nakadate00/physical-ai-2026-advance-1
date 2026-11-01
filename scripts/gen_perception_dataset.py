#!/usr/bin/env python3
"""知覚 CNN 用の合成データセットを生成して data/perception.npz に保存する."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from agripioneer.config import Config
from agripioneer.perception.dataset import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=70)
    ap.add_argument("--poses", type=int, default=40)
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data/perception.npz")
    args = ap.parse_args()

    cfg = Config()
    t0 = time.time()
    print(f"generating {args.worlds * args.poses} samples at {args.res}x{args.res} ...")
    rgb, mask = generate(cfg, args.worlds, args.poses, args.res, args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rgb=rgb, mask=mask)
    print(f"saved {out}  rgb={rgb.shape} mask={mask.shape} "
          f"positive_rate={mask.mean():.3f}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
