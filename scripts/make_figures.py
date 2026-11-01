#!/usr/bin/env python3
"""レポート用の図を docs/figures/ に生成する."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager

from agripioneer.config import Config
from agripioneer.perception.dataset import footprint_label
from agripioneer.perception.camera import PinholeCamera
from agripioneer.perception.unet import GrassUNet, preprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def use_jp_font() -> None:
    for name in ("Hiragino Sans", "Hiragino Kaku Gothic ProN",
                 "Noto Sans CJK JP", "IPAexGothic"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name),
                                  fallback_to_default=False)
            plt.rcParams["font.family"] = name
            return
        except Exception:
            continue


# ------------------------------------------------------------------ 学習曲線

def training_curve(path: pathlib.Path) -> None:
    text = (ROOT / "docs" / "train_perception.log").read_text()
    rows = re.findall(
        r"epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+val_bce=([\d.]+)\s+val_IoU=([\d.]+)",
        text)
    if not rows:
        print("学習ログが見つかりません"); return
    ep = [int(r[0]) for r in rows]
    tl = [float(r[1]) for r in rows]
    iou = [float(r[3]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    ax1.plot(ep, tl, "o-", color="#d46a1e", lw=1.8, ms=4, label="学習 loss")
    ax1.set_xlabel("エポック"); ax1.set_ylabel("学習 loss (BCE + Dice)", color="#d46a1e")
    ax1.tick_params(axis="y", labelcolor="#d46a1e")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(ep, iou, "s-", color="#2a7fbf", lw=1.8, ms=4, label="検証 IoU")
    ax2.set_ylabel("検証 IoU", color="#2a7fbf")
    ax2.tick_params(axis="y", labelcolor="#2a7fbf")
    ax2.axhline(max(iou), ls="--", lw=1, color="#2a7fbf", alpha=0.5)
    ax2.annotate(f"best {max(iou):.3f}", (ep[int(np.argmax(iou))], max(iou)),
                 textcoords="offset points", xytext=(-6, -16), color="#2a7fbf", fontsize=9)
    ax1.set_title("草の株元セグメンテーション U-Net の学習曲線", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print("→", path)


# ------------------------------------------------------------ 知覚の定性結果

def perception_examples(path: pathlib.Path, n: int = 4) -> None:
    from agripioneer.world import World, sample_grass

    cfg = Config()
    blob = torch.load(ROOT / "models" / "grass_unet.pt", map_location="cpu",
                      weights_only=False)
    model = GrassUNet(base=blob.get("base", 16))
    model.load_state_dict(blob["state_dict"]); model.eval()

    res = cfg.robot.cam_res
    cam = PinholeCamera(res, res, cfg.robot.cam_fovy_deg)
    import mujoco

    rng = np.random.default_rng(777)
    rows = []
    for k in range(n):
        f = sample_grass(cfg, rng)
        w = World(cfg, f, (0.0, 0.0), 0.0)
        if k >= 2:  # 刈り取り途中のシーンも見せる
            w.cut_indices(rng.permutation(f.n)[: int(0.5 * f.n)])
        r = mujoco.Renderer(w.model, res, res)
        xy = rng.uniform(-cfg.plot.half, cfg.plot.half, size=2)
        look = rng.uniform(-cfg.plot.half, cfg.plot.half, size=2)
        w.set_pose(tuple(xy), float(np.arctan2(*(look - xy)[::-1])))
        r.update_scene(w.data, camera="onboard")
        rgb = r.render()
        gt = footprint_label(w, cam, cfg)
        with torch.no_grad():
            pr = torch.sigmoid(model(preprocess(rgb)))[0, 0].numpy()
        rows.append((rgb, gt, pr))
        r.close()

    fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.2), dpi=160)
    titles = ["搭載カメラ画像", "教師ラベル (株元)", "U-Net 推定確率"]
    for j, (rgb, gt, pr) in enumerate(rows):
        axes[0][j].imshow(rgb)
        axes[1][j].imshow(gt, cmap="YlGn", vmin=0, vmax=1)
        axes[2][j].imshow(pr, cmap="YlGn", vmin=0, vmax=1)
    for i in range(3):
        axes[i][0].set_ylabel(titles[i], fontsize=10)
        for j in range(n):
            axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
    fig.suptitle("合成データによる学習結果 (右 2 列は刈り取り途中のシーン)", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print("→", path)


# ------------------------------------------------------------ ベンチマーク

def benchmark_figure(path: pathlib.Path, data: pathlib.Path) -> dict:
    rows = json.loads(data.read_text())
    cfg = Config()
    w_blade = cfg.robot.blade_width * 0.88
    fov = np.deg2rad(cfg.robot.cam_fovy_deg)
    w_cam = 2.0 * (0.55 * cfg.robot.map_range) * np.tan(fov / 2.0)

    dens = sorted({r["density"] for r in rows if r["experiment"] == "density"})
    cov, base_d, ai_d, ai_cut, base_cut = [], [], [], [], []
    for d in dens:
        sel = [r for r in rows if r["experiment"] == "density" and r["density"] == d]
        cov.append(100 * np.mean([r["true_grass_coverage"] for r in sel]))
        b = [r["mow_end_distance"] for r in sel if r["mode"] == "baseline"]
        a = [r["mow_end_distance"] for r in sel if r["mode"] == "ai"]
        base_d.append(np.nanmean(b)); ai_d.append(np.nanmean(a))
        base_cut.append(100 * np.mean([r["cut_rate"] for r in sel if r["mode"] == "baseline"]))
        ai_cut.append(100 * np.mean([r["cut_rate"] for r in sel if r["mode"] == "ai"]))

    sizes = sorted({round(r["tsubo"], 2) for r in rows if r["experiment"] == "size"})
    s_base, s_ai = [], []
    for t in sizes:
        sel = [r for r in rows if r["experiment"] == "size" and abs(r["tsubo"] - t) < 0.1]
        s_base.append(np.nanmean([r["mow_end_distance"] for r in sel if r["mode"] == "baseline"]))
        s_ai.append(np.nanmean([r["mow_end_distance"] for r in sel if r["mode"] == "ai"]))

    # 実測への 1 次当てはめと, 両手法が入れ替わる被覆率 (交点) を求める
    cov_a = np.array(cov); ai_a = np.array(ai_d); base_a = np.array(base_d)
    slope, intercept = np.polyfit(cov_a, ai_a, 1)
    base_mean = float(np.mean(base_a))
    crossover = (base_mean - intercept) / slope if slope > 0 else float("nan")

    A = cfg.plot.area
    plt.rcParams.update({"font.size": 11.5, "axes.titlesize": 13,
                         "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11})
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.1), dpi=150)

    x = np.linspace(0, max(cov) * 1.18, 80)
    ax[0].plot(x, slope * x + intercept, "-", color="#d46a1e", lw=1.4, alpha=0.5)
    ax[0].plot(x, A / w_cam + (x / 100) * A / w_blade, ":", color="#555",
               lw=1.4, alpha=0.8, label="単純モデル (草の面積ぶんだけ刈る)")
    ax[0].plot(cov, base_d, "o-", color="#8a8f98", lw=2, label="全面被覆 (ベースライン)")
    ax[0].plot(cov, ai_d, "s-", color="#d46a1e", lw=2, label="知覚駆動 (提案手法)")
    if np.isfinite(crossover) and 0 < crossover < x[-1]:
        ax[0].axvline(crossover, ls="--", color="#b03a2e", lw=1.2, alpha=0.7)
        ax[0].annotate(f"入れ替わり\n{crossover:.0f} %", (crossover, base_mean * 0.55),
                       fontsize=10.5, color="#b03a2e", ha="center")
    ax[0].set_xlabel("草が占める面積の割合 [%]")
    ax[0].set_ylabel("刈り終えるまでの走行距離 [m]")
    ax[0].set_title("① 草の密度に対する走行距離 (1 坪)", fontsize=13)
    ax[0].legend(fontsize=9.5); ax[0].grid(alpha=0.25)

    red = [100 * (1 - a / b) for a, b in zip(ai_d, base_d)]
    bars = ax[1].bar([f"{c:.0f}%" for c in cov], red,
                     color=["#2e8b57" if r > 0 else "#b03a2e" for r in red])
    for b_, r in zip(bars, red):
        ax[1].text(b_.get_x() + b_.get_width() / 2,
                   r + (1.2 if r > 0 else -3.2), f"{r:+.0f}%",
                   ha="center", fontsize=11)
    ax[1].axhline(0, color="#333", lw=1)
    ax[1].set_xlabel("草が占める面積の割合 [%]")
    ax[1].set_ylabel("走行距離の削減率 [%]")
    ax[1].set_title("② 提案手法による距離の削減", fontsize=11)
    ax[1].grid(alpha=0.25, axis="y")

    wdt = 0.34
    idx = np.arange(len(sizes))
    ax[2].bar(idx - wdt / 2, s_base, wdt, color="#8a8f98", label="全面被覆")
    ax[2].bar(idx + wdt / 2, s_ai, wdt, color="#d46a1e", label="知覚駆動")
    for i, (b_, a_) in enumerate(zip(s_base, s_ai)):
        ax[2].text(i, max(b_, a_) * 1.03, f"{100 * (1 - a_ / b_):+.0f}%",
                   ha="center", fontsize=11, color="#c2660f")
    ax[2].set_ylim(0, max(max(s_base), max(s_ai)) * 1.18)
    ax[2].set_xticks(idx); ax[2].set_xticklabels([f"{t:g} 坪" for t in sizes])
    ax[2].set_ylabel("刈り終えるまでの走行距離 [m]")
    ax[2].set_title("③ 区画の大きさ (草の密度は同程度)", fontsize=11)
    ax[2].legend(fontsize=10); ax[2].grid(alpha=0.25, axis="y")

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print("→", path)
    return {"coverage": cov, "base_d": base_d, "ai_d": ai_d,
            "base_cut": base_cut, "ai_cut": ai_cut,
            "sizes": sizes, "s_base": s_base, "s_ai": s_ai,
            "w_blade": w_blade, "w_cam": w_cam,
            "fit_slope": float(slope), "fit_intercept": float(intercept),
            "crossover": float(crossover)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="outputs/benchmark.json")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    use_jp_font()
    FIG.mkdir(parents=True, exist_ok=True)
    if "curve" not in args.skip:
        training_curve(FIG / "training_curve.png")
    if "perception" not in args.skip:
        perception_examples(FIG / "perception_examples.png")
    bm = pathlib.Path(args.benchmark)
    if "benchmark" not in args.skip and bm.exists():
        stats = benchmark_figure(FIG / "benchmark.png", bm)
        (ROOT / "docs" / "benchmark_summary.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
