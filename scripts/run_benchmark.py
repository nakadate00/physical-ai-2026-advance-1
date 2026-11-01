#!/usr/bin/env python3
"""ベースライン (全面被覆) と提案手法 (知覚駆動の被覆) を比較する.

比較軸は 2 つ.

1. 草の被覆率 (主軸): 草が密なら全面を舐めるのと変わらず, 疎なら大きく得をする
   はず, という仮説を検証する. 単純なモデルでは

       全面被覆の走行距離  L_base ~ A / w_blade
       提案手法の走行距離  L_ai   ~ A / w_cam + A_grass / w_blade

   なので, 草が占める面積比 A_grass/A が 1 - w_blade/w_cam を下回ると
   提案手法が有利になる (本構成では w_blade=0.22 m, w_cam~0.79 m).
2. 区画の大きさ: 上式は面積に比例するだけなので, 被覆率が同じなら
   区画を広げても優劣は変わらないはず, という予測を確かめる.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from agripioneer.config import Config
from agripioneer.mission import Mission
from agripioneer.perception.runner import GrassPerception
from agripioneer.planning.coverage import full_coverage_path
from agripioneer.soil.field import SoilField
from agripioneer.soil.recommender import CropRecommender
from agripioneer.world import World, sample_grass

TSUBO = 1.818


#: 草の密度水準. 群生の数・広がり・本数をまとめて変えることで,
#: 区画に占める草の面積をおよそ 5 % 〜 55 % の範囲で連続的に振る.
#: (群生の数だけを増やしても最小間隔の制約で頭打ちになるため)
DENSITY_LEVELS: tuple[dict, ...] = (
    dict(n_clusters=2, cluster_sigma=0.07, min_cluster_gap=0.55,
         n_scattered=2, blades_per_cluster=24),
    dict(n_clusters=3, cluster_sigma=0.09, min_cluster_gap=0.50,
         n_scattered=4, blades_per_cluster=32),
    dict(n_clusters=4, cluster_sigma=0.11, min_cluster_gap=0.45,
         n_scattered=8, blades_per_cluster=42),
    dict(n_clusters=5, cluster_sigma=0.14, min_cluster_gap=0.40,
         n_scattered=14, blades_per_cluster=56),
    dict(n_clusters=6, cluster_sigma=0.18, min_cluster_gap=0.35,
         n_scattered=22, blades_per_cluster=74),
)


def make_config(tsubo: float, level: int) -> Config:
    """tsubo 坪の区画・密度水準 level の設定を作る (群生の数は面積に比例)."""
    d = DENSITY_LEVELS[level]
    base = Config()
    return replace(
        base,
        plot=replace(base.plot, side=TSUBO * np.sqrt(tsubo)),
        grass=replace(
            base.grass,
            n_clusters=max(2, int(round(d["n_clusters"] * tsubo))),
            cluster_sigma=d["cluster_sigma"],
            min_cluster_gap=d["min_cluster_gap"],
            n_scattered=int(round(d["n_scattered"] * tsubo)),
            blades_per_cluster=d["blades_per_cluster"],
        ),
    )


def run_one(cfg: Config, seed: int, mode: str, with_soil: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    field = sample_grass(cfg, rng)
    start = (-cfg.plot.half + 0.17, -cfg.plot.half + 0.17)
    world = World(cfg, field, start, 0.0)

    perception = None if mode == "baseline" else GrassPerception(cfg, world)
    soil = SoilField(cfg, field, np.random.default_rng(seed + 1)) if with_soil else None
    rec = CropRecommender() if (with_soil and mode != "baseline") else None
    mission = Mission(cfg, world, soil=soil, perception=perception,
                      recommender=rec, mode=mode, seed=seed)

    baseline_len = float(
        np.linalg.norm(np.diff(full_coverage_path(cfg), axis=0), axis=1).sum())
    max_time = 90.0 + baseline_len / 0.10

    t0 = time.time()
    while not mission.done and world.time < max_time:
        mission.step(cfg.sim.control_dt)
    wall = time.time() - t0

    # 真の草被覆率 (株元半径 4.5 cm として計算)
    tgt = field.xy[field.height0 > cfg.grass.cut_threshold]
    gm = mission.gmap
    cc = gm.cell_centers.reshape(-1, 2)
    if len(tgt):
        d = np.min(np.linalg.norm(cc[:, None, :] - tgt[None, :, :], axis=-1), axis=1)
        true_cov = float((d <= 0.045).mean())
    else:
        true_cov = 0.0

    r = mission.result
    out = {
        "mode": mode, "seed": seed, "side": cfg.plot.side,
        "tsubo": cfg.plot.area / (TSUBO**2),
        "n_grass": field.n, "n_target": field.n_target,
        "true_grass_coverage": true_cov,
        "sim_time": r.sim_time, "distance": r.distance,
        "survey_distance": r.survey_distance, "mow_distance": r.mow_distance,
        "cut": r.cut, "cut_rate": r.cut_rate,
        "mow_end_time": r.mow_end_time,
        "mow_end_distance": r.mow_end_distance,
        "full_cut_time": r.full_cut_time,
        "map_coverage": r.map_coverage, "n_clusters": r.n_clusters,
        "rounds": r.rounds, "n_inferences": r.n_inferences,
        "baseline_path_length": baseline_len,
        "timed_out": bool(world.time >= max_time),
        "wall_time": wall,
    }
    if perception is not None:
        perception.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="outputs/benchmark.json")
    ap.add_argument("--summary-only", action="store_true",
                    help="再実行せず, 保存済みの結果から要約表だけを表示する")
    args = ap.parse_args()

    if args.summary_only:
        summarize(json.loads(pathlib.Path(args.out).read_text()))
        return

    results: list[dict] = []

    print("=== 実験 1: 草の被覆率 (1 坪固定) ===", flush=True)
    for density in range(len(DENSITY_LEVELS)):
        for seed in range(args.seeds):
            for mode in ("baseline", "ai"):
                cfg = make_config(1.0, level=density)
                r = run_one(cfg, 4000 + seed, mode)
                r["experiment"] = "density"
                r["density"] = density
                results.append(r)
                print(f"  密度L{density} seed={seed} {mode:8s} "
                      f"草被覆={100 * r['true_grass_coverage']:5.1f}% "
                      f"dist={r['mow_end_distance']:7.2f}m "
                      f"time={r['mow_end_time']:6.1f}s "
                      f"cut={100 * r['cut_rate']:5.1f}% ({r['wall_time']:.0f}s)",
                      flush=True)

    print("=== 実験 2: 区画の大きさ (被覆率は同程度) ===", flush=True)
    for tsubo in (1.0, 2.25, 4.0):
        for seed in range(max(2, args.seeds - 1)):
            for mode in ("baseline", "ai"):
                cfg = make_config(tsubo, level=1)
                r = run_one(cfg, 3000 + seed, mode)
                r["experiment"] = "size"
                r["density"] = 1
                results.append(r)
                print(f"  {tsubo:4.2f}坪 seed={seed} {mode:8s} "
                      f"草被覆={100 * r['true_grass_coverage']:5.1f}% "
                      f"dist={r['mow_end_distance']:7.2f}m "
                      f"time={r['mow_end_time']:6.1f}s "
                      f"cut={100 * r['cut_rate']:5.1f}% ({r['wall_time']:.0f}s)",
                      flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n保存: {out}")
    summarize(results)


def summarize(results: list[dict]) -> None:
    def agg(sel, key):
        v = [r[key] for r in results if sel(r) and not np.isnan(r.get(key, np.nan))]
        return float(np.mean(v)) if v else float("nan")

    def block(title: str, groups) -> None:
        print("\n" + "=" * 86)
        print(f"{title:<22}{'手法':<10}{'走行距離':>10}{'所要時間':>10}"
              f"{'刈取率':>9}{'距離削減':>10}{'草被覆':>9}")
        print("-" * 86)
        for label, sel in groups:
            cov = 100 * agg(sel, "true_grass_coverage")
            b = agg(lambda r: sel(r) and r["mode"] == "baseline", "mow_end_distance")
            a = agg(lambda r: sel(r) and r["mode"] == "ai", "mow_end_distance")
            for mode, d in (("baseline", b), ("ai", a)):
                red = "" if mode == "baseline" else f"{100 * (1 - a / b):+9.1f}%"
                t = agg(lambda r, m=mode: sel(r) and r["mode"] == m, "mow_end_time")
                c = 100 * agg(lambda r, m=mode: sel(r) and r["mode"] == m, "cut_rate")
                head = label if mode == "baseline" else ""
                print(f"{head:<22}{mode:<10}{d:>9.2f}m{t:>9.1f}s{c:>8.1f}%"
                      f"{red:>10}{cov if mode == 'baseline' else float('nan'):>8.1f}%")
            print("-" * 86)

    block("草の被覆率 (1坪)",
          [(f"密度水準 {d}", (lambda r, d=d: r["experiment"] == "density"
                             and r["density"] == d))
           for d in range(len(DENSITY_LEVELS))])
    block("区画の大きさ",
          [(f"{t} 坪", (lambda r, t=t: r["experiment"] == "size"
                        and abs(r["tsubo"] - t) < 0.1))
           for t in (1.0, 2.25, 4.0)])


if __name__ == "__main__":
    main()
