#!/usr/bin/env python3
"""頑健性の破壊試験.

「決め打ちの手順をなぞっているだけではないか」を確かめるための実験群.
想定外の状況をわざと作り, 行動が状況に応じて変わるか / 破綻しないかを見る.

  A 基準            既定の初期姿勢
  B 初期姿勢ランダム  置かれる場所と向きが毎回変わる
  C 草の途中出現      探査が終わった後に新しい草むらが現れる (見落とし/再生の模擬)
  D 知覚遮断          U-Net の出力を常に「草なし」に固定する (アブレーション)
  E 外乱             走行中にランダムな外力を加え続ける (スリップ/凹凸の模擬)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from agripioneer.config import Config
from agripioneer.mission import Mission, Phase
from agripioneer.perception.runner import GrassPerception
from agripioneer.world import World, sample_grass


def run(
    seed: int,
    start: tuple[float, float] | None = None,
    yaw: float | None = None,
    blind: bool = False,
    inject_at_phase: bool = False,
    disturbance: float = 0.0,
    max_time: float = 320.0,
) -> dict:
    cfg = Config()
    rng = np.random.default_rng(seed)
    field = sample_grass(cfg, rng)

    lim = cfg.plot.half - 0.12
    if start is None:
        start = (-cfg.plot.half + 0.17, -cfg.plot.half + 0.17)
    if yaw is None:
        yaw = 0.0
    world = World(cfg, field, start, yaw)

    # C: 一部の草を最初は「刈り株」にしておき, 途中で生やす
    hidden: np.ndarray = np.array([], dtype=int)
    if inject_at_phase:
        # 実在する草むらを丸ごと 1 つ選び, 「最初は無かったことにする」.
        # 適当な点の周りを取ると数本しか隠れず試験にならないため,
        # ある株の近傍 n_hide 本をまとめて隠す.
        hid_rng = np.random.default_rng(seed + 99)
        anchor = int(hid_rng.integers(0, field.n))
        d = np.linalg.norm(field.xy - field.xy[anchor], axis=1)
        n_hide = min(40, max(20, field.n // 8))
        hidden = np.argsort(d)[:n_hide]
        if hidden.size:
            world.cut_indices(hidden)
            field.cut[hidden] = False  # 「まだ刈っていない」扱いに戻す
            world.field.height[hidden] = cfg.grass.stubble_height

    perception = GrassPerception(cfg, world, blind=blind)
    mission = Mission(cfg, world, perception=perception, mode="ai", seed=seed)

    dist_rng = np.random.default_rng(seed + 7)
    injected = False
    inject_time = None
    while not mission.done and world.time < max_time:
        # C: 探査が終わった瞬間に草を出現させる
        if inject_at_phase and not injected and mission.phase is not Phase.SURVEY:
            if hidden.size:
                world.grow_indices(hidden)
            injected = True
            inject_time = world.time

        # E: 車体に横向きのランダム外力を加え続ける
        if disturbance > 0.0:
            f = dist_rng.normal(0.0, disturbance, size=3)
            f[2] = 0.0
            world.data.xfrc_applied[world.base_bid, :3] = f

        mission.step(cfg.sim.control_dt)

    r = mission.result
    perception.close()
    return {
        "seed": seed, "start": list(start), "yaw": float(yaw),
        "blind": blind, "inject": inject_at_phase, "disturbance": disturbance,
        "n_target": field.n_target, "cut": r.cut, "cut_rate": r.cut_rate,
        "n_clusters": r.n_clusters, "rounds": r.rounds,
        "mow_end_distance": r.mow_end_distance, "mow_end_time": r.mow_end_time,
        "sim_time": r.sim_time, "map_coverage": r.map_coverage,
        "n_injected": int(hidden.size), "inject_time": inject_time,
        "timed_out": bool(world.time >= max_time),
        "crop": r.recommendations[0].crop.name if r.recommendations else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/robustness.json")
    ap.add_argument("--only", default=None,
                    help="実行する実験の先頭文字 (例: C). 省略時は全部")
    args = ap.parse_args()

    cfg = Config()
    lim = cfg.plot.half - 0.12
    results: list[dict] = []

    def show(tag: str, r: dict) -> None:
        ok = "OK " if r["cut_rate"] >= 1.0 else "NG "
        print(f"  {ok}{tag:28s} 草{r['n_target']:4d}本 → 刈取 {100*r['cut_rate']:5.1f}%  "
              f"草むら{r['n_clusters']:2d}箇所 巡回{r['rounds']}回  "
              f"{r['mow_end_distance']:6.2f}m / {r['mow_end_time']:6.1f}s"
              + ("  [時間切れ]" if r["timed_out"] else ""), flush=True)

    if args.only in (None, "A"):
      print("A 基準 (既定の初期姿勢)", flush=True)
      for s in (20261102, 555):
          r = run(s); r["case"] = "A_base"; results.append(r); show(f"seed={s}", r)

    if args.only in (None, "B"):
      print("B 初期姿勢をランダムに変える", flush=True)
      for s in (101, 202, 303, 404):
          g = np.random.default_rng(s)
          xy = tuple(g.uniform(-lim, lim, size=2))
          yaw = float(g.uniform(-np.pi, np.pi))
          r = run(s, start=xy, yaw=yaw)
          r["case"] = "B_random_start"; results.append(r)
          show(f"seed={s} ({xy[0]:+.2f},{xy[1]:+.2f}) {np.degrees(yaw):+4.0f}deg", r)

    if args.only in (None, "C"):
      print("C 探査の後に草むらが出現する", flush=True)
      for s in (20261102, 555, 303):
          r = run(s, inject_at_phase=True)
          r["case"] = "C_injection"; results.append(r)
          show(f"seed={s} 出現{r['n_injected']:3d}本 @{r['inject_time']:.0f}s", r)

    if args.only in (None, "D"):
      print("D 知覚を遮断する (U-Net 出力を常に 0 に固定)", flush=True)
      for s in (20261102, 555):
          r = run(s, blind=True, max_time=180.0)
          r["case"] = "D_blind"; results.append(r); show(f"seed={s}", r)

    if args.only in (None, "E"):
      print("E 走行中に外乱を加え続ける", flush=True)
      for amp in (12.0, 25.0):
          for s in (20261102, 555):
              r = run(s, disturbance=amp)
              r["case"] = f"E_disturb{amp:.0f}"; results.append(r)
              show(f"seed={s} 外力 sigma={amp:.0f} N", r)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n保存: {out}")

    print("\n" + "=" * 74)
    print(f"{'条件':<22}{'試行':>5}{'刈取 100%':>11}{'平均刈取率':>12}{'平均距離':>11}")
    print("-" * 74)
    for case in sorted({r["case"] for r in results},
                       key=lambda c: (c[0], c)):
        sel = [r for r in results if r["case"] == case]
        n_ok = sum(r["cut_rate"] >= 1.0 for r in sel)
        print(f"{case:<22}{len(sel):>5}{n_ok:>8}/{len(sel):<3}"
              f"{100*np.mean([r['cut_rate'] for r in sel]):>11.1f}%"
              f"{np.nanmean([r['mow_end_distance'] for r in sel]):>10.2f}m")
    print("=" * 74)


if __name__ == "__main__":
    main()
