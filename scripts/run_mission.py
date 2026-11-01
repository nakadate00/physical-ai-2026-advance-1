#!/usr/bin/env python3
"""AgriPioneer のミッションを実行し, HUD 付きの動画を書き出す."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import imageio.v2 as imageio
import mujoco
import numpy as np

from agripioneer.config import Config
from agripioneer.mission import PHASE_LABEL, Mission, Phase
from agripioneer.perception.runner import GrassPerception
from agripioneer.soil.field import PROP_LABELS_JA, SoilField
from agripioneer.soil.recommender import CropRecommender
from agripioneer.viz import (ACCENT, FG, GREEN, MUTED, HudRenderer, HudState,
                             overlay_path, overlay_soil_points, title_card)
from agripioneer.world import World, sample_grass

START_XY = (-0.74, -0.74)
START_YAW = 0.0


def build_report_lines(mission: Mission) -> list[tuple[str, str]]:
    r = mission.result
    if not r.recommendations:
        return []
    lines: list[tuple[str, str]] = []
    v = r.soil
    lines.append(("head", f"計測(中央値): pH {v['pH']:.1f} / EC {v['EC']:.2f} dS/m / "
                          f"有機物 {v['OM']:.1f}%"))
    lines.append(("head", f"            N {v['N']:.1f} / P {v['P']:.0f} / K {v['K']:.0f} "
                          f"mg/100g / 含水率 {v['moisture']:.0f}%"))
    for i, rec in enumerate(r.recommendations, 1):
        lines.append(("item", f"{i}. {rec.crop.name}   {rec.probability * 100:.1f} %"))
        if i == 1:
            lines.append(("sub", f"   {rec.crop.note}"))
    for tip in r.advice[:2]:
        lines.append(("sub", f"   ・{tip}"))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=Config().seed)
    ap.add_argument("--out", default="outputs/agripioneer.mp4")
    ap.add_argument("--max-time", type=float, default=260.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--mode", choices=["ai", "baseline"], default="ai")
    ap.add_argument("--view", choices=["overview", "chase"], default="overview")
    args = ap.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    rng = np.random.default_rng(args.seed)

    field = sample_grass(cfg, rng)
    world = World(cfg, field, START_XY, START_YAW)
    soil = SoilField(cfg, field, np.random.default_rng(args.seed + 1))
    perception = None if args.mode == "baseline" else GrassPerception(cfg, world)
    recommender = None if args.mode == "baseline" else CropRecommender()
    mission = Mission(cfg, world, soil=soil, perception=perception,
                      recommender=recommender, mode=args.mode, seed=args.seed)

    print(f"草 {field.n} 本 (刈り取り対象 {field.n_target} 本) / mode={args.mode}")
    if perception is not None:
        print(f"知覚モデル val IoU = {perception.val_iou:.4f}")

    writer = None
    hud = None
    main_renderer = None
    if not args.no_video:
        hud = HudRenderer(cfg.plot.half)
        main_renderer = mujoco.Renderer(world.model, hud.MAIN_H, hud.MAIN_W)
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(out, fps=args.fps, codec="libx264",
                                    quality=8, macro_block_size=1,
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        for frame, hold in title_frames(hud, cfg, field):
            for _ in range(int(hold * args.fps)):
                writer.append_data(frame)

    dt = cfg.sim.control_dt
    next_frame = 0.0
    t0 = time.time()
    n_frames = 0

    while not mission.done and world.time < args.max_time:
        mission.step(dt)
        if writer is None:
            continue
        while next_frame <= world.time:
            main_renderer.update_scene(world.data, camera=args.view)
            overlay_path(main_renderer.scene, mission.planned_path)
            overlay_soil_points(main_renderer.scene,
                                [s_.xy for s_ in mission.soil_samples])
            st = HudState(
                phase=mission.phase.value,
                phase_label=PHASE_LABEL[mission.phase],
                sim_time=world.time,
                distance=world.distance_travelled,
                cut=mission.result.cut,
                cut_total=mission.result.cut_total,
                map_coverage=mission.result.map_coverage,
                message=mission.message,
                message2=mission.message2,
                onboard_rgb=None if perception is None else perception.last_rgb,
                seg_prob=None if perception is None else perception.last_prob,
                grid_prob=mission.gmap.prob(),
                planned_path=mission.planned_path,
                travelled=np.array(mission.trajectory),
                robot_xy=world.pose()[0],
                robot_yaw=world.pose()[1],
                soil_points=[s.xy for s in mission.soil_samples],
                report_lines=build_report_lines(mission),
            )
            writer.append_data(hud.compose(main_renderer.render(), st))
            next_frame += 1.0 / args.fps
            n_frames += 1

    r = mission.result
    if writer is not None:
        for frame, hold in summary_frames(hud, mission):
            for _ in range(int(hold * args.fps)):
                writer.append_data(frame)
                n_frames += 1
        writer.close()
        print(f"動画: {args.out}  ({n_frames} frames, "
              f"{n_frames / args.fps:.1f} s, 実時間 {time.time() - t0:.0f} s)")

    print("-" * 68)
    print(f"所要時間        {r.sim_time:7.1f} s")
    print(f"走行距離        {r.distance:7.2f} m  "
          f"(探査 {r.survey_distance:.2f} + 刈取 {r.mow_distance:.2f})")
    print(f"刈り取り        {r.cut}/{r.cut_total}  ({100 * r.cut_rate:.1f} %)")
    print(f"刈取作業終了    {r.mow_end_time:7.1f} s / {r.mow_end_distance:.2f} m")
    print(f"地図被覆率      {100 * r.map_coverage:.1f} %")
    print(f"検出クラスタ数  {r.n_clusters}   巡回数 {r.rounds}   "
          f"CNN 推論 {r.n_inferences} 回")
    if r.recommendations:
        print("土壌 (中央値):", {k: round(v, 2) for k, v in r.soil.items()})
        for rec in r.recommendations:
            print(f"  {rec.crop.name:12s} {rec.probability * 100:5.1f} %")
        for a in r.advice:
            print(f"  - {a}")

    metrics = {
        "mode": r.mode, "seed": args.seed, "sim_time": r.sim_time,
        "distance": r.distance, "survey_distance": r.survey_distance,
        "mow_distance": r.mow_distance, "cut": r.cut, "cut_total": r.cut_total,
        "cut_rate": r.cut_rate, "map_coverage": r.map_coverage,
        "mow_end_time": r.mow_end_time,
        "mow_end_distance": r.mow_end_distance,
        "full_cut_time": r.full_cut_time,
        "n_clusters": r.n_clusters, "rounds": r.rounds,
        "n_inferences": r.n_inferences, "planned_length": r.planned_length,
        "perception_iou": r.perception_iou,
        "soil": r.soil,
        "recommendations": [[x.crop.name, x.probability] for x in r.recommendations],
        "advice": r.advice,
    }
    mp = pathlib.Path(args.out).with_suffix(".json")
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"指標: {mp}")

    if perception is not None:
        perception.close()
    if main_renderer is not None:
        main_renderer.close()


def title_frames(hud: HudRenderer, cfg: Config, field) -> list[tuple[np.ndarray, float]]:
    return [
        (title_card(hud, [
            (58, "AgriPioneer", ACCENT),
            (34, "耕作放棄地を自動で開拓する草刈りロボット", FG),
            (22, "", MUTED),
            (24, "課題:  農業の後継者不足と, 人口減少による土地余り", MUTED),
            (24, "第一歩:  1 坪 (1.82 m 角) に無造作に生えた草を, 最適経路で刈り切る", MUTED),
            (22, "", MUTED),
            (24, "① 単眼カメラ + U-Net で草の占有格子地図を作る", GREEN),
            (24, "② 草むらをクラスタ化し, TSP で巡回順を最適化して刈り取る", GREEN),
            (24, "③ 土壌を計測し, 学習済みモデルで最適な作付を提案する", GREEN),
            (22, "", MUTED),
            (22, "MuJoCo 3.17 / PyTorch / scikit-learn   —   動画は等速 (1x) 再生", MUTED),
        ]), 6.0),
    ]


def summary_frames(hud: HudRenderer, mission: Mission) -> list[tuple[np.ndarray, float]]:
    r = mission.result
    lines: list[tuple[int, str, tuple[int, int, int]]] = [
        (52, "ミッション完了", ACCENT),
        (24, "", MUTED),
        (28, f"刈り取り率      {100 * r.cut_rate:.1f} %   ({r.cut}/{r.cut_total} 本)", FG),
        (28, f"所要時間        {r.sim_time:.1f} s", FG),
        (28, f"走行距離        {r.distance:.2f} m  "
             f"(探査 {r.survey_distance:.2f} m + 刈取 {r.mow_distance:.2f} m)", FG),
        (28, f"検出した草むら  {r.n_clusters} 箇所   CNN 推論 {r.n_inferences} 回", FG),
        (24, "", MUTED),
    ]
    if r.recommendations:
        v = r.soil
        lines.append((26, f"土壌: pH {v['pH']:.1f} / EC {v['EC']:.2f} dS/m / "
                          f"有機物 {v['OM']:.1f}% / 含水率 {v['moisture']:.0f}%", MUTED))
        lines.append((24, "", MUTED))
        lines.append((30, "土壌診断 AI による作付提案", GREEN))
        for i, rec in enumerate(r.recommendations, 1):
            lines.append((28, f"  {i}. {rec.crop.name}   {rec.probability * 100:.1f} %", ACCENT))
        lines.append((24, "", MUTED))
        for a in r.advice[:2]:
            lines.append((24, f"  ・{a}", MUTED))
    return [(title_card(hud, lines), 7.0)]


if __name__ == "__main__":
    main()
