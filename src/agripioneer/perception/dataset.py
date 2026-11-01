"""知覚 CNN 用の合成データセットを MuJoCo から自動生成する.

教師ラベルは「カメラ画像の各画素が, 刈り取るべき草の *株元* (地面上の占有領域)
に対応するか」である. 草の立体シルエットではなく地面上のフットプリントを
ラベルにすることで, ネットワークは「見えている草が地面のどこに生えているか」
という遠近を含む推定を学習し, 出力をそのまま占有格子地図へ逆投影できる.

ラベルは草の株元の 3 次元座標をカメラへ順投影して解析的に生成するため,
手作業のアノテーションは一切不要である (ドメインランダマイゼーション付き).
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from ..config import Config
from ..world import World, grass_rgba, sample_grass
from .camera import PinholeCamera

#: 草 1 株が地面を占める半径 [m]
FOOTPRINT_RADIUS = 0.045


def footprint_label(
    world: World, cam: PinholeCamera, cfg: Config
) -> np.ndarray:
    """刈り取り対象の草の株元を画像平面に描画した 2 値ラベルを作る."""
    f = world.field
    tgt = np.flatnonzero((f.height > cfg.grass.cut_threshold) & (~f.cut))
    mask = np.zeros((cam.h, cam.w), dtype=np.uint8)
    if tgt.size == 0:
        return mask

    cam_pos, cam_mat = world.camera_pose()
    base = np.column_stack([f.xy[tgt], np.zeros(tgt.size)])
    uv, depth, vis = cam.project(base, cam_pos, cam_mat)

    # 地面上の半径 r が画像上で占める半径 [px]. 俯角ぶん縦方向は潰れるが,
    # 占有格子の分解能 (3 cm) に対し十分なので等方近似で描く.
    with np.errstate(divide="ignore", invalid="ignore"):
        rad_px = cam.f * FOOTPRINT_RADIUS / np.maximum(depth, 1e-3)

    yy, xx = np.mgrid[0 : cam.h, 0 : cam.w]
    for k in range(tgt.size):
        if not vis[k] or not np.isfinite(rad_px[k]):
            continue
        r = float(np.clip(rad_px[k], 0.8, cam.w))
        u, v = uv[k]
        u0, u1 = int(max(0, u - r - 1)), int(min(cam.w, u + r + 2))
        v0, v1 = int(max(0, v - r - 1)), int(min(cam.h, v + r + 2))
        if u0 >= u1 or v0 >= v1:
            continue
        sub_x = xx[v0:v1, u0:u1] - u
        sub_y = yy[v0:v1, u0:u1] - v
        mask[v0:v1, u0:u1] |= ((sub_x**2 + sub_y**2) <= r * r).astype(np.uint8)
    return mask


def _randomize_appearance(world: World, rng: np.random.Generator) -> None:
    """照明・土色・草色をランダム化し, 見た目の変化に頑健にする."""
    m = world.model
    # 太陽光の方向と強さ
    m.light_pos[0] = rng.uniform([-2.5, -2.5, 2.0], [2.5, 2.5, 4.0])
    d = rng.uniform(0.45, 0.95)
    m.light_diffuse[0] = (d, d, d * rng.uniform(0.92, 1.0))
    amb = rng.uniform(0.28, 0.55)
    m.vis.headlight.ambient = (amb, amb, amb)
    # 土の色味 (明るさを主に変え, 色相のずれは控えめにする)
    tint = rng.uniform(0.72, 1.28) * rng.uniform(0.93, 1.07, size=3)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "soil")
    m.mat_rgba[sid, :3] = np.clip(tint, 0.3, 1.4)
    # 草の色味 (株ごとの色に全体的な色相ずれを掛ける)
    gtint = rng.uniform(0.80, 1.20, size=3)
    gid = world.grass_gid
    m.geom_rgba[gid, :3] = np.clip(m.geom_rgba[gid, :3] * gtint[None, :], 0.0, 1.0)


def _random_grass_cfg(cfg: Config, rng: np.random.Generator) -> Config:
    g = replace(
        cfg.grass,
        n_clusters=int(rng.integers(3, 11)),
        cluster_sigma=float(rng.uniform(0.10, 0.26)),
        blades_per_cluster=int(rng.integers(12, 34)),
        n_scattered=int(rng.integers(8, 55)),
        height_max=float(rng.uniform(0.18, 0.34)),
    )
    return replace(cfg, grass=g)


def generate(
    cfg: Config,
    n_worlds: int,
    poses_per_world: int,
    res: int,
    seed: int,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """(rgb uint8 [N,res,res,3], mask uint8 [N,res,res]) を返す."""
    rng = np.random.default_rng(seed)
    cam = PinholeCamera(res, res, cfg.robot.cam_fovy_deg)
    lim = cfg.plot.half + 0.15

    n = n_worlds * poses_per_world
    rgbs = np.zeros((n, res, res, 3), dtype=np.uint8)
    masks = np.zeros((n, res, res), dtype=np.uint8)

    k = 0
    for wi in range(n_worlds):
        wcfg = _random_grass_cfg(cfg, rng)
        field = sample_grass(wcfg, rng)
        world = World(wcfg, field, (0.0, 0.0), 0.0)

        # 刈り取り途中の状態も学習させる (刈り株を「草」と誤検出しないように)
        frac = float(rng.choice([0.0, 0.0, 0.15, 0.35, 0.6, 0.85]))
        if frac > 0:
            n_cut = int(frac * field.n)
            if rng.random() < 0.5:  # 帯状に刈った跡
                order = np.argsort(field.xy[:, rng.integers(0, 2)])
                world.cut_indices(order[:n_cut])
            else:  # 無作為に刈った跡
                world.cut_indices(rng.permutation(field.n)[:n_cut])

        _randomize_appearance(world, rng)
        renderer = mujoco.Renderer(world.model, res, res)

        for _ in range(poses_per_world):
            if rng.random() < 0.75:
                # 区画内に置き, 区画内の適当な点を向かせる (実運用に近い構図)
                xy = rng.uniform(-cfg.plot.half, cfg.plot.half, size=2)
                look = rng.uniform(-cfg.plot.half, cfg.plot.half, size=2)
                dxy = look - xy
                yaw = float(np.arctan2(dxy[1], dxy[0]) + rng.normal(0, 0.25))
            else:
                xy = rng.uniform(-lim, lim, size=2)
                yaw = rng.uniform(-np.pi, np.pi)
            world.set_pose(tuple(xy), yaw,
                           roll=np.deg2rad(rng.normal(0, 1.5)),
                           pitch=np.deg2rad(rng.normal(0, 2.5)))
            renderer.update_scene(world.data, camera="onboard")
            rgbs[k] = renderer.render()
            masks[k] = footprint_label(world, cam, wcfg)
            k += 1

        renderer.close()
        if progress and (wi + 1) % 10 == 0:
            print(f"  world {wi + 1}/{n_worlds}  samples={k}", flush=True)

    return rgbs, masks
