"""被覆経路計画 (Coverage Path Planning).

1. 探査フェーズ: カメラ視野幅に合わせた牛耕式 (boustrophedon) 経路で区画を一巡し,
   草の占有格子地図を作る.
2. 刈り取りフェーズ: 地図上の草を連結成分でクラスタ化し, 各クラスタの主軸
   (PCA) に沿った刈刃幅の牛耕式経路を生成する. クラスタの訪問順と各経路の
   向きは `planning.tsp` で最適化する.

全面を舐める素朴な被覆に比べ, 草のない領域を走らない分だけ走行距離が短くなる.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..config import Config
from ..perception.mapper import GrassMap
from . import tsp


def _serpentine(lanes: list[np.ndarray]) -> np.ndarray:
    """レーン端点列を往復接続して 1 本のポリラインにする."""
    pts: list[np.ndarray] = []
    for k, (a, b) in enumerate(lanes):
        if k % 2 == 0:
            pts.extend([a, b])
        else:
            pts.extend([b, a])
    return np.array(pts) if pts else np.zeros((0, 2))


# ------------------------------------------------------------------ 探査


def survey_path(cfg: Config) -> np.ndarray:
    """カメラで区画全体を観測するための牛耕式探査経路."""
    h = cfg.plot.half
    # カメラ地上視野の実効幅 (map_range の半分あたりでの横幅) を目安にする
    fov = np.deg2rad(cfg.robot.cam_fovy_deg)
    swath = 2.0 * (0.55 * cfg.robot.map_range) * np.tan(fov / 2.0)
    span = h - 0.12
    n_lanes = max(2, int(np.ceil(2 * span / swath)) + 1)
    ys = np.linspace(-span, span, n_lanes)
    lanes = [(np.array([-span, y]), np.array([span, y])) for y in ys]
    return _serpentine(lanes)


def full_coverage_path(cfg: Config) -> np.ndarray:
    """比較用ベースライン: 地図を使わず区画全面を刈刃幅で舐める経路."""
    h = cfg.plot.half
    swath = cfg.robot.blade_width * 0.88
    span = h - 0.02
    n_lanes = int(np.ceil(2 * span / swath)) + 1
    ys = np.linspace(-span, span, n_lanes)
    lanes = [(np.array([-span, y]), np.array([span, y])) for y in ys]
    return _serpentine(lanes)


# ------------------------------------------------------------ クラスタ分解


@dataclass
class GrassCluster:
    """ひとまとまりの草むらと, それを覆う刈り取り経路."""

    cells_xy: np.ndarray  # (M, 2) 草と判定されたセル中心
    path: np.ndarray  # (P, 2) 刈り取り経路
    area: float  # 推定面積 [m^2]

    @property
    def ends(self) -> np.ndarray:
        return np.array([self.path[0], self.path[-1]])

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())


def _cluster_path(cells_xy: np.ndarray, cfg: Config) -> np.ndarray:
    """クラスタを刈刃幅で覆う経路を主軸方向に沿って生成する."""
    r = cfg.robot.blade_width / 2.0
    swath = cfg.robot.blade_width * 0.85
    c = cells_xy.mean(axis=0)
    q = cells_xy - c

    if len(cells_xy) >= 3:
        # PCA: 分布の長軸方向にレーンを引くとレーン本数が最小になる
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        u_ax, v_ax = vt[0], vt[1]
    else:
        u_ax, v_ax = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    u = q @ u_ax
    v = q @ v_ax

    v_lo, v_hi = v.min(), v.max()
    width = v_hi - v_lo
    n_lanes = max(1, int(np.ceil((width + 2 * r * 0.4) / swath)))
    if n_lanes == 1:
        v_centers = np.array([(v_lo + v_hi) / 2.0])
    else:
        v_centers = np.linspace(v_lo, v_hi, n_lanes)

    lanes = []
    band = max(swath / 2.0, cfg.map.resolution)
    for vc in v_centers:
        sel = np.abs(v - vc) <= band
        if not sel.any():
            continue
        u0, u1 = u[sel].min() - r * 0.55, u[sel].max() + r * 0.55
        a = c + u_ax * u0 + v_ax * vc
        b = c + u_ax * u1 + v_ax * vc
        lanes.append((a, b))

    if not lanes:
        return cells_xy.mean(axis=0, keepdims=True)
    return _serpentine(lanes)


def extract_clusters(
    gmap: GrassMap, cfg: Config, thr: float = 0.6, min_cells: int = 3
) -> list[GrassCluster]:
    """占有格子から草むらを連結成分として切り出し, 各々の刈り取り経路を作る."""
    occ = gmap.prob() > thr
    if not occ.any():
        return []

    # 刈刃幅ぶん膨張させてから連結成分を取る (近接する草むらは 1 回の走行で
    # まとめて刈れるため, 同じクラスタとして扱ったほうが効率が良い)
    rad = max(1, int(round((cfg.robot.blade_width * 0.40) / cfg.map.resolution)))
    yy, xx = np.mgrid[-rad : rad + 1, -rad : rad + 1]
    se = (xx**2 + yy**2) <= rad * rad
    merged = ndimage.binary_dilation(occ, structure=se)

    labels, n = ndimage.label(merged)
    centers = gmap.cell_centers
    out: list[GrassCluster] = []
    cell_area = cfg.map.resolution**2
    for k in range(1, n + 1):
        sel = occ & (labels == k)
        m = int(sel.sum())
        if m < min_cells:
            continue
        cells_xy = centers[sel]
        out.append(
            GrassCluster(cells_xy=cells_xy, path=_cluster_path(cells_xy, cfg),
                         area=m * cell_area)
        )
    return out


# ------------------------------------------------------------ 刈り取り計画


@dataclass
class MowPlan:
    path: np.ndarray
    clusters: list[GrassCluster]
    order: np.ndarray
    flip: np.ndarray
    transit_length: float
    total_length: float


def plan_mowing(
    gmap: GrassMap,
    cfg: Config,
    start_xy: np.ndarray,
    thr: float = 0.6,
    seed: int = 0,
    min_cells: int = 3,
) -> MowPlan:
    """地図から刈り取り経路全体を組み立てる."""
    clusters = extract_clusters(gmap, cfg, thr=thr, min_cells=min_cells)
    if not clusters:
        return MowPlan(np.zeros((0, 2)), [], np.array([], int), np.array([], bool), 0.0, 0.0)

    ends = np.stack([c.ends for c in clusters])  # (K, 2, 2)
    order, flip, transit = tsp.solve(ends, np.asarray(start_xy, dtype=float), seed=seed)

    pts = [np.asarray(start_xy, dtype=float)]
    for k, f in zip(order, flip):
        p = clusters[k].path
        pts.extend(list(p[::-1] if f else p))
    path = np.array(pts)

    lim = cfg.plot.half + 0.22
    path = np.clip(path, -lim, lim)
    total = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return MowPlan(path, clusters, order, flip, transit, total)
