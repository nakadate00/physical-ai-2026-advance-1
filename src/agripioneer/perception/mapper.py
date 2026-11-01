"""CNN のセグメンテーション結果を地面に逆投影し, 草の占有格子地図を作る.

各画素の確率をそのまま逆センサモデル (log odds) として扱い, カメラからの
距離に応じた重みを掛けて地図へ積算する. 遠方の画素は逆投影誤差が大きいため
重みを下げ, `map_range` より遠い画素は捨てる.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .camera import PinholeCamera

L_CLIP = 6.0
#: 1 フレームあたりの更新ゲイン (単独フレームで地図が飽和しないよう抑える)
UPDATE_GAIN = 0.38


class GrassMap:
    """区画を覆う 2 次元の log odds 占有格子."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.res = cfg.map.resolution
        self.n, _ = cfg.map.grid_shape(cfg.plot)
        self.origin = -cfg.plot.half  # 格子 (0,0) の左下 xy
        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)
        self.hits = np.zeros((self.n, self.n), dtype=np.int32)

    # ------------------------------------------------------------ 座標変換
    def xy_to_ij(self, xy: np.ndarray) -> np.ndarray:
        """(..., 2) の xy を格子添字 (..., 2)=(行 i=y, 列 j=x) に変換."""
        idx = np.floor((np.asarray(xy) - self.origin) / self.res).astype(np.int32)
        return idx[..., ::-1]

    def ij_to_xy(self, ij: np.ndarray) -> np.ndarray:
        ij = np.asarray(ij)
        x = self.origin + (ij[..., 1] + 0.5) * self.res
        y = self.origin + (ij[..., 0] + 0.5) * self.res
        return np.stack([x, y], axis=-1)

    @property
    def cell_centers(self) -> np.ndarray:
        g = self.origin + (np.arange(self.n) + 0.5) * self.res
        X, Y = np.meshgrid(g, g, indexing="xy")
        return np.stack([X, Y], axis=-1)  # (n, n, 2) 行=y, 列=x

    # ------------------------------------------------------------ 更新
    def integrate(
        self,
        prob: np.ndarray,
        cam_pos: np.ndarray,
        cam_mat: np.ndarray,
        cam: PinholeCamera,
    ) -> int:
        """確率マップ (H, W) を地図に積算する. 更新したセル数を返す."""
        pts, dist = cam.ground_points(cam_pos, cam_mat)
        rng_max = self.cfg.robot.map_range

        ok = np.isfinite(pts[..., 0]) & (dist < rng_max)
        if not ok.any():
            return 0

        xy = pts[ok]
        ij = self.xy_to_ij(xy)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        if not inside.any():
            return 0
        ij = ij[inside]

        p = np.clip(prob[ok][inside].astype(np.float32), 0.02, 0.98)
        d = dist[ok][inside]
        # 近傍ほど信頼する重み (遠方は逆投影誤差が大きい)
        w = np.clip(1.0 - (d / rng_max) ** 2, 0.0, 1.0).astype(np.float32)

        delta = UPDATE_GAIN * w * np.log(p / (1.0 - p))
        flat = ij[:, 0] * self.n + ij[:, 1]
        acc = np.zeros(self.n * self.n, dtype=np.float32)
        cnt = np.zeros(self.n * self.n, dtype=np.int32)
        np.add.at(acc, flat, delta)
        np.add.at(cnt, flat, 1)

        touched = cnt > 0
        # 1 セルに多数の画素が落ちるため平均を取り, 過剰更新を避ける
        acc[touched] /= cnt[touched]
        self.logodds += acc.reshape(self.n, self.n)
        np.clip(self.logodds, -L_CLIP, L_CLIP, out=self.logodds)
        self.hits += (cnt > 0).reshape(self.n, self.n).astype(np.int32)
        return int(touched.sum())

    def mark_mowed(self, xy_from: np.ndarray, xy_to: np.ndarray) -> None:
        """刈刃が通過したセルを「草なし」に強制更新する (作業記録)."""
        r = self.cfg.robot.blade_width / 2.0
        seg = np.asarray(xy_to) - np.asarray(xy_from)
        c = self.cell_centers.reshape(-1, 2)
        rel = c - np.asarray(xy_from)
        L2 = float(seg @ seg)
        if L2 < 1e-9:
            dist = np.linalg.norm(rel, axis=1)
        else:
            t = np.clip((rel @ seg) / L2, 0.0, 1.0)
            dist = np.linalg.norm(rel - t[:, None] * seg[None, :], axis=1)
        hit = (dist <= r).reshape(self.n, self.n)
        self.logodds[hit] = np.minimum(self.logodds[hit], -2.5)

    # ------------------------------------------------------------ 取り出し
    def prob(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logodds))

    def observed(self) -> np.ndarray:
        return self.hits > 0

    def grass_cells(self, thr: float = 0.6) -> np.ndarray:
        """草ありと判定されたセルの (i, j) 一覧."""
        return np.argwhere(self.prob() > thr)

    def coverage_ratio(self) -> float:
        return float(self.observed().mean())
