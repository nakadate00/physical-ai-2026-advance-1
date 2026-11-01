"""単眼カメラの幾何計算.

MuJoCo のカメラは「-z 方向を向き, +y が画像上方向, +x が画像右方向」である.
地面を平面 z = 0 と仮定して画素と地面の対応を取る (平面ホモグラフィ相当).
"""

from __future__ import annotations

import numpy as np


class PinholeCamera:
    """MuJoCo カメラの内部/外部パラメータを扱う."""

    def __init__(self, height: int, width: int, fovy_deg: float) -> None:
        self.h = int(height)
        self.w = int(width)
        self.fovy = float(fovy_deg)
        # 垂直画角から焦点距離 [px] を求める (MuJoCo は fovy 基準)
        self.f = 0.5 * self.h / np.tan(np.deg2rad(self.fovy) / 2.0)

        v, u = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing="ij")
        # カメラ座標系での画素方向ベクトル (正規化前)
        self.dirs_cam = np.stack(
            [
                (u + 0.5 - self.w / 2.0) / self.f,
                -(v + 0.5 - self.h / 2.0) / self.f,
                -np.ones_like(u, dtype=float),
            ],
            axis=-1,
        )  # (H, W, 3)

    # ------------------------------------------------------------------
    def ground_points(
        self, cam_pos: np.ndarray, cam_mat: np.ndarray, z_plane: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """各画素が地面 z=z_plane と交わる点とカメラからの水平距離を返す.

        Returns:
            pts: (H, W, 2) 交点の xy. 交わらない画素は NaN.
            dist: (H, W) カメラ直下からの水平距離. 交わらない画素は inf.
        """
        d_world = self.dirs_cam @ cam_mat.T  # (H, W, 3)
        dz = d_world[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z_plane - cam_pos[2]) / dz
        valid = (dz < -1e-6) & (t > 0)

        pts = cam_pos[None, None, :2] + t[..., None] * d_world[..., :2]
        pts = np.where(valid[..., None], pts, np.nan)
        dist = np.where(valid, np.linalg.norm(pts - cam_pos[None, None, :2], axis=-1), np.inf)
        return pts, dist

    # ------------------------------------------------------------------
    def project(
        self, pts_world: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """世界座標の点群を画素座標へ投影する.

        Returns:
            uv: (N, 2) 画素座標 (u, v)
            depth: (N,) カメラ光軸方向の奥行き [m]
            visible: (N,) 画面内かつカメラ前方なら True
        """
        rel = np.asarray(pts_world, dtype=float) - cam_pos[None, :]
        p_cam = rel @ cam_mat  # R^T @ rel (cam_mat は列が各軸)
        depth = -p_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.w / 2.0 + self.f * p_cam[:, 0] / depth - 0.5
            v = self.h / 2.0 - self.f * p_cam[:, 1] / depth - 0.5
        uv = np.stack([u, v], axis=-1)
        visible = (depth > 1e-3) & (u > -8) & (u < self.w + 8) & (v > -8) & (v < self.h + 8)
        return uv, depth, visible
