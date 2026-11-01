"""Pure Pursuit による経路追従.

計画経路は折れ線 (レーン端点の列) で与えられるため, そのまま追従すると
端点付近で前方注視距離が縮んで曲率が発散し, ロボットが失速して経路点の
周りを旋回してしまう. これを避けるため, 追従前に経路を一定間隔で密化し,
「経路上を arc length で前方注視する」本来の Pure Pursuit として実装する.

差動二輪は旋回半径に下限がないので, 方位誤差が大きいときはその場旋回
(ゼロターン), 小さいときは曲率追従に切り替えるハイブリッド制御とする.
車輪のスリップは姿勢フィードバックで吸収される.
"""

from __future__ import annotations

import numpy as np

from ..config import Config


def densify(path: np.ndarray, spacing: float = 0.05) -> np.ndarray:
    """折れ線を等間隔の点列に分割する."""
    p = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        return p
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        d = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(d / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.array(out)


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class PurePursuit:
    #: 最近傍点を前方探索する窓幅 [点数]. 折り返し経路へ飛び移るのを防ぐ.
    SEARCH_WINDOW = 14

    def __init__(self, cfg: Config, path: np.ndarray, spacing: float = 0.05) -> None:
        self.cfg = cfg
        self.waypoints = np.asarray(path, dtype=float).reshape(-1, 2)
        self.path = densify(self.waypoints, spacing)
        self.idx = 0
        self.finished = self.path.shape[0] == 0
        self._t = 0.0

    # ------------------------------------------------------------------
    @property
    def target(self) -> np.ndarray:
        return self.path[min(self.idx, len(self.path) - 1)]

    def progress(self) -> float:
        n = len(self.path)
        return 0.0 if n == 0 else min(1.0, self.idx / max(n - 1, 1))

    def remaining_length(self) -> float:
        if self.idx >= len(self.path) - 1:
            return 0.0
        seg = np.diff(self.path[self.idx :], axis=0)
        return float(np.linalg.norm(seg, axis=1).sum())

    # ------------------------------------------------------------------
    def step(self, xy: np.ndarray, yaw: float, dt: float) -> tuple[float, float]:
        """現在姿勢から (並進速度 [m/s], 角速度 [rad/s]) を返す."""
        c = self.cfg.control
        r = self.cfg.robot
        if self.finished:
            return 0.0, 0.0

        n = len(self.path)
        self._t += dt

        # --- 経路上の最近傍点を前方のみ探索して進捗を更新する -------------
        hi = min(n, self.idx + self.SEARCH_WINDOW)
        seg = self.path[self.idx : hi]
        k = int(np.argmin(np.linalg.norm(seg - xy, axis=1)))
        self.idx += k
        if k > 0:
            self._t = 0.0

        # 終端判定
        if self.idx >= n - 1 and np.linalg.norm(self.path[-1] - xy) < c.goal_tol:
            self.finished = True
            return 0.0, 0.0
        # 同じ場所に張り付いた場合は打ち切る (デッドロック回避)
        if self._t > c.waypoint_timeout:
            self.finished = True
            return 0.0, 0.0

        # --- 前方注視点: 経路に沿って lookahead だけ先の点 ----------------
        j = self.idx
        while j < n - 1 and np.linalg.norm(self.path[j] - xy) < c.lookahead:
            j += 1
        look = self.path[j]

        d = look - xy
        dist = float(np.linalg.norm(d))
        alpha = _wrap(np.arctan2(d[1], d[0]) - yaw)

        if abs(alpha) > c.spin_in_place:
            # 大きく向きがずれているときはゼロターン (その場旋回)
            w = np.clip(c.k_ang * alpha, -r.max_ang_vel, r.max_ang_vel)
            return 0.0, float(np.sign(w) * max(abs(w), 0.8))

        # Pure Pursuit の曲率 kappa = 2 sin(alpha) / L
        L = max(dist, c.lookahead_min)
        kappa = 2.0 * np.sin(alpha) / L
        # 方位がずれているときだけ緩やかに減速する
        v = r.max_lin_vel * float(np.clip(np.cos(alpha), 0.35, 1.0))
        v = v / (1.0 + 0.25 * abs(kappa))
        # 経路終端が近づいたら減速して行き過ぎを防ぐ
        v = min(v, 0.22 + 1.1 * self.remaining_length())
        v = float(np.clip(v, 0.08, r.max_lin_vel))
        w = float(np.clip(v * kappa + 1.1 * alpha, -r.max_ang_vel, r.max_ang_vel))
        return v, w
