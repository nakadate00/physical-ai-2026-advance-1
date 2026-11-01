"""AgriPioneer 全体の設定値.

1 坪 = 400/121 m^2 ~= 3.3058 m^2 であり, 正方形に取ると一辺 1.818 m となる.
以降の寸法・制御パラメータはすべてこの区画を基準に定義する.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 1 坪の一辺 [m]
TSUBO_SIDE = 1.818


@dataclass
class PlotConfig:
    """開拓対象となる区画."""

    side: float = TSUBO_SIDE
    #: 区画外周に確保する作業マージン [m]. ロボットの旋回半径を吸収する.
    margin: float = 0.28

    @property
    def half(self) -> float:
        return self.side / 2.0

    @property
    def area(self) -> float:
        return self.side**2


@dataclass
class GrassConfig:
    """草の生成条件."""

    n_clusters: int = 5
    cluster_sigma: float = 0.095
    #: 群生の中心どうしの最小間隔 [m]
    min_cluster_gap: float = 0.60
    blades_per_cluster: int = 46
    n_scattered: int = 6
    height_min: float = 0.07
    height_max: float = 0.32
    radius_min: float = 0.0035
    radius_max: float = 0.008
    #: 刈り取り後に残る刈り株の高さ [m]
    stubble_height: float = 0.018
    #: この高さ以下の草は「刈る必要なし」と見なす [m]
    cut_threshold: float = 0.05


@dataclass
class RobotConfig:
    """差動二輪の自走式草刈りロボット."""

    chassis_len: float = 0.30
    chassis_wid: float = 0.24
    chassis_hgt: float = 0.08
    wheel_radius: float = 0.062
    #: 左右輪間距離 (トレッド) [m]
    wheel_base: float = 0.26
    #: 刈刃の直径 [m]
    blade_width: float = 0.22
    #: 刈刃中心のシャシ前方オフセット [m]
    blade_offset: float = 0.085
    max_lin_vel: float = 0.42
    max_ang_vel: float = 2.2
    max_lin_acc: float = 0.55
    max_ang_acc: float = 3.5
    #: 単眼カメラの取り付け高さ [m]
    cam_height: float = 0.52
    #: カメラの俯角 [deg] (0 = 水平)
    cam_pitch_deg: float = 42.0
    cam_fovy_deg: float = 60.0
    cam_res: int = 96
    #: 知覚結果を地図に反映する最大距離 [m] (遠方ほど逆投影誤差が大きい)
    map_range: float = 1.25


@dataclass
class MapConfig:
    """草占有格子地図."""

    resolution: float = 0.03

    def grid_shape(self, plot: PlotConfig) -> tuple[int, int]:
        n = int(round(plot.side / self.resolution))
        return n, n


@dataclass
class ControlConfig:
    """Pure Pursuit 追従制御."""

    lookahead: float = 0.17
    lookahead_min: float = 0.10
    goal_tol: float = 0.07
    #: その場旋回に切り替える方位誤差 [rad]
    spin_in_place: float = 0.85
    k_ang: float = 2.6
    #: 到達できない経路点を打ち切るまでの時間 [s]
    waypoint_timeout: float = 14.0


@dataclass
class SimConfig:
    timestep: float = 0.004
    #: 制御・知覚の更新周期 [s]
    control_dt: float = 0.05
    #: 知覚 (CNN 推論) の実行周期 [s]
    perception_dt: float = 0.20
    #: 動画のフレームレート [fps]
    video_fps: int = 30
    render_width: int = 960
    render_height: int = 540


@dataclass
class Config:
    plot: PlotConfig = field(default_factory=PlotConfig)
    grass: GrassConfig = field(default_factory=GrassConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    map: MapConfig = field(default_factory=MapConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    seed: int = 20261102
