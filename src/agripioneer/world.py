"""MuJoCo 上に「1 坪の草地 + 自走式草刈りロボット」の世界を構築する.

草は worldbody 直下の静的 geom として配置し, 衝突は無効化している
(ロボットは草をなぎ倒しながら進むため, 物理的な干渉はモデル化しない).
刈り取りは `model.geom_size` / `model.geom_pos` / `model.geom_rgba` を
実行時に書き換えることで表現する. こうすることで数百本の草を扱っても
シミュレーション速度がほとんど落ちない.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import Config

# ---------------------------------------------------------------- 草地の生成


@dataclass
class GrassField:
    """区画内に生えている草の集合."""

    xy: np.ndarray  # (N, 2) 位置 [m]
    height: np.ndarray  # (N,) 現在の草丈 [m]
    height0: np.ndarray  # (N,) 初期の草丈 [m]
    radius: np.ndarray  # (N,) 茎の半径 [m]
    cut: np.ndarray  # (N,) 刈り取り済みフラグ
    tilt: np.ndarray  # (N, 2) 茎の傾き (roll, pitch) [deg]

    @property
    def n(self) -> int:
        return int(self.xy.shape[0])

    @property
    def n_target(self) -> int:
        """刈り取り対象 (初期時点で閾値より高い) の本数."""
        return int(self._target_mask.sum())

    @property
    def n_cut(self) -> int:
        return int((self.cut & self._target_mask).sum())

    @property
    def _target_mask(self) -> np.ndarray:
        return self._tgt

    def set_target_mask(self, mask: np.ndarray) -> None:
        self._tgt = mask


def sample_grass(cfg: Config, rng: np.random.Generator) -> GrassField:
    """クラスタ状 + 散在の草をランダムに生やす.

    現実の耕作放棄地では草は一様分布ではなく群生する. 群生 (草むら) の中心は
    互いに一定以上離れるようサンプリングし (Poisson-disk 風の棄却法),
    群生ごとに広がりと密度を変えることで, 大小さまざまな草むらが点在する
    状況を作る. 一様分布にしてしまうと「全面を舐める」以外に手がなくなり,
    知覚にもとづく経路計画の意味がなくなるため.
    """
    g = cfg.grass
    lim = cfg.plot.half - 0.04

    # --- 群生の中心を最小間隔つきでサンプリング ---------------------------
    centers: list[np.ndarray] = []
    c_lim = cfg.plot.half - 0.26
    for _ in range(30000):
        if len(centers) >= g.n_clusters:
            break
        c = rng.uniform(-c_lim, c_lim, size=2)
        if all(np.linalg.norm(c - o) >= g.min_cluster_gap for o in centers):
            centers.append(c)
    centers_arr = np.array(centers) if centers else rng.uniform(
        -c_lim, c_lim, size=(1, 2))

    # --- 群生ごとに広がりと本数を変える -----------------------------------
    pts: list[np.ndarray] = []
    sigmas = rng.uniform(g.cluster_sigma * 0.65, g.cluster_sigma * 1.6,
                         size=len(centers_arr))
    for c, sg in zip(centers_arr, sigmas):
        # 面積に比例した本数にすると, 大きい草むらほど濃く見える
        n_b = max(6, int(g.blades_per_cluster * (sg / g.cluster_sigma) ** 1.4))
        pts.append(c + rng.normal(0.0, sg, size=(n_b, 2)))
    if g.n_scattered > 0:
        pts.append(rng.uniform(-lim, lim, size=(g.n_scattered, 2)))

    xy = np.clip(np.concatenate(pts, axis=0), -lim, lim)

    # 群生の中心に近いほど背が高くなるようにして, 自然な草むらに見せる
    d = np.min(np.linalg.norm(xy[:, None, :] - centers_arr[None, :, :], axis=-1), axis=1)
    lushness = np.clip(1.0 - d / (3.0 * g.cluster_sigma), 0.0, 1.0)
    base = rng.uniform(g.height_min, g.height_max, size=xy.shape[0])
    height = np.clip(base * (0.55 + 0.75 * lushness), 0.02, g.height_max)
    radius = rng.uniform(g.radius_min, g.radius_max, size=xy.shape[0])

    field = GrassField(
        xy=xy,
        height=height.copy(),
        height0=height.copy(),
        radius=radius,
        cut=np.zeros(xy.shape[0], dtype=bool),
        tilt=rng.normal(0.0, 7.0, size=(xy.shape[0], 2)),
    )
    field.set_target_mask(height > g.cut_threshold)
    return field


def grass_rgba(height: float, cfg: Config) -> tuple[float, float, float, float]:
    """草丈に応じた色. 背が高いほど濃い緑にする."""
    t = float(np.clip((height - 0.02) / (cfg.grass.height_max - 0.02), 0.0, 1.0))
    r = 0.42 - 0.26 * t
    g = 0.72 - 0.26 * t
    b = 0.24 - 0.14 * t
    return (r, g, b, 1.0)


STUBBLE_RGBA = (0.55, 0.52, 0.26, 1.0)

# ---------------------------------------------------------------- MJCF 生成


def _grass_body_xml(field: GrassField, cfg: Config) -> str:
    rows = []
    for i in range(field.n):
        x, y = field.xy[i]
        h = float(field.height[i])
        r = float(field.radius[i])
        cr, cg, cb, ca = grass_rgba(h, cfg)
        tx, ty = field.tilt[i]
        rows.append(
            f'      <geom name="grass{i}" type="capsule" '
            f'pos="{x:.4f} {y:.4f} {h / 2:.4f}" size="{r:.4f} {h / 2:.4f}" '
            f'euler="{tx:.2f} {ty:.2f} 0" '
            f'rgba="{cr:.3f} {cg:.3f} {cb:.3f} {ca:.2f}" '
            f'contype="0" conaffinity="0" group="0"/>'
        )
    return "\n".join(rows)


def _plot_marker_xml(cfg: Config) -> str:
    """1 坪区画の外周を示す視覚マーカー (物理的な壁ではない)."""
    h = cfg.plot.half
    t = 0.012
    posts = []
    for sx, sy, ln, ax in (
        (0.0, h, h, "x"),
        (0.0, -h, h, "x"),
        (h, 0.0, h, "y"),
        (-h, 0.0, h, "y"),
    ):
        size = f"{ln:.4f} {t:.4f} {t:.4f}" if ax == "x" else f"{t:.4f} {ln:.4f} {t:.4f}"
        posts.append(
            f'      <geom type="box" pos="{sx:.4f} {sy:.4f} {t:.4f}" size="{size}" '
            f'rgba="0.95 0.85 0.25 1" contype="0" conaffinity="0" group="0"/>'
        )
    return "\n".join(posts)


def build_mjcf(cfg: Config, field: GrassField, start_xy: tuple[float, float], start_yaw: float) -> str:
    r = cfg.robot
    wr = r.wheel_radius
    hl, hw, hh = r.chassis_len / 2, r.chassis_wid / 2, r.chassis_hgt / 2
    cam_z = r.cam_height - wr
    p = np.deg2rad(r.cam_pitch_deg)
    # カメラ右方向 = 機体 -y, カメラ上方向 = 機体 (sin p, 0, cos p)
    xyaxes = f"0 -1 0 {np.sin(p):.6f} 0 {np.cos(p):.6f}"
    sx, sy = start_xy
    caster = 0.024

    return f"""
<mujoco model="agripioneer">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{cfg.sim.timestep}" integrator="implicitfast" gravity="0 0 -9.81"
          cone="elliptic" impratio="3"/>
  <visual>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.55 0.55 0.52" specular="0.08 0.08 0.08"/>
    <map znear="0.02" zfar="50" shadowclip="2"/>
    <quality shadowsize="4096" offsamples="4"/>
    <global offwidth="1920" offheight="1080" azimuth="130" elevation="-28"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.46 0.64 0.86" rgb2="0.86 0.91 0.96" width="256" height="256"/>
    <texture name="soiltex" type="2d" builtin="flat" rgb1="0.41 0.32 0.23"
             mark="random" markrgb="0.30 0.23 0.16" random="0.45" width="512" height="512"/>
    <material name="soil" texture="soiltex" texrepeat="9 9"
              specular="0.02" shininess="0.01" reflectance="0.0"/>
    <material name="shell" rgba="0.90 0.42 0.12 1" specular="0.35" shininess="0.45"/>
    <material name="dark" rgba="0.13 0.14 0.16 1" specular="0.2" shininess="0.3"/>
    <material name="steel" rgba="0.62 0.64 0.68 1" specular="0.6" shininess="0.7"/>
  </asset>

  <default>
    <default class="wheel">
      <!-- MuJoCo は 2 つの geom の摩擦係数を「最大値」で合成するため, 地面側の
           摩擦に引きずられないよう priority を上げて車輪側の値を強制採用させる -->
      <geom type="cylinder" size="{wr:.4f} 0.016" euler="90 0 0" material="dark"
            friction="1.6 0.004 0.0004" priority="2" condim="3" mass="0.5"
            solref="0.012 1.1" solimp="0.95 0.99 0.002"/>
    </default>
    <default class="caster">
      <!-- 自在キャスタ (ボールキャスタ) 相当. 旋回を妨げないよう低摩擦にする -->
      <geom type="sphere" size="{caster:.4f}" material="steel"
            friction="0.02 0.001 0.0001" priority="2" condim="3" mass="0.08"
            solref="0.012 1.1" solimp="0.95 0.99 0.002"/>
    </default>
  </default>

  <worldbody>
    <light name="sun" directional="true" pos="1.2 -1.4 3.2" dir="-0.3 0.4 -1"
           diffuse="0.72 0.72 0.68" specular="0.12 0.12 0.12" castshadow="true"/>
    <geom name="ground" type="plane" size="6 6 0.1" material="soil"
          friction="1.3 0.01 0.001" condim="4"/>

{_plot_marker_xml(cfg)}

{_grass_body_xml(field, cfg)}

    <camera name="overview" pos="0 -1.92 1.74" xyaxes="1 0 0 0 0.67 0.74" fovy="48"/>
    <camera name="topdown" pos="0 0 2.55" xyaxes="1 0 0 0 1 0" fovy="46"/>

    <body name="base" pos="{sx:.4f} {sy:.4f} {wr:.4f}" euler="0 0 {np.rad2deg(start_yaw):.3f}">
      <freejoint name="base_free"/>
      <!-- 車体原点 = 駆動輪軸の中点. 重心をわずかに前寄りに置き, 前方キャスタと
           駆動輪の三点支持で接地荷重の 7 割以上を駆動輪に掛ける. -->
      <inertial pos="0.080 0 0.012" mass="7.5" diaginertia="0.07 0.10 0.12"/>

      <geom name="chassis" type="box" pos="0.060 0 {0.012 + hh:.4f}" size="{hl:.4f} {hw:.4f} {hh:.4f}"
            material="shell" friction="0.4 0.005 0.0005" priority="2" condim="3"/>
      <geom name="hood" type="box" pos="0.115 0 {0.012 + 2 * hh + 0.018:.4f}"
            size="{hl * 0.52:.4f} {hw * 0.78:.4f} 0.018" material="dark"
            contype="0" conaffinity="0"/>
      <geom name="mast" type="capsule" fromto="0.130 0 {0.012 + 2 * hh:.4f} 0.165 0 {cam_z:.4f}"
            size="0.009" material="steel" contype="0" conaffinity="0"/>
      <geom name="camcase" type="box" pos="0.170 0 {cam_z:.4f}" size="0.024 0.034 0.020"
            material="dark" contype="0" conaffinity="0"/>
      <geom name="blade" type="cylinder" pos="{r.blade_offset:.4f} 0 -0.042"
            size="{r.blade_width / 2:.4f} 0.005" rgba="0.74 0.76 0.80 1"
            contype="0" conaffinity="0"/>
      <geom name="bladehub" type="cylinder" pos="{r.blade_offset:.4f} 0 -0.028"
            size="0.026 0.014" material="dark" contype="0" conaffinity="0"/>

      <camera name="onboard" pos="0.170 0 {cam_z:.4f}" xyaxes="{xyaxes}"
              fovy="{r.cam_fovy_deg}"/>
      <camera name="chase" pos="-0.92 0 0.72" xyaxes="0 -1 0 0.46 0 0.89" fovy="60"/>

      <body name="wheel_l" pos="0 {r.wheel_base / 2:.4f} 0">
        <joint name="wheel_l" type="hinge" axis="0 1 0" damping="0.01" armature="0.004"/>
        <geom class="wheel"/>
      </body>
      <body name="wheel_r" pos="0 {-r.wheel_base / 2:.4f} 0">
        <joint name="wheel_r" type="hinge" axis="0 1 0" damping="0.01" armature="0.004"/>
        <geom class="wheel"/>
      </body>
      <body name="caster_f" pos="0.185 0 {-wr + caster:.4f}">
        <geom class="caster"/>
      </body>
      <!-- 後方キャスタは通常は浮いており, 加速時の「ウィリー」を止める保険 -->
      <body name="caster_b" pos="-0.145 0 {-wr + caster + 0.005:.4f}">
        <geom class="caster"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="mot_l" joint="wheel_l" kv="28" ctrlrange="-30 30"/>
    <velocity name="mot_r" joint="wheel_r" kv="28" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
""".strip()


# ---------------------------------------------------------------- World


class World:
    """MuJoCo モデルと草地の状態をまとめて扱うラッパ."""

    def __init__(
        self,
        cfg: Config,
        field: GrassField,
        start_xy: tuple[float, float] = (0.0, 0.0),
        start_yaw: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.field = field
        xml = build_mjcf(cfg, field, start_xy, start_yaw)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.grass_gid = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"grass{i}")
                for i in range(field.n)
            ],
            dtype=np.int32,
        )
        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "onboard")

        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self._prev_blade: np.ndarray | None = None
        self.distance_travelled = 0.0
        self._prev_xy = np.array(start_xy, dtype=float)

        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ 状態取得
    @property
    def time(self) -> float:
        return float(self.data.time)

    def pose(self) -> tuple[np.ndarray, float]:
        """(xy, yaw) を返す."""
        pos = self.data.xpos[self.base_bid]
        mat = self.data.xmat[self.base_bid].reshape(3, 3)
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        return pos[:2].copy(), yaw

    def blade_center(self) -> np.ndarray:
        pos = self.data.xpos[self.base_bid]
        mat = self.data.xmat[self.base_bid].reshape(3, 3)
        off = mat @ np.array([self.cfg.robot.blade_offset, 0.0, 0.0])
        return (pos + off)[:2].copy()

    # ------------------------------------------------------------ 指令
    def set_wheel_speeds(self, wl: float, wr: float) -> None:
        self.data.ctrl[0] = float(np.clip(wl, -30.0, 30.0))
        self.data.ctrl[1] = float(np.clip(wr, -30.0, 30.0))

    def set_body_twist(self, v: float, w: float, dt: float | None = None) -> None:
        """並進速度 v [m/s] と角速度 w [rad/s] を車輪角速度に変換して与える.

        実機と同様に加減速をレート制限する. 無制限に指令すると駆動反力で
        車体がピッチングし (ウィリー), 駆動輪が浮いて空転するため.
        """
        r = self.cfg.robot
        v = float(np.clip(v, -r.max_lin_vel, r.max_lin_vel))
        w = float(np.clip(w, -r.max_ang_vel, r.max_ang_vel))
        if dt is not None:
            dv = r.max_lin_acc * dt
            dw = r.max_ang_acc * dt
            v = float(np.clip(v, self._cmd_v - dv, self._cmd_v + dv))
            w = float(np.clip(w, self._cmd_w - dw, self._cmd_w + dw))
        self._cmd_v, self._cmd_w = v, w
        vl = v - w * r.wheel_base / 2.0
        vr = v + w * r.wheel_base / 2.0
        self.set_wheel_speeds(vl / r.wheel_radius, vr / r.wheel_radius)

    # ------------------------------------------------------------ 進行
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        xy, _ = self.pose()
        self.distance_travelled += float(np.linalg.norm(xy - self._prev_xy))
        self._prev_xy = xy

    # ------------------------------------------------------------ 刈り取り
    def mow(self) -> int:
        """刈刃の掃引領域に入った草を刈り取る. 新たに刈れた本数を返す.

        制御周期の間にロボットが移動した分を線分として扱い, 線分と草の距離で
        判定する (離散化による刈り残しを防ぐ).
        """
        c = self.blade_center()
        prev = self._prev_blade if self._prev_blade is not None else c
        self._prev_blade = c.copy()

        f = self.field
        alive = (~f.cut) & (f.height > self.cfg.grass.stubble_height + 1e-4)
        if not alive.any():
            return 0

        seg = c - prev
        L2 = float(seg @ seg)
        rel = f.xy[alive] - prev
        if L2 < 1e-9:
            dist = np.linalg.norm(rel, axis=1)
        else:
            t = np.clip((rel @ seg) / L2, 0.0, 1.0)
            dist = np.linalg.norm(rel - t[:, None] * seg[None, :], axis=1)

        hit_local = dist <= (self.cfg.robot.blade_width / 2.0)
        if not hit_local.any():
            return 0

        idx = np.flatnonzero(alive)[hit_local]
        self.cut_indices(idx)
        return int(len(idx))

    def cut_indices(self, idx: np.ndarray) -> None:
        """指定した草を刈り取った状態 (刈り株) に書き換える."""
        stub = self.cfg.grass.stubble_height
        f = self.field
        for i in np.atleast_1d(idx):
            gid = self.grass_gid[i]
            self.model.geom_size[gid, 1] = stub / 2.0
            self.model.geom_pos[gid, 2] = stub / 2.0
            self.model.geom_quat[gid] = (1.0, 0.0, 0.0, 0.0)
            self.model.geom_rgba[gid] = STUBBLE_RGBA
            f.height[i] = stub
            f.cut[i] = True

    def grow_indices(self, idx: np.ndarray) -> None:
        """刈り株状態の草を元の草丈に戻す (草の再生 / 途中出現の模擬)."""
        f = self.field
        for i in np.atleast_1d(idx):
            gid = self.grass_gid[i]
            h = float(f.height0[i])
            self.model.geom_size[gid, 1] = h / 2.0
            self.model.geom_pos[gid, 2] = h / 2.0
            self.model.geom_rgba[gid] = grass_rgba(h, self.cfg)
            f.height[i] = h
            f.cut[i] = False

    def set_pose(self, xy: tuple[float, float], yaw: float, roll: float = 0.0,
                 pitch: float = 0.0) -> None:
        """ロボットを指定姿勢へ瞬間移動させる (学習データ生成用)."""
        import mujoco as _mj

        quat = np.zeros(4)
        _mj.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), "xyz")
        self.data.qpos[0:2] = xy
        self.data.qpos[2] = self.cfg.robot.wheel_radius
        self.data.qpos[3:7] = quat
        self.data.qvel[:] = 0.0
        _mj.mj_forward(self.model, self.data)

    def camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """搭載カメラの (位置, 回転行列) を返す."""
        return (
            self.data.cam_xpos[self.cam_id].copy(),
            self.data.cam_xmat[self.cam_id].reshape(3, 3).copy(),
        )

    # ------------------------------------------------------------ 進捗
    def mow_progress(self) -> tuple[int, int]:
        return self.field.n_cut, self.field.n_target
