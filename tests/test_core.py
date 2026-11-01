"""主要コンポーネントの回帰テスト."""

from __future__ import annotations

import numpy as np
import pytest

from agripioneer.config import Config
from agripioneer.control.pure_pursuit import PurePursuit, densify
from agripioneer.perception.camera import PinholeCamera
from agripioneer.perception.mapper import GrassMap
from agripioneer.planning import tsp
from agripioneer.planning.coverage import full_coverage_path, plan_mowing, survey_path
from agripioneer.soil.crops import CROPS, rule_scores
from agripioneer.soil.field import PROP_NAMES, SoilField
from agripioneer.world import World, sample_grass


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config()


# ------------------------------------------------------------------ world


def test_grass_sampling_in_plot(cfg: Config) -> None:
    f = sample_grass(cfg, np.random.default_rng(0))
    assert f.n > 50
    assert np.all(np.abs(f.xy) <= cfg.plot.half)
    assert f.n_cut == 0


def test_robot_tracks_commanded_velocity(cfg: Config) -> None:
    """車輪が滑り切らず, 指令並進速度をおおむね達成できること."""
    f = sample_grass(cfg, np.random.default_rng(0))
    w = World(cfg, f, (-0.7, -0.7), 0.0)
    for _ in range(40):  # 2 秒の助走
        w.set_body_twist(0.35, 0.0, 0.05)
        w.step(13)
    p0, t0 = w.pose()[0].copy(), w.time
    for _ in range(40):
        w.set_body_twist(0.35, 0.0, 0.05)
        w.step(13)
    v = np.linalg.norm(w.pose()[0] - p0) / (w.time - t0)
    assert v == pytest.approx(0.35, rel=0.15)


def test_mowing_shortens_grass(cfg: Config) -> None:
    f = sample_grass(cfg, np.random.default_rng(1))
    w = World(cfg, f, tuple(f.xy[0] - np.array([0.085, 0.0])), 0.0)
    before = f.height[0]
    assert w.mow() >= 1
    assert f.height[0] < before
    assert f.height[0] == pytest.approx(cfg.grass.stubble_height)


# ------------------------------------------------------------------ 知覚


def test_camera_projection_roundtrip(cfg: Config) -> None:
    """地面への逆投影と画像への順投影が互いに逆になること."""
    f = sample_grass(cfg, np.random.default_rng(2))
    w = World(cfg, f, (-0.5, -0.3), 0.4)
    cam = PinholeCamera(96, 96, cfg.robot.cam_fovy_deg)
    pos, mat = w.camera_pose()
    pts, _ = cam.ground_points(pos, mat)
    probe = np.array([[pts[48, 48, 0], pts[48, 48, 1], 0.0],
                      [pts[20, 70, 0], pts[20, 70, 1], 0.0]])
    uv, _, vis = cam.project(probe, pos, mat)
    assert vis.all()
    assert uv[0] == pytest.approx([48.0, 48.0], abs=1e-3)
    assert uv[1] == pytest.approx([70.0, 20.0], abs=1e-3)


def test_map_integration_marks_grass(cfg: Config) -> None:
    """確率 1 の観測を入れると, カメラ前方のセルが草ありになること."""
    f = sample_grass(cfg, np.random.default_rng(3))
    w = World(cfg, f, (0.0, -0.6), np.pi / 2)
    cam = PinholeCamera(96, 96, cfg.robot.cam_fovy_deg)
    gmap = GrassMap(cfg)
    pos, mat = w.camera_pose()
    for _ in range(6):
        gmap.integrate(np.ones((96, 96)), pos, mat, cam)
    assert gmap.coverage_ratio() > 0.05
    ahead = gmap.xy_to_ij(np.array([0.0, -0.2]))
    assert gmap.prob()[ahead[0], ahead[1]] > 0.8


# ------------------------------------------------------------------ 計画


def test_survey_is_shorter_than_full_coverage(cfg: Config) -> None:
    """カメラ観測幅は刈刃幅より広いので, 探査経路のほうが必ず短いこと."""
    s = np.linalg.norm(np.diff(survey_path(cfg), axis=0), axis=1).sum()
    b = np.linalg.norm(np.diff(full_coverage_path(cfg), axis=0), axis=1).sum()
    assert s < b * 0.6


def test_plan_covers_only_detected_grass(cfg: Config) -> None:
    gmap = GrassMap(cfg)
    ij = gmap.xy_to_ij(np.array([[0.4, 0.4], [0.42, 0.44], [0.38, 0.43]]))
    gmap.logodds[ij[:, 0], ij[:, 1]] = 5.0
    plan = plan_mowing(gmap, cfg, np.array([-0.7, -0.7]))
    assert len(plan.clusters) == 1
    # 経路は検出した草むらの近傍に収まっているはず
    assert np.all(plan.path[1:, :] > 0.0)


def test_tsp_improves_on_naive_order() -> None:
    rng = np.random.default_rng(0)
    k = 9
    ends = np.stack([np.stack([p, p + 0.05]) for p in rng.uniform(-1, 1, (k, 2))])
    start = np.array([-1.0, -1.0])
    order, flip, cost = tsp.solve(ends, start, iters=4000, seed=1)
    naive = tsp._tour_cost(np.arange(k), np.zeros(k, bool), ends, start)
    assert sorted(order.tolist()) == list(range(k))
    assert cost <= naive


# ------------------------------------------------------------------ 制御


def test_densify_keeps_shape() -> None:
    p = densify(np.array([[0.0, 0.0], [1.0, 0.0]]), spacing=0.1)
    assert len(p) == 11
    assert np.allclose(p[0], [0, 0]) and np.allclose(p[-1], [1, 0])


def test_pure_pursuit_reaches_goal(cfg: Config) -> None:
    """運動学モデル上で経路を追従し終えられること."""
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.6], [0.0, 0.6]])
    pp = PurePursuit(cfg, path)
    xy, yaw, dt = np.array([0.0, 0.0]), 0.0, 0.05
    for _ in range(2000):
        v, w = pp.step(xy, yaw, dt)
        yaw += w * dt
        xy = xy + v * dt * np.array([np.cos(yaw), np.sin(yaw)])
        if pp.finished:
            break
    assert pp.finished
    assert np.linalg.norm(xy - path[-1]) < 0.12


# ------------------------------------------------------------------ 土壌


def test_soil_values_in_range(cfg: Config) -> None:
    f = sample_grass(cfg, np.random.default_rng(4))
    sf = SoilField(cfg, f, np.random.default_rng(5))
    s = sf.probe(np.array([0.2, -0.3]))
    assert set(s.values) == set(PROP_NAMES)
    assert 4.0 <= s.values["pH"] <= 8.0


def test_crop_rules_prefer_matching_soil() -> None:
    """強酸性の土壌ではブルーベリーがホウレンソウより高評価になること."""
    acid = {"pH": np.array([4.8]), "EC": np.array([0.2]), "OM": np.array([5.0]),
            "N": np.array([8.0]), "P": np.array([25.0]), "K": np.array([25.0]),
            "moisture": np.array([30.0]), "solar": np.array([17.0])}
    s = rule_scores(acid)[0]
    names = [c.name for c in CROPS]
    assert s[names.index("ブルーベリー")] > s[names.index("ホウレンソウ")]
