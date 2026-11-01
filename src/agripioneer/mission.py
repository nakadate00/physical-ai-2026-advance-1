"""ミッション全体の状態機械.

    SURVEY  探査: 牛耕式に走りながら CNN で草の占有格子地図を作る
    PLAN    計画: 地図を連結成分でクラスタ化し, TSP で巡回順を最適化
    MOW     刈取: 計画経路を Pure Pursuit で追従し, 刈刃掃引で草を刈る
    VERIFY  検証: 更新後の地図を見て刈り残しがあれば MOW に戻る
    SOIL    診断: 複数地点で土壌を計測し, 学習済みモデルで作付を提案
    DONE    完了

ベースラインモード (`mode="baseline"`) では知覚も計画も行わず, 区画全面を
刈刃幅で舐める素朴な被覆経路のみを実行する. 比較実験に用いる.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import Config
from .control.pure_pursuit import PurePursuit
from .perception.mapper import GrassMap
from .perception.runner import GrassPerception
from .planning import tsp
from .planning.coverage import MowPlan, full_coverage_path, plan_mowing, survey_path
from .soil.field import SoilField, SoilSample, aggregate
from .soil.recommender import CropRecommender, Recommendation
from .world import World


class Phase(str, Enum):
    SURVEY = "SURVEY"
    PLAN = "PLAN"
    MOW = "MOW"
    VERIFY = "VERIFY"
    SOIL = "SOIL"
    DONE = "DONE"


PHASE_LABEL = {
    Phase.SURVEY: "探査中",
    Phase.PLAN: "経路計画中",
    Phase.MOW: "刈り取り中",
    Phase.VERIFY: "刈り残し確認",
    Phase.SOIL: "土壌診断中",
    Phase.DONE: "完了",
}

#: 計画結果を見せるための静止時間 [s]
PLAN_HOLD = 2.2
#: 土壌サンプリング 1 地点あたりの静止時間 [s]
PROBE_HOLD = 1.1
#: 刈り残し再計画の最大回数
MAX_MOW_ROUNDS = 4
#: 1 巡目の草検出しきい値. 巡を重ねるごとに下げ, 確信度の低い草も拾いに行く
#: (最初から低くすると誤検出に振り回されて走行距離が伸びるため)
GRASS_THR_FIRST = 0.60
GRASS_THR_STEP = 0.09
GRASS_THR_MIN = 0.33


@dataclass
class MissionResult:
    mode: str
    sim_time: float = 0.0
    distance: float = 0.0
    mow_distance: float = 0.0
    survey_distance: float = 0.0
    cut: int = 0
    cut_total: int = 0
    map_coverage: float = 0.0
    n_clusters: int = 0
    n_inferences: int = 0
    planned_length: float = 0.0
    transit_length: float = 0.0
    rounds: int = 0
    #: 刈り取り作業を終えた時点の時刻 [s] と走行距離 [m] (手法間の公平な比較用).
    #: 土壌診断フェーズは含めない.
    mow_end_time: float = float("nan")
    mow_end_distance: float = float("nan")
    #: 刈り取り率が 100 % に達した時刻 (達しなければ NaN)
    full_cut_time: float = float("nan")
    soil: dict[str, float] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    perception_iou: float = float("nan")

    @property
    def cut_rate(self) -> float:
        return 0.0 if self.cut_total == 0 else self.cut / self.cut_total


class Mission:
    def __init__(
        self,
        cfg: Config,
        world: World,
        soil: SoilField | None = None,
        perception: GrassPerception | None = None,
        recommender: CropRecommender | None = None,
        mode: str = "ai",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.world = world
        self.soil = soil
        self.perception = perception
        self.recommender = recommender
        self.mode = mode
        self.seed = seed

        self.gmap = GrassMap(cfg)
        self.trajectory: list[np.ndarray] = [world.pose()[0].copy()]
        self.result = MissionResult(mode=mode, cut_total=world.field.n_target)
        if perception is not None:
            self.result.perception_iou = perception.val_iou

        self.plan: MowPlan | None = None
        self.soil_samples: list[SoilSample] = []
        self.soil_points: list[np.ndarray] = []
        self.message = ""
        self.message2 = ""
        self._hold = 0.0
        self._probe_i = 0
        self._round = 0
        self._t_perc = 0.0
        self._phase_t0 = 0.0
        self._dist0 = 0.0

        if mode == "baseline":
            self.phase = Phase.MOW
            self.controller = PurePursuit(cfg, full_coverage_path(cfg))
            self.result.planned_length = float(
                np.linalg.norm(np.diff(self.controller.path, axis=0), axis=1).sum()
            )
            self.message = "地図を使わず区画全面を走査 (ベースライン)"
        else:
            self.phase = Phase.SURVEY
            self.controller = PurePursuit(cfg, survey_path(cfg))
            self.message = "牛耕式に走査して草の地図を作成"

    # ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE and self._hold <= 0.0

    @property
    def planned_path(self) -> np.ndarray | None:
        if self.phase in (Phase.SOIL, Phase.DONE):
            return None
        return self.controller.path if self.controller.path.size else None

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """制御周期 dt [s] ぶんミッションを進める."""
        world = self.world
        xy, yaw = world.pose()

        if self._hold > 0.0:
            self._hold -= dt
            world.set_body_twist(0.0, 0.0, dt)
            world.step(int(round(dt / self.cfg.sim.timestep)))
            if self._hold <= 0.0:
                self._on_hold_end()
            self._record(xy)
            return

        v, w = self.controller.step(xy, yaw, dt)
        world.set_body_twist(v, w, dt)

        blade_before = world.blade_center()
        world.step(int(round(dt / self.cfg.sim.timestep)))

        # 実機の芝刈り機と同じく刈刃は走行中つねに回しておく. 探査の走行で
        # たまたま刈刃の下に入った草も刈れるので, その分だけ後工程が減る.
        if self.phase in (Phase.MOW, Phase.SURVEY, Phase.VERIFY):
            world.mow()
            self.gmap.mark_mowed(blade_before, world.blade_center())

        # 知覚は走行中つねに回す (刈り取り中の地図更新にも使う)
        self._t_perc += dt
        if (self.perception is not None
                and self.phase in (Phase.SURVEY, Phase.MOW, Phase.VERIFY)
                and self._t_perc >= self.cfg.sim.perception_dt):
            self._t_perc = 0.0
            self.perception.observe(self.gmap)

        if self.phase is Phase.SOIL:
            self._maybe_probe(world.pose()[0])

        self._record(world.pose()[0])

        if self.controller.finished and self._hold <= 0.0:
            self._advance()

    # ------------------------------------------------------------------
    def _record(self, xy: np.ndarray) -> None:
        if np.linalg.norm(xy - self.trajectory[-1]) > 0.015:
            self.trajectory.append(xy.copy())
        r = self.result
        r.sim_time = self.world.time
        r.distance = self.world.distance_travelled
        r.cut, r.cut_total = self.world.mow_progress()
        if r.cut_total > 0 and r.cut >= r.cut_total and np.isnan(r.full_cut_time):
            r.full_cut_time = self.world.time
        r.map_coverage = self.gmap.coverage_ratio()
        if self.perception is not None:
            r.n_inferences = self.perception.n_inferences

    # ------------------------------------------------------------------
    def _advance(self) -> None:
        xy, _ = self.world.pose()

        if self.phase is Phase.SURVEY:
            self.result.survey_distance = self.world.distance_travelled
            self._enter_plan(xy)

        elif self.phase is Phase.MOW:
            if self.mode == "baseline":
                self.result.mow_distance = (
                    self.world.distance_travelled - self.result.survey_distance)
                self._mark_mow_end()
                self._enter_done()
            else:
                self._enter_verify(xy)

        elif self.phase is Phase.VERIFY:
            self._enter_soil(xy)

        elif self.phase is Phase.SOIL:
            self._enter_done()

    # ------------------------------------------------------------------
    def _enter_plan(self, xy: np.ndarray) -> None:
        self.phase = Phase.PLAN
        self.plan = plan_mowing(self.gmap, self.cfg, xy, seed=self.seed)
        self.controller = PurePursuit(self.cfg, self.plan.path)
        self.result.n_clusters = len(self.plan.clusters)
        self.result.planned_length = self.plan.total_length
        self.result.transit_length = self.plan.transit_length
        area = sum(c.area for c in self.plan.clusters)
        self.message = (f"草むら {len(self.plan.clusters)} 箇所を検出 "
                        f"(推定面積 {area:.2f} m² / {self.cfg.plot.area:.2f} m²)")
        self.message2 = (f"TSP で巡回順を最適化 → 計画経路長 "
                         f"{self.plan.total_length:.2f} m")
        self._hold = PLAN_HOLD

    def _mark_mow_end(self) -> None:
        self.result.mow_end_time = self.world.time
        self.result.mow_end_distance = self.world.distance_travelled

    def _enter_verify(self, xy: np.ndarray) -> None:
        self._round += 1
        self.result.rounds = self._round
        self.result.mow_distance = (
            self.world.distance_travelled - self.result.survey_distance)
        thr = max(GRASS_THR_MIN, GRASS_THR_FIRST - GRASS_THR_STEP * self._round)
        plan = plan_mowing(self.gmap, self.cfg, xy, thr=thr,
                           seed=self.seed + self._round,
                           min_cells=1 if self._round >= 1 else 3)
        leftover = sum(c.area for c in plan.clusters)

        if plan.clusters and self._round < MAX_MOW_ROUNDS:
            self.phase = Phase.MOW
            self.plan = plan
            self.controller = PurePursuit(self.cfg, plan.path)
            self.message = (f"刈り残し {len(plan.clusters)} 箇所 "
                            f"({leftover:.3f} m²) を検出 — 再計画")
            self.message2 = (f"第 {self._round + 1} 巡目 "
                             f"(検出しきい値を {thr:.2f} へ引き下げ)")
            self._hold = 1.4
        else:
            self.phase = Phase.VERIFY
            self._mark_mow_end()
            self.message = "刈り残しなし — 耕耘可能な状態になりました"
            self.message2 = ""
            self.controller = PurePursuit(self.cfg, np.zeros((0, 2)))
            self._hold = 1.4

    def _enter_soil(self, xy: np.ndarray) -> None:
        if self.soil is None or self.recommender is None:
            self._enter_done()
            return
        self.phase = Phase.SOIL
        h = self.cfg.plot.half * 0.52
        pts = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [0.0, 0.0]])
        ends = np.stack([np.stack([p, p]) for p in pts])
        order, _, _ = tsp.solve(ends, xy, iters=1500, seed=self.seed)
        self.soil_points = [pts[i] for i in order]
        self.controller = PurePursuit(self.cfg, np.array(self.soil_points))
        self._probe_i = 0
        self.message = "土壌センサで 5 地点をサンプリング"
        self.message2 = "pH / EC / 有機物 / N・P・K / 含水率 / 日射"

    def _enter_done(self) -> None:
        self.phase = Phase.DONE
        self.controller = PurePursuit(self.cfg, np.zeros((0, 2)))
        if self.soil_samples and self.recommender is not None:
            agg = aggregate(self.soil_samples)
            recs = self.recommender.recommend(agg, k=3)
            self.result.soil = agg
            self.result.recommendations = recs
            self.result.advice = self.recommender.advice(agg, recs[0].crop)
            self.message = f"推奨作付: {recs[0].crop.name} (確信度 {recs[0].probability:.0%})"
            self.message2 = "1 坪の開拓完了 — 耕耘と作付に進めます"
        else:
            self.message = "被覆走査 完了"
            self.message2 = ""
        self._hold = 6.0

    # ------------------------------------------------------------------
    def _on_hold_end(self) -> None:
        if self.phase is Phase.PLAN:
            self.phase = Phase.MOW
            self.message = "最適順に草むらを刈り取り"
            self.message2 = ""
        elif self.phase is Phase.VERIFY and self.controller.path.size == 0:
            self._enter_soil(self.world.pose()[0])
        elif self.phase is Phase.SOIL and self.controller.finished:
            self._enter_done()

    # ------------------------------------------------------------------
    def _maybe_probe(self, xy: np.ndarray) -> None:
        """土壌サンプリング地点に十分近づいたら計測する."""
        if self.soil is None or self._probe_i >= len(self.soil_points):
            return
        p = self.soil_points[self._probe_i]
        if np.linalg.norm(xy - p) < 0.11:
            s = self.soil.probe(p)  # noqa: E501
            self.soil_samples.append(s)
            self._probe_i += 1
            self.message = (f"地点 {self._probe_i}/{len(self.soil_points)} 計測: "
                            f"pH {s.values['pH']:.1f} / "
                            f"EC {s.values['EC']:.2f} dS/m / "
                            f"有機物 {s.values['OM']:.1f}%")
            self.message2 = (f"N {s.values['N']:.1f} / P {s.values['P']:.0f} / "
                             f"K {s.values['K']:.0f} mg/100g, "
                             f"含水率 {s.values['moisture']:.0f}%")
            self._hold = PROBE_HOLD
