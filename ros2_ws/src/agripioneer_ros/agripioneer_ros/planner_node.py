"""占有格子地図から刈り取り経路を計画するノード.

起動直後は探査 (牛耕式) 経路を配信し, 地図の被覆率が閾値を超えた時点で
草むらのクラスタ分解 + TSP による巡回順最適化を行い, 刈り取り経路に切り替える.

    sub : /grass/map   nav_msgs/OccupancyGrid
    sub : /odom        nav_msgs/Odometry
    sub : /controller/finished  std_msgs/Bool  経路を走り終えた通知
    pub : /plan        nav_msgs/Path
    pub : /mission/phase  std_msgs/String
    srv : ~/replan     std_srvs/Trigger   (手動で再計画させたいとき)
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from agripioneer.config import Config
from agripioneer.perception.mapper import GrassMap
from agripioneer.planning.coverage import plan_mowing, survey_path


class PlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("planner_node")
        self.declare_parameter("coverage_threshold", 0.97)
        self.declare_parameter("grass_threshold", 0.6)
        self.declare_parameter("max_rounds", 4)

        self.cfg = Config()
        self.gmap = GrassMap(self.cfg)
        self.xy = np.zeros(2)
        self.phase = "SURVEY"
        self.rounds = 0
        self.coverage = 0.0

        self.pub_path = self.create_publisher(Path, "plan", 2)
        self.pub_phase = self.create_publisher(String, "mission/phase", 2)
        self.create_subscription(OccupancyGrid, "grass/map", self._on_map, 2)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.create_subscription(Bool, "controller/finished", self._on_finished, 2)
        self.create_service(Trigger, "~/replan", self._on_replan)

        self._publish(survey_path(self.cfg))
        self.get_logger().info("探査経路を配信しました (SURVEY)")

    # ------------------------------------------------------------------
    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.xy = np.array([p.x, p.y])

    def _on_map(self, msg: OccupancyGrid) -> None:
        n = msg.info.width
        data = np.array(msg.data, dtype=np.int16).reshape(n, n)
        observed = data >= 0
        p = np.clip(np.where(observed, data, 50) / 100.0, 1e-3, 1 - 1e-3)
        self.gmap.logodds = np.log(p / (1 - p)).astype(np.float32)
        self.gmap.hits = observed.astype(np.int32)
        self.coverage = float(observed.mean())

        if self.phase == "SURVEY" and self.coverage >= float(
            self.get_parameter("coverage_threshold").value
        ):
            self._replan()

    def _on_finished(self, msg: Bool) -> None:
        """経路を走り終えたら必ず再計画する (探査の取りこぼし対策も兼ねる)."""
        if msg.data and self.phase != "DONE":
            self.get_logger().info(
                f"経路追従の完了を受信 (地図被覆率 {100 * self.coverage:.1f} %) — 再計画します")
            self._replan()

    def _on_replan(self, request, response):
        self._replan()
        response.success = True
        response.message = f"phase={self.phase}"
        return response

    # ------------------------------------------------------------------
    def _replan(self) -> None:
        # 巡を重ねるごとに検出しきい値を下げ, 確信度の低い刈り残しも拾いに行く
        thr = max(0.33, float(self.get_parameter("grass_threshold").value)
                  - 0.09 * max(0, self.rounds - 1))
        plan = plan_mowing(self.gmap, self.cfg, self.xy, thr=thr, seed=self.rounds,
                           min_cells=1 if self.rounds >= 2 else 3)
        self.rounds += 1
        if not plan.clusters or self.rounds > int(self.get_parameter("max_rounds").value):
            self.phase = "DONE"
            self._publish(np.zeros((0, 2)))
            self.get_logger().info("刈り残しなし — 刈り取り完了")
            return
        self.phase = "MOW"
        self._publish(plan.path)
        self.get_logger().info(
            f"[第 {self.rounds} 巡] しきい値 {thr:.2f} / 草むら {len(plan.clusters)} 箇所 / "
            f"計画経路 {plan.total_length:.2f} m (うち移動 {plan.transit_length:.2f} m) を配信")

    def _publish(self, path: np.ndarray) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for p in np.asarray(path).reshape(-1, 2):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)
        self.pub_phase.publish(String(data=self.phase))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
