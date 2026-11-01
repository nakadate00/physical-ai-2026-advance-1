"""経路を Pure Pursuit で追従し速度指令を出すノード.

    sub : /plan      nav_msgs/Path
    sub : /odom      nav_msgs/Odometry
    pub : /cmd_vel   geometry_msgs/Twist
    pub : /controller/finished  std_msgs/Bool
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool

from agripioneer.config import Config
from agripioneer.control.pure_pursuit import PurePursuit


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("controller_node")
        self.declare_parameter("control_hz", 20.0)
        self.cfg = Config()
        self.pp: PurePursuit | None = None
        self.xy = np.zeros(2)
        self.yaw = 0.0
        self._announced = False

        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_fin = self.create_publisher(Bool, "controller/finished", 2)
        self.create_subscription(Path, "plan", self._on_path, 2)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        hz = float(self.get_parameter("control_hz").value)
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self._tick)

    # ------------------------------------------------------------------
    def _on_path(self, msg: Path) -> None:
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self.pp = PurePursuit(self.cfg, pts) if len(pts) else None
        self._announced = False
        self.get_logger().info(f"経路を受信: {len(pts)} 点")

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.xy = np.array([p.x, p.y])
        self.yaw = float(np.arctan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z))

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        t = Twist()
        if self.pp is not None and not self.pp.finished:
            v, w = self.pp.step(self.xy, self.yaw, self.dt)
            t.linear.x, t.angular.z = float(v), float(w)
        elif self.pp is not None and not self._announced:
            self._announced = True
            self.pub_fin.publish(Bool(data=True))
            self.get_logger().info("経路追従を完了しました")
        self.pub.publish(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
