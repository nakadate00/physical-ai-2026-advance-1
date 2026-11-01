"""MuJoCo シミュレータを ROS 2 のセンサ/アクチュエータとして公開するノード.

    sub : /cmd_vel            geometry_msgs/Twist     速度指令
    pub : /odom               nav_msgs/Odometry       自己位置 (シミュレータ真値)
    pub : /camera/image_raw   sensor_msgs/Image       搭載カメラ RGB
    pub : /camera/pose        geometry_msgs/PoseStamped  カメラ姿勢 (逆投影用)
    pub : /mow/progress       std_msgs/Int32MultiArray   [刈取済, 対象本数]
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

import mujoco

from agripioneer.config import Config
from agripioneer.world import World, sample_grass


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).flatten())
    return q  # (w, x, y, z)


class SimulatorNode(Node):
    def __init__(self) -> None:
        super().__init__("simulator_node")
        self.declare_parameter("seed", 20261102)
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("camera_hz", 5.0)
        self.declare_parameter("blade_enabled", True)

        seed = int(self.get_parameter("seed").value)
        self.cfg = Config()
        rng = np.random.default_rng(seed)
        field = sample_grass(self.cfg, rng)
        h = self.cfg.plot.half
        self.world = World(self.cfg, field, (-h + 0.17, -h + 0.17), 0.0)
        res = self.cfg.robot.cam_res
        self.renderer = mujoco.Renderer(self.world.model, res, res)

        self.cmd = (0.0, 0.0)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.pub_img = self.create_publisher(Image, "camera/image_raw", 2)
        self.pub_cam = self.create_publisher(PoseStamped, "camera/pose", 2)
        self.pub_mow = self.create_publisher(Int32MultiArray, "mow/progress", 10)

        hz = float(self.get_parameter("control_hz").value)
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self._tick)
        self.create_timer(1.0 / float(self.get_parameter("camera_hz").value),
                          self._publish_camera)
        self.create_timer(10.0, self._log_progress)
        self.get_logger().info(
            f"MuJoCo 起動: 草 {field.n} 本 (対象 {field.n_target} 本), "
            f"区画 {self.cfg.plot.side:.3f} m 角")

    # ------------------------------------------------------------------
    def _on_cmd(self, msg: Twist) -> None:
        self.cmd = (float(msg.linear.x), float(msg.angular.z))

    def _tick(self) -> None:
        self.world.set_body_twist(self.cmd[0], self.cmd[1], self.dt)
        self.world.step(int(round(self.dt / self.cfg.sim.timestep)))
        if bool(self.get_parameter("blade_enabled").value):
            self.world.mow()

        xy, yaw = self.world.pose()
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(xy[0])
        odom.pose.pose.position.y = float(xy[1])
        odom.pose.pose.orientation.z = float(np.sin(yaw / 2))
        odom.pose.pose.orientation.w = float(np.cos(yaw / 2))
        self.pub_odom.publish(odom)

        cut, total = self.world.mow_progress()
        self.pub_mow.publish(Int32MultiArray(data=[int(cut), int(total)]))

    def _log_progress(self) -> None:
        cut, total = self.world.mow_progress()
        pct = 0.0 if total == 0 else 100.0 * cut / total
        self.get_logger().info(
            f"t={self.world.time:6.1f}s  走行 {self.world.distance_travelled:6.2f} m  "
            f"刈り取り {cut}/{total} ({pct:.1f} %)")

    # ------------------------------------------------------------------
    def _publish_camera(self) -> None:
        self.renderer.update_scene(self.world.data, camera="onboard")
        rgb = self.renderer.render()
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height, msg.width = rgb.shape[0], rgb.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = rgb.shape[1] * 3
        msg.data = rgb.tobytes()
        self.pub_img.publish(msg)

        pos, mat = self.world.camera_pose()
        q = _mat_to_quat(mat)
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos)
        ps.pose.orientation.w = float(q[0])
        ps.pose.orientation.x = float(q[1])
        ps.pose.orientation.y = float(q[2])
        ps.pose.orientation.z = float(q[3])
        self.pub_cam.publish(ps)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
