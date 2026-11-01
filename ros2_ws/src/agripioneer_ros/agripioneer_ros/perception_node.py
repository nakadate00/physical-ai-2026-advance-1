"""カメラ画像から草の占有格子地図を作るノード.

    sub : /camera/image_raw   sensor_msgs/Image
    sub : /camera/pose        geometry_msgs/PoseStamped
    pub : /grass/map          nav_msgs/OccupancyGrid   (0-100, 未観測は -1)
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image

import mujoco
import torch

from agripioneer.config import Config
from agripioneer.perception.camera import PinholeCamera
from agripioneer.perception.mapper import GrassMap
from agripioneer.perception.runner import DEFAULT_MODEL
from agripioneer.perception.unet import GrassUNet, preprocess


def _quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.array([q.w, q.x, q.y, q.z], dtype=float))
    return m.reshape(3, 3)


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("model_path", str(DEFAULT_MODEL))
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("threads", 2)

        self.cfg = Config()
        res = self.cfg.robot.cam_res
        self.cam = PinholeCamera(res, res, self.cfg.robot.cam_fovy_deg)
        self.gmap = GrassMap(self.cfg)

        blob = torch.load(str(self.get_parameter("model_path").value),
                          map_location="cpu", weights_only=False)
        self.model = GrassUNet(base=blob.get("base", 16))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        torch.set_num_threads(int(self.get_parameter("threads").value))
        self.get_logger().info(f"U-Net 読込 (val IoU={blob.get('val_iou', float('nan')):.4f})")

        self.cam_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.create_subscription(PoseStamped, "camera/pose", self._on_pose, 2)
        self.create_subscription(Image, "camera/image_raw", self._on_image, 2)
        self.pub = self.create_publisher(OccupancyGrid, "grass/map", 2)
        self.create_timer(1.0 / float(self.get_parameter("publish_hz").value),
                          self._publish)

    # ------------------------------------------------------------------
    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.cam_pose = (np.array([p.x, p.y, p.z]),
                         _quat_to_mat(msg.pose.orientation))

    def _on_image(self, msg: Image) -> None:
        if self.cam_pose is None:
            return
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(preprocess(rgb)))[0, 0].numpy()
        pos, mat = self.cam_pose
        self.gmap.integrate(prob, pos, mat, self.cam)

    # ------------------------------------------------------------------
    def _publish(self) -> None:
        g = OccupancyGrid()
        g.header.stamp = self.get_clock().now().to_msg()
        g.header.frame_id = "map"
        g.info.resolution = float(self.gmap.res)
        g.info.width = g.info.height = int(self.gmap.n)
        g.info.origin.position.x = float(self.gmap.origin)
        g.info.origin.position.y = float(self.gmap.origin)
        g.info.origin.orientation.w = 1.0
        occ = np.rint(self.gmap.prob() * 100).astype(np.int8)
        occ[~self.gmap.observed()] = -1
        g.data = occ.flatten().tolist()
        self.pub.publish(g)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
