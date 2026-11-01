"""学習済み U-Net をシミュレータ上で動かす推論ラッパ."""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np
import torch

from ..config import Config
from ..world import World
from .camera import PinholeCamera
from .mapper import GrassMap
from .unet import GrassUNet, preprocess

DEFAULT_MODEL = pathlib.Path(__file__).resolve().parents[3] / "models" / "grass_unet.pt"


class GrassPerception:
    """搭載カメラ画像 -> 草の株元セグメンテーション -> 占有格子への積算."""

    def __init__(self, cfg: Config, world: World, model_path: str | pathlib.Path | None = None,
                 threads: int = 4, blind: bool = False) -> None:
        self.cfg = cfg
        self.world = world
        res = cfg.robot.cam_res
        self.cam = PinholeCamera(res, res, cfg.robot.cam_fovy_deg)
        self.renderer = mujoco.Renderer(world.model, res, res)

        p = pathlib.Path(model_path) if model_path else DEFAULT_MODEL
        if not p.exists():
            raise FileNotFoundError(
                f"学習済み知覚モデルが見つかりません: {p}\n"
                "`python scripts/gen_perception_dataset.py && "
                "python scripts/train_perception.py` を実行してください"
            )
        blob = torch.load(p, map_location="cpu", weights_only=False)
        self.model = GrassUNet(base=blob.get("base", 16))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.val_iou = float(blob.get("val_iou", float("nan")))
        torch.set_num_threads(threads)

        #: True にすると推論結果を常に「草なし」にする (知覚のアブレーション用).
        #: 経路が本当に知覚由来かを確かめるための破壊試験に使う.
        self.blind = blind
        self.last_rgb: np.ndarray | None = None
        self.last_prob: np.ndarray | None = None
        self.n_inferences = 0

    # ------------------------------------------------------------------
    def observe(self, gmap: GrassMap) -> int:
        """1 回の知覚サイクル. 地図に積算したセル数を返す."""
        self.renderer.update_scene(self.world.data, camera="onboard")
        rgb = self.renderer.render()
        if self.blind:
            prob = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        else:
            with torch.no_grad():
                logits = self.model(preprocess(rgb))
                prob = torch.sigmoid(logits)[0, 0].numpy()
        self.last_rgb = rgb
        self.last_prob = prob
        self.n_inferences += 1

        cam_pos, cam_mat = self.world.camera_pose()
        return gmap.integrate(prob, cam_pos, cam_mat, self.cam)

    def close(self) -> None:
        self.renderer.close()
