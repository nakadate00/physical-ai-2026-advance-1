"""草の株元セグメンテーション用の小型 U-Net.

CPU だけで数分〜十数分で学習でき, 推論も 1 フレーム 10 ms 程度で済む規模に
抑えてある (パラメータ約 0.5 M). 入力 128x128 RGB, 出力 128x128 のロジット.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class GrassUNet(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        b = base
        self.enc1 = _block(3, b)
        self.enc2 = _block(b, b * 2)
        self.enc3 = _block(b * 2, b * 4)
        self.bott = _block(b * 4, b * 6)
        self.dec3 = _block(b * 6 + b * 4, b * 4)
        self.dec2 = _block(b * 4 + b * 2, b * 2)
        self.dec1 = _block(b * 2 + b, b)
        self.head = nn.Conv2d(b, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))

        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return self.head(d1)


#: 画像の正規化定数 (0-1 スケール後に適用)
IMG_MEAN = (0.43, 0.40, 0.30)
IMG_STD = (0.22, 0.21, 0.19)


def preprocess(rgb_uint8, device: str = "cpu") -> torch.Tensor:
    """HWC uint8 (単枚) または NHWC uint8 を正規化済みテンソルにする."""
    import numpy as np

    arr = np.asarray(rgb_uint8)
    single = arr.ndim == 3
    if single:
        arr = arr[None]
    x = torch.from_numpy(np.ascontiguousarray(arr)).float().div_(255.0)
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(IMG_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMG_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1.0 - num / den).mean()
