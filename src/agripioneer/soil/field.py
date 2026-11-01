"""区画の土壌特性を空間的に連続な場としてシミュレートする.

MuJoCo は土壌化学をモデル化しないため, 土壌センサの計測値は本モジュールで
生成した合成場から読み出す (この点はレポートでも明示している).
現実との整合を取るため, 以下の相関を入れてある.

* 草が密生していた場所 = 有機物と可給態窒素が多い (植生由来の還元)
* 有機物が多い場所 = 保水性が高い
* 低い場所 (窪地) = 含水率が高く pH がやや下がる
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..world import GrassField

#: (名称, 単位, 下限, 上限) センサが取得する 8 項目
PROPERTIES: tuple[tuple[str, str, float, float], ...] = (
    ("pH", "", 4.0, 8.0),
    ("EC", "dS/m", 0.05, 1.30),
    ("OM", "%", 0.5, 8.0),
    ("N", "mg/100g", 1.0, 25.0),
    ("P", "mg/100g", 2.0, 80.0),
    ("K", "mg/100g", 5.0, 60.0),
    ("moisture", "%", 10.0, 45.0),
    ("solar", "MJ/m2/d", 8.0, 22.0),
)
PROP_NAMES = tuple(p[0] for p in PROPERTIES)

PROP_LABELS_JA = {
    "pH": "pH",
    "EC": "EC (塩類濃度)",
    "OM": "有機物含量",
    "N": "可給態窒素",
    "P": "有効態リン酸",
    "K": "交換性カリ",
    "moisture": "体積含水率",
    "solar": "日射量",
}


def _smooth_field(
    xy: np.ndarray, rng: np.random.Generator, n_bumps: int = 6, scale: float = 0.55
) -> np.ndarray:
    """RBF の和で滑らかな空間場 (平均 0, 概ね分散 1) を作る."""
    centers = rng.uniform(-1.3, 1.3, size=(n_bumps, 2))
    amps = rng.normal(0.0, 1.0, size=n_bumps)
    d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    f = (amps[None, :] * np.exp(-d2 / (2 * scale**2))).sum(-1)
    s = f.std()
    return f / s if s > 1e-6 else f


@dataclass
class SoilSample:
    xy: np.ndarray
    values: dict[str, float]


class SoilField:
    """区画全体の土壌特性場. `probe()` でセンサ計測を模擬する."""

    def __init__(self, cfg: Config, grass: GrassField, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self._grass_xy = grass.xy.copy()
        self._grass_h = grass.height0.copy()

        # 各項目の空間変動パターン (区画ごとに異なる地力のむら)
        self._bumps = {name: (rng.uniform(-1.3, 1.3, size=(6, 2)),
                              rng.normal(0.0, 1.0, size=6)) for name in PROP_NAMES}
        # 区画全体のベースライン (0-1 の相対値). 耕作放棄地を想定して
        # pH はやや酸性, 養分はやや不足寄りに設定する.
        self._base = {
            "pH": rng.uniform(0.22, 0.46),
            "EC": rng.uniform(0.12, 0.35),
            "OM": rng.uniform(0.25, 0.55),
            "N": rng.uniform(0.15, 0.45),
            "P": rng.uniform(0.10, 0.40),
            "K": rng.uniform(0.20, 0.50),
            "moisture": rng.uniform(0.35, 0.62),
            "solar": rng.uniform(0.55, 0.85),
        }
        self._noise = {
            "pH": 0.06, "EC": 0.03, "OM": 0.12, "N": 0.5,
            "P": 1.6, "K": 1.2, "moisture": 0.8, "solar": 0.25,
        }

    # ------------------------------------------------------------------
    def _grass_density(self, xy: np.ndarray) -> np.ndarray:
        """その地点の草のバイオマス密度 (0-1 に正規化)."""
        d2 = ((xy[:, None, :] - self._grass_xy[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * 0.18**2)) * self._grass_h[None, :]
        s = w.sum(-1)
        return np.clip(s / 1.4, 0.0, 1.0)

    def _spatial(self, name: str, xy: np.ndarray) -> np.ndarray:
        centers, amps = self._bumps[name]
        d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        f = (amps[None, :] * np.exp(-d2 / (2 * 0.55**2))).sum(-1)
        s = f.std() if f.std() > 1e-6 else 1.0
        return f / s

    def true_values(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        """ノイズを含まない真値 (可視化・検証用)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        g = self._grass_density(xy)
        out: dict[str, np.ndarray] = {}
        rel: dict[str, np.ndarray] = {}
        for name, unit, lo, hi in PROPERTIES:
            r = self._base[name] + 0.17 * self._spatial(name, xy)
            if name == "OM":
                r = r + 0.30 * g  # 草が多い場所は有機物が蓄積している
            elif name == "N":
                r = r + 0.26 * g
            elif name == "solar":
                r = r - 0.18 * g  # 草丈が高い場所は日照が遮られがち
            rel[name] = r
        # 二次的な相関: 有機物が多いほど保水性が高い
        rel["moisture"] = rel["moisture"] + 0.22 * (rel["OM"] - 0.45)
        # 含水が多いほど還元的で pH が下がる
        rel["pH"] = rel["pH"] - 0.10 * (rel["moisture"] - 0.5)

        for name, unit, lo, hi in PROPERTIES:
            out[name] = lo + (hi - lo) * np.clip(rel[name], 0.0, 1.0)
        return out

    def probe(self, xy: np.ndarray) -> SoilSample:
        """1 地点の土壌センサ計測 (計測ノイズ付き)."""
        xy = np.asarray(xy, dtype=float)
        t = self.true_values(xy[None, :])
        vals = {}
        for name, unit, lo, hi in PROPERTIES:
            v = float(t[name][0]) + float(self.rng.normal(0.0, self._noise[name]))
            vals[name] = float(np.clip(v, lo, hi))
        return SoilSample(xy=xy.copy(), values=vals)


def aggregate(samples: list[SoilSample]) -> dict[str, float]:
    """複数地点の計測を中央値で代表値にまとめる (外れ値に強い)."""
    return {
        name: float(np.median([s.values[name] for s in samples])) for name in PROP_NAMES
    }


def to_vector(values: dict[str, float]) -> np.ndarray:
    return np.array([values[n] for n in PROP_NAMES], dtype=float)
