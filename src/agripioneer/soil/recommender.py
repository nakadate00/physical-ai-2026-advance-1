"""学習済みモデルによる推奨作物の推論."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import joblib
import numpy as np

from .crops import CROPS, CROP_NAMES, Crop, improvement_advice
from .field import PROP_NAMES

DEFAULT_MODEL = pathlib.Path(__file__).resolve().parents[3] / "models" / "soil_crop_model.joblib"


@dataclass
class Recommendation:
    crop: Crop
    probability: float


class CropRecommender:
    def __init__(self, path: str | pathlib.Path | None = None) -> None:
        p = pathlib.Path(path) if path else DEFAULT_MODEL
        if not p.exists():
            raise FileNotFoundError(
                f"学習済み土壌モデルが見つかりません: {p}\n"
                "`python scripts/train_soil_model.py` を実行してください"
            )
        blob = joblib.load(p)
        self.model = blob["model"]
        self.classes: list[str] = blob["classes"]
        self.features: list[str] = blob["features"]
        self.test_acc: float = float(blob.get("test_acc", float("nan")))

    def recommend(self, values: dict[str, float], k: int = 3) -> list[Recommendation]:
        x = np.array([[values[n] for n in self.features]], dtype=float)
        proba = self.model.predict_proba(x)[0]
        idx = np.argsort(proba)[::-1][:k]
        return [
            Recommendation(crop=CROPS[CROP_NAMES.index(self.classes[i])],
                           probability=float(proba[i]))
            for i in idx
        ]

    @staticmethod
    def advice(values: dict[str, float], crop: Crop) -> list[str]:
        return improvement_advice(values, crop)
