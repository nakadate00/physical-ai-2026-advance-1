"""作物ごとの土壌適性ルール.

公開されている栽培の目安 (適正 pH, 要求肥沃度, 水分・日照条件) を
簡略化した適性関数として実装したもの. これを使って合成データを作り,
機械学習モデル (勾配ブースティング) に蒸留する.

注意: ここでの数値はあくまで一般的な目安を模した簡易モデルであり,
実際の営農判断に用いるには地域・品種ごとの検証が必要である.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _band(x: float | np.ndarray, lo: float, hi: float, soft: float) -> np.ndarray:
    """[lo, hi] の内側で 1, 外側で soft 幅をかけて 0 に落ちる台形メンバシップ."""
    x = np.asarray(x, dtype=float)
    left = np.clip((x - (lo - soft)) / soft, 0.0, 1.0)
    right = np.clip(((hi + soft) - x) / soft, 0.0, 1.0)
    return np.minimum(left, right)


@dataclass
class Crop:
    name: str
    ph: tuple[float, float]
    ph_soft: float = 0.6
    #: 必要な可給態窒素 [mg/100g] の下限・上限
    n: tuple[float, float] = (3.0, 20.0)
    p: tuple[float, float] = (5.0, 70.0)
    k: tuple[float, float] = (8.0, 55.0)
    om: tuple[float, float] = (1.0, 7.5)
    moisture: tuple[float, float] = (15.0, 40.0)
    solar: tuple[float, float] = (10.0, 22.0)
    #: 塩類集積への弱さ (EC がこの値を超えると急に不適)
    ec_max: float = 0.9
    note: str = ""
    #: 耕作放棄地の初年度に向くか
    pioneer: bool = False
    #: 収益性・需要を加味した選好重み. 緑肥は「育つが売り物ではない」ため低い
    value: float = 1.0

    def score(self, v: dict[str, np.ndarray]) -> np.ndarray:
        """0-1 の適性スコア.

        各項目のメンバシップを単純に掛け合わせると, 1 項目でも外れた瞬間に
        0 になり「どの作物も不適」ばかりになる. 実際の営農では不足分を施肥で
        補えるため, 重み付き幾何平均を取り, 下限 0.05 を置いて緩やかに評価する.
        pH は矯正に手間と時間がかかるため重みを大きくしている.
        """
        floor = 0.05
        terms = (
            (2.2, _band(v["pH"], *self.ph, self.ph_soft)),
            (1.0, _band(v["N"], *self.n, 4.0)),
            (0.7, _band(v["P"], *self.p, 12.0)),
            (0.7, _band(v["K"], *self.k, 10.0)),
            (1.0, _band(v["OM"], *self.om, 1.2)),
            (1.3, _band(v["moisture"], *self.moisture, 6.0)),
            (1.1, _band(v["solar"], *self.solar, 3.0)),
        )
        wsum = sum(w for w, _ in terms)
        acc = sum(w * np.log(t + floor) for w, t in terms) / wsum
        s = np.exp(acc)
        # 塩類集積は施肥で救えないので, 上限超過は乗算ゲートで厳しく効かせる
        s = s * np.clip((self.ec_max + 0.25 - v["EC"]) / 0.25, 0.0, 1.0)
        return s * self.value


CROPS: tuple[Crop, ...] = (
    Crop("ブルーベリー", ph=(4.3, 5.3), ph_soft=0.35, n=(2.0, 12.0), p=(3.0, 40.0),
         k=(5.0, 35.0), om=(3.0, 8.0), moisture=(22.0, 42.0), solar=(12.0, 22.0),
         ec_max=0.5, value=1.06, note="強酸性土壌をそのまま活かせる. 中和不要で初期費用が小さい"),
    Crop("ジャガイモ", ph=(5.0, 6.0), n=(4.0, 16.0), p=(10.0, 70.0), k=(15.0, 55.0),
         om=(1.5, 6.0), moisture=(15.0, 32.0), solar=(12.0, 22.0), ec_max=0.8,
         value=1.00, note="弱酸性を好み, そうか病回避のため石灰施用を控えられる"),
    Crop("サツマイモ", ph=(5.5, 6.5), n=(1.0, 9.0), p=(5.0, 50.0), k=(18.0, 60.0),
         om=(0.5, 4.5), moisture=(12.0, 28.0), solar=(14.0, 22.0), ec_max=0.7,
         value=0.94, note="痩せ地・乾燥に強い. 窒素が多いとつるぼけする", pioneer=True),
    Crop("ソバ", ph=(5.0, 6.5), n=(1.0, 10.0), p=(3.0, 45.0), k=(5.0, 45.0),
         om=(0.5, 5.0), moisture=(13.0, 33.0), solar=(11.0, 22.0), ec_max=0.7,
         value=0.80, note="生育期間が短く地力が低くても収穫できる. 遊休地の初年度に定番",
         pioneer=True),
    Crop("ダイコン", ph=(5.5, 6.8), n=(5.0, 18.0), p=(10.0, 70.0), k=(12.0, 55.0),
         om=(1.5, 6.0), moisture=(16.0, 34.0), solar=(11.0, 22.0), ec_max=0.8,
         value=1.00, note="深く耕せる砕土性の良い畑に向く"),
    Crop("エダマメ", ph=(6.0, 6.8), n=(1.0, 11.0), p=(8.0, 60.0), k=(10.0, 50.0),
         om=(1.0, 5.5), moisture=(16.0, 36.0), solar=(13.0, 22.0), ec_max=0.7,
         value=0.98, note="根粒菌が窒素を固定するため窒素肥料が少なくて済む", pioneer=True),
    Crop("トマト", ph=(6.0, 6.8), n=(7.0, 20.0), p=(20.0, 80.0), k=(18.0, 60.0),
         om=(2.0, 7.0), moisture=(14.0, 30.0), solar=(15.0, 22.0), ec_max=0.9,
         value=1.08, note="日射と排水の良さが要る. 肥沃度が高い区画向き"),
    Crop("ネギ", ph=(6.0, 7.2), n=(6.0, 19.0), p=(10.0, 70.0), k=(12.0, 55.0),
         om=(1.5, 6.5), moisture=(15.0, 33.0), solar=(11.0, 22.0), ec_max=0.9,
         value=1.00, note="中性寄りの土壌を好み, 連作障害が出にくい"),
    Crop("ホウレンソウ", ph=(6.5, 7.5), n=(8.0, 22.0), p=(15.0, 80.0), k=(15.0, 60.0),
         om=(2.0, 7.5), moisture=(20.0, 40.0), solar=(10.0, 20.0), ec_max=0.6,
         value=1.02, note="酸性に非常に弱い. 石灰で pH を上げてから作付けする"),
    Crop("ライムギ(緑肥)", ph=(4.5, 7.5), ph_soft=0.9, n=(1.0, 22.0), p=(2.0, 80.0),
         k=(5.0, 60.0), om=(0.5, 8.0), moisture=(10.0, 42.0), solar=(8.0, 22.0),
         ec_max=1.1,
         value=0.50, note="適応範囲が極めて広い. まず地力を回復させたい区画に蒔く",
         pioneer=True),
)

CROP_NAMES: tuple[str, ...] = tuple(c.name for c in CROPS)


def rule_scores(values: dict[str, np.ndarray]) -> np.ndarray:
    """(N, n_crops) のルールベース適性スコア."""
    return np.stack([c.score(values) for c in CROPS], axis=-1)


# ------------------------------------------------------------ 土壌改良の助言

def improvement_advice(v: dict[str, float], target: Crop) -> list[str]:
    """計測値と目標作物から, ルールベースで改良作業を提案する."""
    tips: list[str] = []
    lo, hi = target.ph
    if v["pH"] < lo - 0.1:
        # 苦土石灰 100 g/m2 で概ね pH 0.5 上がる, という現場の目安を使う
        need = (lo + 0.2 - v["pH"]) / 0.5 * 100.0
        tips.append(f"pH が {v['pH']:.1f} と低い → 苦土石灰を約 {need:.0f} g/m2 施用して "
                    f"pH {lo:.1f}〜{hi:.1f} へ矯正")
    elif v["pH"] > hi + 0.1:
        tips.append(f"pH が {v['pH']:.1f} と高い → ピートモス等で酸度を戻す")
    if v["OM"] < 3.0:
        tips.append(f"有機物が {v['OM']:.1f}% と不足 → 完熟堆肥 2〜3 kg/m2 を鋤き込み")
    if v["N"] < target.n[0] + 2.0:
        tips.append(f"可給態窒素が {v['N']:.1f} mg/100g と少ない → 元肥で窒素を補給")
    if v["P"] < target.p[0] + 5.0:
        tips.append(f"有効態リン酸が {v['P']:.0f} mg/100g と少ない → 過リン酸石灰を追加")
    if v["EC"] > target.ec_max:
        tips.append(f"EC が {v['EC']:.2f} dS/m と高い → 湛水による除塩を検討")
    if v["moisture"] > 36.0:
        tips.append(f"含水率 {v['moisture']:.0f}% と過湿 → 明渠・暗渠で排水を改善")
    if not tips:
        tips.append("主要項目はいずれも適正範囲内. 耕耘後ただちに作付け可能")
    return tips
