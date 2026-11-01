#!/usr/bin/env python3
"""土壌計測値 -> 推奨作物 の分類モデルを学習する.

作物ごとの適性ルール (agripioneer.soil.crops) から合成データを作り,
勾配ブースティング決定木に蒸留する. ルールをそのまま使わず学習モデルに
するのは, (1) 欠測・ノイズを含む実計測に対して滑らかに外挿できる,
(2) 確率として上位候補を順位付けできる, (3) 将来, 実地の収量データで
そのまま再学習できる, という利点があるため.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from agripioneer.soil.crops import CROP_NAMES, rule_scores
from agripioneer.soil.field import PROPERTIES, PROP_NAMES


def synth(n: int, rng: np.random.Generator):
    """合成土壌サンプルと, ルールから導いた推奨作物ラベルを作る.

    一様分布だけでは現実にはまず出ない極端な土壌ばかりになるため,
    「よくある農地の値域に寄せた分布 (7 割)」と「範囲全体の一様分布 (3 割)」の
    混合から特徴量を引く. 後者は未知の土壌への外挿性能を確保するため.
    """
    v = {}
    n_typ = int(n * 0.7)
    for name, unit, lo, hi in PROPERTIES:
        # 中央寄りのベータ分布 + 一様分布の混合
        typ = lo + (hi - lo) * rng.beta(2.4, 2.4, size=n_typ)
        uni = rng.uniform(lo, hi, size=n - n_typ)
        v[name] = np.concatenate([typ, uni])

    s = rule_scores(v)
    ls = np.log(np.clip(s, 1e-9, None))
    # 学習ラベルには Gumbel ノイズを混ぜ, 境界付近の判断を滑らかにする.
    # 評価はノイズなしのルール出力 (y_clean) に対して行う.
    g = rng.gumbel(0.0, 0.10, size=s.shape)
    y = np.argmax(ls + g, axis=1)
    y_clean = np.argmax(ls, axis=1)
    X = np.stack([v[n_] for n_ in PROP_NAMES], axis=1)
    return X, y, y_clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90000)
    ap.add_argument("--out", default="models/soil_crop_model.joblib")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    X, y, y_clean = synth(args.n, rng)
    keep = np.bincount(y, minlength=len(CROP_NAMES))
    print("class counts:", dict(zip(CROP_NAMES, keep.tolist())))

    Xtr, Xte, ytr, yte, _, yte_clean = train_test_split(
        X, y, y_clean, test_size=0.2, random_state=0, stratify=y)
    clf = HistGradientBoostingClassifier(
        max_iter=350, learning_rate=0.09, max_depth=None, max_leaf_nodes=31,
        l2_regularization=0.5, early_stopping=True, validation_fraction=0.1,
        random_state=0,
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)
    labels = np.arange(len(CROP_NAMES))
    acc = accuracy_score(yte_clean, p.argmax(1))
    top3 = top_k_accuracy_score(yte_clean, p, k=3, labels=labels)
    acc_noisy = accuracy_score(yte, p.argmax(1))
    print(f"[vs ルール出力] accuracy = {acc:.4f}   top-3 accuracy = {top3:.4f}")
    print(f"[vs ノイズ付きラベル] accuracy = {acc_noisy:.4f} "
          f"(ラベル自体に意図的な揺らぎを入れているため上限が低い)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": list(PROP_NAMES),
                 "classes": list(CROP_NAMES), "test_acc": acc, "test_top3": top3}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
