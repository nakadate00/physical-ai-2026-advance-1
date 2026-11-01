"""巡回順の最適化.

草むら (クラスタ) ごとに生成した刈り取り経路を, どの順番で・どちら向きに
たどると総移動距離が最小になるかを解く. 各クラスタは始点と終点を持ち,
経路を逆向きにたどることもできるため, 向き付き巡回路問題になる.
2-opt / Or-opt 近傍による局所探索を, 焼きなまし法で繰り返す.
"""

from __future__ import annotations

import numpy as np


def _tour_cost(
    order: np.ndarray, flip: np.ndarray, ends: np.ndarray, start: np.ndarray
) -> float:
    """ends: (K, 2, 2) 各クラスタの [始点, 終点]. start: 出発地点 xy."""
    cur = start
    total = 0.0
    for k, f in zip(order, flip):
        a, b = (ends[k, 1], ends[k, 0]) if f else (ends[k, 0], ends[k, 1])
        total += float(np.linalg.norm(a - cur))
        cur = b
    return total


def solve(
    ends: np.ndarray,
    start: np.ndarray,
    iters: int = 6000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """焼きなまし + 2-opt で (訪問順, 反転フラグ, 総移動距離) を返す."""
    k = len(ends)
    if k == 0:
        return np.array([], dtype=int), np.array([], dtype=bool), 0.0
    if k == 1:
        best = min(
            (_tour_cost(np.array([0]), np.array([f]), ends, start), f) for f in (False, True)
        )
        return np.array([0]), np.array([best[1]]), best[0]

    rng = np.random.default_rng(seed)

    # 初期解: 最近傍法 (向きも含めて最も近い端点へ貪欲に進む)
    order, flip = [], []
    remain = set(range(k))
    cur = start.copy()
    while remain:
        best = None
        for i in remain:
            for f in (False, True):
                a = ends[i, 1] if f else ends[i, 0]
                d = float(np.linalg.norm(a - cur))
                if best is None or d < best[0]:
                    best = (d, i, f)
        _, i, f = best
        order.append(i)
        flip.append(f)
        cur = ends[i, 0] if f else ends[i, 1]
        remain.discard(i)
    order = np.array(order)
    flip = np.array(flip)

    cost = _tour_cost(order, flip, ends, start)
    best_order, best_flip, best_cost = order.copy(), flip.copy(), cost

    T0, T1 = max(cost * 0.08, 1e-3), 1e-4
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(iters - 1, 1))
        o, f = order.copy(), flip.copy()
        m = rng.integers(0, 3)
        if m == 0 and k >= 2:  # 2-opt: 区間反転 (向きも反転する)
            i, j = sorted(rng.choice(k, size=2, replace=False))
            o[i : j + 1] = o[i : j + 1][::-1]
            f[i : j + 1] = ~f[i : j + 1][::-1]
        elif m == 1:  # Or-opt: 1 要素の移動
            i = rng.integers(0, k)
            j = rng.integers(0, k)
            v, vf = o[i], f[i]
            o = np.delete(o, i)
            f = np.delete(f, i)
            o = np.insert(o, j, v)
            f = np.insert(f, j, vf)
        else:  # 向きだけ反転
            i = rng.integers(0, k)
            f[i] = ~f[i]

        c = _tour_cost(o, f, ends, start)
        if c < cost or rng.random() < np.exp(-(c - cost) / max(T, 1e-9)):
            order, flip, cost = o, f, c
            if c < best_cost:
                best_order, best_flip, best_cost = o.copy(), f.copy(), c

    return best_order, best_flip, best_cost
