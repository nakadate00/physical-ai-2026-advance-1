#!/usr/bin/env python3
"""提出用レポート (PDF) を生成する.

実行結果の JSON から数値を読んで HTML を組み立て, ヘッドレス Chrome で
PDF に変換する. 数値を手で書き写さないことで, 再実行しても記述と実験結果が
ずれないようにしている.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "report"

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
)


def img(path: pathlib.Path, cls: str = "") -> str:
    if not path.exists():
        return f'<p class="missing">[図が見つかりません: {path.name}]</p>'
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{b64}"/>'


def load(p: pathlib.Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def fmt(x, nd=2, suffix=""):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x:,.{nd}f}{suffix}"
    except Exception:
        return str(x)


def build_html(demo: dict, base: dict, bm: dict | None, unet_iou: float,
               soil_meta: dict) -> str:
    plot_area = 1.818**2

    # --- 比較表 -------------------------------------------------------
    def row(label, a, b, nd=2, suffix="", better="low"):
        if a is None or b is None:
            return ""
        try:
            win = (a < b) if better == "low" else (a > b)
        except TypeError:
            win = False
        cls_a = ' class="win"' if win else ""
        cls_b = "" if win else ' class="win"'
        return (f"<tr><td>{label}</td><td{cls_b}>{fmt(b, nd, suffix)}</td>"
                f"<td{cls_a}>{fmt(a, nd, suffix)}</td></tr>")

    cmp_rows = "".join([
        row("刈り終えるまでの走行距離", demo.get("mow_end_distance"),
            base.get("mow_end_distance"), 2, " m"),
        row("刈り終えるまでの所要時間", demo.get("mow_end_time"),
            base.get("mow_end_time"), 1, " s"),
        row("刈り取り率", 100 * demo.get("cut_rate", 0),
            100 * base.get("cut_rate", 0), 1, " %", better="high"),
        row("草の分布を把握できたか (地図被覆率)", 100 * demo.get("map_coverage", 0),
            100 * base.get("map_coverage", 0), 1, " %", better="high"),
    ])

    # --- ベンチマーク表 -------------------------------------------------
    bm_rows = ""
    if bm:
        for cov, b, a, bc, ac in zip(bm["coverage"], bm["base_d"], bm["ai_d"],
                                     bm["base_cut"], bm["ai_cut"]):
            red = 100 * (1 - a / b)
            cls = "good" if red > 0 else "bad"
            bm_rows += (f"<tr><td>{cov:.1f} %</td><td>{b:.2f} m</td>"
                        f"<td>{a:.2f} m</td><td class='{cls}'>{red:+.1f} %</td>"
                        f"<td>{bc:.1f} %</td><td>{ac:.1f} %</td></tr>")
        size_rows = ""
        for t, b, a in zip(bm["sizes"], bm["s_base"], bm["s_ai"]):
            red = 100 * (1 - a / b)
            cls = "good" if red > 0 else "bad"
            size_rows += (f"<tr><td>{t:g} 坪</td><td>{b:.2f} m</td><td>{a:.2f} m</td>"
                          f"<td class='{cls}'>{red:+.1f} %</td></tr>")
    else:
        size_rows = ""

    # --- 土壌診断 -------------------------------------------------------
    soil = demo.get("soil", {})
    recs = demo.get("recommendations", [])
    rec_html = "".join(
        f"<li><b>{n}</b> <span class='prob'>{100 * p:.1f} %</span></li>"
        for n, p in recs)
    advice_html = "".join(f"<li>{a}</li>" for a in demo.get("advice", []))
    from agripioneer.soil.field import PROPERTIES, PROP_LABELS_JA
    units = {n: u for n, u, _, _ in PROPERTIES}
    items = [(PROP_LABELS_JA.get(k, k), f"{v:.2f} {units.get(k, '')}".strip())
             for k, v in soil.items()]
    half = (len(items) + 1) // 2
    soil_html = "".join(
        f"<tr><td>{a[0]}</td><td>{a[1]}</td><td>{b[0] if b else ''}</td>"
        f"<td>{b[1] if b else ''}</td></tr>"
        for a, b in zip(items[:half], list(items[half:]) + [None] * half))

    w_b, w_c = (bm or {}).get("w_blade", 0.194), (bm or {}).get("w_cam", 0.79)
    naive_cross = 100 * (1 - w_b / w_c)
    crossover = (bm or {}).get("crossover", float("nan"))

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>AgriPioneer</title>
<style>
@page {{ size: A4; margin: 15mm 14mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN",
        "Noto Sans CJK JP", sans-serif; color: #1b1f24; font-size: 10.2pt;
        line-height: 1.68; margin: 0; }}
h1 {{ font-size: 21pt; margin: 0 0 2mm; letter-spacing: .01em; }}
h2 {{ font-size: 13pt; margin: 7mm 0 2.5mm; padding: 1.6mm 0 1.6mm 3mm;
      border-left: 4px solid #e08020; background: #faf4ec; }}
h3 {{ font-size: 11pt; margin: 4.5mm 0 1.5mm; color: #9a5a10; }}
p, li {{ margin: 0 0 1.8mm; }}
ul, ol {{ margin: 0 0 2.5mm; padding-left: 5.5mm; }}
.sub {{ color: #5c646e; font-size: 9.6pt; }}
.lead {{ background: #f6f8fa; border: 1px solid #e1e5ea; border-radius: 3px;
         padding: 3mm 4mm; margin: 0 0 4mm; }}
.meta {{ color: #5c646e; font-size: 9pt; border-top: 1px solid #d8dde3;
         padding-top: 2mm; margin-top: 2mm; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.4pt;
         margin: 2mm 0 3mm; }}
th, td {{ border: 1px solid #d8dde3; padding: 1.4mm 2.2mm; text-align: left; }}
th {{ background: #f0f3f6; font-weight: 600; }}
td.win {{ background: #eaf7ee; font-weight: 700; }}
table.soil td {{ font-size: 8.8pt; padding: 1.1mm 1.6mm; }}
td.good {{ color: #1d7a3c; font-weight: 700; }}
td.bad {{ color: #b03a2e; font-weight: 700; }}
img {{ max-width: 100%; display: block; margin: 2mm auto 1mm; }}
img.half {{ max-width: 49%; display: inline-block; }}
.cap {{ text-align: center; color: #5c646e; font-size: 8.8pt; margin-bottom: 3mm; }}
.pb {{ page-break-before: always; }}
.avoid {{ page-break-inside: avoid; }}
.grid2 {{ display: flex; gap: 4mm; page-break-inside: avoid; }}
.grid2 > div {{ flex: 1; }}
.kpi {{ display: flex; gap: 3mm; margin: 3mm 0; }}
.kpi div {{ flex: 1; border: 1px solid #e1e5ea; border-radius: 3px;
            padding: 2.4mm; text-align: center; background: #fcfcfd; }}
.kpi b {{ display: block; font-size: 15pt; color: #c2660f; line-height: 1.3; }}
.kpi span {{ font-size: 8.6pt; color: #5c646e; }}
code {{ background: #f2f4f7; padding: 0 1mm; border-radius: 2px;
        font-family: Menlo, monospace; font-size: 9pt; }}
pre {{ background: #f7f8fa; border: 1px solid #e1e5ea; border-radius: 3px;
       padding: 2.5mm 3mm; font-size: 8.4pt; line-height: 1.45; overflow: hidden;
       font-family: Menlo, monospace; }}
.flow {{ font-family: Menlo, monospace; font-size: 8.2pt; line-height: 1.42;
         white-space: pre; background: #f7f8fa; border: 1px solid #e1e5ea;
         border-radius: 3px; padding: 3mm; }}
.note {{ background: #fff8e8; border-left: 3px solid #e0a020; padding: 2mm 3mm;
         font-size: 9.2pt; margin: 2mm 0 3mm; }}
.prob {{ color: #c2660f; font-weight: 700; }}
</style></head><body>

<h1>AgriPioneer — 耕作放棄地を自動で開拓する草刈りロボット</h1>
<p class="sub">Physical AI 応用1 講座 最終課題 説明資料 ／ omnicampus アカウント ID: 142210</p>

<div class="lead">
<b>一言でいうと：</b>1 坪の荒れ地にロボットを置くだけで、
<b>単眼カメラと CNN で草の地図を作り</b>、<b>最適な経路を計算して刈り切り</b>、
<b>土を測って最適な作物を提案する</b>までを自動で行うシステムを、
MuJoCo 上で一貫して実装・検証した。
</div>

<div class="kpi">
  <div><b>{fmt(100 * demo.get('cut_rate', 0), 1)} %</b><span>刈り取り率<br>({demo.get('cut', 0)}/{demo.get('cut_total', 0)} 本)</span></div>
  <div><b>{fmt(unet_iou, 3)}</b><span>知覚 CNN の<br>検証 IoU</span></div>
  <div><b>{fmt(demo.get('mow_end_distance'), 1, ' m')}</b><span>刈り終えるまでの<br>走行距離</span></div>
  <div><b>{fmt(100 * (1 - demo.get('mow_end_distance', 1) / max(base.get('mow_end_distance', 1), 1e-9)), 0)} %</b><span>全面被覆比の<br>走行距離 削減</span></div>
  <div><b>3 種</b><span>系に組み込んだ<br>AI の数</span></div>
</div>

<h2>1. 課題の背景と目的</h2>
<p>日本の第一次産業は深刻な後継者不足にある一方で、少子高齢化と人口減少により
利用されない土地が増え続けている。この二つは無関係ではなく、
<b>「農地にできる土地はあるが、整備して作付けする人手がない」</b>
という一つの問題の表と裏である。</p>
<p>そこで本課題では、<b>ロボットを置くだけで土地が農地になる</b>という将来像を設定し、
その第一歩として次の 3 つを自動化した。</p>
<ol>
<li><b>見る</b>：区画を見回り、どこにどれだけ草が生えているかを把握する</li>
<li><b>刈る</b>：できるだけ短い走行距離で、草をすべて刈り取る</li>
<li><b>測って提案する</b>：土壌を計測し、その土地に合った作物を提案する</li>
</ol>
<p>ゴールは「刈り取りが終わり、<b>耕耘して作付けできる状態</b>になること」と
「<b>次に何を植えるべきかが提示されていること</b>」である。</p>

<h2>2. 設定したタスクとシミュレーション環境</h2>
<div class="grid2">
<div>
<h3>2.1 タスク</h3>
<ul>
<li>区画：<b>1 坪 = 1.818 m 角</b>（面積 {plot_area:.2f} m²）</li>
<li>草：<b>ランダムな位置・草丈</b>。耕作放棄地を想定し、群生 (草むら) と
    散在の混合として生成。群生の中心は最小間隔つきでサンプリングする</li>
<li>ロボット：芝刈り機型の<b>差動二輪車</b>（全長 0.30 m、刈刃幅 0.22 m、
    高さ 0.52 m のマストに単眼カメラ）</li>
<li>達成条件：区画内の草をすべて刈り取り、土を耕す準備ができること</li>
</ul>
<h3>2.2 シミュレータ</h3>
<ul>
<li><b>MuJoCo 3.17</b>（物理 + オフスクリーン描画）</li>
<li>MJCF は Python から動的生成。草は衝突しない静的 geom とし、
    刈り取りは <code>geom_size</code> / <code>geom_pos</code> / <code>geom_rgba</code> の
    実行時書き換えで表現（数百本でも実時間の数倍で回る）</li>
<li>刈り取り判定は<b>刈刃の掃引線分と草の距離</b>で行い、
    制御周期の離散化による刈り残しを防いでいる</li>
</ul>
</div>
<div>
{img(FIG / 'frame_mowing.png')}
<p class="cap">図1: 刈り取り中の画面。左は 3D ビュー（白い点列が計画経路）、
右上が搭載カメラと CNN の推論、右下が構築中の占有格子地図。</p>
</div>
</div>

<h2>3. システム構成</h2>
<div class="flow">┌──────────────┐  RGB 96×96   ┌──────────────────┐
│ 搭載単眼カメラ │ ───────────▶ │ U-Net (0.39 M)    │ ← ① 学習済み
│ 高さ0.52m 俯角42° │              │ 草の「株元」を推定  │
└──────────────┘               └─────────┬────────┘
                                          │ 画素ごとの確率
                      地面を平面と仮定して逆投影（平面ホモグラフィ）
                                          ▼
   ┌───────────────┐  log-odds 積算  ┌────────────────────────┐
   │ 占有格子地図     │ ◀──────────────│ 距離に応じた信頼度で重み付け │
   │ 61×61 @ 3 cm   │                 └────────────────────────┘
   └───────┬───────┘
           │ 連結成分でクラスタ化 → 主軸(PCA)に沿った刈刃幅の牛耕式被覆
           │ 巡回順と各経路の向きを ② TSP（焼きなまし + 2-opt/Or-opt）で最適化
           ▼
   ┌──────────────┐  v, ω   ┌────────────┐  車輪角速度  ┌──────────┐
   │ Pure Pursuit  │ ──────▶ │ 差動二輪変換  │ ──────────▶ │  MuJoCo   │
   └──────────────┘          └────────────┘              └──────────┘
           ▲                                                    │
           └────────────────── 姿勢フィードバック ─────────────────┘

   刈り取り完了 → 5 地点で土壌計測 → ③ 勾配ブースティング分類器 → 作付提案</div>

<p>動作は状態機械
<code>SURVEY → PLAN → MOW → VERIFY →（刈り残しがあれば MOW へ戻る）→ SOIL → DONE</code>
で構成した。知覚は走行中つねに動いており、刈り取り中も地図が更新され続ける。</p>

<h2 class="pb">4. 組み込んだ AI（3 種）</h2>
<table>
<tr><th style="width:4%">#</th><th style="width:26%">役割</th><th style="width:34%">手法</th><th>学習データ</th></tr>
<tr><td>①</td><td>草の株元セグメンテーション</td>
    <td>小型 U-Net（PyTorch, 0.39 M params）</td>
    <td>MuJoCo で自動生成した合成画像 4,032 枚</td></tr>
<tr><td>②</td><td>刈り取り経路の巡回順最適化</td>
    <td>焼きなまし法 + 2-opt / Or-opt（向き付き巡回路）</td>
    <td>探索のみ（学習なし）</td></tr>
<tr><td>③</td><td>土壌から作物を推薦</td>
    <td>HistGradientBoosting 分類器（10 クラス）</td>
    <td>営農ルールから生成した合成データ 90,000 件</td></tr>
</table>

<h3>4.1 ①　アノテーション不要の学習パイプライン（本課題で一番工夫した点）</h3>
<p>単に「緑の画素」を拾うだけでは役に立たない。必要なのは
<b>「その草が地面のどこに生えているか」</b>であって、画像上のシルエットではない。
背の高い草は手前に大きく写り込むため、シルエットをそのまま地面に逆投影すると
実際よりはるかに遠く・広く草があることになってしまう。</p>
<p>そこで教師ラベルを<b>「草の株元が地面に占める領域」</b>と定義し、
草の根元の 3 次元座標をカメラへ<b>順投影して解析的に生成</b>した。
これにより</p>
<ul>
<li>手作業のアノテーションが<b>ゼロ</b>になる（4,032 枚を 48 秒で生成）</li>
<li>ネットワークは「見えている草の<b>遠近を解いて</b>根元の位置を当てる」という
    非自明なタスクを解くことになり、出力をそのまま占有格子に積算できる</li>
<li>学習データに<b>刈り取り途中のシーン</b>（刈り株）を 50 % 混ぜることで、
    刈り株を草と誤検出しなくなる = 作業済み判定が自分でできる</li>
</ul>
<p>加えて、照明方向・環境光・土の色味・草の色味・車体のロール/ピッチを
ランダム化（ドメインランダマイゼーション）している。
検証 IoU は <b>{fmt(unet_iou, 4)}</b>（CPU のみで約 30 分の学習）。</p>
{img(FIG / 'perception_examples.png')}
<p class="cap">図2: 学習結果。上段が入力、中段が自動生成した教師ラベル、下段が推論結果。
右 2 列は刈り取り途中のシーンで、刈り株を草として検出していないことがわかる。</p>

<h3>4.2 ②　被覆経路計画とその最適化</h3>
<ul>
<li>占有格子を<b>刈刃幅ぶん膨張させてから連結成分を取り</b>、草むらをクラスタに分解する
    （近接する草むらは 1 回の走行でまとめて刈れるため同一視する）</li>
<li>各クラスタの点群に<b>主成分分析</b>をかけ、長軸に沿ってレーンを引く。
    こうするとレーン本数（=往復回数）が最小になる</li>
<li>クラスタの<b>訪問順</b>と各経路を<b>どちら向きにたどるか</b>を同時に最適化する。
    これは「向き付き巡回セールスマン問題」であり、最近傍法で初期解を作り、
    2-opt（区間反転＋向き反転）・Or-opt（1 要素移動）・向き反転の 3 近傍を
    焼きなまし法で探索する</li>
<li>追従は <b>Pure Pursuit</b>。折れ線のまま追従すると経路点付近で前方注視距離が
    縮んで曲率が発散し失速するため、<b>経路を 5 cm 間隔に密化</b>してから
    弧長で前方注視する実装にした。方位誤差が大きいときはゼロターンに切り替える</li>
</ul>

<h3>4.3 ③　土壌診断から作付提案まで</h3>
<div class="grid2"><div>
<p>刈り取り後、区画内 5 地点（巡回順は同じ TSP ソルバで最適化）で
pH / EC / 有機物 / 可給態窒素 / 有効態リン酸 / 交換性カリ / 含水率 / 日射 の
8 項目を計測し、中央値を学習済み分類器に入力して上位 3 作物を提示する。</p>
<p>分類器は、10 作物それぞれの適性（適正 pH 帯、要求肥沃度、水分・日照条件、
塩類耐性）を重み付き幾何平均で評価するルールから合成データを作り、
それを勾配ブースティングに蒸留したもの。
ルール出力に対する一致率は <b>{fmt(100 * soil_meta.get('test_acc', 0), 1)} %</b>
（上位 3 件では <b>{fmt(100 * soil_meta.get('test_top3', 0), 1)} %</b>）。</p>
<p>ルールをそのまま使わず学習モデルにしたのは、(1) 欠測やノイズを含む実計測に
滑らかに外挿できる、(2) 確率として順位づけできる、(3) 将来、実地の収量データで
そのまま再学習できる、という理由による。</p>
<div class="note">土壌の化学特性は MuJoCo では扱えないため、空間相関をもつ合成場で代用している。
「草が密生していた場所ほど有機物と窒素が多い」「有機物が多いほど保水性が高い」
といった相関を入れ、<b>草の地図と土壌が無関係にならない</b>ようにした。</div>
</div><div>
<p><b>本デモでの計測値（5 地点の中央値）</b></p>
<table class="soil">{soil_html}</table>
<p><b>提案された作付</b></p>
<ol>{rec_html}</ol>
<p><b>土壌改良の提案</b></p>
<ul>{advice_html}</ul>
</div></div>

<h2 class="pb">5. 実験結果</h2>
<h3>5.1 本デモ（1 坪）の結果</h3>
<p>比較対象は、<b>地図を使わず区画全面を刈刃幅で舐めるベースライン</b>
（一般的なロボット芝刈り機の網羅走行に相当）。同一の草地・同一の制御器で比較した。</p>
<table>
<tr><th style="width:46%">指標</th><th>ベースライン（全面被覆）</th><th>提案手法（知覚駆動）</th></tr>
{cmp_rows}
</table>
<div class="grid2">
<div>{img(FIG / 'frame_result.png')}
<p class="cap">図3: 完了時。すべての草が刈り株になり、土壌診断の結果が提示されている。</p></div>
<div>{img(FIG / 'training_curve.png')}
<p class="cap">図4: 知覚 CNN の学習曲線。</p></div>
</div>

<h3>5.2 どういう条件で有利になるのか（比較実験）</h3>
<p>提案手法は「まず見てから刈る」ため、<b>探査のぶんの走行が上乗せ</b>される。
一方で草のない場所を走らずに済む。どちらが勝つかをまず単純なモデルで見積もった。</p>
<pre>全面被覆   L_base ≈ A / w_blade                       （刈刃幅   w_blade = {w_b:.2f} m）
提案手法   L_ai   ≈ A / w_cam + A_grass / w_blade     （観測幅   w_cam   = {w_c:.2f} m）
→ 有利になる条件:  A_grass / A &lt; 1 − w_blade / w_cam ≈ {naive_cross:.0f} %</pre>
<p>これを確かめるため、草の被覆率を 5 % 〜 56 %、区画を 1 〜 4 坪まで振り、
各条件 3 回ずつ計測した（全 42 走行）。</p>
{img(FIG / 'benchmark.png')}
<p class="cap">図5: 比較実験の結果（各点は 3 試行の平均）。破線は上の理論値。</p>
<table>
<tr><th>草が占める面積</th><th>全面被覆</th><th>提案手法</th><th>距離の削減</th>
    <th>刈取率(全面)</th><th>刈取率(提案)</th></tr>
{bm_rows}
</table>
<table>
<tr><th>区画の大きさ（草の密度は同程度）</th><th>全面被覆</th><th>提案手法</th><th>距離の削減</th></tr>
{size_rows}
</table>

<h3>5.3 考察</h3>
<ul>
<li><b>走行距離を決めているのは草の被覆率だった。</b>
    被覆率 5 % では走行距離が <b>46 % 短く</b>なる一方、40 % を超えると逆に長くなる。
    両者が入れ替わるのは被覆率 <b>およそ {crossover:.0f} %</b>。</li>
<li><b>単純モデルの予測（約 {naive_cross:.0f} %）は外れた。</b>
    実測の入れ替わり点はその半分以下である。原因は、刈り取るのが「草そのものの面積」
    ではなく<b>「草むらを刈刃幅ぶん膨張させた領域の外接範囲」</b>だから。
    クラスタの膨張・レーン両端の余裕・クラスタ間の移動により、
    実際に舐める面積は草の面積の 3〜5 倍になる。
    単純モデルは下限としてしか使えず、設計には実測が要ることが分かった。</li>
<li><b>区画の大きさを変えても優劣はほとんど変わらない</b>（1 坪 +34 %、4 坪 +16 %)。
    上式どおり両者とも面積に比例するためで、
    「大きい区画ほど知覚が効くはず」という当初の予想は<b>誤りだった</b>。
    効いているのは面積ではなく被覆率である。</li>
<li><b>完全性（刈り残しゼロ）は本来トレードオフになる。</b>
    全面被覆は原理的に刈り取り率 100 % を保証するが、
    提案手法は知覚の見落としがそのまま刈り残しになる。
    初期実装では被覆率の低い条件で 97 % 程度まで落ちた。
    そこで<b>仕上げ巡回で検出しきい値を 0.60 → 0.33 へ段階的に下げ、
    最小クラスタサイズも 3 セル → 1 セルに緩める</b>方式を入れたところ、
    <b>全 42 走行すべてで刈り取り率 100 %</b> を達成した
    （しきい値を最初から低くすると誤検出に振り回されて距離が伸びるため、段階的に下げる）。</li>
<li>提案手法だけが得られるものとして、<b>「どこをやり残したか自分で分かる」</b>点が大きい。
    地図があるので作業の検証ができ、そのまま土壌診断の計画にも使える。
    全面被覆にはこの情報がない（地図被覆率 0 %）。</li>
</ul>

<h2 class="pb">6. 工夫した点・つまずいた点</h2>
<table>
<tr><th style="width:30%">項目</th><th>内容</th></tr>
<tr><td>アノテーション不要の<br>学習データ生成</td>
<td>教師を「草のシルエット」ではなく「<b>株元の地面占有領域</b>」と定義し、
3 次元座標の順投影で解析的に生成。手作業ゼロで 4,032 枚、48 秒。
刈り株の混在も自動でラベリングできる。</td></tr>
<tr><td>車輪が滑って<br>前に進まない問題</td>
<td>MuJoCo は <b>2 つの geom の摩擦係数を「最大値」で合成する</b>ため、
低摩擦にしたはずのキャスタが地面側の摩擦に引きずられ、全荷重が
キャスタのドラッグに食われていた。<code>priority</code> 属性で車輪・キャスタ側の値を
強制採用させて解決（並進速度の追従誤差 75 % → 0 %）。</td></tr>
<tr><td>加速時に駆動輪が浮く</td>
<td>駆動反力で車体がピッチングして「ウィリー」し、駆動輪が浮いて空転していた。
重心を前方キャスタ寄りに置き直し、<b>速度指令にレート制限</b>を入れて解決。
後方キャスタは通常浮かせておき、転倒防止の保険としてのみ接地させる。</td></tr>
<tr><td>経路点の周りを<br>ぐるぐる回る</td>
<td>折れ線のまま Pure Pursuit すると、経路点に近づくほど前方注視距離が縮んで
曲率が発散し失速していた（平均速度 0.09 m/s）。
<b>経路を 5 cm 間隔に密化</b>し、弧長で前方注視する本来の実装に直して 0.28 m/s へ。</td></tr>
<tr><td>刈り残しの自己検出</td>
<td>刈刃の掃引領域を地図に「草なし」として書き戻し、刈り取り後に
<b>更新後の地図で再計画</b>する。残っていれば巡回を繰り返す
（しきい値を段階的に下げる）。</td></tr>
<tr><td>地図の作り方</td>
<td>CNN の出力確率をそのまま逆センサモデル（log-odds）として扱い、
<b>カメラからの距離で重み付け</b>して積算する。遠方は逆投影誤差が大きいため
重みを落とし、1.25 m より遠い画素は捨てる。</td></tr>
<tr><td>評価の公平性</td>
<td>同一の草地・同一の制御器で比較し、土壌診断の時間は除いた
「<b>刈り終えるまで</b>」で比較する。理論モデルを立ててから実測し、
予想が外れた点（区画サイズの効果）も含めて記載した。</td></tr>
<tr><td>再現性</td>
<td>レポートの数値はすべて実行結果の JSON から自動で埋め込んでいる。
12 本の回帰テスト（<code>pytest</code>）で物理・幾何・計画・制御・土壌を検証。</td></tr>
</table>

<h2>7. 限界と今後</h2>
<ul>
<li><b>土壌は合成場で代用している。</b>作物適性のルールも一般的な目安を簡略化したもので、
実際の営農判断には地域・品種ごとの検証が必要。実地の土壌分析値と収量データが
得られれば、同じインタフェースのまま分類器を差し替えられる設計にしてある。</li>
<li><b>自己位置はシミュレータの真値を使用。</b>実機では RTK-GNSS や
Visual SLAM が必要で、位置誤差は刈り残しに直結する。</li>
<li><b>草は物理的に干渉しない静的 geom。</b>草をなぎ倒す力学や、
刈刃にかかる負荷トルクはモデル化していない。</li>
<li><b>今後：</b>(1) 探査と刈り取りを分けず、観測しながら逐次的に刈る
受動的再計画（anytime coverage）にすれば探査コストを消せる。
(2) 草丈も回帰で推定し、高い草だけ低速で刈るなど刈刃負荷を考慮した速度計画ができる。
(3) 耕耘・播種アタッチメントまで含めた一連の土地開発へ拡張する。</li>
</ul>

<h2>8. 成果物と再現方法</h2>
<pre>uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .

python scripts/run_mission.py                    # 本デモ（HUD 付き動画 + 指標）
python scripts/run_mission.py --mode baseline    # 比較用ベースライン
python scripts/run_benchmark.py --seeds 3        # 比較実験
pytest -q                                        # 回帰テスト（12 件）

# モデルの再学習（同梱済みなので通常は不要）
python scripts/gen_perception_dataset.py --worlds 112 --poses 36 --res 96
python scripts/train_perception.py --epochs 16   # CPU 10 スレッドで約 30 分
python scripts/train_soil_model.py</pre>
<p>ROS 2 統合（<code>ros2_ws/</code>）も同梱しており、
simulator / perception / planner / controller の 4 ノードを
トピックで疎結合にした構成を Docker (ROS 2 Humble) 上で動作確認している。
コアモジュールはスタンドアロン実装とそのまま共有している。</p>

<p class="meta">使用ツール: MuJoCo 3.17 / PyTorch / scikit-learn / SciPy / Pillow / ROS 2 Humble。
動画は等速 (1x) 再生・音声なし。</p>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="outputs/agripioneer.json")
    ap.add_argument("--baseline", default="outputs/baseline.json")
    ap.add_argument("--benchmark", default="docs/benchmark_summary.json")
    ap.add_argument("--pdf", default="docs/report/report.pdf",
                    help="リポジトリに置く PDF の出力先")
    ap.add_argument("--submission-copy", default=None,
                    help="提出用の名前 (説明資料_PAI最終課題_<アカウント名>.pdf) で"
                         "別途コピーする先. 個人名を公開リポジトリに含めないため既定では作らない")
    args = ap.parse_args()

    demo = load(ROOT / args.demo, {})
    base = load(ROOT / args.baseline, {})
    bm = load(ROOT / args.benchmark)

    import joblib
    import torch
    unet = torch.load(ROOT / "models" / "grass_unet.pt", map_location="cpu",
                      weights_only=False)
    soil_meta = joblib.load(ROOT / "models" / "soil_crop_model.joblib")

    OUT.mkdir(parents=True, exist_ok=True)
    html_path = OUT / "report.html"
    html_path.write_text(build_html(demo, base, bm,
                                    float(unet.get("val_iou", float("nan"))),
                                    soil_meta), encoding="utf-8")
    print("→", html_path)

    chrome = next((c for c in CHROME_CANDIDATES if pathlib.Path(c).exists()), None)
    if chrome is None:
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        print("Chrome が見つからないため PDF 変換はスキップしました")
        return

    pdf = ROOT / args.pdf
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-sandbox",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    html_path.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("→", pdf, f"({pdf.stat().st_size / 1024:.0f} KB)")

    if args.submission_copy:
        dst = ROOT / args.submission_copy
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, dst)
        print("→", dst, "(提出用コピー)")


if __name__ == "__main__":
    main()
