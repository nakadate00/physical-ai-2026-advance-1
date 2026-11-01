"""動画フレームの合成 (HUD).

MuJoCo のレンダリング結果に, 搭載カメラ映像・CNN のセグメンテーション結果・
占有格子地図・計画経路・進捗ステータスを重ねて 1 枚のフレームにする.
毎フレーム matplotlib を呼ぶと遅いため, 描画はすべて numpy と Pillow で行う.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ フォント

FONT_CANDIDATES = (
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
)


def _find_font() -> str | None:
    for p in FONT_CANDIDATES:
        if pathlib.Path(p).exists():
            return p
    return None


class Fonts:
    """日本語を含む HUD 用フォント一式."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or _find_font()
        self._cache: dict[int, ImageFont.FreeTypeFont] = {}

    def __call__(self, size: int) -> ImageFont.ImageFont:
        if size not in self._cache:
            if self.path:
                self._cache[size] = ImageFont.truetype(self.path, size)
            else:  # 日本語フォントが無い環境へのフォールバック
                self._cache[size] = ImageFont.load_default(size)
        return self._cache[size]


# ------------------------------------------------------------------ 配色

BG = (18, 21, 26)
PANEL = (28, 32, 39)
LINE = (58, 64, 74)
FG = (232, 236, 242)
MUTED = (146, 156, 170)
ACCENT = (255, 168, 38)
GREEN = (86, 198, 122)
BLUE = (84, 165, 246)
RED = (238, 102, 94)
PURPLE = (186, 134, 252)

PHASE_COLORS = {
    "SURVEY": BLUE,
    "PLAN": PURPLE,
    "MOW": ACCENT,
    "SOIL": GREEN,
    "DONE": GREEN,
}

# ------------------------------------------------------------------ 画像操作


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def heat_rgb(p: np.ndarray) -> np.ndarray:
    """0-1 の確率マップを「土色 → 黄 → 緑」のカラーマップで着色する."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    c0 = np.array([62, 48, 36])  # 草なし (土)
    c1 = np.array([150, 130, 48])  # 不確か
    c2 = np.array([74, 200, 104])  # 草あり
    out = np.where(
        (p < 0.5)[..., None],
        c0 + (c1 - c0) * (p / 0.5)[..., None],
        c1 + (c2 - c1) * ((p - 0.5) / 0.5)[..., None],
    )
    return out.astype(np.uint8)


# ------------------------------------------------------------------ HUD


@dataclass
class HudState:
    """1 フレーム分の表示内容."""

    phase: str = "SURVEY"
    phase_label: str = "探査中"
    sim_time: float = 0.0
    distance: float = 0.0
    cut: int = 0
    cut_total: int = 0
    map_coverage: float = 0.0
    message: str = ""
    message2: str = ""
    onboard_rgb: np.ndarray | None = None
    seg_prob: np.ndarray | None = None
    grid_prob: np.ndarray | None = None
    planned_path: np.ndarray | None = None
    travelled: np.ndarray | None = None
    robot_xy: np.ndarray | None = None
    robot_yaw: float = 0.0
    soil_points: list[np.ndarray] = field(default_factory=list)
    clusters: list[np.ndarray] = field(default_factory=list)
    report_lines: list[tuple[str, str]] = field(default_factory=list)


class HudRenderer:
    """フレーム合成器. 解像度 1600x900 固定."""

    W, H = 1600, 900
    MAIN_W, MAIN_H = 1120, 720
    SIDE_W = 480
    CAM_H = 240
    MAP_H = 480
    BAR_H = 180

    def __init__(self, plot_half: float, font_path: str | None = None) -> None:
        self.fonts = Fonts(font_path)
        self.plot_half = plot_half

    # ------------------------------------------------------------------
    def _map_to_px(self, xy: np.ndarray, size: int, pad: int) -> np.ndarray:
        """世界座標 -> 地図パネルの画素座標."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        s = (size - 2 * pad) / (2 * self.plot_half)
        u = pad + (xy[:, 0] + self.plot_half) * s
        v = pad + (self.plot_half - xy[:, 1]) * s
        return np.stack([u, v], axis=-1)

    # ------------------------------------------------------------------
    def _draw_map_panel(self, st: HudState) -> Image.Image:
        size = self.MAP_H
        pad = 26
        img = Image.new("RGB", (self.SIDE_W, size), PANEL)
        d = ImageDraw.Draw(img)
        x0 = (self.SIDE_W - size) // 2

        if st.grid_prob is not None:
            g = heat_rgb(st.grid_prob)[::-1]  # 行 0 を下 (y 最小) にする
            tile = Image.fromarray(g).resize((size - 2 * pad, size - 2 * pad), Image.NEAREST)
            img.paste(tile, (x0 + pad, pad))
        d.rectangle([x0 + pad - 1, pad - 1, x0 + pad + size - 2 * pad, pad + size - 2 * pad],
                    outline=(120, 108, 60), width=2)

        def px(p):
            q = self._map_to_px(p, size, pad)
            q[:, 0] += x0
            return q

        # 計画経路
        if st.planned_path is not None and len(st.planned_path) >= 2:
            q = px(st.planned_path)
            d.line([tuple(v) for v in q], fill=(255, 255, 255, 180), width=2, joint="curve")
        # 走行軌跡
        if st.travelled is not None and len(st.travelled) >= 2:
            q = px(st.travelled)
            d.line([tuple(v) for v in q], fill=ACCENT, width=3, joint="curve")
        # 土壌サンプリング点
        for p in st.soil_points:
            u, v = px(np.asarray(p))[0]
            d.ellipse([u - 6, v - 6, u + 6, v + 6], outline=GREEN, width=2)
            d.line([u - 9, v, u + 9, v], fill=GREEN, width=1)
            d.line([u, v - 9, u, v + 9], fill=GREEN, width=1)
        # ロボット
        if st.robot_xy is not None:
            u, v = px(np.asarray(st.robot_xy))[0]
            c, s = np.cos(-st.robot_yaw), np.sin(-st.robot_yaw)
            tri = [(13, 0), (-8, 8), (-8, -8)]
            pts = [(u + c * a - s * b, v + s * a + c * b) for a, b in tri]
            d.polygon(pts, fill=RED, outline=(255, 255, 255))

        d.text((14, 6), "草の占有格子地図 (CNN 推定)", font=self.fonts(17), fill=FG)
        return img

    # ------------------------------------------------------------------
    def _draw_cam_panel(self, st: HudState) -> Image.Image:
        img = Image.new("RGB", (self.SIDE_W, self.CAM_H), PANEL)
        d = ImageDraw.Draw(img)
        s = self.CAM_H - 28
        if st.onboard_rgb is not None:
            img.paste(Image.fromarray(resize(st.onboard_rgb, s, s)), (6, 22))
        if st.seg_prob is not None:
            over = heat_rgb(st.seg_prob)
            img.paste(Image.fromarray(resize(over, s, s)), (self.SIDE_W - s - 6, 22))
        d.text((8, 3), "搭載カメラ", font=self.fonts(16), fill=MUTED)
        d.text((self.SIDE_W - s - 4, 3), "U-Net 推論 (草の株元)", font=self.fonts(16), fill=MUTED)
        return img

    # ------------------------------------------------------------------
    def _draw_bar(self, st: HudState) -> Image.Image:
        img = Image.new("RGB", (self.W, self.BAR_H), PANEL)
        d = ImageDraw.Draw(img)
        col = PHASE_COLORS.get(st.phase, FG)

        d.rectangle([0, 0, self.W, 4], fill=col)
        d.text((26, 18), "AgriPioneer", font=self.fonts(30), fill=FG)
        d.text((26, 58), "耕作放棄地 自動開拓ロボット", font=self.fonts(18), fill=MUTED)
        d.text((26, 88), "1 坪 (1.82 m 角) の自動草刈り → 土壌診断 → 作付提案",
               font=self.fonts(16), fill=MUTED)

        # フェーズ表示
        d.rounded_rectangle([330, 30, 620, 84], radius=12, fill=col)
        d.text((350, 40), st.phase_label, font=self.fonts(30), fill=(16, 18, 22))

        def metric(x: int, label: str, value: str, color=FG) -> None:
            d.text((x, 26), label, font=self.fonts(17), fill=MUTED)
            d.text((x, 48), value, font=self.fonts(32), fill=color)

        metric(660, "経過時間", f"{st.sim_time:5.1f} s")
        metric(820, "走行距離", f"{st.distance:5.2f} m")
        pct = 0 if st.cut_total == 0 else 100.0 * st.cut / st.cut_total
        metric(990, "刈り取り", f"{st.cut}/{st.cut_total}", GREEN if pct > 99 else ACCENT)
        metric(1180, "地図被覆率", f"{100 * st.map_coverage:4.1f} %", BLUE)

        # 刈り取り進捗バー
        bx0, bx1, by = 660, 1420, 112
        d.rounded_rectangle([bx0, by, bx1, by + 20], radius=10, fill=(44, 50, 60))
        if pct > 0:
            d.rounded_rectangle([bx0, by, bx0 + (bx1 - bx0) * pct / 100.0, by + 20],
                                radius=10, fill=GREEN)
        d.text((bx0, by + 26), f"刈り取り進捗 {pct:5.1f} %", font=self.fonts(16), fill=MUTED)

        d.text((1450, 26), "MuJoCo", font=self.fonts(17), fill=MUTED)
        d.text((1450, 48), "simulation", font=self.fonts(17), fill=MUTED)
        d.text((1450, 90), "等速再生 (1x)", font=self.fonts(16), fill=MUTED)
        d.text((1450, 112), "音声なし", font=self.fonts(16), fill=MUTED)
        return img

    # ------------------------------------------------------------------
    def compose(self, main_rgb: np.ndarray, st: HudState) -> np.ndarray:
        frame = Image.new("RGB", (self.W, self.H), BG)
        frame.paste(Image.fromarray(resize(main_rgb, self.MAIN_W, self.MAIN_H)), (0, 0))
        frame.paste(self._draw_cam_panel(st), (self.MAIN_W, 0))
        frame.paste(self._draw_map_panel(st), (self.MAIN_W, self.CAM_H))
        frame.paste(self._draw_bar(st), (0, self.MAIN_H))

        d = ImageDraw.Draw(frame)
        d.line([self.MAIN_W, 0, self.MAIN_W, self.MAIN_H], fill=LINE, width=2)
        d.line([self.MAIN_W, self.CAM_H, self.W, self.CAM_H], fill=LINE, width=2)

        if st.message:
            self._draw_caption(frame, st)
        # 最終レポート (土壌診断結果) を左下にオーバーレイ
        if st.report_lines:
            self._draw_report(frame, st.report_lines)
        return np.asarray(frame)

    # ------------------------------------------------------------------
    def _draw_caption(self, frame: Image.Image, st: HudState) -> None:
        """メインビュー下端に字幕帯を重ねる."""
        h = 74 if st.message2 else 48
        band = Image.new("RGBA", (self.MAIN_W, h), (12, 14, 18, 205))
        d = ImageDraw.Draw(band)
        col = PHASE_COLORS.get(st.phase, FG)
        d.rectangle([0, 0, 5, h], fill=col)
        d.text((20, 9), st.message, font=self.fonts(23), fill=FG)
        if st.message2:
            d.text((20, 43), st.message2, font=self.fonts(19), fill=MUTED)
        frame.paste(band, (0, self.MAIN_H - h), band)

    # ------------------------------------------------------------------
    def _draw_report(self, frame: Image.Image, lines: list[tuple[str, str]]) -> None:
        w, h = 700, 54 + 30 * len(lines)
        x0, y0 = 30, self.MAIN_H - h - 96
        panel = Image.new("RGBA", (w, h), (18, 21, 26, 232))
        d = ImageDraw.Draw(panel)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=12, outline=GREEN, width=2)
        d.text((18, 12), "土壌診断 AI  作付提案", font=self.fonts(24), fill=GREEN)
        y = 48
        for kind, text in lines:
            color = {"head": FG, "item": ACCENT, "sub": MUTED}.get(kind, FG)
            size = {"head": 20, "item": 21, "sub": 17}.get(kind, 18)
            d.text((22, y), text, font=self.fonts(size), fill=color)
            y += 30
        frame.paste(panel, (x0, y0), panel)


def title_card(
    renderer: HudRenderer, lines: list[tuple[int, str, tuple[int, int, int]]]
) -> np.ndarray:
    """タイトル / まとめ用の静止フレーム."""
    img = Image.new("RGB", (renderer.W, renderer.H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, renderer.W, 6], fill=ACCENT)
    y = 150
    for size, text, color in lines:
        d.text((110, y), text, font=renderer.fonts(size), fill=color)
        y += int(size * 1.75)
    return np.asarray(img)


# ------------------------------------------------------------------ 3D 重畳

def add_scene_marker(
    scene, pos: np.ndarray, rgba: tuple[float, float, float, float],
    size: float = 0.012, geom_type: str = "sphere",
) -> None:
    """MuJoCo のシーンへ装飾用 geom を 1 つ追加する."""
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    t = {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
         "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}[geom_type]
    sz = np.array([size, size, size], dtype=np.float64)
    mujoco.mjv_initGeom(g, t, sz, np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def overlay_path(scene, path: np.ndarray | None, rgba=(1.0, 1.0, 1.0, 0.85),
                 radius: float = 0.011, z: float = 0.014, max_pts: int = 90) -> None:
    """計画経路を 3D ビュー上に点列として描く."""
    if path is None or len(path) < 2:
        return
    step = max(1, len(path) // max_pts)
    for p in np.asarray(path)[::step]:
        add_scene_marker(scene, (p[0], p[1], z), rgba, radius)


def overlay_soil_points(scene, points: list[np.ndarray],
                        rgba=(0.34, 0.78, 0.48, 0.9)) -> None:
    """土壌サンプリング地点に目印を立てる."""
    for p in points:
        add_scene_marker(scene, (p[0], p[1], 0.05), rgba, 0.022, "cylinder")
