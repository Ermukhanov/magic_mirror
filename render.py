"""
Модуль рендеринга «Сиқырлы айна»:
- Сказочные яркие дневные и вечерние фоны для каждого из 8 героев
- Параллакс облаков, мерцающих звёзд, радуги, орла в степи и комет
- Система частиц салюта (Fireworks) с гравитацией и хвостами
- Контрастная голограмма на светлом фоне с сохранением настоящего лица ребёнка
"""
import os
import math
import time
import random
import threading
from collections import deque
import cv2
import numpy as np
import config
from config import W, H, SW, SH, AA
from tracking import geom, ip, clean_person_mask
from heroes import apoly, draw_back, draw_front
from ai_transform import get_ai_manager

_stage, _tint = {}, {}


class VideoWorldManager:
    """
    Менеджер живых видео-миров (Этап 3, video_worlds: true):
    - Загружает бесшовные циклы 1280x720 30 FPS из assets/<hero_id>/world.mp4.
    - Декодирование в фоновом демон-потоке с буфером (не блокирует 30+ FPS рендер-поток).
    - Бесшовный цикл вперёд-назад (ping-pong loop): 0..N-1..0 без резкого шва на стыке.
    - При отсутствии файла или выключении флага безопасно возвращает None (фоллбэк на stage_bg).
    """

    def __init__(self):
        self._cache = {}
        self._missing = set()
        self._loading = set()
        self._lock = threading.Lock()

    def _load_worker(self, hero_id):
        v_path = os.path.join(config.BASE_DIR, "assets", hero_id, "world.mp4")
        if not os.path.exists(v_path):
            with self._lock:
                self._missing.add(hero_id)
                self._loading.discard(hero_id)
            return

        frames = []
        cap = cv2.VideoCapture(v_path)
        while cap.isOpened():
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if fr.shape[1] != W or fr.shape[0] != H:
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_LINEAR)
            mean_b = float(fr.mean())
            if mean_b < 125.0:
                scale_b = min(2.8, 135.0 / max(15.0, mean_b))
                fr = cv2.convertScaleAbs(fr, alpha=scale_b, beta=int(18 * (scale_b - 0.5)))
            frames.append(fr)
        cap.release()

        with self._lock:
            if frames:
                self._cache[hero_id] = frames
            else:
                self._missing.add(hero_id)
            self._loading.discard(hero_id)

    def get_frame(self, hero_id, t, is_flying=False):
        if not bool(config.VIDEO_WORLDS):
            return None
        with self._lock:
            if hero_id in self._missing:
                return None
            if hero_id in self._cache:
                frames = self._cache[hero_id]
            else:
                if hero_id not in self._loading:
                    self._loading.add(hero_id)
                    threading.Thread(target=self._load_worker, args=(hero_id,), daemon=True).start()
                return None

        n = len(frames)
        if n == 0:
            return None
        if n == 1:
            return frames[0].copy()

        # Бесшовный цикл вперёд-назад (ping-pong loop):
        # 0 -> 1 -> ... -> n-1 -> n-2 -> ... -> 1 -> 0
        cycle_len = 2 * (n - 1)
        fps = 30.0 * (1.5 if is_flying else 1.0)
        phase = int(t * fps) % cycle_len
        idx = phase if phase < n else cycle_len - phase
        return frames[idx].copy()


_WORLD_MGR = None


def get_world_manager():
    global _WORLD_MGR
    if _WORLD_MGR is None:
        _WORLD_MGR = VideoWorldManager()
    return _WORLD_MGR


def get_stage_or_video_bg(hero, t, is_flying=False):
    """
    Возвращает живой видео-мир (video_worlds: true) или процедурный сказочный фон с параллаксом.
    Гарантирует праздничную солнечную яркость (без тёмных мрачных локаций).
    """
    v_frame = get_world_manager().get_frame(hero["id"], t, is_flying=is_flying)
    if v_frame is not None:
        if is_flying:
            for i in range(8):
                sx = int((i * 157 + t * 400) % W)
                cv2.line(v_frame, (sx, 0), (sx, 90), (255, 255, 255), 2, AA)
        return v_frame

    bg = stage_bg(hero)
    draw_sky_parallax(bg, hero, t, is_flying=is_flying)
    return bg
yy = np.arange(2 * H)
_prof = np.clip(np.where(yy % 4 < 2, 255, 245), 0, 255).astype(np.uint8)
SCAN = np.repeat(np.repeat(_prof[:, None, None], W, 1), 3, 2)

FIREWORK_COLORS = [
    (60, 235, 255),   # золотисто-жёлтый
    (120, 255, 80),   # изумрудно-зелёный
    (255, 110, 240),  # розовый неон
    (255, 220, 60),   # небесно-голубой
    (70, 140, 255),   # оранжевый
]


def spawn_firework(fireworks, cx, cy, count=42):
    angles = np.random.uniform(0, 2 * math.pi, count).astype(np.float32)
    speeds = np.random.uniform(140, 440, count).astype(np.float32)
    vx = np.cos(angles) * speeds
    vy = np.sin(angles) * speeds - 60.0
    col = random.choice(FIREWORK_COLORS)
    col2 = random.choice(FIREWORK_COLORS)
    for i in range(count):
        c = col if i % 2 == 0 else col2
        fireworks.append([float(cx), float(cy), float(vx[i]), float(vy[i]), 1.0, c])


def update_and_draw_fireworks(img, fireworks, dt):
    if not fireworks:
        return
    alive = []
    for p in fireworks:
        x, y, vx, vy, life, col = p
        px, py = x - vx * 0.035, y - vy * 0.035
        x += vx * dt
        y += vy * dt
        vy += 360.0 * dt
        vx *= (1.0 - 0.8 * dt)
        life -= dt * 1.15
        if life > 0 and 0 <= x < W and 0 <= y < H:
            r = max(2, int(life * 5))
            c_fade = tuple(min(255, int(ch * max(0.3, life))) for ch in col)
            cv2.line(img, (int(px), int(py)), (int(x), int(y)), c_fade, max(1, r - 1), AA)
            cv2.circle(img, (int(x), int(y)), r, (255, 255, 255) if life > 0.65 else c_fade, -1, AA)
            p[0], p[1], p[2], p[3], p[4] = x, y, vx, vy, life
            alive.append(p)
    fireworks[:] = alive


def _draw_rainbow(img, cx, cy, r_outer, thickness=26, alpha=0.26):
    ov = img.copy()
    rainbow_cols = [
        (80, 80, 255),
        (60, 165, 255),
        (60, 240, 255),
        (90, 235, 90),
        (255, 210, 80),
        (255, 120, 180),
    ]
    step = max(2, thickness // len(rainbow_cols))
    for i, col in enumerate(rainbow_cols):
        rr = r_outer - i * step
        cv2.ellipse(ov, (cx, cy), (rr, int(rr * 0.68)), 0, 190, 350, col, step + 1, AA)
    img[:] = cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0)


def _draw_sunbeams(img, sx, sy, col=(180, 245, 255), alpha=0.18):
    ov = img.copy()
    cv2.circle(ov, (sx, sy), 68, (220, 250, 255), -1, AA)
    cv2.circle(ov, (sx, sy), 42, (255, 255, 255), -1, AA)
    for i in range(9):
        ang = math.radians(15 + i * 18)
        ang2 = ang + 0.065
        pts = np.array([
            [sx, sy],
            [int(sx + math.cos(ang) * 1450), int(sy + math.sin(ang) * 950)],
            [int(sx + math.cos(ang2) * 1450), int(sy + math.sin(ang2) * 950)],
        ], np.int32)
        cv2.fillPoly(ov, [pts], col, AA)
    img[:] = cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0)


def _draw_yurt(img, cx, by, w, h):
    wall_h = int(h * 0.55)
    cv2.rectangle(img, (cx - w // 2, by - wall_h), (cx + w // 2, by), (242, 245, 248), -1, AA)
    cv2.ellipse(img, (cx, by - wall_h), (w // 2, int(h * 0.48)), 0, 180, 360, (248, 250, 252), -1, AA)
    cv2.ellipse(img, (cx, by - wall_h - int(h * 0.46)), (int(w * 0.18), int(h * 0.08)), 0, 0, 360, (40, 195, 250), -1, AA)
    cv2.rectangle(img, (cx - w // 2, by - wall_h - 4), (cx + w // 2, by - wall_h + 6), (40, 185, 245), -1, AA)
    cv2.rectangle(img, (cx - w // 7, by - int(wall_h * 0.72)), (cx + w // 7, by), (60, 90, 160), -1, AA)
    cv2.polylines(img, [np.array([[cx - w // 2, by], [cx - w // 2, by - wall_h], [cx + w // 2, by - wall_h], [cx + w // 2, by]], np.int32)], False, (180, 195, 210), 2, AA)


def stage_bg(hero):
    """Кэшируемый яркий сказочный фон для каждого героя."""
    k = hero["id"]
    if k not in _stage:
        ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        hz = int(H * 0.66)

        if k in ("alpamys", "tomiris"):
            sky_top = np.array((245, 165, 55), np.float32) if k == "alpamys" else np.array((235, 150, 95), np.float32)
            sky_bot = np.array((210, 238, 255), np.float32)
            img = np.repeat(sky_top * (1 - ys) + sky_bot * ys, W, axis=1).astype(np.uint8)
            _draw_sunbeams(img, 210 if k == "alpamys" else W - 210, 105, (190, 245, 255), 0.22)
            _draw_rainbow(img, W // 2 + 120, hz + 30, 520, 28, 0.22)
            m_pts = np.array([[0, hz], [160, hz - 170], [340, hz - 95], [560, hz - 210],
                              [780, hz - 110], [1020, hz - 195], [1190, hz - 125], [W, hz]], np.int32)
            cv2.fillPoly(img, [m_pts], (195, 165, 140), AA)
            for px, py in ((160, hz - 170), (560, hz - 210), (1020, hz - 195)):
                cap = np.array([[px, py], [px - 55, py + 55], [px, py + 42], [px + 55, py + 55]], np.int32)
                cv2.fillPoly(img, [cap], (250, 252, 255), AA)
            cv2.ellipse(img, (W // 4, hz + 120), (520, 165), 0, 180, 360, (85, 195, 115), -1, AA)
            cv2.ellipse(img, (3 * W // 4, hz + 130), (580, 175), 0, 180, 360, (75, 180, 100), -1, AA)
            cv2.rectangle(img, (0, hz), (W, H), (70, 175, 95), -1)
            _draw_yurt(img, 185, hz + 12, 92, 62)
            _draw_yurt(img, 305, hz + 22, 68, 46)
            _draw_yurt(img, W - 210, hz + 16, 86, 58)
            rs_ = np.random.RandomState(11)
            for _ in range(45):
                fx, fy = rs_.randint(40, W - 40), rs_.randint(hz + 15, H - 15)
                cv2.circle(img, (fx, fy), 4, (50, 70, 245) if rs_.rand() > 0.4 else (40, 220, 255), -1, AA)

        elif k in ("elsa", "superheroine"):
            sky_top = np.array((195, 115, 55), np.float32)
            sky_bot = np.array((255, 235, 195), np.float32)
            img = np.repeat(sky_top * (1 - ys) + sky_bot * ys, W, axis=1).astype(np.uint8)
            ov = img.copy()
            for band, bcol in ((130, (210, 255, 110)), (195, (255, 160, 235)), (250, (255, 240, 120))):
                pts = [(x, int(band + 36 * math.sin(x * 0.009 + band))) for x in range(0, W + 32, 32)]
                pts_bot = [(x, y + 55) for x, y in reversed(pts)]
                cv2.fillPoly(ov, [np.array(pts + pts_bot, np.int32)], bcol, AA)
            img = cv2.addWeighted(cv2.GaussianBlur(ov, (0, 0), 15), 0.48, img, 0.52, 0)
            peaks = np.array([[0, hz], [140, hz - 210], [290, hz - 110], [470, hz - 250],
                              [680, hz - 130], [890, hz - 235], [1110, hz - 160], [W, hz]], np.int32)
            cv2.fillPoly(img, [peaks], (245, 215, 175), AA)
            cv2.polylines(img, [peaks], False, (255, 250, 240), 3, AA)
            cv2.rectangle(img, (0, hz), (W, H), (245, 225, 195), -1)

        elif k == "ladybug":
            sky_top = np.array((155, 85, 185), np.float32)
            sky_bot = np.array((125, 205, 255), np.float32)
            img = np.repeat(sky_top * (1 - ys) + sky_bot * ys, W, axis=1).astype(np.uint8)
            _draw_sunbeams(img, W // 2, hz - 40, (150, 220, 255), 0.20)
            tx = W - 250
            tower = np.array([[tx - 65, hz], [tx - 18, hz - 160], [tx - 6, hz - 310],
                              [tx + 6, hz - 310], [tx + 18, hz - 160], [tx + 65, hz]], np.int32)
            cv2.fillPoly(img, [tower], (95, 65, 115), AA)
            cv2.circle(img, (tx, hz - 315), 8, (140, 245, 255), -1, AA)
            cv2.rectangle(img, (0, hz), (W, H), (95, 70, 110), -1)
            rs_ = np.random.RandomState(23)
            for bx in range(30, W - 80, 95):
                bh = rs_.randint(55, 125)
                cv2.rectangle(img, (bx, hz - bh), (bx + 78, hz), (110, 78, 125), -1)
                for wy in range(hz - bh + 14, hz - 10, 22):
                    for wx in range(bx + 12, bx + 66, 22):
                        cv2.rectangle(img, (wx, wy), (wx + 10, wy + 12), (120, 235, 255), -1)

        elif k == "astronaut":
            sky_top = np.array((115, 45, 55), np.float32)
            sky_bot = np.array((175, 95, 125), np.float32)
            img = np.repeat(sky_top * (1 - ys) + sky_bot * ys, W, axis=1).astype(np.uint8)
            ov = img.copy()
            cv2.circle(ov, (340, 230), 210, (220, 110, 190), -1, AA)
            cv2.circle(ov, (940, 260), 230, (240, 185, 80), -1, AA)
            img = cv2.addWeighted(cv2.GaussianBlur(ov, (0, 0), 45), 0.38, img, 0.62, 0)
            cv2.circle(img, (230, 185), 64, (90, 180, 255), -1, AA)
            cv2.ellipse(img, (230, 185), (115, 26), -18, 0, 360, (190, 235, 255), 5, AA)
            cv2.circle(img, (1040, 165), 48, (180, 120, 255), -1, AA)
            cv2.ellipse(img, (W // 2, H + 90), (W // 2 + 180, 240), 0, 180, 360, (175, 165, 190), -1, AA)

        elif k == "fairy":
            sky_top = np.array((235, 195, 95), np.float32)
            sky_bot = np.array((195, 248, 225), np.float32)
            img = np.repeat(sky_top * (1 - ys) + sky_bot * ys, W, axis=1).astype(np.uint8)
            _draw_sunbeams(img, W // 2, 80, (190, 255, 245), 0.24)
            _draw_rainbow(img, W // 2, hz + 50, 560, 32, 0.30)
            for tx in (110, 260, W - 260, W - 110):
                cv2.rectangle(img, (tx - 16, hz - 140), (tx + 16, hz + 20), (85, 115, 155), -1, AA)
                cv2.circle(img, (tx, hz - 175), 78, (125, 215, 115), -1, AA)
                cv2.circle(img, (tx - 35, hz - 145), 55, (145, 228, 130), -1, AA)
                cv2.circle(img, (tx + 35, hz - 145), 55, (115, 205, 105), -1, AA)
            cv2.rectangle(img, (0, hz), (W, H), (105, 200, 115), -1)

        else:  # spider, iron
            sky_top = np.array((245, 155, 55), np.float32)
            sky_bot = np.array((215, 235, 255), np.float32)
            img = np.repeat(sky_top * (1 - ys) + sky_bot * ys, W, axis=1).astype(np.uint8)
            _draw_sunbeams(img, W - 220, 95, (190, 240, 255), 0.22)
            rs_ = np.random.RandomState(31)
            for bx in range(20, W - 60, 88):
                bh = rs_.randint(95, 255)
                bcol = (210, 175, 145) if k in ("iron", "superhero") else (185, 155, 135)
                cv2.rectangle(img, (bx, hz - bh), (bx + 72, hz), bcol, -1, AA)
                cv2.rectangle(img, (bx, hz - bh), (bx + 72, hz), (245, 235, 220), 2, AA)
                for wy in range(hz - bh + 16, hz - 12, 24):
                    for wx in range(bx + 12, bx + 60, 20):
                        cv2.rectangle(img, (wx, wy), (wx + 11, wy + 14), (245, 245, 255), -1)
            cv2.rectangle(img, (0, hz), (W, H), (165, 155, 150), -1)

        gl = tuple(min(255, int(c * 0.85 + 40)) for c in hero["glow"])
        cv2.line(img, (0, hz), (W, hz), (255, 255, 255), 3, AA)
        cv2.line(img, (0, hz + 2), (W, hz + 2), gl, 2, AA)
        for i in range(-10, 11):
            cv2.line(img, (W // 2 + i * 28, hz), (W // 2 + i * 165, H), gl, 1, AA)

        _stage[k] = img
    return _stage[k].copy()


_CLOUD_SPECS = [
    (120, 95, 95, 36, 22.0),
    (420, 140, 115, 42, 32.0),
    (780, 85, 105, 38, 26.0),
    (1080, 130, 90, 34, 36.0),
    (260, 215, 80, 30, 18.0),
    (920, 205, 85, 32, 29.0),
]
_STAR_SPECS = [(int((i * 173) % (W - 40)) + 20, int((i * 79) % 270) + 18, i) for i in range(28)]


def draw_sky_parallax(bg, hero, t, is_flying=False):
    """Живое небо: плывущие облака (параллакс), мерцающие звёзды, орёл в степи / светлячки / комета."""
    k = hero["id"]
    speed_mult = 6.5 if is_flying else 1.0

    for sx, sy, idx in _STAR_SPECS:
        if is_flying:
            sy = int((sy + t * 260.0 + idx * 35) % (H * 0.62))
        tw = 0.5 + 0.5 * math.sin(t * 4.5 + idx * 1.3)
        if tw > 0.35:
            r = 2 if tw < 0.8 else 3
            cv2.circle(bg, (sx, sy), r, (255, 255, 255), -1, AA)
            if tw > 0.86:
                cv2.line(bg, (sx - 6, sy), (sx + 6, sy), (255, 255, 255), 1, AA)
                cv2.line(bg, (sx, sy - 6), (sx, sy + 6), (255, 255, 255), 1, AA)

    if k != "astronaut":
        for bx, by, cw, ch, spd in _CLOUD_SPECS:
            cx = int((bx + t * spd * speed_mult) % (W + 260)) - 130
            cy = int(by + (t * 140.0 % 300 if is_flying else 0)) % int(H * 0.58)
            c_col = (248, 250, 255) if k != "ladybug" else (235, 225, 255)
            cv2.ellipse(bg, (cx, cy), (cw, ch), 0, 0, 360, c_col, -1, AA)
            cv2.ellipse(bg, (cx - int(cw * 0.45), cy + 8), (int(cw * 0.65), int(ch * 0.78)), 0, 0, 360, c_col, -1, AA)
            cv2.ellipse(bg, (cx + int(cw * 0.45), cy + 6), (int(cw * 0.68), int(ch * 0.82)), 0, 0, 360, c_col, -1, AA)

    if k in ("alpamys", "tomiris"):
        ex = int(W * 0.5 + math.cos(t * 0.65) * 340)
        ey = int(125 + math.sin(t * 1.3) * 28)
        flap = int(math.sin(t * 5.5) * 14)
        e_col = (35, 165, 245)
        pts_l = np.array([[ex, ey], [ex - 28, ey - 10 - flap // 2], [ex - 58, ey - flap]], np.int32)
        pts_r = np.array([[ex, ey], [ex + 28, ey - 10 - flap // 2], [ex + 58, ey - flap]], np.int32)
        cv2.polylines(bg, [pts_l, pts_r], False, (255, 255, 255), 5, AA)
        cv2.polylines(bg, [pts_l, pts_r], False, e_col, 3, AA)

    if k == "fairy":
        for i in range(12):
            fx = int((i * 117 + math.sin(t * 1.8 + i) * 45) % W)
            fy = int(H * 0.42 + math.cos(t * 2.2 + i * 1.7) * 140)
            cv2.circle(bg, (fx, fy), 5, (140, 255, 255), -1, AA)
            cv2.circle(bg, (fx, fy), 2, (255, 255, 255), -1, AA)

    if k == "astronaut":
        cx = int((t * 220 * speed_mult) % (W + 400)) - 200
        cy = int(90 + (cx * 0.18))
        cv2.line(bg, (cx - 85, cy - 16), (cx, cy), (255, 235, 150), 3, AA)
        cv2.circle(bg, (cx, cy), 6, (255, 255, 255), -1, AA)


def foot_base(p, g):
    ys = [p[i][1] for i in (27, 28) if p[i][2] > 0.5]
    by = max(ys) + 10 if ys else max(H - 35.0, g["hip"][1] + g["th"] * 0.8)
    return float(g["neck"][0]), float(min(by, H - 28)), g["sw"] * 1.5


def stage_fx(bg, p, g, hero, t, dt, parts):
    """Проектор под ногами: кольца, световой конус, поднимающиеся частицы."""
    bx, by, bw = foot_base(p, g)
    glow = hero["glow"]
    top = max(30.0, by - (by - 30) * 0.95)
    apoly(bg, [(bx - bw * .5, by), (bx + bw * .5, by), (bx + bw * 1.15, top), (bx - bw * 1.15, top)], glow, 0.10)
    cv2.ellipse(bg, (int(bx), int(by)), (int(bw * .55), int(bw * .12)), 0, 0, 360, tuple(int(c * .6) for c in glow), -1, AA)
    for i in range(3):
        k = (t * 0.8 + i / 3) % 1
        rr = bw * (0.4 + k * 0.9)
        cv2.ellipse(bg, (int(bx), int(by)), (int(rr), int(rr * .22)), 0, 0, 360, tuple(int(c * (1 - k)) for c in glow), 2, AA)
    parts[:, 1] = (parts[:, 1] - parts[:, 2] * dt * 0.3) % 1
    for x, y, s in parts:
        cv2.circle(bg, (int(bx + x * bw * 1.1), int(by - y * (by - top))), 1 + int(s * 2), glow, -1)


def holo_person(frame, hero, t, g, bbox=None):
    """Тело ребёнка становится голограммой, лицо остаётся настоящим. Рендерится в ROI силуэта для 60+ FPS."""
    k = hero["id"]
    if k not in _tint:
        _tint[k] = np.full((H, W, 3), [min(255, int(c * 1.0) + 40) for c in hero["glow"]], np.uint8)
    bx0, by0, bx1, by1 = bbox if bbox is not None else (0, 0, W, H)
    f_roi = frame[by0:by1, bx0:bx1]
    t_roi = _tint[k][by0:by1, bx0:bx1]
    off = int(t * 140) % H
    s_roi = SCAN[off + by0:off + by1, bx0:bx1, 0]

    gray = cv2.convertScaleAbs(cv2.cvtColor(f_roi, cv2.COLOR_BGR2GRAY), alpha=1.35, beta=22)
    gray_scan = cv2.multiply(gray, s_roi, scale=1.0 / 255.0)
    holo = cv2.multiply(cv2.merge([gray_scan, gray_scan, gray_scan]), t_roi, scale=1.0 / 255.0)
    fl = 0.55 + 0.08 * random.random()
    person_roi = cv2.addWeighted(f_roi, 0.78, holo, 0.42, 16)
    if person_roi.shape[1] > 6:
        person_roi[:, 3:, 2] = person_roi[:, :-3, 2]
        person_roi[:, :-3, 0] = person_roi[:, 3:, 0]

    out_person_roi = person_roi
    # Лицо без эффектов (всегда настоящее!)
    R = max(8, int(g["hr"] * 1.45))
    cx, cy = ip(g["nose"])
    x0, y0, x1, y1 = max(cx - R, bx0), max(cy - R, by0), min(cx + R, bx1), min(cy + R, by1)
    if x1 > x0 and y1 > y0:
        a = np.zeros((2 * R, 2 * R), np.uint8)
        cv2.circle(a, (R, R), int(R * .85), 255, -1)
        k_sz = max(5, min(25, (int(R * 0.18) // 2) * 2 + 1))
        a = cv2.GaussianBlur(a, (k_sz, k_sz), 0)[y0 - (cy - R):y1 - (cy - R), x0 - (cx - R):x1 - (cx - R)]
        af = a.astype(np.float32) * (1.0 / 255.0)
        rx0, ry0, rx1, ry1 = x0 - bx0, y0 - by0, x1 - bx0, y1 - by0
        fg_face = frame[y0:y1, x0:x1]
        # Яркое солнечное освещение лица ребёнка (чистый тон кожи без темноты)
        mean_face = float(cv2.mean(fg_face)[0] * 0.114 + cv2.mean(fg_face)[1] * 0.587 + cv2.mean(fg_face)[2] * 0.299)
        scale_f = min(2.1, 148.0 / max(30.0, mean_face))
        fg_face_bright = cv2.convertScaleAbs(fg_face, alpha=scale_f, beta=int(24 * (scale_f - 0.5)))
        bg_face = out_person_roi[ry0:ry1, rx0:rx1]
        out_person_roi[ry0:ry1, rx0:rx1] = cv2.blendLinear(fg_face_bright, bg_face, af, 1.0 - af)
    return out_person_roi


def aura(out, mask_small, hero, bbox=None):
    """Контрастная неоновая обводка голограммы в активной области силуэта (ROI)."""
    m = (mask_small * 255).astype(np.uint8)
    big_outer = cv2.dilate(m, np.ones((11, 11), np.uint8))
    big_inner = cv2.dilate(m, np.ones((6, 6), np.uint8))
    ring = cv2.GaussianBlur(cv2.subtract(big_inner, m), (5, 5), 0)
    glow = hero["glow"]
    col_s = cv2.merge([
        cv2.convertScaleAbs(ring, alpha=min(255, glow[0] * 1.9) / 255.0),
        cv2.convertScaleAbs(ring, alpha=min(255, glow[1] * 1.9) / 255.0),
        cv2.convertScaleAbs(ring, alpha=min(255, glow[2] * 1.9) / 255.0),
    ])
    bx0, by0, bx1, by1 = bbox if bbox is not None else (0, 0, W, H)
    rw, rh = bx1 - bx0, by1 - by0
    if rw > 0 and rh > 0:
        sx0 = max(0, int(bx0 * SW / W))
        sy0 = max(0, int(by0 * SH / H))
        sx1 = min(SW, int(bx1 * SW / W) + 1)
        sy1 = min(SH, int(by1 * SH / H) + 1)
        crop_s = col_s[sy0:sy1, sx0:sx1]
        if crop_s.size > 0:
            col_roi = cv2.resize(crop_s, (rw, rh), interpolation=cv2.INTER_LINEAR)
            roi = out[by0:by1, bx0:bx1]
            out[by0:by1, bx0:bx1] = cv2.add(roi, col_roi)


def draw_ai_vortex_animation(out, g, hero, t):
    """
    Анимация вихря превращения (transformation vortex / energy rings)
    вокруг ребёнка во время ожидания готовности ИИ при ai_transform = 'on'.
    """
    neck, hip, sw, th = g["neck"], g["hip"], g["sw"], g["th"]
    cx = int((neck[0] + hip[0]) * 0.5)
    cy = int((neck[1] + hip[1]) * 0.5)
    glow = hero.get("glow", (0, 255, 255))
    accent = hero.get("accent", (255, 200, 50))

    # 1. Вращающиеся кольца энергии вихря
    for i in range(5):
        ang = int((t * 200.0 + i * 45.0) % 360)
        scale_r = 0.50 + 0.22 * math.sin(t * 3.2 + i * 1.1)
        rx = max(12, int(sw * scale_r))
        ry = max(6, int(rx * 0.38))
        c_ring = glow if i % 2 == 0 else accent
        cv2.ellipse(out, (cx, cy), (rx, ry), ang, 0, 360, c_ring, 2, AA)

    # 2. Светящиеся спиральные частицы, закручивающиеся вокруг тела
    for pi in range(10):
        ang_p = t * 4.2 + pi * (2.0 * math.pi / 10.0)
        r_dist = sw * (0.45 + 0.30 * math.sin(t * 2.5 + pi))
        px = int(cx + math.cos(ang_p) * r_dist)
        py = int(cy + math.sin(ang_p) * (r_dist * 0.55))
        cv2.circle(out, (px, py), 3, (255, 255, 255), -1, AA)
        cv2.circle(out, (px, py), 6, glow, 1, AA)

    # 3. Пульсирующий бейдж «ИИ дайындалуда… / ИИ готовится…»
    dots = "." * (int(t * 2.5) % 4)
    vortex_txt = f"ИИ дайындалуда{dots} / ИИ готовится{dots}"
    font_scale = 0.50
    (tw, th_t), _ = cv2.getTextSize(vortex_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bw, bh = tw + 36, 34
    bx0, by0 = cx - bw // 2, cy - bh // 2
    bx1, by1 = bx0 + bw, by0 + bh

    if 0 <= bx0 and bx1 <= W and 0 <= by0 and by1 <= H:
        roi = out[by0:by1, bx0:bx1]
        dark = np.full_like(roi, (12, 16, 24))
        out[by0:by1, bx0:bx1] = cv2.addWeighted(dark, 0.82, roi, 0.18, 0)
        cv2.rectangle(out, (bx0, by0), (bx1, by1), glow, 1, AA)
        cv2.putText(
            out, vortex_txt,
            (bx0 + 18, (by0 + by1 + th_t) // 2 - 1),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, AA
        )


def compose(frame, p, mask_s, hero, t, dt, parts, is_flying=False):
    g = geom(p)
    bg = get_stage_or_video_bg(hero, t, is_flying=is_flying)
    stage_fx(bg, p, g, hero, t, dt, parts)
    draw_back(bg, g, hero, t)

    m_u8 = (mask_s * 242).astype(np.uint8)
    coords = cv2.findNonZero(m_u8)
    if coords is not None:
        sx, sy, sw_m, sh_m = cv2.boundingRect(coords)
        scale_x, scale_y = W / SW, H / SH
        bx0 = max(0, int(sx * scale_x) - 36)
        by0 = max(0, int(sy * scale_y) - 36)
        bx1 = min(W, int((sx + sw_m) * scale_x) + 36)
        by1 = min(H, int((sy + sh_m) * scale_y) + 36)
    else:
        bx0, by0, bx1, by1 = 0, 0, W, H

    ai_active = False
    ai_waiting = False
    ai_mode = str(config.AI_TRANSFORM).lower()
    if bx1 > bx0 and by1 > by0:
        person_roi = None
        m8_roi = None
        rw, rh = bx1 - bx0, by1 - by0
        if ai_mode != "off":
            try:
                ai_mgr = get_ai_manager()
                if ai_mgr.is_enabled():
                    # Передаём кадр в ИИ только тогда, когда GPU свободен и ждёт следующий кадр
                    if ai_mgr.needs_frame():
                        clean_s = clean_person_mask(mask_s, erode_px=2)
                        ai_mgr.submit_frame(frame, clean_s, p, (bx0, by0, bx1, by1), hero, int(t * 25))
                    if ai_mgr.is_active_for_hero(hero, t):
                        person_roi = ai_mgr.render_ai_person_roi(frame, p, hero, t, bbox=(bx0, by0, bx1, by1))
                        if person_roi is not None:
                            ai_active = True
                            m_roi_s = mask_s[int(by0 * SH / H):int(by1 * SH / H), int(bx0 * SW / W):int(bx1 * SW / W)]
                            if m_roi_s.size > 0:
                                m8_roi = cv2.resize((m_roi_s * 248.0).astype(np.uint8), (rw, rh), interpolation=cv2.INTER_LINEAR)
                            else:
                                m8_roi = np.full((rh, rw), 248, dtype=np.uint8)

                # Режим strict "on": пока ИИ загружается или готовит первый кадр (до 60 сек)
                if ai_mode == "on" and not ai_active:
                    elapsed = time.perf_counter() - ai_mgr.start_time
                    if elapsed < 60.0 and not ai_mgr.failed:
                        ai_waiting = True
            except Exception as e_comp:
                import traceback
                print(f"[COMPOSE EXCEPTION] {e_comp}\n{traceback.format_exc()}")
                person_roi = None
                ai_active = False
                ai_waiting = False

        if person_roi is None:
            person_roi = holo_person(frame, hero, t, g, bbox=(bx0, by0, bx1, by1))
            m_roi_s = mask_s[int(by0 * SH / H):int(by1 * SH / H), int(bx0 * SW / W):int(bx1 * SW / W)]
            if m_roi_s.size > 0:
                m8_roi = cv2.resize((m_roi_s * 242.0).astype(np.uint8), (rw, rh), interpolation=cv2.INTER_LINEAR)
            else:
                m8_roi = np.full((rh, rw), 242, dtype=np.uint8)

        if not ai_active and random.random() < 0.03 and (by1 - by0) > 70:
            rh_box = by1 - by0
            y0_g = random.randint(0, rh_box - 50)
            hh_g = random.randint(8, min(36, rh_box - y0_g))
            sh = random.randint(-22, 22)
            person_roi[y0_g:y0_g + hh_g] = np.roll(person_roi[y0_g:y0_g + hh_g], sh, axis=1)
            m8_roi[y0_g:y0_g + hh_g] = np.roll(m8_roi[y0_g:y0_g + hh_g], sh, axis=1)

        # Сверхбыстрое наложение cv2.blendLinear (0.4 мс)
        bg_roi = bg[by0:by1, bx0:bx1]
        w_fg = (m8_roi.astype(np.float32) * (1.0 / 255.0))
        bg[by0:by1, bx0:bx1] = cv2.blendLinear(person_roi, bg_roi, w_fg, 1.0 - w_fg)
    out = bg
    aura(out, mask_s, hero, bbox=(bx0, by0, bx1, by1))
    if ai_waiting:
        draw_ai_vortex_animation(out, g, hero, t)
    draw_front(out, g, hero, t, ai_active=(ai_active or ai_waiting))
    return out, g
