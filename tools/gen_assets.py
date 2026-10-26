"""
Генератор ассетов для «Сиқырлы айна» (Этап 3 и Этап 4):
1. Видео-миры (assets/<hero_id>/world.mp4):
   - 1280x720, 30 FPS, бесшовный зацикленный 3-секундный видео-мир (90 кадров) для всех 8 героев.
2. Видеоэффекты атак и суперсил (assets/<hero_id>/fx_<action>.mp4):
   - fx_fly, fx_beam, fx_shield, fx_special для каждого героя.
3. Графика героев (assets/<hero_id>/*.png):
   - Высококачественные прозрачные RGBA PNG-спрайты: armor.png, back.png, crown.png, glove.png.
"""
import os
import sys
import math
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

W, H = 1280, 720
FPS = 30
FRAMES_WORLD = 90  # 3.0 секунды при 30 FPS — идеальный бесшовный цикл

HERO_IDS = ["alpamys", "tomiris", "spider", "elsa", "ladybug", "iron", "astronaut", "fairy"]

AA = cv2.LINE_AA


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def smooth_phase(i, total_frames=FRAMES_WORLD):
    """Возвращает фазу [0..1] для гладкого синусоидального зацикливания."""
    return (2.0 * math.pi * i) / float(total_frames)


# ==============================================================================
# 1. ГЕНЕРАЦИЯ ВИДЕО-МИРОВ (1280x720, 30 FPS, 90 кадров)
# ==============================================================================

def draw_alpamys_frame(f_idx):
    """Мир Алпамыса: золотистая степь, восходящее солнце с лучами, скачущий Тұлпар на горизонте, юрты и тюльпаны."""
    ph = smooth_phase(f_idx)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Небо: градиент от лазурного к золотистому
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top_c = np.array([245, 175, 45], np.float32)
    bot_c = np.array([210, 240, 255], np.float32)
    img[:] = (top_c * (1 - ys) + bot_c * ys).astype(np.uint8)

    # Солнце и вращающиеся лучи
    sx, sy = 240, 140
    cv2.circle(img, (sx, sy), 80, (180, 245, 255), -1, AA)
    cv2.circle(img, (sx, sy), 55, (255, 255, 255), -1, AA)
    for k in range(12):
        ang = ph * 0.5 + k * (math.pi / 6)
        p_far = (int(sx + math.cos(ang) * 900), int(sy + math.sin(ang) * 600))
        cv2.line(img, (sx, sy), p_far, (200, 245, 255), 2, AA)

    # Горы на заднем плане
    hz = int(H * 0.64)
    mounts = np.array([
        [0, hz], [180, hz - 190], [360, hz - 110], [580, hz - 230],
        [820, hz - 130], [1040, hz - 210], [W, hz]
    ], np.int32)
    cv2.fillPoly(img, [mounts], (190, 160, 135), AA)
    for px, py in ((180, hz - 190), (580, hz - 230), (1040, hz - 210)):
        snow = np.array([[px, py], [px - 60, py + 60], [px, py + 48], [px + 60, py + 60]], np.int32)
        cv2.fillPoly(img, [snow], (250, 252, 255), AA)

    # Изумрудно-зелёная равнина
    cv2.rectangle(img, (0, hz), (W, H), (75, 185, 105), -1)
    
    # Юрты
    for yx, yy_b, yw, yh in ((190, hz + 14, 96, 64), (320, hz + 24, 72, 48), (W - 220, hz + 18, 90, 60)):
        cv2.rectangle(img, (yx - yw // 2, yy_b - yh // 2), (yx + yw // 2, yy_b), (242, 245, 248), -1, AA)
        cv2.ellipse(img, (yx, yy_b - yh // 2), (yw // 2, yh // 2), 0, 180, 360, (248, 250, 252), -1, AA)
        cv2.circle(img, (yx, yy_b - yh), 8, (40, 195, 250), -1, AA)

    # Скачущий силуэт золотого коня Тұлпара на горизонте
    hx = int((f_idx / float(FRAMES_WORLD) * (W + 200))) - 100
    hy = hz - 28 + int(math.sin(ph * 4.0) * 8)
    cv2.ellipse(img, (hx, hy), (26, 14), 0, 0, 360, (30, 160, 245), -1, AA)
    cv2.circle(img, (hx + 22, hy - 14), 9, (30, 160, 245), -1, AA)
    cv2.line(img, (hx + 18, hy - 8), (hx + 30, hy - 18), (30, 160, 245), 4, AA)
    # Грива и хвост из пламени
    cv2.line(img, (hx - 24, hy), (hx - 40 - int(math.sin(ph*8)*6), hy - 8), (40, 215, 255), 3, AA)

    # Степные тюльпаны, колышущиеся на ветру
    for idx_f in range(35):
        tx = int((idx_f * 37 + 15) % (W - 30) + 15)
        ty = int(hz + 25 + (idx_f * 53) % (H - hz - 40))
        sway = int(math.sin(ph * 2.0 + idx_f) * 6)
        cv2.line(img, (tx, ty), (tx + sway, ty - 18), (50, 140, 70), 2, AA)
        cv2.circle(img, (tx + sway, ty - 20), 5, (40, 60, 245) if idx_f % 2 == 0 else (40, 220, 255), -1, AA)

    return img


def draw_tomiris_frame(f_idx):
    """Мир Томирис: вечерняя царская степь саков, пурпурный закат, золотой орёл и лагерь с искрами."""
    ph = smooth_phase(f_idx)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top_c = np.array([215, 120, 85], np.float32)
    bot_c = np.array([160, 215, 255], np.float32)
    img[:] = (top_c * (1 - ys) + bot_c * ys).astype(np.uint8)

    hz = int(H * 0.64)
    # Золотой закатный диск
    cv2.circle(img, (W - 280, hz - 60), 90, (80, 210, 255), -1, AA)
    cv2.circle(img, (W - 280, hz - 60), 65, (160, 240, 255), -1, AA)

    # Рельеф пурпурных холмов
    peaks = np.array([[0, hz], [200, hz - 140], [450, hz - 80], [720, hz - 180], [980, hz - 100], [W, hz]], np.int32)
    cv2.fillPoly(img, [peaks], (165, 95, 125), AA)
    cv2.rectangle(img, (0, hz), (W, H), (120, 80, 105), -1)

    # Парящий беркут
    ex = int(W * 0.5 + math.sin(ph) * 280)
    ey = int(140 + math.cos(ph * 2.0) * 35)
    flap = int(math.sin(ph * 6.0) * 16)
    pts_l = np.array([[ex, ey], [ex - 36, ey - 10 - flap // 2], [ex - 72, ey - flap]], np.int32)
    pts_r = np.array([[ex, ey], [ex + 36, ey - 10 - flap // 2], [ex + 72, ey - flap]], np.int32)
    cv2.polylines(img, [pts_l, pts_r], False, (40, 210, 255), 4, AA)

    # Искры от степного костра
    for i in range(25):
        spk_x = int(240 + math.sin(ph * 3.0 + i * 1.5) * 45)
        spk_y = int(hz + 30 - ((f_idx * 4 + i * 28) % 180))
        cv2.circle(img, (spk_x, spk_y), 3, (60, 220, 255), -1, AA)

    return img


def draw_spider_frame(f_idx):
    """Мир Человека-паука: ночной мегаполис, неоновые небоскрёбы, пульсирующая паутина и световой поток дорог."""
    ph = smooth_phase(f_idx)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img[:] = (np.array([90, 35, 45], np.float32) * (1 - ys) + np.array([160, 70, 85], np.float32) * ys).astype(np.uint8)

    hz = int(H * 0.70)
    # Небоскрёбы
    rs_ = np.random.RandomState(42)
    for bx in range(20, W - 40, 85):
        bh = rs_.randint(140, 360)
        cv2.rectangle(img, (bx, hz - bh), (bx + 72, hz), (65, 45, 55), -1)
        cv2.rectangle(img, (bx, hz - bh), (bx + 72, hz), (140, 85, 105), 2)
        # Окна со светом
        for wy in range(hz - bh + 16, hz - 12, 24):
            for wx in range(bx + 12, bx + 60, 18):
                lit = ((wx + wy + int(ph * 10)) % 7) == 0
                c_win = (120, 240, 255) if lit else (180, 140, 90)
                cv2.rectangle(img, (wx, wy), (wx + 10, wy + 14), c_win, -1)

    cv2.rectangle(img, (0, hz), (W, H), (45, 35, 42), -1)

    # Неоновые линии паутины между крышами с бегущими импульсами
    web_center = (W // 2, 160)
    for i_w in range(8):
        ang_w = i_w * (math.pi / 4)
        p_end = (int(web_center[0] + math.cos(ang_w) * 550), int(web_center[1] + math.sin(ang_w) * 380))
        cv2.line(img, web_center, p_end, (190, 80, 240), 1, AA)
        # Бегущий неоновый узел
        t_node = (ph + i_w * 0.7) % (2 * math.pi) / (2 * math.pi)
        nx = int(web_center[0] + (p_end[0] - web_center[0]) * t_node)
        ny = int(web_center[1] + (p_end[1] - web_center[1]) * t_node)
        cv2.circle(img, (nx, ny), 5, (255, 230, 120), -1, AA)

    return img


def draw_elsa_frame(f_idx):
    """Мир Эльзы: ледяной дворец, танцующее северное сияние, кристаллы и кружащиеся снежинки."""
    ph = smooth_phase(f_idx)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img[:] = (np.array([215, 135, 55], np.float32) * (1 - ys) + np.array([255, 245, 215], np.float32) * ys).astype(np.uint8)

    # Северное сияние (Aurora Borealis)
    ov = img.copy()
    for band, bcol in ((120, (190, 255, 90)), (175, (255, 140, 220)), (230, (255, 230, 90))):
        pts = [(x, int(band + 38 * math.sin(x * 0.008 + ph * 2.0 + band * 0.01))) for x in range(0, W + 32, 32)]
        pts_bot = [(x, y + 60) for x, y in reversed(pts)]
        cv2.fillPoly(ov, [np.array(pts + pts_bot, np.int32)], bcol, AA)
    img = cv2.addWeighted(cv2.GaussianBlur(ov, (0, 0), 17), 0.45, img, 0.55, 0)

    hz = int(H * 0.66)
    # Ледяные кристаллы и горы
    peaks = np.array([[0, hz], [160, hz - 220], [320, hz - 120], [520, hz - 260],
                      [720, hz - 140], [920, hz - 240], [1120, hz - 150], [W, hz]], np.int32)
    cv2.fillPoly(img, [peaks], (245, 220, 185), AA)
    cv2.polylines(img, [peaks], False, (255, 252, 245), 3, AA)
    cv2.rectangle(img, (0, hz), (W, H), (250, 235, 210), -1)

    # Падающие кружащиеся кристаллы льда
    for i in range(40):
        sx = int((i * 39 + math.sin(ph + i) * 35) % W)
        sy = int(((f_idx * 5 + i * 27) % H))
        cv2.circle(img, (sx, sy), 3, (255, 255, 255), -1, AA)
        if i % 3 == 0:
            cv2.line(img, (sx - 5, sy), (sx + 5, sy), (255, 255, 255), 1, AA)
            cv2.line(img, (sx, sy - 5), (sx, sy + 5), (255, 255, 255), 1, AA)

    return img


def draw_ladybug_frame(f_idx):
    """Мир Ледибаг: сумерки над Парижем, светящаяся Эйфелева башня с вращающимся прожектором и лепестки роз."""
    ph = smooth_phase(f_idx)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img[:] = (np.array([165, 80, 195], np.float32) * (1 - ys) + np.array([135, 210, 255], np.float32) * ys).astype(np.uint8)

    hz = int(H * 0.68)
    tx = W - 260
    # Эйфелева башня
    tower = np.array([[tx - 70, hz], [tx - 20, hz - 180], [tx - 8, hz - 330],
                      [tx + 8, hz - 330], [tx + 20, hz - 180], [tx + 70, hz]], np.int32)
    cv2.fillPoly(img, [tower], (95, 65, 115), AA)
    cv2.circle(img, (tx, hz - 335), 10, (140, 245, 255), -1, AA)

    # Вращающийся луч прожектора
    beam_ang = ph
    ov_b = img.copy()
    bx_far = int(tx + math.cos(beam_ang) * 900)
    by_far = int(hz - 335 + math.sin(beam_ang) * 350)
    b_pts = np.array([[tx, hz - 335], [bx_far - 40, by_far], [bx_far + 40, by_far]], np.int32)
    cv2.fillPoly(ov_b, [b_pts], (200, 250, 255), AA)
    img = cv2.addWeighted(ov_b, 0.22, img, 0.78, 0)

    # Крыши Парижа
    cv2.rectangle(img, (0, hz), (W, H), (95, 70, 110), -1)

    # Парящие искры божьих коровок
    for i in range(25):
        lx = int((i * 51 + math.sin(ph * 2.0 + i) * 60) % W)
        ly = int(hz - 40 - ((f_idx * 3 + i * 29) % 240))
        cv2.circle(img, (lx, ly), 5, (45, 45, 250), -1, AA)
        cv2.circle(img, (lx, ly), 2, (255, 255, 255), -1, AA)

    return img


def draw_iron_frame(f_idx):
    """Мир Железного героя: лаборатория будущего, энерго-реакторы, голографические кольца и потоки энергии."""
    ph = smooth_phase(f_idx)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img[:] = (np.array([180, 75, 45], np.float32) * (1 - ys) + np.array([240, 185, 95], np.float32) * ys).astype(np.uint8)

    hz = int(H * 0.68)
    cv2.rectangle(img, (0, hz), (W, H), (65, 55, 60), -1)
    
    # Сетка пола с перспективой
    for i in range(-12, 13):
        cv2.line(img, (W // 2 + i * 35, hz), (W // 2 + i * 190, H), (90, 225, 255), 1, AA)

    # Центральный реактор на стене
    rcx, rcy = W // 2, hz - 180
    pulse = int(math.sin(ph * 3.0) * 12)
    for r_i in (140, 105, 70):
        cv2.circle(img, (rcx, rcy), r_i + pulse // 2, (70, 210, 255), 2, AA)
    cv2.circle(img, (rcx, rcy), 45, (255, 255, 255), -1, AA)

    # Вертикальные энерго-потоки
    for k in range(8):
        ex = int((k * 155 + 50) % W)
        ey = int((f_idx * 8 + k * 80) % hz)
        cv2.line(img, (ex, ey), (ex, ey + 40), (80, 225, 255), 2, AA)

    return img


def draw_astronaut_frame(f_idx):
    """Мир Космонавта: глубокий космос, вращающаяся Земля, цветная туманность и кометы."""
    ph = smooth_phase(f_idx)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img[:] = (np.array([85, 25, 35], np.float32) * (1 - ys) + np.array([140, 55, 95], np.float32) * ys).astype(np.uint8)

    # Космическая туманность
    ov = img.copy()
    cv2.circle(ov, (340, 230), 220, (220, 110, 190), -1, AA)
    cv2.circle(ov, (940, 260), 240, (240, 185, 80), -1, AA)
    img = cv2.addWeighted(cv2.GaussianBlur(ov, (0, 0), 45), 0.38, img, 0.62, 0)

    # Земля с атмосферным свечением
    ex, ey = 250, 220
    cv2.circle(img, (ex, ey), 85, (220, 160, 45), -1, AA)
    cv2.circle(img, (ex, ey), 88, (255, 240, 160), 3, AA)
    # Облака на планете
    cv2.ellipse(img, (ex - 20, ey - 10), (50, 18), int(math.degrees(ph)), 0, 360, (255, 255, 255), -1, AA)

    # Марс
    cv2.circle(img, (W - 240, 180), 46, (60, 110, 235), -1, AA)

    # Мерцающие звёзды
    for i in range(50):
        sx = int((i * 137) % W)
        sy = int((i * 73) % (H - 50))
        tw = 0.5 + 0.5 * math.sin(ph * 4.0 + i)
        if tw > 0.4:
            cv2.circle(img, (sx, sy), 2 if tw < 0.8 else 3, (255, 255, 255), -1, AA)

    # Пролетающая комета
    cx = int((f_idx / float(FRAMES_WORLD) * (W + 400))) - 200
    cy = int(90 + cx * 0.18)
    cv2.line(img, (cx - 95, cy - 18), (cx, cy), (255, 240, 160), 3, AA)
    cv2.circle(img, (cx, cy), 6, (255, 255, 255), -1, AA)

    return img


def draw_fairy_frame(f_idx):
    """Мир Феи: волшебный биолюминесцентный лес, гигантские светящиеся грибы, бабочки и звёздная пыльца."""
    ph = smooth_phase(f_idx)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img[:] = (np.array([235, 185, 85], np.float32) * (1 - ys) + np.array([215, 250, 225], np.float32) * ys).astype(np.uint8)

    hz = int(H * 0.65)
    # Волшебные деревья
    for tx in (120, 280, W - 280, W - 120):
        cv2.rectangle(img, (tx - 18, hz - 160), (tx + 18, hz + 20), (85, 115, 155), -1, AA)
        cv2.circle(img, (tx, hz - 190), 85, (125, 220, 115), -1, AA)
    cv2.rectangle(img, (0, hz), (W, H), (105, 210, 120), -1)

    # Светящиеся волшебные грибы
    for mx in (220, 420, W - 420, W - 220):
        pulse_m = int(math.sin(ph * 3.0 + mx) * 5)
        cv2.rectangle(img, (mx - 6, hz - 25), (mx + 6, hz), (240, 245, 255), -1, AA)
        cv2.ellipse(img, (mx, hz - 25), (28 + pulse_m, 18), 0, 180, 360, (255, 140, 240), -1, AA)

    # Парящие волшебные бабочки
    for i in range(16):
        fx = int((i * 123 + math.sin(ph * 2.0 + i) * 65) % W)
        fy = int(hz - 60 + math.cos(ph * 3.0 + i * 1.5) * 110)
        cv2.circle(img, (fx, fy), 5, (140, 255, 255), -1, AA)
        cv2.circle(img, (fx, fy), 2, (255, 255, 255), -1, AA)

    return img


WORLD_GENERATORS = {
    "alpamys": draw_alpamys_frame,
    "tomiris": draw_tomiris_frame,
    "spider": draw_spider_frame,
    "elsa": draw_elsa_frame,
    "ladybug": draw_ladybug_frame,
    "iron": draw_iron_frame,
    "astronaut": draw_astronaut_frame,
    "fairy": draw_fairy_frame,
}


def generate_world_videos():
    """Создаёт 8 видео-миров (assets/<hero_id>/world.mp4) 1280x720 30 FPS 90 кадров."""
    print("\n[GEN_ASSETS] 1) Генерация 8 видео-миров (assets/<hero_id>/world.mp4, 1280x720, 30 FPS)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for hid in HERO_IDS:
        h_dir = os.path.join(ASSETS_DIR, hid)
        ensure_dir(h_dir)
        vid_path = os.path.join(h_dir, "world.mp4")
        writer = cv2.VideoWriter(vid_path, fourcc, float(FPS), (W, H))
        draw_fn = WORLD_GENERATORS.get(hid, draw_alpamys_frame)
        for i in range(FRAMES_WORLD):
            frame = draw_fn(i)
            writer.write(frame)
        writer.release()
        sz_kb = os.path.getsize(vid_path) / 1024.0
        print(f"  [WORLD] {hid:10s} -> {vid_path} ({sz_kb:.1f} KB, 90 кадров)")


# ==============================================================================
# 2. ГЕНЕРАЦИЯ ВИДЕОЭФФЕКТОВ FX (assets/<hero_id>/fx_<action>.mp4)
# ==============================================================================

def generate_fx_clips():
    """Создаёт видеоэффекты атак и способностей (assets/<hero_id>/fx_<action>.mp4)."""
    print("\n[GEN_ASSETS] 2) Генерация видеоэффектов атак (assets/<hero_id>/fx_*.mp4)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fw, fh = 480, 480
    n_frames = 45  # 1.5 секунды

    for hid in HERO_IDS:
        h_dir = os.path.join(ASSETS_DIR, hid)
        ensure_dir(h_dir)

        # 1. fx_fly.mp4 (энергетический восходящий поток / ускорители)
        fly_path = os.path.join(h_dir, "fx_fly.mp4")
        w_fly = cv2.VideoWriter(fly_path, fourcc, float(FPS), (fw, fh))
        for i in range(n_frames):
            fr = np.zeros((fh, fw, 3), dtype=np.uint8)
            t_k = i / float(n_frames)
            for j in range(3):
                r = int((t_k * 300 + j * 90) % 220)
                alpha = max(0.0, 1.0 - r / 220.0)
                col = tuple(int(c * alpha) for c in (40, 220, 255))
                cv2.circle(fr, (fw // 2, fh // 2 + 50), r + 20, col, 4, AA)
            # Вертикальные струи ускорения
            for sx in (fw // 2 - 60, fw // 2 + 60):
                cv2.line(fr, (sx, fh // 2 + 60), (sx, fh - 10), (255, 240, 150), 6, AA)
                cv2.circle(fr, (sx, fh // 2 + 60), 14, (255, 255, 255), -1, AA)
            w_fly.write(fr)
        w_fly.release()

        # 2. fx_beam.mp4 (лазерный / плазменный импульсный луч)
        beam_path = os.path.join(h_dir, "fx_beam.mp4")
        w_beam = cv2.VideoWriter(beam_path, fourcc, float(FPS), (fw, fh))
        for i in range(n_frames):
            fr = np.zeros((fh, fw, 3), dtype=np.uint8)
            pulse = math.sin(i * 0.8) * 6
            cv2.line(fr, (20, fh // 2), (fw - 20, fh // 2), (255, 255, 255), int(12 + pulse), AA)
            cv2.line(fr, (20, fh // 2), (fw - 20, fh // 2), (40, 190, 255), int(22 + pulse), AA)
            for k in range(8):
                spark_x = int(20 + (k * 55 + i * 22) % (fw - 40))
                spark_y = int(fh // 2 + math.sin(i * 1.5 + k) * 32)
                cv2.circle(fr, (spark_x, spark_y), 4, (120, 255, 255), -1, AA)
            w_beam.write(fr)
        w_beam.release()

        # 3. fx_shield.mp4 (светящийся силовой шестиугольный щит)
        shield_path = os.path.join(h_dir, "fx_shield.mp4")
        w_shield = cv2.VideoWriter(shield_path, fourcc, float(FPS), (fw, fh))
        for i in range(n_frames):
            fr = np.zeros((fh, fw, 3), dtype=np.uint8)
            sc = 120 + int(math.sin(i * 0.4) * 15)
            hexa = [(int(fw // 2 + sc * math.cos(math.pi / 3 * k + i * 0.05)),
                     int(fh // 2 + sc * math.sin(math.pi / 3 * k + i * 0.05))) for k in range(6)]
            cv2.polylines(fr, [np.array(hexa, np.int32)], True, (255, 255, 255), 4, AA)
            cv2.polylines(fr, [np.array(hexa, np.int32)], True, (40, 215, 255), 8, AA)
            cv2.circle(fr, (fw // 2, fh // 2), sc - 20, (60, 160, 255), 2, AA)
            w_shield.write(fr)
        w_shield.release()

        # 4. fx_special.mp4 (уникальная суперсила каждого героя)
        spec_path = os.path.join(h_dir, "fx_special.mp4")
        w_spec = cv2.VideoWriter(spec_path, fourcc, float(FPS), (fw, fh))
        for i in range(n_frames):
            fr = np.zeros((fh, fw, 3), dtype=np.uint8)
            t_s = i / float(n_frames)
            ang = t_s * 4.0 * math.pi
            r_sp = int(t_s * 180)
            cx, cy = int(fw // 2 + math.cos(ang) * r_sp), int(fh // 2 + math.sin(ang) * r_sp)
            cv2.circle(fr, (cx, cy), 16, (40, 235, 255), -1, AA)
            cv2.circle(fr, (cx, cy), 10, (255, 255, 255), -1, AA)
            for p_i in range(12):
                px = int(fw // 2 + math.cos(ang + p_i * 0.5) * (r_sp * 0.75))
                py = int(fh // 2 + math.sin(ang + p_i * 0.5) * (r_sp * 0.75))
                cv2.circle(fr, (px, py), 4, (255, 215, 120), -1, AA)
            w_spec.write(fr)
        w_spec.release()

        print(f"  [FX] {hid:10s} -> fx_fly, fx_beam, fx_shield, fx_special созданы")


# ==============================================================================
# 3. ГЕНЕРАЦИЯ ГРАФИКИ ГЕРОЕВ (RGBA PNG 512x512)
# ==============================================================================

def create_armor_png(hero_id):
    img = np.zeros((512, 512, 4), dtype=np.uint8)
    cx, cy = 256, 256
    
    palettes = {
        "alpamys":   dict(c1=(40, 195, 255), c2=(30, 40, 190), trim=(255, 255, 255)),
        "tomiris":   dict(c1=(150, 40, 190), c2=(40, 190, 245), trim=(255, 255, 255)),
        "spider":    dict(c1=(50, 50, 220), c2=(220, 60, 40), trim=(255, 255, 255)),
        "elsa":      dict(c1=(245, 210, 120), c2=(255, 245, 210), trim=(255, 255, 255)),
        "ladybug":   dict(c1=(45, 45, 235), c2=(25, 25, 30), trim=(255, 255, 255)),
        "iron":      dict(c1=(40, 60, 215), c2=(50, 190, 255), trim=(255, 255, 255)),
        "astronaut": dict(c1=(230, 230, 235), c2=(210, 120, 40), trim=(255, 255, 255)),
        "fairy":     dict(c1=(220, 160, 245), c2=(140, 240, 160), trim=(255, 255, 255)),
    }
    pal = palettes.get(hero_id, palettes["alpamys"])
    
    # Нагрудная пластина
    pts = np.array([
        [cx - 160, cy - 140], [cx + 160, cy - 140],
        [cx + 120, cy + 120], [cx, cy + 170], [cx - 120, cy + 120]
    ], np.int32)
    b, g, r = pal["c1"]
    cv2.fillPoly(img, [pts], (b, g, r, 255), AA)
    tb, tg, tr_ = pal["trim"]
    cv2.polylines(img, [pts], True, (tb, tg, tr_, 255), 8, AA)
    
    # Центральный орнамент / эмблема
    b2, g2, r2 = pal["c2"]
    cv2.circle(img, (cx, cy), 68, (b2, g2, r2, 255), -1, AA)
    cv2.circle(img, (cx, cy), 68, (255, 255, 255, 255), 5, AA)
    cv2.circle(img, (cx, cy), 28, (255, 255, 255, 255), -1, AA)
    return img


def create_back_png(hero_id):
    img = np.zeros((512, 512, 4), dtype=np.uint8)
    cx, cy = 256, 256

    if hero_id in ("alpamys", "tomiris", "iron"):
        # Королевский широкий плащ
        c_cape = (35, 45, 210) if hero_id != "tomiris" else (150, 45, 120)
        pts = np.array([
            [cx - 110, cy - 180], [cx + 110, cy - 180],
            [cx + 210, cy + 210], [cx, cy + 190], [cx - 210, cy + 210]
        ], np.int32)
        b, g, r = c_cape
        cv2.fillPoly(img, [pts], (b, g, r, 240), AA)
        cv2.polylines(img, [pts], True, (40, 215, 255, 255), 8, AA)
    elif hero_id == "fairy":
        # Крылья бабочки
        for s_dir in (-1, 1):
            w_pts = np.array([
                [cx, cy - 60],
                [cx + s_dir * 220, cy - 160],
                [cx + s_dir * 230, cy + 40],
                [cx + s_dir * 140, cy + 150],
                [cx, cy + 60]
            ], np.int32)
            cv2.fillPoly(img, [w_pts], (245, 170, 235, 220), AA)
            cv2.polylines(img, [w_pts], True, (255, 255, 255, 255), 5, AA)
    elif hero_id == "astronaut":
        # Ранец космонавта
        cv2.rectangle(img, (cx - 140, cy - 130), (cx + 140, cy + 130), (225, 225, 230, 245), -1, AA)
        cv2.rectangle(img, (cx - 140, cy - 130), (cx + 140, cy + 130), (60, 160, 255, 255), 6, AA)
        for sx in (cx - 70, cx + 70):
            cv2.circle(img, (sx, cy + 145), 24, (50, 200, 255, 255), -1, AA)
    else:
        # Для паука / ледибаг / эльзы — легкая аура / накидка
        cv2.ellipse(img, (cx, cy), (180, 210), 0, 0, 360, (255, 220, 130, 180), -1, AA)
        cv2.ellipse(img, (cx, cy), (180, 210), 0, 0, 360, (255, 255, 255, 255), 6, AA)

    return img


def create_crown_png(hero_id):
    img = np.zeros((512, 512, 4), dtype=np.uint8)
    cx, cy = 256, 256

    if hero_id in ("alpamys", "tomiris"):
        # Корона / боевой шлем-венец сакской царицы
        pts = np.array([
            [cx - 140, cy + 30], [cx - 140, cy - 60], [cx - 70, cy - 10],
            [cx, cy - 90], [cx + 70, cy - 10], [cx + 140, cy - 60], [cx + 140, cy + 30]
        ], np.int32)
        cv2.fillPoly(img, [pts], (40, 210, 255, 255), AA)
        cv2.polylines(img, [pts], True, (255, 255, 255, 255), 6, AA)
        for px, py in ((cx - 140, cy - 60), (cx, cy - 90), (cx + 140, cy - 60)):
            cv2.circle(img, (px, py), 12, (30, 40, 220, 255), -1, AA)
    elif hero_id in ("elsa", "fairy"):
        # Хрустальная диадема / венок
        cv2.ellipse(img, (cx, cy + 20), (130, 45), 0, 180, 360, (255, 235, 140, 240), 8, AA)
        for i_s in (-100, -50, 0, 50, 100):
            cv2.line(img, (cx + i_s, cy + 20), (cx + i_s, cy - 40 - abs(i_s) // 3), (255, 255, 255, 255), 5, AA)
    else:
        # Нимб / высокотехнологичный сканер над головой
        cv2.ellipse(img, (cx, cy), (140, 48), 0, 0, 360, (70, 220, 255, 230), 8, AA)
        cv2.ellipse(img, (cx, cy), (140, 48), 0, 0, 360, (255, 255, 255, 255), 3, AA)

    return img


def create_glove_png(hero_id):
    img = np.zeros((512, 512, 4), dtype=np.uint8)
    cx, cy = 256, 256
    # Боевая перчатка / наруч с энерго-узлом
    cv2.circle(img, (cx, cy), 130, (40, 195, 255, 240), -1, AA)
    cv2.circle(img, (cx, cy), 130, (255, 255, 255, 255), 8, AA)
    cv2.circle(img, (cx, cy), 65, (255, 240, 150, 255), -1, AA)
    cv2.circle(img, (cx, cy), 35, (255, 255, 255, 255), -1, AA)
    return img


def generate_hero_art_pngs():
    """Создаёт PNG-ассеты (armor.png, back.png, crown.png, glove.png) для всех 8 героев."""
    print("\n[GEN_ASSETS] 3) Генерация PNG-ассетов героев (assets/<hero_id>/*.png)...")
    for hid in HERO_IDS:
        h_dir = os.path.join(ASSETS_DIR, hid)
        ensure_dir(h_dir)
        cv2.imwrite(os.path.join(h_dir, "armor.png"), create_armor_png(hid))
        cv2.imwrite(os.path.join(h_dir, "back.png"), create_back_png(hid))
        cv2.imwrite(os.path.join(h_dir, "crown.png"), create_crown_png(hid))
        cv2.imwrite(os.path.join(h_dir, "glove.png"), create_glove_png(hid))
        print(f"  [ART] {hid:10s} -> armor.png, back.png, crown.png, glove.png созданы")


def main():
    print("=" * 65)
    print("СИҚЫРЛЫ АЙНА: ГЕНЕРАТОР АССЕТОВ (ЭТАПЫ 3 И 4)")
    print("=" * 65)
    generate_world_videos()
    generate_fx_clips()
    generate_hero_art_pngs()
    print("\n[GEN_ASSETS] Все ассеты успешно созданы в папке assets/!")


if __name__ == "__main__":
    main()
