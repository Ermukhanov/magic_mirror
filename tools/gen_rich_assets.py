"""
Генератор высококачественной кинематографической графики (assets/<hero_id>/*.png)
Создаёт детализированные RGBA PNG-спрайты для всех 8 героев:
1. armor.png  - 512x512 нагрудная броня / корсаж с 3D-градиентами, бликами и рельефом
2. back.png   - 512x512 плащ / крылья / ранец с драпировкой складок и свечением
3. crown.png  - 512x512 королевская корона / шлем / нимб / диадема
4. glove.png  - 256x256 боевые наручи / перчатки с манжетами и заклёпками
"""
import os
import math
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
AA = cv2.LINE_AA


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def add_metallic_shading(img_bgr, mask, highlight_dir=(-0.6, -0.8)):
    """Добавляет реалистичные 3D-блики металла и затенение по нормалям."""
    dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    max_d = max(1.0, np.max(dist))
    norm_d = dist / max_d

    # Градиенты нормалей
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-5
    nx = gx / mag
    ny = gy / mag

    lx, ly = highlight_dir
    dot = np.clip(nx * lx + ny * ly, 0.0, 1.0)
    specular = np.power(dot, 4.0) * 110.0
    ambient = 0.85 + 0.15 * norm_d

    shaded = img_bgr.astype(np.float32) * ambient[:, :, None] + specular[:, :, None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


# ==============================================================================
# 1. АЛПАМЫС БАТЫР (Золотой сауыт, кошкар-муйиз, алый плащ, богатырский шлем)
# ==============================================================================
def gen_alpamys_assets(h_dir):
    ensure_dir(h_dir)
    cx, cy = 256, 256

    # 1.1 ARMOR.PNG (Золотая кованая кираса батыра с орнаментом қошқар мүйіз)
    arm = np.zeros((512, 512, 4), dtype=np.uint8)
    plate_pts = np.array([
        [cx - 165, cy - 145], [cx + 165, cy - 145],
        [cx + 140, cy + 85],  [cx, cy + 185], [cx - 140, cy + 85]
    ], np.int32)
    m_plate = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_plate, [plate_pts], 1.0)

    # Золотая основа с градиентом
    col_plate = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        k = y / 512.0
        # Градиент от полированного золота к медно-золотому
        b = int(35 + 25 * k)
        g = int(185 + 45 * (1.0 - k))
        r = int(245 + 10 * (1.0 - k))
        col_plate[y, :] = (b, g, r)

    shaded_plate = add_metallic_shading(col_plate, m_plate)
    arm[:, :, :3] = shaded_plate
    arm[:, :, 3] = (m_plate * 255).astype(np.uint8)

    # Кожаная оторочка по краям
    cv2.polylines(arm, [plate_pts], True, (25, 45, 140, 255), 10, AA)
    cv2.polylines(arm, [plate_pts], True, (80, 230, 255, 255), 4, AA)

    # Чешуйчатые сегменты брони (ламеллярные пластины)
    for row in range(3):
        ry = cy - 80 + row * 45
        cv2.line(arm, (cx - 130 + row * 8, ry), (cx + 130 - row * 8, ry), (25, 120, 190, 255), 3, AA)
        cv2.line(arm, (cx - 130 + row * 8, ry + 2), (cx + 130 - row * 8, ry + 2), (120, 245, 255, 255), 2, AA)

    # Центральный казахский орнамент «Қошқар мүйіз» (рога барана)
    c_m = (cx, cy + 25)
    cv2.circle(arm, c_m, 62, (25, 35, 175, 255), -1, AA)  # алая сердцевина
    cv2.circle(arm, c_m, 62, (60, 220, 255, 255), 5, AA)  # золотой обод
    cv2.circle(arm, c_m, 22, (50, 225, 255, 255), -1, AA)
    # Завитки рогов
    cv2.ellipse(arm, (cx - 26, cy + 15), (28, 22), -30, 40, 310, (60, 225, 255, 255), 6, AA)
    cv2.ellipse(arm, (cx + 26, cy + 15), (28, 22), 30, -130, 140, (60, 225, 255, 255), 6, AA)
    cv2.imwrite(os.path.join(h_dir, "armor.png"), arm)

    # 1.2 BACK.PNG (Шелково-бархатный алый плащ с золотой каймой и драпировкой)
    back = np.zeros((512, 512, 4), dtype=np.uint8)
    cape_pts = np.array([
        [cx - 110, cy - 200], [cx + 110, cy - 200],
        [cx + 225, cy + 225], [cx + 70, cy + 195],
        [cx, cy + 215],       [cx - 70, cy + 195],
        [cx - 225, cy + 225]
    ], np.int32)
    m_cape = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_cape, [cape_pts], 1.0)

    # Текстура складок бархата
    col_cape = np.zeros((512, 512, 3), dtype=np.uint8)
    for x in range(512):
        fold = math.sin((x - cx) * 0.08) * 35.0
        b = int(np.clip(32 + fold * 0.2, 10, 60))
        g = int(np.clip(38 + fold * 0.3, 15, 80))
        r = int(np.clip(215 + fold, 140, 255))
        col_cape[:, x] = (b, g, r)

    back[:, :, :3] = col_cape
    back[:, :, 3] = (m_cape * 248).astype(np.uint8)
    cv2.polylines(back, [cape_pts], True, (50, 215, 255, 255), 8, AA)
    cv2.imwrite(os.path.join(h_dir, "back.png"), back)

    # 1.3 CROWN.PNG (Богатырский золотой шлем-айдар со стреловидным навершием)
    cr = np.zeros((512, 512, 4), dtype=np.uint8)
    helm_pts = np.array([
        [cx - 135, cy + 50], [cx - 120, cy - 35],
        [cx - 50, cy - 85],  [cx, cy - 145],
        [cx + 50, cy - 85],  [cx + 120, cy - 35],
        [cx + 135, cy + 50]
    ], np.int32)
    cv2.fillPoly(cr, [helm_pts], (40, 205, 255, 255), AA)
    cv2.polylines(cr, [helm_pts], True, (80, 235, 255, 255), 6, AA)
    cv2.line(cr, (cx, cy - 145), (cx, cy - 185), (80, 240, 255, 255), 8, AA)
    cv2.circle(cr, (cx, cy - 192), 12, (30, 45, 235, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "crown.png"), cr)

    # 1.4 GLOVE.PNG (Золотые кованые наручи сауыт жеңсесі)
    gl = np.zeros((256, 256, 4), dtype=np.uint8)
    gcx, gcy = 128, 128
    cv2.rectangle(gl, (gcx - 55, gcy - 75), (gcx + 55, gcy + 75), (35, 195, 255, 255), -1, AA)
    cv2.rectangle(gl, (gcx - 55, gcy - 75), (gcx + 55, gcy + 75), (70, 235, 255, 255), 6, AA)
    cv2.line(gl, (gcx - 55, gcy), (gcx + 55, gcy), (20, 40, 160, 255), 4, AA)
    cv2.circle(gl, (gcx, gcy), 18, (30, 35, 215, 255), -1, AA)
    cv2.circle(gl, (gcx, gcy), 18, (70, 235, 255, 255), 3, AA)
    cv2.imwrite(os.path.join(h_dir, "glove.png"), gl)


# ==============================================================================
# 2. ТОМИРИС (Царский сакский доспех, пурпур, рубиновое солнце, золотая диадема)
# ==============================================================================
def gen_tomiris_assets(h_dir):
    ensure_dir(h_dir)
    cx, cy = 256, 256

    # 2.1 ARMOR.PNG (Сакский золотой чешуйчатый панцирь с царским рубином)
    arm = np.zeros((512, 512, 4), dtype=np.uint8)
    plate_pts = np.array([
        [cx - 160, cy - 140], [cx + 160, cy - 140],
        [cx + 135, cy + 90],  [cx, cy + 180], [cx - 135, cy + 90]
    ], np.int32)
    m_plate = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_plate, [plate_pts], 1.0)

    col_plate = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        k = y / 512.0
        col_plate[y, :] = (int(125 * (1 - k) + 40 * k), int(45 + 140 * k), int(160 + 80 * (1 - k)))

    arm[:, :, :3] = add_metallic_shading(col_plate, m_plate)
    arm[:, :, 3] = (m_plate * 255).astype(np.uint8)
    cv2.polylines(arm, [plate_pts], True, (70, 215, 255, 255), 8, AA)

    # Царский рубиновый медальон Сакской царицы
    c_m = (cx, cy + 25)
    cv2.circle(arm, c_m, 65, (45, 195, 255, 255), -1, AA)
    cv2.circle(arm, c_m, 50, (30, 20, 185, 255), -1, AA)  # гранёный рубин
    cv2.circle(arm, c_m, 20, (120, 80, 255, 255), -1, AA)  # сверкающий блик
    for i in range(8):
        ang = i * math.pi / 4.0
        p1 = (int(cx + math.cos(ang) * 52), int(cy + 25 + math.sin(ang) * 52))
        p2 = (int(cx + math.cos(ang) * 74), int(cy + 25 + math.sin(ang) * 74))
        cv2.line(arm, p1, p2, (80, 235, 255, 255), 4, AA)
    cv2.imwrite(os.path.join(h_dir, "armor.png"), arm)

    # 2.2 BACK.PNG (Царский пурпурный плащ с золотой каймой)
    back = np.zeros((512, 512, 4), dtype=np.uint8)
    cape_pts = np.array([
        [cx - 105, cy - 200], [cx + 105, cy - 200],
        [cx + 215, cy + 220], [cx, cy + 195], [cx - 215, cy + 220]
    ], np.int32)
    cv2.fillPoly(back, [cape_pts], (140, 35, 160, 245), AA)
    cv2.polylines(back, [cape_pts], True, (60, 210, 255, 255), 8, AA)
    cv2.imwrite(os.path.join(h_dir, "back.png"), back)

    # 2.3 CROWN.PNG (Сакская золотая тиара с крылатыми грифонами)
    cr = np.zeros((512, 512, 4), dtype=np.uint8)
    tiara_pts = np.array([
        [cx - 145, cy + 40], [cx - 130, cy - 50], [cx - 65, cy - 15],
        [cx, cy - 95],       [cx + 65, cy - 15], [cx + 130, cy - 50],
        [cx + 145, cy + 40]
    ], np.int32)
    cv2.fillPoly(cr, [tiara_pts], (45, 210, 255, 255), AA)
    cv2.polylines(cr, [tiara_pts], True, (90, 245, 255, 255), 6, AA)
    cv2.circle(cr, (cx, cy - 45), 14, (30, 25, 215, 255), -1, AA)  # рубин
    cv2.imwrite(os.path.join(h_dir, "crown.png"), cr)

    # 2.4 GLOVE.PNG (Золотые сакские браслеты)
    gl = np.zeros((256, 256, 4), dtype=np.uint8)
    gcx, gcy = 128, 128
    cv2.rectangle(gl, (gcx - 55, gcy - 70), (gcx + 55, gcy + 70), (50, 205, 255, 255), -1, AA)
    cv2.rectangle(gl, (gcx - 55, gcy - 70), (gcx + 55, gcy + 70), (80, 235, 255, 255), 6, AA)
    cv2.circle(gl, (gcx, gcy), 16, (30, 25, 205, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "glove.png"), gl)


# ==============================================================================
# 3. ӨРМЕКШІ-АДАМ (Текстура нано-костюма, рельефная паутина, чёрный паук)
# ==============================================================================
def gen_spider_assets(h_dir):
    ensure_dir(h_dir)
    cx, cy = 256, 256

    # 3.1 ARMOR.PNG (Сверхдетализированный красно-синий костюм супергероя)
    arm = np.zeros((512, 512, 4), dtype=np.uint8)
    plate_pts = np.array([
        [cx - 150, cy - 135], [cx + 150, cy - 135],
        [cx + 125, cy + 95],  [cx, cy + 175], [cx - 125, cy + 95]
    ], np.int32)
    m_plate = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_plate, [plate_pts], 1.0)

    # Красная основа с нано-сотами
    col = np.zeros((512, 512, 3), dtype=np.uint8)
    col[:] = (35, 38, 225)
    for y in range(0, 512, 8):
        cv2.line(col, (0, y), (512, y), (28, 30, 195), 1)

    arm[:, :, :3] = add_metallic_shading(col, m_plate)
    arm[:, :, 3] = (m_plate * 255).astype(np.uint8)
    cv2.polylines(arm, [plate_pts], True, (25, 25, 30, 255), 7, AA)

    # Радиальная паутина
    c_m = (cx, cy + 25)
    for r in (38, 75, 115, 155):
        cv2.ellipse(arm, c_m, (r, int(r * 0.85)), 0, 0, 360, (25, 25, 30, 255), 3, AA)
    for ang_d in range(0, 360, 30):
        rad = math.radians(ang_d)
        p2 = (int(cx + math.cos(rad) * 165), int(cy + 25 + math.sin(rad) * 145))
        cv2.line(arm, c_m, p2, (25, 25, 30, 255), 3, AA)

    # Рельефный чёрный паук на груди
    cv2.ellipse(arm, c_m, (14, 22), 0, 0, 360, (20, 20, 22, 255), -1, AA)
    cv2.circle(arm, (cx, cy + 4), 9, (20, 20, 22, 255), -1, AA)
    # 8 лап паука
    for s_x in (-1, 1):
        for idx_l, dy in enumerate((-12, -4, 4, 12)):
            p_joint = (cx + s_x * 28, cy + 18 + dy * 2)
            p_tip = (cx + s_x * 48, cy + 10 + dy * 4)
            cv2.line(arm, (cx + s_x * 8, cy + 25 + dy), p_joint, (20, 20, 22, 255), 4, AA)
            cv2.line(arm, p_joint, p_tip, (20, 20, 22, 255), 3, AA)

    cv2.imwrite(os.path.join(h_dir, "armor.png"), arm)

    # 3.2 BACK.PNG (Паучья аура / накидка с паутиной)
    back = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.ellipse(back, (cx, cy), (210, 210), 0, 0, 360, (215, 70, 30, 160), -1, AA)
    for r in (80, 140, 200):
        cv2.circle(back, (cx, cy), r, (255, 255, 255, 200), 2, AA)
    cv2.imwrite(os.path.join(h_dir, "back.png"), back)

    # 3.3 CROWN.PNG (Паучий нимб-паутина над головой)
    cr = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.ellipse(cr, (cx, cy), (160, 48), 0, 0, 360, (30, 40, 235, 255), 6, AA)
    cv2.ellipse(cr, (cx, cy), (160, 48), 0, 0, 360, (255, 255, 255, 255), 2, AA)
    cv2.imwrite(os.path.join(h_dir, "crown.png"), cr)

    # 3.4 GLOVE.PNG (Перчатки с паутиной и веб-шутером)
    gl = np.zeros((256, 256, 4), dtype=np.uint8)
    gcx, gcy = 128, 128
    cv2.rectangle(gl, (gcx - 50, gcy - 65), (gcx + 50, gcy + 65), (35, 38, 225, 255), -1, AA)
    cv2.rectangle(gl, (gcx - 50, gcy - 65), (gcx + 50, gcy + 65), (25, 25, 30, 255), 5, AA)
    cv2.circle(gl, (gcx, gcy), 14, (240, 240, 245, 255), -1, AA)  # сопло веб-шутера
    cv2.circle(gl, (gcx, gcy), 8, (30, 30, 35, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "glove.png"), gl)


# ==============================================================================
# 4. ЭЛЬЗА (Кристальный корсаж, ледяные грани, сияющая снежинка, диадема)
# ==============================================================================
def gen_elsa_assets(h_dir):
    ensure_dir(h_dir)
    cx, cy = 256, 256

    # 4.1 ARMOR.PNG (Кристальный гранёный ледяной лиф)
    arm = np.zeros((512, 512, 4), dtype=np.uint8)
    plate_pts = np.array([
        [cx - 150, cy - 130], [cx + 150, cy - 130],
        [cx + 120, cy + 90],  [cx, cy + 170], [cx - 120, cy + 90]
    ], np.int32)
    m_plate = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_plate, [plate_pts], 1.0)

    col = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        k = y / 512.0
        col[y, :] = (int(255 - 20 * k), int(225 - 40 * k), int(120 + 80 * k))

    arm[:, :, :3] = add_metallic_shading(col, m_plate)
    arm[:, :, 3] = (m_plate * 255).astype(np.uint8)
    cv2.polylines(arm, [plate_pts], True, (255, 255, 255, 255), 6, AA)

    # Сияющая 6-лучевая ледяная снежинка
    c_m = (cx, cy + 25)
    for i in range(6):
        ang = i * math.pi / 3.0
        p_end = (int(cx + math.cos(ang) * 58), int(cy + 25 + math.sin(ang) * 58))
        cv2.line(arm, c_m, p_end, (255, 255, 255, 255), 4, AA)
        for b_ang in (-0.5, 0.5):
            pb = (int(p_end[0] + math.cos(ang + b_ang) * 16), int(p_end[1] + math.sin(ang + b_ang) * 16))
            cv2.line(arm, p_end, pb, (255, 250, 220, 255), 2, AA)
    cv2.circle(arm, c_m, 12, (255, 255, 255, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "armor.png"), arm)

    # 4.2 BACK.PNG (Шлейф из сияющего морозного льда)
    back = np.zeros((512, 512, 4), dtype=np.uint8)
    cape_pts = np.array([
        [cx - 95, cy - 190], [cx + 95, cy - 190],
        [cx + 205, cy + 225], [cx, cy + 200], [cx - 205, cy + 225]
    ], np.int32)
    cv2.fillPoly(back, [cape_pts], (255, 225, 140, 210), AA)
    cv2.polylines(back, [cape_pts], True, (255, 255, 255, 255), 5, AA)
    cv2.imwrite(os.path.join(h_dir, "back.png"), back)

    # 4.3 CROWN.PNG (Ледяная кристаллическая диадема)
    cr = np.zeros((512, 512, 4), dtype=np.uint8)
    tiara = np.array([
        [cx - 130, cy + 35], [cx - 110, cy - 55], [cx - 55, cy - 15],
        [cx, cy - 105],      [cx + 55, cy - 15], [cx + 110, cy - 55],
        [cx + 130, cy + 35]
    ], np.int32)
    cv2.fillPoly(cr, [tiara], (255, 235, 165, 255), AA)
    cv2.polylines(cr, [tiara], True, (255, 255, 255, 255), 5, AA)
    cv2.circle(cr, (cx, cy - 40), 10, (255, 255, 255, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "crown.png"), cr)

    # 4.4 GLOVE.PNG (Ледяные перчатки со снежинками)
    gl = np.zeros((256, 256, 4), dtype=np.uint8)
    gcx, gcy = 128, 128
    cv2.rectangle(gl, (gcx - 50, gcy - 65), (gcx + 50, gcy + 65), (255, 220, 140, 255), -1, AA)
    cv2.rectangle(gl, (gcx - 50, gcy - 65), (gcx + 50, gcy + 65), (255, 255, 255, 255), 5, AA)
    cv2.circle(gl, (gcx, gcy), 14, (255, 255, 255, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "glove.png"), gl)


# ==============================================================================
# 5. ЛЕДИБАГ (Алый костюм, объёмные глянцевые чёрные точки, воротничок)
# ==============================================================================
def gen_ladybug_assets(h_dir):
    ensure_dir(h_dir)
    cx, cy = 256, 256

    # 5.1 ARMOR.PNG (Сверхглянцевый костюм Ледибаг)
    arm = np.zeros((512, 512, 4), dtype=np.uint8)
    plate_pts = np.array([
        [cx - 150, cy - 135], [cx + 150, cy - 135],
        [cx + 125, cy + 95],  [cx, cy + 175], [cx - 125, cy + 95]
    ], np.int32)
    m_plate = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_plate, [plate_pts], 1.0)

    col = np.zeros((512, 512, 3), dtype=np.uint8)
    col[:] = (40, 42, 235)  # алый глянец

    arm[:, :, :3] = add_metallic_shading(col, m_plate)
    arm[:, :, 3] = (m_plate * 255).astype(np.uint8)
    cv2.polylines(arm, [plate_pts], True, (25, 25, 30, 255), 6, AA)

    # Фирменные чёрные точки божьей коровки с бликами
    dots = [
        (cx, cy + 25, 26),
        (cx - 75, cy - 35, 22), (cx + 75, cy - 35, 22),
        (cx - 65, cy + 85, 20), (cx + 65, cy + 85, 20),
    ]
    for dx, dy, dr in dots:
        cv2.circle(arm, (dx, dy), dr, (22, 22, 25, 255), -1, AA)
        # Блик на глянцевой точке
        cv2.circle(arm, (dx - int(dr * 0.35), dy - int(dr * 0.35)), int(dr * 0.28), (95, 95, 110, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "armor.png"), arm)

    # 5.2 BACK.PNG (Крылышки божьей коровки)
    back = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.ellipse(back, (cx - 80, cy), (95, 160), -20, 0, 360, (35, 38, 220, 210), -1, AA)
    cv2.ellipse(back, (cx + 80, cy), (95, 160), 20, 0, 360, (35, 38, 220, 210), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "back.png"), back)

    # 5.3 CROWN.PNG (Усики божьей коровки)
    cr = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.line(cr, (cx - 20, cy + 30), (cx - 60, cy - 70), (25, 25, 30, 255), 5, AA)
    cv2.line(cr, (cx + 20, cy + 30), (cx + 60, cy - 70), (25, 25, 30, 255), 5, AA)
    cv2.circle(cr, (cx - 62, cy - 74), 10, (35, 38, 235, 255), -1, AA)
    cv2.circle(cr, (cx + 62, cy - 74), 10, (35, 38, 235, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "crown.png"), cr)

    # 5.4 GLOVE.PNG (Перчатки с чёрными пятнышками)
    gl = np.zeros((256, 256, 4), dtype=np.uint8)
    gcx, gcy = 128, 128
    cv2.rectangle(gl, (gcx - 50, gcy - 65), (gcx + 50, gcy + 65), (40, 42, 235, 255), -1, AA)
    cv2.rectangle(gl, (gcx - 50, gcy - 65), (gcx + 50, gcy + 65), (25, 25, 30, 255), 5, AA)
    cv2.circle(gl, (gcx, gcy), 14, (22, 22, 25, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "glove.png"), gl)


# ==============================================================================
# 6. ТЕМІР БАТЫР (Железный Батыр: титаново-золотая броня, сияющий дуговой реактор)
# ==============================================================================
def gen_iron_assets(h_dir):
    ensure_dir(h_dir)
    cx, cy = 256, 256

    # 6.1 ARMOR.PNG (Сверхтехнологичная броня с дуговым реактором)
    arm = np.zeros((512, 512, 4), dtype=np.uint8)
    plate_pts = np.array([
        [cx - 165, cy - 140], [cx + 165, cy - 140],
        [cx + 140, cy + 85],  [cx, cy + 185], [cx - 140, cy + 85]
    ], np.int32)
    m_plate = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_plate, [plate_pts], 1.0)

    # Бордово-красный титан
    col = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        k = y / 512.0
        col[y, :] = (int(30 + 15 * k), int(35 + 20 * k), int(215 - 35 * k))

    arm[:, :, :3] = add_metallic_shading(col, m_plate)
    arm[:, :, 3] = (m_plate * 255).astype(np.uint8)
    cv2.polylines(arm, [plate_pts], True, (45, 205, 255, 255), 8, AA)

    # Золотые пластины на ключицах
    cv2.rectangle(arm, (cx - 140, cy - 125), (cx - 40, cy - 70), (45, 205, 255, 255), -1, AA)
    cv2.rectangle(arm, (cx + 40, cy - 125), (cx + 140, cy - 70), (45, 205, 255, 255), -1, AA)

    # СИЯЮЩИЙ ДУГОВОЙ РЕАКТОР (Arc Reactor)
    c_m = (cx, cy + 25)
    cv2.circle(arm, c_m, 62, (255, 240, 180, 255), -1, AA)  # внешнее свечение
    cv2.circle(arm, c_m, 50, (30, 35, 45, 255), -1, AA)     # тёмный кожух
    cv2.circle(arm, c_m, 40, (255, 215, 60, 255), 5, AA)    # неоновое кольцо
    cv2.circle(arm, c_m, 26, (255, 255, 255, 255), -1, AA)  # ядро сингулярности
    for idx_s in range(10):
        ang = idx_s * 2.0 * math.pi / 10.0
        p_in = (int(cx + math.cos(ang) * 28), int(cy + 25 + math.sin(ang) * 28))
        p_out = (int(cx + math.cos(ang) * 44), int(cy + 25 + math.sin(ang) * 44))
        cv2.line(arm, p_in, p_out, (255, 240, 120, 255), 3, AA)

    cv2.imwrite(os.path.join(h_dir, "armor.png"), arm)

    # 6.2 BACK.PNG (Репульсорные закрылки)
    back = np.zeros((512, 512, 4), dtype=np.uint8)
    cape_pts = np.array([
        [cx - 100, cy - 190], [cx + 100, cy - 190],
        [cx + 200, cy + 210], [cx, cy + 185], [cx - 200, cy + 210]
    ], np.int32)
    cv2.fillPoly(back, [cape_pts], (35, 42, 205, 240), AA)
    cv2.polylines(back, [cape_pts], True, (45, 205, 255, 255), 8, AA)
    cv2.imwrite(os.path.join(h_dir, "back.png"), back)

    # 6.3 CROWN.PNG (Шлем Железного Человека с золотой маской)
    cr = np.zeros((512, 512, 4), dtype=np.uint8)
    h_pts = np.array([
        [cx - 120, cy + 45], [cx - 110, cy - 45], [cx, cy - 105],
        [cx + 110, cy - 45], [cx + 120, cy + 45]
    ], np.int32)
    cv2.fillPoly(cr, [h_pts], (45, 205, 255, 255), AA)
    cv2.polylines(cr, [h_pts], True, (35, 45, 220, 255), 8, AA)
    cv2.line(cr, (cx - 65, cy - 15), (cx - 20, cy - 15), (255, 255, 255, 255), 6, AA)
    cv2.line(cr, (cx + 20, cy - 15), (cx + 65, cy - 15), (255, 255, 255, 255), 6, AA)
    cv2.imwrite(os.path.join(h_dir, "crown.png"), cr)

    # 6.4 GLOVE.PNG (Перчатка с репульсором)
    gl = np.zeros((256, 256, 4), dtype=np.uint8)
    gcx, gcy = 128, 128
    cv2.rectangle(gl, (gcx - 55, gcy - 70), (gcx + 55, gcy + 70), (35, 42, 215, 255), -1, AA)
    cv2.rectangle(gl, (gcx - 55, gcy - 70), (gcx + 55, gcy + 70), (45, 205, 255, 255), 6, AA)
    cv2.circle(gl, (gcx, gcy), 18, (255, 255, 255, 255), -1, AA)  # репульсор
    cv2.circle(gl, (gcx, gcy), 24, (255, 220, 80, 255), 3, AA)
    cv2.imwrite(os.path.join(h_dir, "glove.png"), gl)


# ==============================================================================
# 7. ҒАРЫШКЕР (Белый скафандр космонавта с телеметрией, ранцем и гербом)
# ==============================================================================
def gen_astronaut_assets(h_dir):
    ensure_dir(h_dir)
    cx, cy = 256, 256

    # 7.1 ARMOR.PNG (Грудной пульт жизнеобеспечения скафандра)
    arm = np.zeros((512, 512, 4), dtype=np.uint8)
    plate_pts = np.array([
        [cx - 155, cy - 130], [cx + 155, cy - 130],
        [cx + 130, cy + 95],  [cx, cy + 175], [cx - 130, cy + 95]
    ], np.int32)
    m_plate = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_plate, [plate_pts], 1.0)

    # Белая высокопрочная ткань
    col = np.zeros((512, 512, 3), dtype=np.uint8)
    col[:] = (240, 242, 245)

    arm[:, :, :3] = add_metallic_shading(col, m_plate)
    arm[:, :, 3] = (m_plate * 255).astype(np.uint8)
    cv2.polylines(arm, [plate_pts], True, (70, 80, 225, 255), 6, AA)

    # Пульт телеметрии с тумблерами и индикаторами
    cv2.rectangle(arm, (cx - 95, cy - 35), (cx + 95, cy + 95), (60, 65, 75, 255), -1, AA)
    cv2.rectangle(arm, (cx - 95, cy - 35), (cx + 95, cy + 95), (200, 205, 215, 255), 3, AA)
    # Светодиоды
    cv2.circle(arm, (cx - 65, cy + 15), 10, (50, 220, 80, 255), -1, AA)   # зелёный
    cv2.circle(arm, (cx - 30, cy + 15), 10, (40, 215, 255, 255), -1, AA)  # жёлтый
    cv2.circle(arm, (cx + 5, cy + 15), 10, (255, 120, 40, 255), -1, AA)   # синий
    cv2.circle(arm, (cx + 40, cy + 15), 10, (40, 50, 240, 255), -1, AA)   # красный
    # Шкала давления
    cv2.rectangle(arm, (cx - 70, cy + 50), (cx + 70, cy + 75), (25, 30, 35, 255), -1, AA)
    cv2.rectangle(arm, (cx - 66, cy + 54), (cx + 25, cy + 71), (255, 210, 50, 255), -1, AA)

    cv2.imwrite(os.path.join(h_dir, "armor.png"), arm)

    # 7.2 BACK.PNG (Ранец космонавта)
    back = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.rectangle(back, (cx - 150, cy - 140), (cx + 150, cy + 140), (225, 228, 235, 245), -1, AA)
    cv2.rectangle(back, (cx - 150, cy - 140), (cx + 150, cy + 140), (60, 160, 255, 255), 6, AA)
    for sx in (cx - 75, cx + 75):
        cv2.circle(back, (sx, cy + 155), 26, (50, 200, 255, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "back.png"), back)

    # 7.3 CROWN.PNG (Обод шлема скафандра с золотым светофильтром)
    cr = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.ellipse(cr, (cx, cy), (165, 55), 0, 0, 360, (235, 238, 245, 255), 12, AA)
    cv2.ellipse(cr, (cx, cy), (165, 55), 0, 0, 360, (50, 210, 255, 255), 4, AA)
    cv2.imwrite(os.path.join(h_dir, "crown.png"), cr)

    # 7.4 GLOVE.PNG (Перчатки космонавта с датчиками давления)
    gl = np.zeros((256, 256, 4), dtype=np.uint8)
    gcx, gcy = 128, 128
    cv2.rectangle(gl, (gcx - 55, gcy - 70), (gcx + 55, gcy + 70), (235, 238, 242, 255), -1, AA)
    cv2.rectangle(gl, (gcx - 55, gcy - 70), (gcx + 55, gcy + 70), (60, 75, 225, 255), 6, AA)
    cv2.circle(gl, (gcx, gcy), 16, (40, 185, 255, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "glove.png"), gl)


# ==============================================================================
# 8. ПЕРІ (Волшебные крылья бабочки, изумрудный корсаж, лепестковая диадема)
# ==============================================================================
def gen_fairy_assets(h_dir):
    ensure_dir(h_dir)
    cx, cy = 256, 256

    # 8.1 ARMOR.PNG (Цветочный лепестковый корсаж с золотым цветком)
    arm = np.zeros((512, 512, 4), dtype=np.uint8)
    plate_pts = np.array([
        [cx - 140, cy - 120], [cx + 140, cy - 120],
        [cx + 115, cy + 90],  [cx, cy + 165], [cx - 115, cy + 90]
    ], np.int32)
    m_plate = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(m_plate, [plate_pts], 1.0)

    # Изумрудный шелк
    col = np.zeros((512, 512, 3), dtype=np.uint8)
    col[:] = (95, 215, 80)

    arm[:, :, :3] = add_metallic_shading(col, m_plate)
    arm[:, :, 3] = (m_plate * 255).astype(np.uint8)
    cv2.polylines(arm, [plate_pts], True, (215, 160, 255, 255), 6, AA)

    # Волшебный цветок в центре
    c_m = (cx, cy + 25)
    for i in range(5):
        ang = i * 2.0 * math.pi / 5.0 - math.pi / 2.0
        pet = (int(cx + math.cos(ang) * 26), int(cy + 25 + math.sin(ang) * 26))
        cv2.circle(arm, pet, 18, (215, 140, 255, 255), -1, AA)
        cv2.circle(arm, pet, 18, (255, 255, 255, 255), 2, AA)
    cv2.circle(arm, c_m, 14, (60, 235, 255, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "armor.png"), arm)

    # 8.2 BACK.PNG (Мерцающие прозрачные крылья бабочки)
    back = np.zeros((512, 512, 4), dtype=np.uint8)
    for s_dir in (-1, 1):
        w_pts = np.array([
            [cx, cy - 40],
            [cx + s_dir * 230, cy - 170],
            [cx + s_dir * 240, cy + 50],
            [cx + s_dir * 150, cy + 180],
            [cx, cy + 80]
        ], np.int32)
        cv2.fillPoly(back, [w_pts], (235, 160, 250, 195), AA)
        cv2.polylines(back, [w_pts], True, (255, 255, 255, 255), 4, AA)
        for i_v in range(4):
            v_end = (int(cx + s_dir * (140 + i_v * 25)), int(cy - 120 + i_v * 75))
            cv2.line(back, (cx, cy + 20), v_end, (255, 255, 255, 180), 2, AA)
    cv2.imwrite(os.path.join(h_dir, "back.png"), back)

    # 8.3 CROWN.PNG (Цветочный венец феи с сияющими росинками)
    cr = np.zeros((512, 512, 4), dtype=np.uint8)
    cv2.ellipse(cr, (cx, cy), (150, 42), 0, 0, 360, (115, 220, 85, 255), 8, AA)
    for ang_d in range(0, 360, 45):
        rad = math.radians(ang_d)
        pt = (int(cx + math.cos(rad) * 150), int(cy + math.sin(rad) * 42))
        cv2.circle(cr, pt, 9, (220, 150, 255, 255), -1, AA)
        cv2.circle(cr, pt, 4, (255, 255, 255, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "crown.png"), cr)

    # 8.4 GLOVE.PNG (Браслеты из цветов)
    gl = np.zeros((256, 256, 4), dtype=np.uint8)
    gcx, gcy = 128, 128
    cv2.rectangle(gl, (gcx - 50, gcy - 60), (gcx + 50, gcy + 60), (110, 220, 80, 255), -1, AA)
    cv2.rectangle(gl, (gcx - 50, gcy - 60), (gcx + 50, gcy + 60), (215, 160, 255, 255), 5, AA)
    cv2.circle(gl, (gcx, gcy), 15, (230, 165, 255, 255), -1, AA)
    cv2.imwrite(os.path.join(h_dir, "glove.png"), gl)


def main():
    generators = {
        "alpamys":   gen_alpamys_assets,
        "tomiris":   gen_tomiris_assets,
        "spider":    gen_spider_assets,
        "elsa":      gen_elsa_assets,
        "ladybug":   gen_ladybug_assets,
        "iron":      gen_iron_assets,
        "astronaut": gen_astronaut_assets,
        "fairy":     gen_fairy_assets,
    }
    print("=" * 70)
    print("ГЕНЕРАЦИЯ ДЕТАЛИЗИРОВАННЫХ RGBA PNG АССЕТОВ ДЛЯ ВСЕХ 8 ГЕРОЕВ")
    print("=" * 70)
    for hid, fn in generators.items():
        h_dir = os.path.join(ASSETS_DIR, hid)
        fn(h_dir)
        print(f"  [OK] {hid:10s} -> armor.png, back.png, crown.png, glove.png созданы с 3D-шейдингом!")
    print("=" * 70)
    print("Все ассеты успешно обновлены!")


if __name__ == "__main__":
    main()
