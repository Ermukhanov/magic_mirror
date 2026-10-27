"""
Модуль реального ИИ-превращения ребёнка в героя «Сиқырлы айна» (Этап 2: ai_transform.py):
- Отдельный фоновый поток на GPU (SD-Turbo fp16 + TAESD, локально из models/, 100% офлайн).
- Профессиональное качество прорисовки костюмов для ВСЕХ 8 ГЕРОЕВ:
  - Использование высококачественных PNG-ассетов брони, плащей, крыльев и наручей (assets/<hero>/).
  - Сбалансированная сила ИИ (strength = 0.48, 2 шага) для реалистичных металлических бликов,
    естественных складок ткани и отсутствия мутаций/деформаций.
- Полное устранение задержек и отставания тела от головы:
  - 2D-аффинное позовое выравнивание (warp_ai_to_live_pose) по текущему положению скелета
    гарантирует моментальный отклик и плавные 30+ FPS без "плавающей головы"!
- 100% настоящее живое лицо, причёска, шея и кисти рук (hands_live: true) с естественным
  нахлёстом воротника и отсутствием серого гало.
"""
import os
import math
import time
import logging
import threading
import cv2
import numpy as np

import config
from config import W, H, AA
from tracking import ip, clean_person_mask
from heroes import load_hero_png, overlay_rgba_sprite

MODELS_DIR = os.path.join(config.BASE_DIR, "models")
SD_TURBO_DIR = os.path.join(MODELS_DIR, "sd-turbo")
TAESD_DIR = os.path.join(MODELS_DIR, "taesd")

# Промпты кинематографического качества для каждого из 8 героев (яркое праздничное освещение)
HERO_PROMPTS = {
    "alpamys": "masterpiece character art of a heroic Kazakh batyr child, shining polished gold armor plate with steppe ornaments, flowing bright red cape, golden vambraces, bright vibrant daylight lighting, brilliant bright colors, cinematic 8k masterpiece",
    "tomiris": "masterpiece character art of a royal Saka warrior queen child, shining gold and vibrant purple armor plate, ruby medallion, royal purple cape, bright vibrant daylight lighting, brilliant bright colors, cinematic 8k masterpiece",
    "spider": "masterpiece character art of a child superhero, vivid bright red and electric blue superhero suit, sharp black spiderweb texture and spider emblem, bright daylight lighting, brilliant bright colors, cinematic 8k masterpiece",
    "elsa": "masterpiece character art of a child ice princess, sparkling light blue crystal frost gown, glowing ice snowflake crystal bodice, translucent shimmering cape, bright vibrant daylight lighting, brilliant bright colors, cinematic 8k masterpiece",
    "ladybug": "masterpiece character art of a child superheroine, vibrant scarlet red superhero suit with crisp bold black polka dots, high collar, bright daylight lighting, brilliant bright colors, cinematic 8k masterpiece",
    "iron": "masterpiece character art of a child superhero, sleek metallic crimson and bright gold high-tech armor, glowing radiant cyan chest reactor, bright daylight lighting, brilliant bright colors, cinematic 8k masterpiece",
    "astronaut": "masterpiece character art of a child astronaut, pristine bright white pressurized space suit, telemetry chest control module, mission patches, bright sunny daylight lighting, brilliant colors, cinematic 8k masterpiece",
    "fairy": "masterpiece character art of a child magical forest fairy, glowing emerald green petal dress, rose corsage, shimmering translucent rainbow butterfly wings, bright daylight lighting, brilliant bright colors, cinematic 8k masterpiece",
}
NEGATIVE_PROMPT = "dark, underexposed, gloomy, dim, shadows, night, murky, silhouette, bad anatomy, extra limbs, deformed hands, distorted face, blurry, ugly, low quality, artifacts, text, logo"

# Опорные цвета костюмов для 8 героев (яркие, сочные, праздничные тона BGR)
HERO_SKETCH_PALETTES = {
    "alpamys": dict(
        tunic=(35, 40, 220),     # яркий царский багряный кафтан
        armor=(45, 195, 255),    # сияющее батырское золото
        cape=(30, 35, 205),      # струящийся алый плащ
        belt=(35, 110, 210),     # кожаный пояс с золотой пряжкой
        bracer=(45, 195, 255),   # золотые наручи
        trim=(80, 225, 255),     # античный золотой орнамент
        bg_fill=(85, 120, 195),  # светлый фон
    ),
    "tomiris": dict(
        tunic=(155, 45, 140),    # царский сакский пурпур
        armor=(45, 195, 255),    # древнее сакское золото
        cape=(125, 35, 110),     # глубокий пурпурный плащ
        belt=(45, 195, 255),     # золотой пояс
        bracer=(45, 195, 255),   # золотые наручи
        trim=(80, 225, 255),     # золотая филигрань и рубин
        bg_fill=(125, 95, 165),
    ),
    "spider": dict(
        tunic=(210, 65, 35),     # яркий насыщенный синий
        armor=(35, 45, 235),     # пламенный алый паучий жилет
        cape=None,
        belt=(35, 45, 235),
        bracer=(35, 45, 235),
        trim=(25, 25, 30),       # чёткая паутина
        bg_fill=(120, 75, 130),
    ),
    "elsa": dict(
        tunic=(255, 195, 100),   # кристальный лазурный
        armor=(255, 230, 165),   # ледяной морозный корсет
        cape=(255, 220, 145),    # струящийся полупрозрачный шлейф
        belt=(255, 235, 185),
        bracer=(255, 225, 155),
        trim=(255, 255, 255),    # сияющий алмазный кристалл
        bg_fill=(210, 180, 130),
    ),
    "ladybug": dict(
        tunic=(35, 40, 235),     # сочный рубиново-алый костюм
        armor=(35, 45, 245),
        cape=None,
        belt=(25, 25, 30),
        bracer=(35, 40, 235),
        trim=(20, 20, 25),       # матовые точки
        bg_fill=(85, 85, 165),
    ),
    "iron": dict(
        tunic=(35, 40, 215),     # титановый винно-красный
        armor=(45, 195, 255),    # сияющее полированное золото
        cape=None,
        belt=(35, 45, 215),
        bracer=(35, 45, 215),
        trim=(255, 245, 120),    # сияющий дуговой реактор
        bg_fill=(85, 105, 180),
    ),
    "astronaut": dict(
        tunic=(240, 245, 250),   # композитный лунный белый
        armor=(200, 205, 215),   # титановый модуль
        cape=(210, 215, 225),    # ранец
        belt=(205, 95, 55),
        bracer=(225, 230, 235),
        trim=(220, 95, 55),      # синие эмблемы
        bg_fill=(165, 170, 180),
    ),
    "fairy": dict(
        tunic=(80, 210, 65),     # изумрудный мох
        armor=(205, 120, 255),   # орхидейно-розовый корсаж
        cape=(240, 175, 255),    # радужные крылья бабочки
        belt=(70, 215, 245),
        bracer=(195, 130, 245),
        trim=(120, 245, 255),    # золотисто-янтарная пыльца
        bg_fill=(145, 185, 145),
    ),
}

_yy_ai = np.arange(2 * H)
_scan_ai_u8 = np.where(_yy_ai % 4 < 2, 255, 248).astype(np.uint8)
SCAN_AI = np.repeat(np.repeat(_scan_ai_u8[:, None, None], W, 1), 3, 2)


def compute_face_and_hands_mask(h_img, w_img, p_local, hands_live=True):
    """
    Строит мягкую маску (0.0..1.0) головы, шеи и кистей рук.
    Шея плавно заходит под воротник брони, исключая резкие границы и зазоры.
    """
    ds = 4
    sw_g = max(16, w_img // ds)
    sh_g = max(16, h_img // ds)
    live_mask_s = np.zeros((sh_g, sw_g), dtype=np.float32)

    if p_local is None:
        cv2.ellipse(
            live_mask_s,
            (sw_g // 2, int(sh_g * 0.24)),
            (int(sw_g * 0.16), int(sh_g * 0.13)),
            0, 0, 360, 1.0, -1,
        )
        live_mask_s = cv2.GaussianBlur(live_mask_s, (11, 11), 3.0)
        return cv2.resize(live_mask_s, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    scale_w = w_img / 640.0
    nose = p_local[0, :2] / ds
    ls, rs = p_local[11, :2] / ds, p_local[12, :2] / ds
    sw = float(np.linalg.norm(ls - rs))
    if sw < (28.0 * scale_w) / ds:
        sw = float(sw_g * 0.34)

    le_ear, re_ear = p_local[7, :2] / ds, p_local[8, :2] / ds
    ear_d = float(np.linalg.norm(le_ear - re_ear)) if (p_local[7, 2] > 0.3 and p_local[8, 2] > 0.3) else (sw * 0.52)
    ear_d = max(18.0, ear_d)

    # Голова и волосы: радиус по расстоянию между ушами (по ТЗ пользователя)
    rx_head = max(8, int(ear_d * 0.68))
    ry_head = max(10, int(ear_d * 0.82))
    head_cx = int(nose[0])
    head_cy = int(nose[1] - ear_d * 0.16)
    cv2.ellipse(live_mask_s, (head_cx, head_cy), (rx_head, ry_head), 0, 0, 360, 1.0, -1)

    # Шея: узкая область до линии воротника брони (не перекрывает нагрудные доспехи!)
    neck = (ls + rs) * 0.5
    neck_w = max(4, int(ear_d * 0.36))
    neck_bot = int(neck[1] + (sw * 0.06))
    neck_top = int(head_cy + ry_head * 0.55)
    if neck_bot > neck_top:
        cv2.rectangle(live_mask_s, (head_cx - neck_w, neck_top), (head_cx + neck_w, neck_bot), 1.0, -1)

    # Кисти рук (только ладони/пальцы, предплечья получают броню наручей!)
    if hands_live:
        r_hand = max(4, int(ear_d * 0.30))
        for wr_i, ind_i, pnk_i, el_i in ((15, 19, 17, 13), (16, 20, 18, 14)):
            if p_local[wr_i, 2] > 0.45:
                wr = p_local[wr_i, :2] / ds
                if p_local[ind_i, 2] > 0.40:
                    palm = wr * 0.25 + (p_local[ind_i, :2] / ds) * 0.75
                elif p_local[el_i, 2] > 0.40:
                    palm = wr + (wr - (p_local[el_i, :2] / ds)) * 0.22
                else:
                    palm = wr
                cv2.circle(live_mask_s, ip(palm), r_hand, 1.0, -1)

    k_blur = max(5, min(17, (int(ear_d * 0.22) // 2) * 2 + 1))
    live_mask_s = cv2.GaussianBlur(live_mask_s, (k_blur, k_blur), k_blur * 0.35)
    live_mask = cv2.resize(live_mask_s, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    return np.clip(live_mask, 0.0, 1.0)


def warp_ai_to_live_pose(ai_img, p_ai, p_live, out_w, out_h):
    """
    Вычисляет плавное 2D аффинное преобразование (перенос + масштабирование + поворот торса)
    из позы p_ai (в момент генерации ИИ) в текущую позу p_live (в момент рендера 30+ FPS).
    Благодаря этому ИИ-костюм движется синхронно с телом ребёнка БЕЗ задержек и отставаний!
    """
    if ai_img is None:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    ai_base = cv2.resize(ai_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    if p_ai is None or p_live is None:
        return ai_base

    if (
        p_ai[11, 2] > 0.25 and p_ai[12, 2] > 0.25
        and p_live[11, 2] > 0.25 and p_live[12, 2] > 0.25
    ):
        s_ls, s_rs = p_ai[11, :2], p_ai[12, :2]
        d_ls, d_rs = p_live[11, :2], p_live[12, :2]

        s_sw = max(20.0, float(np.linalg.norm(s_rs - s_ls)))
        d_sw = max(20.0, float(np.linalg.norm(d_rs - d_ls)))
        scale = float(np.clip(d_sw / s_sw, 0.85, 1.18))

        s_neck = (s_ls + s_rs) * 0.5
        d_neck = (d_ls + d_rs) * 0.5

        ang_s = math.atan2(float(s_rs[1] - s_ls[1]), float(s_rs[0] - s_ls[0]))
        ang_d = math.atan2(float(d_rs[1] - d_ls[1]), float(d_rs[0] - d_ls[0]))
        d_theta = float(np.clip(ang_d - ang_s, -0.25, 0.25))

        cos_t = math.cos(d_theta) * scale
        sin_t = math.sin(d_theta) * scale

        tx = float(d_neck[0] - (s_neck[0] * cos_t - s_neck[1] * sin_t))
        ty = float(d_neck[1] - (s_neck[0] * sin_t + s_neck[1] * cos_t))

        M = np.float32([
            [cos_t, -sin_t, tx],
            [sin_t,  cos_t, ty],
        ])

        return cv2.warpAffine(
            ai_base, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    return ai_base

    return cv2.resize(ai_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def build_costume_sketch(frame_bgr, person_mask, p_local, hero, hands_live=True):
    """
    Строит ВЫСОКОДЕТАЛИЗИРОВАННЫЙ ЭСКИЗ КОСТЮМА с наложением реалистичных PNG-доспехов,
    плащей и текстур по скелету перед отправкой в SD-Turbo.
    """
    hid = hero["id"]
    pal = HERO_SKETCH_PALETTES.get(hid, HERO_SKETCH_PALETTES["alpamys"])
    h_img, w_img = frame_bgr.shape[:2]

    face_hands_mask = compute_face_and_hands_mask(h_img, w_img, p_local, hands_live=hands_live)
    body_only_mask = np.clip(person_mask * (1.0 - face_hands_mask * 0.92), 0.0, 1.0)

    # Определяем ключевые точки торса
    has_shoulders = (p_local is not None and p_local[11, 2] > 0.25 and p_local[12, 2] > 0.25)
    if has_shoulders:
        ls = p_local[11, :2].copy()
        rs = p_local[12, :2].copy()
        if ls[0] > rs[0]:
            sh_l, sh_r = rs, ls
        else:
            sh_l, sh_r = ls, rs
        sw = max(40.0 * (w_img / 640.0), float(np.linalg.norm(sh_r - sh_l)))
        neck = (sh_l + sh_r) * 0.5
        if p_local[0, 2] > 0.25:
            min_neck_y = float(p_local[0, 1] + sw * 0.42)
            if neck[1] < min_neck_y:
                dy_n = min_neck_y - neck[1]
                neck[1] += dy_n
                sh_l[1] += dy_n
                sh_r[1] += dy_n
        if (
            p_local[23, 2] > 0.35 and p_local[24, 2] > 0.35
            and min(p_local[23, 1], p_local[24, 1]) > neck[1] + sw * 0.65
            and min(p_local[23, 1], p_local[24, 1]) < h_img - 10
        ):
            lh, rh = p_local[23, :2].copy(), p_local[24, :2].copy()
            if lh[0] > rh[0]:
                hp_l, hp_r = rh, lh
            else:
                hp_l, hp_r = lh, rh
            hip = (hp_l + hp_r) * 0.5
            th = max(sw * 0.95, float(np.linalg.norm(hip - neck)))
        else:
            th = sw * 1.28
            hip = np.array([neck[0], min(h_img - 8.0, neck[1] + th)], dtype=np.float32)
            hp_l = np.array([neck[0] - sw * 0.36, hip[1]], dtype=np.float32)
            hp_r = np.array([neck[0] + sw * 0.36, hip[1]], dtype=np.float32)
    elif p_local is not None and p_local[0, 2] > 0.25:
        nose_pt = p_local[0, :2]
        le_e, re_e = p_local[7, :2], p_local[8, :2]
        ed = float(np.linalg.norm(le_e - re_e)) if (p_local[7, 2] > 0.3 and p_local[8, 2] > 0.3) else (w_img * 0.22)
        sw = max(80.0 * (w_img / 640.0), ed * 2.1)
        neck = np.array([nose_pt[0], nose_pt[1] + ed * 1.05], dtype=np.float32)
        sh_l = np.array([neck[0] - sw * 0.5, neck[1]], dtype=np.float32)
        sh_r = np.array([neck[0] + sw * 0.5, neck[1]], dtype=np.float32)
        th = sw * 1.25
        hip = np.array([neck[0], min(h_img - 8.0, neck[1] + th)], dtype=np.float32)
        hp_l = np.array([neck[0] - sw * 0.36, hip[1]], dtype=np.float32)
        hp_r = np.array([neck[0] + sw * 0.36, hip[1]], dtype=np.float32)
    else:
        ys, xs = np.where(person_mask > 0.35)
        if len(xs) > 50:
            x0_m, x1_m = int(np.percentile(xs, 10)), int(np.percentile(xs, 90))
            y0_m, y1_m = int(np.min(ys)), int(np.max(ys))
            cx_m = (x0_m + x1_m) * 0.5
            sw = max(80.0, (x1_m - x0_m) * 0.68)
            neck = np.array([cx_m, y0_m + (y1_m - y0_m) * 0.34], dtype=np.float32)
            sh_l = np.array([cx_m - sw * 0.5, neck[1]], dtype=np.float32)
            sh_r = np.array([cx_m + sw * 0.5, neck[1]], dtype=np.float32)
            th = max(sw * 1.15, (y1_m - neck[1]) * 0.88)
            hip = np.array([cx_m, min(h_img - 10.0, neck[1] + th)], dtype=np.float32)
            hp_l = np.array([cx_m - sw * 0.36, hip[1]], dtype=np.float32)
            hp_r = np.array([cx_m + sw * 0.36, hip[1]], dtype=np.float32)
        else:
            cx_m = w_img * 0.5
            sw = w_img * 0.36
            neck = np.array([cx_m, h_img * 0.38], dtype=np.float32)
            sh_l = np.array([cx_m - sw * 0.5, neck[1]], dtype=np.float32)
            sh_r = np.array([cx_m + sw * 0.5, neck[1]], dtype=np.float32)
            th = sw * 1.25
            hip = np.array([cx_m, min(h_img - 10.0, neck[1] + th)], dtype=np.float32)
            hp_l = np.array([cx_m - sw * 0.36, hip[1]], dtype=np.float32)
            hp_r = np.array([cx_m + sw * 0.36, hip[1]], dtype=np.float32)

    ang_deg = math.degrees(math.atan2(float(sh_r[1] - sh_l[1]), float(sh_r[0] - sh_l[0])))
    sketch = frame_bgr.copy()

    # 1. ПЕРЕКРАСКА ТЁМНОЙ ОДЕЖДЫ ТЕЛА В ФИРМЕННЫЙ ЦВЕТ КОСТЮМА С СОХРАНЕНИЕМ СКЛАДОК
    gray_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    shade_u8 = cv2.addWeighted(gray_3, 0.40, np.full_like(gray_3, 168), 1.0, 0)
    tunic_img = np.full_like(frame_bgr, pal["tunic"], dtype=np.uint8)
    recolored_body = cv2.multiply(tunic_img, shade_u8, scale=1.0 / 215.0)

    w_body = (body_only_mask * 0.88).astype(np.float32)
    sketch = cv2.blendLinear(recolored_body, sketch, w_body, 1.0 - w_body)

    armor_layer = sketch.copy()

    # 2. ПЛАЩ / КРЫЛЬЯ / РАНЕЦ ЗА СПИНОЙ (С ИСПОЛЬЗОВАНИЕМ РЕАЛИСТИЧНОГО PNG-АРТА)
    back_png = load_hero_png(hid, "back")
    if back_png is not None:
        cx_b = float(neck[0])
        cy_b = float(neck[1] + th * 0.50)
        overlay_rgba_sprite(armor_layer, back_png, cx_b, cy_b, sw * 2.35, th * 1.55, -ang_deg)
    elif pal["cape"] is not None:
        c_cape = pal["cape"]
        c_trim = pal["trim"]
        if hid == "fairy":
            for s_dir in (-1, 1):
                sh_pt = sh_l if s_dir < 0 else sh_r
                w_upper = np.array([
                    sh_pt,
                    sh_pt + [s_dir * sw * 0.95, -th * 0.42],
                    sh_pt + [s_dir * sw * 1.15, th * 0.12],
                    sh_pt + [s_dir * sw * 0.25, th * 0.38],
                ], np.int32)
                w_lower = np.array([
                    sh_pt + [s_dir * sw * 0.15, th * 0.25],
                    sh_pt + [s_dir * sw * 0.88, th * 0.35],
                    sh_pt + [s_dir * sw * 0.68, th * 0.92],
                    sh_pt + [s_dir * sw * 0.10, th * 0.65],
                ], np.int32)
                cv2.fillPoly(armor_layer, [w_upper, w_lower], c_cape, AA)
                cv2.polylines(armor_layer, [w_upper, w_lower], True, c_trim, 4, AA)
        elif hid == "astronaut":
            p0 = (int(sh_l[0] - sw * 0.15), int(neck[1] - th * 0.14))
            p1 = (int(sh_r[0] + sw * 0.15), int(neck[1] + th * 0.62))
            cv2.rectangle(armor_layer, p0, p1, c_cape, -1, AA)
            cv2.rectangle(armor_layer, p0, p1, (130, 140, 155), 4, AA)
        else:
            cape_bot_y = min(h_img - 4, int(hip[1] + th * 0.38))
            top_y = int(neck[1] - 4)
            cape_pts = np.array([
                [int(sh_l[0] - sw * 0.15), top_y],
                [int(sh_r[0] + sw * 0.15), top_y],
                [int(hp_r[0] + sw * 0.65), cape_bot_y],
                [int(hp_l[0] - sw * 0.65), cape_bot_y],
            ], np.int32)

            # Вертикальный градиент плаща (от освещённых плеч к глубокому бархату внизу)
            c_top = tuple(min(255, int(c * 1.25)) for c in c_cape)
            c_bot = tuple(max(10, int(c * 0.70)) for c in c_cape)
            cape_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(cape_mask, [cape_pts], 255)

            y_span = max(1, cape_bot_y - top_y)
            grad_1d = np.linspace(0.0, 1.0, y_span, dtype=np.float32)
            for ch in range(3):
                col_grad = (c_top[ch] * (1.0 - grad_1d) + c_bot[ch] * grad_1d).astype(np.uint8)
                col_block = np.tile(col_grad[:, None], (1, w_img))
                armor_layer[top_y:cape_bot_y, :, ch] = np.where(
                    cape_mask[top_y:cape_bot_y] > 0,
                    col_block,
                    armor_layer[top_y:cape_bot_y, :, ch]
                )

            # Складки плаща (вертикальные драпировки: чередующиеся тени и блики на гребнях)
            n_folds = 7
            c_shadow = tuple(max(5, int(c * 0.52)) for c in c_cape)
            c_highlight = tuple(min(255, int(c * 1.35)) for c in c_cape)
            for f_i in range(n_folds):
                frac = (f_i + 1) / float(n_folds + 1)
                p_top = (int(sh_l[0] + (sh_r[0] - sh_l[0]) * frac), top_y)
                bot_x0 = int(hp_l[0] - sw * 0.65)
                bot_x1 = int(hp_r[0] + sw * 0.65)
                p_bot = (int(bot_x0 + (bot_x1 - bot_x0) * frac), cape_bot_y)
                col_fold = c_shadow if f_i % 2 == 0 else c_highlight
                th_fold = max(2, int(sw * 0.035)) if f_i % 2 == 0 else max(1, int(sw * 0.02))
                cv2.line(armor_layer, p_top, p_bot, col_fold, th_fold, AA)

            cv2.polylines(armor_layer, [cape_pts], True, c_trim, 4, AA)

    # 3. РУКАВА И НАРУЧИ НА ПРЕДПЛЕЧЬЯХ (С ИСПОЛЬЗОВАНИЕМ GLOVE PNG)
    glove_png = load_hero_png(hid, "glove")
    if p_local is not None:
        sleeve_th = max(16, int(sw * 0.26))
        bracer_th = max(14, int(sw * 0.22))
        for sh_i, el_i, wr_i in ((11, 13, 15), (12, 14, 16)):
            if p_local[sh_i, 2] > 0.25 and p_local[el_i, 2] > 0.25:
                p_sh = p_local[sh_i, :2]
                p_el = p_local[el_i, :2]
                cv2.line(armor_layer, ip(p_sh), ip(p_el), pal["tunic"], sleeve_th, AA)
                cv2.line(armor_layer, ip(p_sh), ip(p_el), pal["trim"], 2, AA)
                if p_local[wr_i, 2] > 0.25:
                    p_wr = p_local[wr_i, :2]
                    p_br_start = p_el + (p_wr - p_el) * 0.18
                    p_br_end = p_el + (p_wr - p_el) * 0.78
                    if glove_png is not None:
                        br_cx, br_cy = (p_br_start + p_br_end) * 0.5
                        br_w = max(16.0, float(np.linalg.norm(p_br_end - p_br_start)) * 1.1)
                        overlay_rgba_sprite(armor_layer, glove_png, br_cx, br_cy, br_w, br_w, 0.0)
                    else:
                        cv2.line(armor_layer, ip(p_br_start), ip(p_br_end), pal["bracer"], bracer_th, AA)
                        cv2.line(armor_layer, ip(p_br_start), ip(p_br_end), pal["trim"], 3, AA)

    # 4. НАГРУДНАЯ БРОНЯ (С ИСПОЛЬЗОВАНИЕМ ARMOR PNG АССЕТА)
    chest_top_y = int(neck[1] + th * 0.04)
    chest_bot_y = int(neck[1] + th * 0.78)
    armor_png = load_hero_png(hid, "armor")
    cx_a = float((neck[0] + hip[0]) * 0.5)
    cy_a = float(neck[1] + th * 0.44)
    if armor_png is not None:
        overlay_rgba_sprite(armor_layer, armor_png, cx_a, cy_a, sw * 1.55, th * 1.18, -ang_deg)
    else:
        chest_pts = np.array([
            [int(sh_l[0] - sw * 0.08), chest_top_y],
            [int(sh_r[0] + sw * 0.08), chest_top_y],
            [int(neck[0] + sw * 0.44), int(neck[1] + th * 0.45)],
            [int(hp_r[0] + sw * 0.06), chest_bot_y],
            [int(hp_l[0] - sw * 0.06), chest_bot_y],
            [int(neck[0] - sw * 0.44), int(neck[1] + th * 0.45)],
        ], np.int32)
        # Вертикальный металлический градиент кирасы (светотень и объём)
        ch_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(ch_mask, [chest_pts], 255)
        ch_span = max(1, chest_bot_y - chest_top_y)
        grad_ch = np.linspace(0.0, 1.0, ch_span, dtype=np.float32)
        c_arm_hi = tuple(min(255, int(c * 1.30 + 18)) for c in pal["armor"])
        c_arm_lo = tuple(max(10, int(c * 0.72)) for c in pal["armor"])
        for ch in range(3):
            col_grad = (c_arm_hi[ch] * (1.0 - grad_ch) + c_arm_lo[ch] * grad_ch).astype(np.uint8)
            col_block = np.tile(col_grad[:, None], (1, w_img))
            armor_layer[chest_top_y:chest_bot_y, :, ch] = np.where(
                ch_mask[chest_top_y:chest_bot_y] > 0,
                col_block,
                armor_layer[chest_top_y:chest_bot_y, :, ch]
            )
        cv2.polylines(armor_layer, [chest_pts], True, pal["trim"], 4, AA)

        inner_pts = np.array([
            [int(sh_l[0] + sw * 0.06), int(chest_top_y + th * 0.07)],
            [int(sh_r[0] - sw * 0.06), int(chest_top_y + th * 0.07)],
            [int(hp_r[0] - sw * 0.04), int(chest_bot_y - th * 0.12)],
            [int(hp_l[0] + sw * 0.04), int(chest_bot_y - th * 0.12)],
        ], np.int32)
        hi_col = tuple(min(255, int(c * 1.15 + 22)) for c in pal["armor"])
        cv2.polylines(armor_layer, [inner_pts], True, pal["trim"], 2, AA)

        # 5. ИНДИВИДУАЛЬНЫЕ ЭМБЛЕМЫ И УЗОРЫ ДЛЯ ВСЕХ 8 ГЕРОЕВ (только при отсутствии PNG-ассета)
        chest_c = (int(neck[0]), int(neck[1] + th * 0.34))

        if hid == "spider":
            for r_w in (int(sw * 0.18), int(sw * 0.32), int(sw * 0.46)):
                cv2.ellipse(armor_layer, chest_c, (r_w, int(r_w * 0.85)), 0, 0, 360, (25, 25, 30), 2, AA)
            for ang_d in range(0, 360, 30):
                rad = math.radians(ang_d)
                p_end = (int(chest_c[0] + math.cos(rad) * sw * 0.52), int(chest_c[1] + math.sin(rad) * sw * 0.45))
                cv2.line(armor_layer, chest_c, p_end, (25, 25, 30), 2, AA)
            sp_r = max(6, int(sw * 0.06))
            cv2.ellipse(armor_layer, chest_c, (sp_r, int(sp_r * 1.35)), 0, 0, 360, (20, 20, 25), -1, AA)
            cv2.circle(armor_layer, (chest_c[0], chest_c[1] - int(sp_r * 0.9)), int(sp_r * 0.6), (20, 20, 25), -1, AA)
            for s_leg in (-1, 1):
                for i_leg, dy_l in enumerate((-0.6, -0.1, 0.4, 0.9)):
                    l0 = (chest_c[0] + int(s_leg * sp_r * 0.8), chest_c[1] + int(dy_l * sp_r))
                    l1 = (chest_c[0] + int(s_leg * sp_r * 2.1), chest_c[1] + int((dy_l - 0.4) * sp_r))
                    l2 = (chest_c[0] + int(s_leg * sp_r * 2.8), chest_c[1] + int((dy_l + 0.5) * sp_r))
                    cv2.polylines(armor_layer, [np.array([l0, l1, l2], np.int32)], False, (20, 20, 25), 2, AA)

        elif hid == "ladybug":
            col_pts = np.array([
                [int(sh_l[0] + sw * 0.12), int(neck[1] - 4)],
                [int(sh_r[0] - sw * 0.12), int(neck[1] - 4)],
                [int(sh_r[0] - sw * 0.16), int(neck[1] + th * 0.12)],
                [int(sh_l[0] + sw * 0.16), int(neck[1] + th * 0.12)],
            ], np.int32)
            cv2.fillPoly(armor_layer, [col_pts], (20, 20, 25), AA)
            cv2.circle(armor_layer, chest_c, max(14, int(sw * 0.13)), (35, 35, 230), -1, AA)
            cv2.circle(armor_layer, chest_c, max(14, int(sw * 0.13)), (20, 20, 25), 3, AA)
            cv2.line(armor_layer, (chest_c[0], chest_c[1] - int(sw * 0.12)), (chest_c[0], chest_c[1] + int(sw * 0.12)), (20, 20, 25), 3, AA)
            spot_r = max(7, int(sw * 0.08))
            for dx_k, dy_k in ((-0.28, -0.14), (0.28, -0.14), (0.0, 0.01), (-0.24, 0.18), (0.24, 0.18), (-0.38, 0.04), (0.38, 0.04)):
                sx = int(chest_c[0] + dx_k * sw)
                sy = int(chest_c[1] + dy_k * th)
                cv2.circle(armor_layer, (sx, sy), spot_r, (20, 20, 25), -1, AA)

        elif hid in ("alpamys", "tomiris"):
            # Казахский национальный орнамент «қошқар мүйіз» (рога барана) и царский медальон
            orn_r = int(sw * 0.18)
            cv2.circle(armor_layer, chest_c, orn_r + 6, pal["cape"], -1, AA)
            cv2.circle(armor_layer, chest_c, orn_r + 6, pal["trim"], 4, AA)
            for s_k in (-1, 1):
                h_cx = int(chest_c[0] + s_k * orn_r * 0.55)
                cv2.ellipse(armor_layer, (h_cx, chest_c[1]), (int(orn_r * 0.52), int(orn_r * 0.40)), 0, -45, 225, pal["trim"], max(3, int(sw * 0.035)), AA)
                cv2.ellipse(armor_layer, (h_cx, chest_c[1]), (int(orn_r * 0.30), int(orn_r * 0.22)), 0, -45, 225, (255, 255, 255), 2, AA)
            cv2.circle(armor_layer, chest_c, max(6, int(sw * 0.06)), (50, 40, 230) if hid == "tomiris" else pal["trim"], -1, AA)
            cv2.circle(armor_layer, chest_c, max(6, int(sw * 0.06)), (120, 240, 255) if hid == "tomiris" else (255, 255, 255), 2, AA)

        elif hid == "iron":
            r_arc = max(15, int(sw * 0.15))
            cv2.circle(armor_layer, chest_c, r_arc + 8, (35, 45, 190), -1, AA)
            cv2.circle(armor_layer, chest_c, r_arc + 3, (40, 205, 255), 4, AA)
            cv2.circle(armor_layer, chest_c, r_arc, (255, 225, 80), -1, AA)
            cv2.circle(armor_layer, chest_c, int(r_arc * 0.65), (255, 255, 255), -1, AA)

        elif hid == "elsa":
            r_sn = max(16, int(sw * 0.17))
            for i_s in range(6):
                ang = i_s * math.pi / 3.0
                p_a = (int(chest_c[0] - math.cos(ang) * r_sn), int(chest_c[1] - math.sin(ang) * r_sn))
                p_b = (int(chest_c[0] + math.cos(ang) * r_sn), int(chest_c[1] + math.sin(ang) * r_sn))
                cv2.line(armor_layer, p_a, p_b, (255, 255, 255), 4, AA)
                p_mid = (int(chest_c[0] + math.cos(ang) * r_sn * 0.6), int(chest_c[1] + math.sin(ang) * r_sn * 0.6))
                v_ang1 = ang + math.pi / 4.0
                v_ang2 = ang - math.pi / 4.0
                b_len = r_sn * 0.32
                cv2.line(armor_layer, p_mid, (int(p_mid[0] + math.cos(v_ang1) * b_len), int(p_mid[1] + math.sin(v_ang1) * b_len)), (255, 245, 220), 2, AA)
                cv2.line(armor_layer, p_mid, (int(p_mid[0] + math.cos(v_ang2) * b_len), int(p_mid[1] + math.sin(v_ang2) * b_len)), (255, 245, 220), 2, AA)
            cv2.circle(armor_layer, chest_c, max(6, int(r_sn * 0.28)), (255, 220, 110), -1, AA)

        elif hid == "astronaut":
            pw, ph = int(sw * 0.38), int(th * 0.32)
            px0, py0 = chest_c[0] - pw // 2, chest_c[1] - ph // 2
            cv2.rectangle(armor_layer, (px0, py0), (px0 + pw, py0 + ph), (225, 228, 235), -1, AA)
            cv2.rectangle(armor_layer, (px0, py0), (px0 + pw, py0 + ph), (110, 120, 135), 3, AA)
            cv2.rectangle(armor_layer, (px0 + 6, py0 + 6), (px0 + pw - 6, py0 + ph // 2 - 2), (40, 45, 55), -1, AA)
            led_y = py0 + ph // 2 + 8
            cv2.circle(armor_layer, (px0 + 14, led_y), 5, (255, 220, 60), -1, AA)
            cv2.circle(armor_layer, (px0 + 28, led_y), 5, (80, 235, 110), -1, AA)
            cv2.circle(armor_layer, (px0 + 42, led_y), 5, (50, 60, 240), -1, AA)
            cv2.line(armor_layer, (px0 + pw - 30, led_y - 4), (px0 + pw - 6, led_y - 4), (200, 70, 40), 3, AA)
            cv2.line(armor_layer, (px0 + pw - 30, led_y + 4), (px0 + pw - 6, led_y + 4), (50, 60, 230), 3, AA)

        elif hid == "fairy":
            fl_r = max(14, int(sw * 0.14))
            for i_pet in range(5):
                p_ang = i_pet * 2.0 * math.pi / 5.0 - math.pi / 2.0
                pet_c = (int(chest_c[0] + math.cos(p_ang) * fl_r * 0.72), int(chest_c[1] + math.sin(p_ang) * fl_r * 0.72))
                cv2.circle(armor_layer, pet_c, int(fl_r * 0.55), (215, 150, 255), -1, AA)
                cv2.circle(armor_layer, pet_c, int(fl_r * 0.55), (120, 245, 255), 2, AA)
            cv2.circle(armor_layer, chest_c, int(fl_r * 0.42), (90, 235, 80), -1, AA)
            cv2.circle(armor_layer, chest_c, int(fl_r * 0.42), (255, 255, 255), 2, AA)

    # 6. НАПЛЕЧНИКИ (PAULDRONS)
    if hid in ("alpamys", "tomiris", "iron", "astronaut", "fairy"):
        pr_x, pr_y = int(sw * 0.24), int(sw * 0.15)
        for sh_pt, ang_p in ((sh_l, -18), (sh_r, 18)):
            p_col = pal["armor"] if hid != "fairy" else (100, 220, 70)
            cv2.ellipse(armor_layer, ip(sh_pt), (pr_x, pr_y), ang_p, 0, 360, p_col, -1, AA)
            cv2.ellipse(armor_layer, ip(sh_pt), (pr_x, pr_y), ang_p, 0, 360, pal["trim"], 3, AA)

    # 7. ШИРОКИЙ ПОЯС
    belt_y0 = int(chest_bot_y - th * 0.08)
    belt_y1 = int(chest_bot_y + th * 0.08)
    belt_pts = np.array([
        [int(hp_l[0] - sw * 0.10), belt_y0],
        [int(hp_r[0] + sw * 0.10), belt_y0],
        [int(hp_r[0] + sw * 0.08), belt_y1],
        [int(hp_l[0] - sw * 0.08), belt_y1],
    ], np.int32)
    cv2.fillPoly(armor_layer, [belt_pts], pal["belt"], AA)
    cv2.polylines(armor_layer, [belt_pts], True, pal["trim"], 4, AA)
    cv2.circle(armor_layer, (int(neck[0]), (belt_y0 + belt_y1) // 2), max(10, int(sw * 0.09)), pal["trim"], -1, AA)

    # 8. ПЛАВНЫЙ НАХЛЁСТ ВОРОТНИКА НА ШЕЮ
    collar_poly = np.array([
        [int(neck[0] - sw * 0.24), int(neck[1] - th * 0.02)],
        [int(neck[0] + sw * 0.24), int(neck[1] - th * 0.02)],
        [int(neck[0] + sw * 0.18), int(neck[1] + th * 0.10)],
        [int(neck[0] - sw * 0.18), int(neck[1] + th * 0.10)],
    ], np.int32)
    cv2.fillPoly(armor_layer, [collar_poly], pal["armor"], AA)
    cv2.polylines(armor_layer, [collar_poly], True, pal["trim"], 2, AA)

    # Накладываем плотный эскиз костюма, сохраняя живую голову и кисти рук
    w_armor = ((1.0 - face_hands_mask) * 0.94).astype(np.float32)
    sketch = cv2.blendLinear(armor_layer, sketch, w_armor, 1.0 - w_armor)

    # 9. УДАЛЕНИЕ СЕРОГО ГАЛО: расширение силуэта и заливка фоном ауры
    fg_mask = np.clip(person_mask, 0.0, 1.0).astype(np.float32)
    kern_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    dilated_sketch = cv2.dilate(sketch, kern_dil, iterations=1)
    bg_color_img = np.full_like(sketch, pal["bg_fill"], dtype=np.uint8)
    halo_free_bg = cv2.addWeighted(dilated_sketch, 0.65, bg_color_img, 0.35, 0)

    final_input_for_ai = cv2.blendLinear(sketch, halo_free_bg, fg_mask, 1.0 - fg_mask)
    return final_input_for_ai, face_hands_mask


class AITransformManager:
    """
    Управляет третьим потоком (GPU) для ИИ-превращения в реальном времени (Режим А + fallback Режим Б):
    - Полная поддержка всех 8 героев с мгновенным переключением без задержек и артефактов.
    - SD-Turbo fp16 + TAESD с ускорением через CUDA Graph (15-22+ FPS при strength=0.48).
    """

    def __init__(self):
        rec = config.CFG.get("ai_recommended", {})
        self.ai_w = int(rec.get("width", 384))
        self.ai_h = int(rec.get("height", 512))
        self.steps = int(rec.get("steps", 2))
        self.strength = float(rec.get("strength", 0.48))
        self.ema_alpha = float(rec.get("ema_alpha", 0.62))

        self.lock = threading.Lock()
        self.running = True
        self.ready = False
        self.failed = False
        self.status = "loading" if str(config.AI_TRANSFORM).lower() != "off" else "disabled"
        self.error_msg_short = ""
        self.error_msg = ""
        self.start_time = time.perf_counter()
        self.ready_time = 0.0
        self.device_name = "GPU"

        # Входная задача для GPU-потока
        self.req_crop_bgr = None
        self.req_mask_crop = None
        self.req_p_crop = None
        self.req_p_full = None
        self.req_bbox = None
        self.req_hero = None
        self.req_idx = -1

        # Готовый результат GPU-потока
        self.out_bgr_small = None
        self.out_bbox = None
        self.out_p_full = None
        self.out_hero_id = None
        self.out_ts = 0.0
        self.ai_fps = 0.0
        self.ai_ms = 0.0
        self.vram_gb = 0.0

        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._gpu_worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def is_enabled(self):
        mode = str(config.AI_TRANSFORM).lower()
        if mode == "off" or self.failed:
            return False
        return True

    def get_status_info(self):
        """Возвращает информацию для углового индикатора статуса ИИ."""
        mode = str(config.AI_TRANSFORM).lower()
        if mode == "off":
            return {
                "status": "disabled",
                "label": "ИИ: выкл [A]",
                "color": (150, 150, 150),
                "fps": 0.0,
                "ms": 0.0,
                "vram": 0.0,
                "details": "ИИ: выкл [A] (процедурный режим)",
            }
        if self.failed or self.status == "error":
            reason = self.error_msg_short if self.error_msg_short else "сбой"
            return {
                "status": "error",
                "label": f"ИИ: ошибка ({reason})",
                "color": (50, 50, 240),
                "fps": 0.0,
                "ms": 0.0,
                "vram": 0.0,
                "details": f"ИИ: ошибка ({reason}) | откат",
            }
        if not self.ready:
            elapsed = time.perf_counter() - self.start_time
            return {
                "status": "loading",
                "label": "ИИ: загрузка…",
                "color": (30, 205, 255),
                "fps": 0.0,
                "ms": 0.0,
                "vram": 0.0,
                "details": f"ИИ: загрузка ({elapsed:.0f}с) | SD-Turbo fp16",
            }
        with self.lock:
            fps_val = self.ai_fps
            ms_val = self.ai_ms
            vram_val = self.vram_gb
        return {
            "status": "ready",
            "label": "ИИ: готов",
            "color": (60, 230, 90),
            "fps": fps_val,
            "ms": ms_val,
            "vram": vram_val,
            "details": f"ИИ: готов | {fps_val:.1f} FPS ({ms_val:.0f} мс) | VRAM: {vram_val:.1f} ГБ",
        }

    def needs_frame(self):
        """Возвращает True только тогда, когда GPU-поток свободен и ждёт новый кадр."""
        if not self.is_enabled() or not self.ready:
            return False
        with self.lock:
            return self.req_crop_bgr is None

    def is_active_for_hero(self, hero, now=None):
        """Проверяет, есть ли свежий ИИ-кадр для текущего героя без просадки FPS и зависаний."""
        if not self.is_enabled() or not self.ready:
            return False
        t_real = time.perf_counter()
        with self.lock:
            if self.out_bgr_small is None or self.out_hero_id != hero["id"]:
                return False
            if (t_real - self.out_ts) > 3.0:
                return False
            if str(config.AI_TRANSFORM).lower() == "auto" and 0.0 < self.ai_fps < 2.0:
                return False
        return True

    def submit_frame(self, frame_full, clean_mask_full, p_full, bbox, hero, frame_idx):
        """
        Отправляет вертикальное портретное окно 540x720 (пропорция 3:4 = 384x512)
        вокруг центра человека в GPU-поток, не блокируя основной рендер 30+ FPS.
        """
        if not self.is_enabled() or not self.ready:
            return
        bx0, by0, bx1, by1 = bbox
        if bx1 - bx0 < 40 or by1 - by0 < 60:
            return
        win_w = min(W, max(640, (bx1 - bx0) + 40))
        if p_full is not None and p_full[11, 2] > 0.25 and p_full[12, 2] > 0.25:
            cx_p = int((p_full[11, 0] + p_full[12, 0]) * 0.5)
        else:
            cx_p = (bx0 + bx1) // 2
        wx0 = int(np.clip(cx_p - win_w // 2, 0, W - win_w))
        wx1 = wx0 + win_w
        wy0, wy1 = 0, H

        with self.lock:
            if self.req_idx == frame_idx:
                return
            self.req_crop_bgr = frame_full[wy0:wy1, wx0:wx1].copy()
            self.req_mask_crop = clean_mask_full[wy0:wy1, wx0:wx1].copy()
            if p_full is not None:
                p_c = p_full.copy()
                p_c[:, 0] -= wx0
                p_c[:, 1] -= wy0
                self.req_p_crop = p_c
                self.req_p_full = p_full.copy()
            else:
                self.req_p_crop = None
                self.req_p_full = None
            self.req_bbox = (wx0, wy0, wx1, wy1)
            self.req_hero = hero
            self.req_idx = frame_idx

    def render_ai_person_roi(self, frame_full, p_full, hero, t, bbox):
        """
        Формирует область силуэта ребёнка (ROI) с ИИ-костюмом + настоящим живым лицом и кистями рук
        + 2D-аффинным позовым выравниванием (warp_ai_to_live_pose) для 30+ FPS без отставания!
        """
        with self.lock:
            if self.out_bgr_small is None or self.out_hero_id != hero["id"]:
                return None
            ai_small = self.out_bgr_small.copy()
            ai_bbox = self.out_bbox
            ai_p = self.out_p_full.copy() if self.out_p_full is not None else None

        bx0, by0, bx1, by1 = bbox
        rw, rh = bx1 - bx0, by1 - by0
        if rw <= 4 or rh <= 4:
            return None

        # 1. Аффинное выравнивание ИИ-кадра по скелету (zero lag, 30+ FPS)
        if ai_bbox is not None:
            aw = max(4, ai_bbox[2] - ai_bbox[0])
            ah = max(4, ai_bbox[3] - ai_bbox[1])
            ai_win_full = cv2.resize(ai_small, (aw, ah), interpolation=cv2.INTER_LINEAR)

            p_ai_crop = None
            if ai_p is not None:
                p_ai_crop = ai_p.copy()
                p_ai_crop[:, 0] -= ai_bbox[0]
                p_ai_crop[:, 1] -= ai_bbox[1]
            p_live_crop = None
            if p_full is not None:
                p_live_crop = p_full.copy()
                p_live_crop[:, 0] -= ai_bbox[0]
                p_live_crop[:, 1] -= ai_bbox[1]

            ai_warped = warp_ai_to_live_pose(ai_win_full, p_ai_crop, p_live_crop, aw, ah)

            rx0 = max(0, bx0 - ai_bbox[0])
            ry0 = max(0, by0 - ai_bbox[1])
            rx1 = min(aw, rx0 + rw)
            ry1 = min(ah, ry0 + rh)
            if rx1 > rx0 and ry1 > ry0:
                ai_roi = cv2.resize(ai_warped[ry0:ry1, rx0:rx1], (rw, rh), interpolation=cv2.INTER_LINEAR)
            else:
                ai_roi = cv2.resize(ai_small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        else:
            ai_roi = cv2.resize(ai_small, (rw, rh), interpolation=cv2.INTER_LINEAR)

        # 2. Фильтр резкости (unsharp mask) для чётких металлических граней
        ai_sharp = cv2.addWeighted(ai_roi, 1.25, cv2.GaussianBlur(ai_roi, (0, 0), 2.0), -0.25, 0)

        # 3. Сочное осветление костюма ИИ (сияющее золото, полированная броня, сочные цвета)
        ai_bright = cv2.convertScaleAbs(ai_sharp, alpha=1.22, beta=20)
        off_y = int(t * 120) % H
        scan_roi = SCAN_AI[off_y + by0:off_y + by1, bx0:bx1]
        ai_holo = cv2.multiply(ai_bright, scan_roi, scale=1.0 / 255.0)
        if rw > 6:
            ai_holo[:, 1:, 2] = ai_holo[:, :-1, 2]
            ai_holo[:, :-1, 0] = ai_holo[:, 1:, 0]

        # 4. Вклеиваем настоящее живое лицо, причёску, шею и кисти рук с лёгким авто-освещением
        live_roi = frame_full[by0:by1, bx0:bx1]
        live_bright = cv2.convertScaleAbs(live_roi, alpha=1.12, beta=12)
        p_local = None
        if p_full is not None:
            p_local = p_full.copy()
            p_local[:, 0] -= bx0
            p_local[:, 1] -= by0

        face_hands_m = compute_face_and_hands_mask(
            rh, rw, p_local, hands_live=bool(config.HANDS_LIVE)
        ).astype(np.float32)
        person_roi = cv2.blendLinear(live_bright, ai_holo, face_hands_m, 1.0 - face_hands_m)
        return person_roi

    def _gpu_worker_loop(self):
        """Фоновый GPU-поток с SD-Turbo (fp16) + TAESD + CUDA Graph для всех 8 героев."""
        try:
            if not os.path.isdir(SD_TURBO_DIR) or not os.path.isdir(TAESD_DIR):
                missing = []
                if not os.path.isdir(SD_TURBO_DIR):
                    missing.append("sd-turbo")
                if not os.path.isdir(TAESD_DIR):
                    missing.append("taesd")
                self.failed = True
                self.status = "error"
                self.error_msg_short = "нет models/"
                self.error_msg = f"Модели {', '.join(missing)} не найдены в {MODELS_DIR}"
                print(f"[AI_TRANSFORM] Ошибка: {self.error_msg}")
                logging.warning(f"[AI_TRANSFORM] {self.error_msg}")
                return

            import torch
            if not torch.cuda.is_available():
                self.failed = True
                self.status = "error"
                self.error_msg_short = "нет CUDA"
                self.error_msg = "CUDA недоступна, видеокарта NVIDIA не найдена"
                print(f"[AI_TRANSFORM] Ошибка: {self.error_msg}")
                logging.warning(f"[AI_TRANSFORM] {self.error_msg}")
                return

            from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny
            from diffusers.models.attention_processor import AttnProcessor2_0

            device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            print("[AI_TRANSFORM] Загрузка моделей SD-Turbo (fp16) + TAESD в видеопамять...")
            try:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    SD_TURBO_DIR,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )
            except Exception:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    SD_TURBO_DIR,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )

            taesd = AutoencoderTiny.from_pretrained(
                TAESD_DIR,
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            pipe.vae = taesd
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
            print("[AI_TRANSFORM] Модели загружены в VRAM. Кодирование промптов 8 героев...")

            prompts_cfg = config.CFG.get("ai_prompts", HERO_PROMPTS)
            seeds_cfg = config.CFG.get("ai_hero_seeds", {})

            # Предварительное кодирование промптов для ВСЕХ 8 ГЕРОЕВ
            embeds_map = {}
            with torch.inference_mode():
                for hid, default_pr in HERO_PROMPTS.items():
                    pr = prompts_cfg.get(hid, default_pr)
                    emb, _ = pipe.encode_prompt(
                        prompt=pr,
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    embeds_map[hid] = emb
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

            scaling_factor = float(pipe.vae.config.scaling_factor)
            alphas_cumprod_cpu = pipe.scheduler.alphas_cumprod.cpu().numpy()
            lat_h, lat_w = self.ai_h // 8, self.ai_w // 8

            # Фиксированный шум для каждого из 8 героев против мерцания
            noises_map = {}
            for hid in HERO_PROMPTS:
                seed_val = int(seeds_cfg.get(hid, 1001))
                gen = torch.Generator(device=device).manual_seed(seed_val)
                n1 = torch.randn((1, 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
                n2 = torch.randn((1, 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
                noises_map[hid] = (n1, n2)

            static_in = torch.zeros((1, 3, self.ai_h, self.ai_w), device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
            static_emb = embeds_map["alpamys"].clone()
            static_n1, static_n2 = noises_map["alpamys"][0].clone(), noises_map["alpamys"][1].clone()
            static_t = torch.tensor([699, 349], device=device, dtype=torch.long)
            static_sig = torch.tensor([2.5, 1.1, 0.0], device=device, dtype=torch.float16)

            # Захват CUDA Graph для рабочего количества шагов (1 или 2)
            graphs = {}
            st_target = 1 if self.steps == 1 else 2

            def _build_graph(st_cnt):
                def _step_fn(st=st_cnt):
                    lat = pipe.vae.encode(static_in).latents * scaling_factor
                    x = lat + static_n1 * static_sig[0]
                    for idx_s in range(st):
                        sig = static_sig[idx_s]
                        sig_next = static_sig[idx_s + 1]
                        c_in = x / torch.sqrt(sig * sig + 1.0)
                        eps = pipe.unet(c_in, static_t[idx_s], encoder_hidden_states=static_emb, return_dict=False)[0]
                        if idx_s == st - 1:
                            x = x - sig * eps
                        else:
                            sig_up = torch.sqrt((sig_next ** 2) * (sig ** 2 - sig_next ** 2) / (sig ** 2))
                            sig_down = torch.sqrt(sig_next ** 2 - sig_up ** 2)
                            x = x + (sig_down - sig) * eps + static_n2 * sig_up
                    return pipe.vae.decode(x / scaling_factor, return_dict=False)[0]

                s_stream = torch.cuda.Stream()
                s_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_stream), torch.inference_mode():
                    for _ in range(2):
                        out_buf = _step_fn()
                torch.cuda.current_stream().wait_stream(s_stream)
                g_c = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_c), torch.inference_mode():
                    out_buf = _step_fn()
                return g_c, out_buf

            print(f"[AI_TRANSFORM] Прогрев и фиксация графа CUDA (steps={st_target})...")
            graphs[st_target] = _build_graph(st_target)

            self.ready = True
            self.status = "ready"
            self.ready_time = time.perf_counter()
            self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
            load_dt = self.ready_time - self.start_time
            print(f"[AI_TRANSFORM] ИИ готов к работе ({self.device_name}, прогрев {load_dt:.1f} сек, SD-Turbo + TAESD)")
            logging.info(f"[AI_TRANSFORM] GPU-поток SD-Turbo + TAESD успешно инициализирован ({self.device_name})")

            last_done_idx = -1
            prev_ema = None
            prev_hero_id = None
            prev_sched_key = None

            while self.running:
                if not self.is_enabled():
                    time.sleep(0.05)
                    continue

                with self.lock:
                    if self.req_crop_bgr is None or self.req_idx == last_done_idx or self.req_hero is None:
                        fr_crop = None
                    else:
                        fr_crop = self.req_crop_bgr
                        mk_crop = self.req_mask_crop
                        p_crop = self.req_p_crop
                        p_full = self.req_p_full
                        bbox = self.req_bbox
                        hero = self.req_hero
                        last_done_idx = self.req_idx
                        self.req_crop_bgr = None

                if fr_crop is None:
                    time.sleep(0.003)
                    continue

                t0 = time.perf_counter()
                hid = hero["id"]
                fh_orig, fw_orig = fr_crop.shape[:2]
                fr_small = cv2.resize(fr_crop, (self.ai_w, self.ai_h), interpolation=cv2.INTER_LINEAR)
                if mk_crop is None:
                    mk_small = np.ones((self.ai_h, self.ai_w), dtype=np.float32)
                else:
                    mk_small = cv2.resize(mk_crop, (self.ai_w, self.ai_h), interpolation=cv2.INTER_LINEAR)

                p_small = None
                if p_crop is not None:
                    p_small = p_crop.copy()
                    p_small[:, 0] *= (self.ai_w / float(max(1, fw_orig)))
                    p_small[:, 1] *= (self.ai_h / float(max(1, fh_orig)))

                if bool(config.SKETCH_CONDITIONING):
                    inp_small, _ = build_costume_sketch(
                        fr_small, mk_small, p_small, hero, hands_live=bool(config.HANDS_LIVE)
                    )
                else:
                    pal = HERO_SKETCH_PALETTES.get(hid, HERO_SKETCH_PALETTES["alpamys"])
                    kern_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                    dil = cv2.dilate(fr_small, kern_dil, iterations=1)
                    bg_c = np.full_like(fr_small, pal["bg_fill"], dtype=np.uint8)
                    halo_free = cv2.addWeighted(dil, 0.65, bg_c, 0.35, 0)
                    w_fg = np.clip(mk_small, 0.0, 1.0).astype(np.float32)
                    inp_small = cv2.blendLinear(fr_small, halo_free, w_fg, 1.0 - w_fg)

                inp_rgb = cv2.cvtColor(inp_small, cv2.COLOR_BGR2RGB)
                steps = 1 if self.steps == 1 else 2
                strength = float(np.clip(self.strength, 0.35, 0.75))

                with torch.inference_mode():
                    sched_key = (steps, round(strength, 3))
                    if sched_key != prev_sched_key:
                        t0_idx = int(round(strength * 999))
                        t1_idx = max(1, int(round(t0_idx * 0.48)))
                        ab0 = float(alphas_cumprod_cpu[t0_idx])
                        ab1 = float(alphas_cumprod_cpu[t1_idx])
                        sig0 = math.sqrt(max(1e-5, (1.0 - ab0) / ab0))
                        sig1 = math.sqrt(max(1e-5, (1.0 - ab1) / ab1))
                        if steps == 1:
                            static_t.copy_(torch.tensor([t0_idx, t1_idx], dtype=torch.long))
                            static_sig.copy_(torch.tensor([sig0, 0.0, 0.0], dtype=torch.float16))
                        else:
                            static_t.copy_(torch.tensor([t0_idx, t1_idx], dtype=torch.long))
                            static_sig.copy_(torch.tensor([sig0, sig1, 0.0], dtype=torch.float16))
                        prev_sched_key = sched_key

                    in_t = (
                        (torch.from_numpy(inp_rgb).to(device=device, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0) / 127.5) - 1.0
                    ).contiguous(memory_format=torch.channels_last)
                    static_in.copy_(in_t)

                    # Переключение на эмбеддинги и шум выбранного героя
                    if prev_hero_id != hid:
                        static_emb.copy_(embeds_map.get(hid, embeds_map["alpamys"]))
                        n1, n2 = noises_map.get(hid, noises_map["alpamys"])
                        static_n1.copy_(n1)
                        static_n2.copy_(n2)
                        prev_ema = None

                    if steps not in graphs:
                        graphs[steps] = _build_graph(steps)
                    g_c, out_buf = graphs[steps]
                    g_c.replay()
                    out_rgb = ((out_buf[0].permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()

                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                if prev_ema is not None and prev_ema.shape == out_bgr.shape and prev_hero_id == hid:
                    a_ema = self.ema_alpha
                    out_bgr = cv2.addWeighted(out_bgr, a_ema, prev_ema, 1.0 - a_ema, 0)
                prev_ema = out_bgr.copy()
                prev_hero_id = hid

                dt_ai = max(1e-3, time.perf_counter() - t0)
                vram_now = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                now_done = time.perf_counter()

                with self.lock:
                    self.out_bgr_small = out_bgr
                    self.out_bbox = bbox
                    self.out_p_full = p_full
                    self.out_hero_id = hid
                    self.out_ts = now_done
                    self.ai_ms = dt_ai * 1000.0
                    self.ai_fps = (1.0 / dt_ai) if self.ai_fps == 0 else (self.ai_fps * 0.72 + (1.0 / dt_ai) * 0.28)
                    self.vram_gb = vram_now

        except Exception as e:
            import traceback
            self.failed = True
            self.status = "error"
            self.error_msg_short = f"{type(e).__name__}"[:20]
            self.error_msg = f"{e}\n{traceback.format_exc()}"
            print(f"[AI_TRANSFORM ERROR] Ошибка инициализации/работы ИИ: {e}")
            print(f"[AI_TRANSFORM ERROR]\n{traceback.format_exc()}")
            logging.error(f"[AI_TRANSFORM ERROR] Откат на процедурный костюм: {e}")


_AI_MANAGER = None


def get_ai_manager():
    global _AI_MANAGER
    if _AI_MANAGER is None:
        _AI_MANAGER = AITransformManager()
        _AI_MANAGER.start()
    return _AI_MANAGER
