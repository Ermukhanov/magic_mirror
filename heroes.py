"""
Модуль героев и графических ассетов «Сиқырлы айна».
Содержит:
- 8 героев (Алпамыс, Томирис, Өрмекші-адам, Эльза, Ледибаг, Темір батыр, Ғарышкер, Пері)
- Загрузчик прозрачных PNG из assets/<hero_id>/ с автоматическим запасным рендером OpenCV
- Отрисовку элементов за спиной (плащ, крылья, ранец, паутина) и на теле (броня, эмблема, корона/нимб над головой).
Лицо ребёнка никогда не закрывается!
"""
import os
import math
import cv2
import numpy as np
import config
from config import BASE_DIR, AA
from tracking import lerp, ip
import i18n

HEROES = [
    dict(id="alpamys", name=("Алпамыс", "Алпамыс"), special="tulpar",
         cape=(30, 40, 190), cape2=(70, 90, 255), glow=(40, 190, 255), accent=(40, 190, 245),
         emblem="ram", armor=True, pauldron=True, crown=False, halo=False, wings=False, pack=False),
    dict(id="tomiris", name=("Томирис", "Томирис"), special="arrows",
         cape=(150, 40, 110), cape2=(220, 110, 190), glow=(255, 120, 220), accent=(60, 200, 255),
         emblem="diamond", armor=True, pauldron=True, crown=True, halo=False, wings=False, pack=False),
    dict(id="spider", name=("Өрмекші-адам", "Человек-паук"), special="web",
         cape=None, cape2=None, glow=(60, 60, 255), accent=(235, 70, 35),
         emblem="spider", armor=True, pauldron=False, crown=False, halo=False, wings=False, pack=False),
    dict(id="elsa", name=("Эльза", "Эльза"), special="ice_wave",
         cape=(245, 210, 120), cape2=(255, 245, 210), glow=(255, 225, 120), accent=(255, 245, 200),
         emblem="snowflake", armor=False, pauldron=False, crown=True, halo=False, wings=False, pack=False),
    dict(id="ladybug", name=("Ледибаг", "Ледибаг"), special="lucky_charm",
         cape=None, cape2=None, glow=(70, 50, 255), accent=(40, 40, 235),
         emblem="spots", armor=True, pauldron=False, crown=False, halo=False, wings=False, pack=False),
    dict(id="iron", name=("Темір батыр", "Железный герой"), special="repulsor",
         cape=(35, 45, 200), cape2=(60, 195, 255), glow=(80, 225, 255), accent=(40, 60, 215),
         emblem="reactor", armor=True, pauldron=True, crown=False, halo=False, wings=False, pack=False),
    dict(id="astronaut", name=("Ғарышкер", "Космонавт"), special="zero_g",
         cape=(200, 200, 210), cape2=(255, 255, 255), glow=(255, 230, 100), accent=(70, 70, 255),
         emblem="panel", armor=False, pauldron=False, crown=False, halo=False, wings=False, pack=True),
    dict(id="fairy", name=("Пері", "Фея"), special="magic_dust",
         cape=None, cape2=None, glow=(255, 200, 255), wing=(255, 170, 230), accent=(150, 255, 255),
         emblem="gem", armor=False, pauldron=False, crown=False, halo=True, wings=True, pack=False),
]

_png_cache = {}


def apoly(img, pts, col, alpha=1.0, edge=None, th=2):
    pts = np.array(pts, np.int32)
    hh, ww = img.shape[:2]
    if alpha >= 1:
        cv2.fillPoly(img, [pts], col, AA)
    else:
        x, y, w, h = cv2.boundingRect(pts)
        x0, y0, x1, y1 = max(x - 4, 0), max(y - 4, 0), min(x + w + 4, ww), min(y + h + 4, hh)
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1].copy()
            ov = roi.copy()
            cv2.fillPoly(ov, [pts - [x0, y0]], col, AA)
            img[y0:y1, x0:x1] = cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0)
    if edge is not None:
        cv2.polylines(img, [pts], True, edge, th, AA)


def star_pts(c, r, rot=-math.pi / 2, inner=0.42):
    return [(c[0] + (r if i % 2 == 0 else r * inner) * math.cos(rot + i * math.pi / 5),
             c[1] + (r if i % 2 == 0 else r * inner) * math.sin(rot + i * math.pi / 5)) for i in range(10)]


def hero_display_name(hero):
    key = f"hero_{hero['id']}"
    val = i18n.tr(key, "")
    if val and val != key:
        return val
    return hero["name"][0 if i18n.get_lang() == "kk" else 1]


def load_hero_png(hero_id, part_name):
    """Загружает RGBA PNG-ассет из assets/<hero_id>/<part_name>.png (если hero_art: true) и кэширует в памяти."""
    if not getattr(config, "HERO_ART", True):
        return None
    key = (hero_id, part_name)
    if key in _png_cache:
        return _png_cache[key]
    p = os.path.join(BASE_DIR, "assets", hero_id, f"{part_name}.png")
    if os.path.exists(p):
        try:
            rgba = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                _png_cache[key] = rgba
                return rgba
        except Exception:
            pass
    _png_cache[key] = None
    return None


def overlay_rgba_sprite(img, rgba, cx, cy, target_w, target_h, angle_deg=0.0):
    """Накладывает RGBA PNG-спрайт с поворотом и масштабом по скелету."""
    if rgba is None or target_w < 4 or target_h < 4:
        return False
    tw, th = int(target_w), int(target_h)
    resized = cv2.resize(rgba, (tw, th), interpolation=cv2.INTER_LINEAR)
    if abs(angle_deg) > 1.0:
        M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angle_deg, 1.0)
        resized = cv2.warpAffine(resized, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    x0, y0 = int(cx - tw // 2), int(cy - th // 2)
    hh, ww = img.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(ww, x0 + tw), min(hh, y0 + th)
    if dx1 <= dx0 or dy1 <= dy0:
        return False
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    sub = resized[sy0:sy1, sx0:sx1]
    a = sub[:, :, 3:4]
    a3 = cv2.merge([a[:, :, 0], a[:, :, 0], a[:, :, 0]])
    roi = img[dy0:dy1, dx0:dx1]
    img[dy0:dy1, dx0:dx1] = cv2.add(
        cv2.multiply(sub[:, :, :3], a3, scale=1.0 / 255.0),
        cv2.multiply(roi, cv2.bitwise_not(a3), scale=1.0 / 255.0),
    )
    return True


def draw_back(img, g, hero, t):
    """За спиной ребёнка: плащ, крылья, ранец (с поддержкой PNG из assets/<hero_id>/back.png)."""
    sw, th = g["sw"], g["th"]
    ls, rs, lh, rh = g["ls"], g["rs"], g["lh"], g["rh"]
    back_png = load_hero_png(hero["id"], "back")
    if back_png is not None:
        cx, cy = (g["neck"][0] + g["hip"][0]) * 0.5, g["neck"][1] + th * 0.55
        ang = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
        if overlay_rgba_sprite(img, back_png, cx, cy, sw * 2.4, th * 1.65, -ang):
            return

    sway = math.sin(t * 3.5) * sw * 0.16
    if hero["id"] == "spider":
        c = (g["neck"] + g["hip"]) * 0.5
        for r_w in (int(sw * 0.95), int(sw * 1.25)):
            cv2.circle(img, ip(c), r_w, (200, 200, 255), 1, AA)
        for i in range(8):
            a = i * math.pi / 4 + math.sin(t * 2) * 0.08
            p_end = c + np.array([math.cos(a), math.sin(a)]) * (sw * 1.32)
            cv2.line(img, ip(c), ip(p_end), (190, 190, 255), 1, AA)
    if hero["cape"] is not None:
        L = th * (0.75 if hero["id"] == "alpamys" else 1.2)
        w = lambda k: math.sin(t * 4 + k) * sw * 0.10
        pts = [ls + [-sw * .05, 0], rs + [sw * .05, 0], rh + [sw * .55 + sway + w(1), L * .75],
               (rh + lh) / 2 + [sway * 1.2 + w(2), L * .9], lh + [-sw * .55 + sway + w(3), L * .75]]
        apoly(img, pts, hero["cape"], 0.90 if hero["id"] != "elsa" else 0.72, hero["glow"], 2)
        c = np.mean(pts, 0)
        apoly(img, [c + (q - c) * 0.62 for q in pts], hero["cape2"], 0.38)
        if hero["id"] == "elsa":
            for i in range(7):
                sx = int(c[0] + math.sin(t * 2.5 + i * 1.7) * sw * 0.55)
                sy = int(c[1] + ((i * 29 + t * 40) % max(10, int(L * 0.65))) - L * 0.25)
                cv2.circle(img, (sx, sy), 3, (255, 255, 255), -1, AA)
    if hero["pack"]:
        pts = [ls + [-sw * .32, -th * .08], rs + [sw * .32, -th * .08], rh + [sw * .32, th * .18], lh + [-sw * .32, th * .18]]
        apoly(img, pts, (190, 190, 200), 0.95, hero["glow"], 3)
        for s in (-1, 1):
            c = g["neck"] + [s * sw * .55, th * .25]
            cv2.circle(img, ip(c), int(sw * .12), hero["glow"], -1, AA)
            cv2.circle(img, ip(c), int(sw * .12), (255, 255, 255), 2, AA)
    if hero["wings"]:
        flap = math.sin(t * 7) * 9
        specs = []
        for s in (-1, 1):
            specs.append((g["neck"] + [s * sw * .95, -th * .22], (int(sw * 1.25), int(sw * .42)), -s * (36 + flap)))
            specs.append((g["neck"] + [s * sw * .8, th * .38], (int(sw * .9), int(sw * .3)), s * (30 + flap * .6)))
        for c, ax, ang in specs:
            cv2.ellipse(img, ip(c), ax, ang, 0, 360, hero["wing"], -1, AA)
            cv2.ellipse(img, ip(c), ax, ang, 0, 360, (255, 255, 255), 2, AA)
            cv2.ellipse(img, ip(c), (int(ax[0] * .6), int(ax[1] * .55)), ang, 0, 360, hero["glow"], 2, AA)
        for i in range(12):
            a = t * 1.7 + i * 1.9
            c = g["neck"] + [math.cos(a) * sw * 1.8, math.sin(a * 1.3) * th * .65]
            cv2.circle(img, ip(c), 2 + i % 3, (255, 255, 255), -1, AA)


def emblem(img, c, r, kind, hero, t):
    glow, acc = hero["glow"], hero["accent"]
    ci = ip(c)
    if kind == "star":
        apoly(img, star_pts(c, r * 1.15), acc, 1.0, (255, 255, 255), 2)
    elif kind == "diamond":
        d = [(c[0], c[1] - r), (c[0] + r * .85, c[1]), (c[0], c[1] + r * 1.15), (c[0] - r * .85, c[1])]
        apoly(img, d, acc, 1.0, (255, 255, 255), 3)
        apoly(img, star_pts(c, r * .5), (255, 255, 255), 1.0)
    elif kind == "ram":
        for s, (a0, a1) in ((1, (-40, 230)), (-1, (-50, 220))):
            cv2.ellipse(img, (int(c[0] + s * r * .6), ci[1]), (int(r * .55), int(r * .55)), 0, a0, a1, (255, 255, 255), max(3, int(r * .22)), AA)
            cv2.ellipse(img, (int(c[0] + s * r * .6), ci[1]), (int(r * .55), int(r * .55)), 0, a0, a1, (30, 40, 190), max(1, int(r * .1)), AA)
        cv2.circle(img, ci, int(r * .22), (30, 40, 190), -1, AA)
    elif kind == "spider":
        cv2.ellipse(img, ci, (int(r * 0.35), int(r * 0.48)), 0, 0, 360, (25, 25, 30), -1, AA)
        cv2.circle(img, (ci[0], int(ci[1] - r * 0.45)), int(r * 0.22), (25, 25, 30), -1, AA)
        for s in (-1, 1):
            for ang in (-0.6, -0.2, 0.25, 0.65):
                p1 = (int(ci[0] + s * r * 0.75 * math.cos(ang)), int(ci[1] + r * 0.75 * math.sin(ang)))
                cv2.line(img, ci, p1, (25, 25, 30), max(2, int(r * 0.12)), AA)
    elif kind == "snowflake":
        for i in range(6):
            ang = i * math.pi / 3 + t * 0.5
            p1 = (int(ci[0] + math.cos(ang) * r * 1.1), int(ci[1] + math.sin(ang) * r * 1.1))
            cv2.line(img, ci, p1, (255, 255, 255), max(2, int(r * 0.14)), AA)
        cv2.circle(img, ci, int(r * 0.35), glow, -1, AA)
    elif kind == "spots":
        cv2.circle(img, ci, int(r * 0.95), (40, 40, 235), -1, AA)
        cv2.circle(img, ci, int(r * 0.95), (255, 255, 255), 2, AA)
        for dx, dy in ((0, -0.42), (-0.42, 0.1), (0.42, 0.1), (-0.25, 0.52), (0.25, 0.52)):
            cv2.circle(img, (int(ci[0] + dx * r), int(ci[1] + dy * r)), max(3, int(r * 0.18)), (20, 20, 25), -1, AA)
    elif kind == "reactor":
        pulse = 1.0 + 0.12 * math.sin(t * 10.0)
        rr = int(r * 0.9 * pulse)
        cv2.circle(img, ci, rr + 5, (40, 190, 255), 3, AA)
        cv2.circle(img, ci, rr, (255, 240, 180), -1, AA)
        cv2.circle(img, ci, int(rr * 0.65), (255, 255, 255), -1, AA)
    elif kind == "panel":
        cv2.rectangle(img, (int(c[0] - r), int(c[1] - r * .7)), (int(c[0] + r), int(c[1] + r * .7)), (45, 40, 50), -1, AA)
        cv2.rectangle(img, (int(c[0] - r), int(c[1] - r * .7)), (int(c[0] + r), int(c[1] + r * .7)), glow, 2, AA)
        for i, col in enumerate(((0, 0, 255), (0, 255, 0), (0, 220, 255))):
            on = (int(t * 3) + i) % 3 != 0
            cv2.circle(img, (int(c[0] + (i - 1) * r * .6), ci[1]), max(3, int(r * .2)), col if on else (60, 60, 60), -1, AA)
    else:  # gem
        cv2.circle(img, ci, int(r * .8), glow, -1, AA)
        cv2.circle(img, ci, int(r * .8), (255, 255, 255), 3, AA)
        cv2.circle(img, (ci[0] - int(r * .25), ci[1] - int(r * .25)), int(r * .2), (255, 255, 255), -1, AA)


def draw_hero_headpiece(img, g, hero, t):
    """
    Фирменные головные уборы, маски и шлемы для всех 8 персонажей.
    ВАЖНО: Лицо, глаза, носик и улыбка ребёнка ВСЕГДА остаются полностью открытыми и яркими!
    """
    hid = hero["id"]
    nx, ny = g["nose"][0], g["nose"][1]
    hr = max(18.0, float(g["hr"]))
    sw = g["sw"]
    glow = hero["glow"]
    acc = hero["accent"]

    # Пользовательский PNG-ассет (если есть)
    crown_png = load_hero_png(hid, "crown")
    if crown_png is not None:
        overlay_rgba_sprite(img, crown_png, nx, ny - hr * 1.55, hr * 2.1, hr * 1.35)
        return

    if hid == "spider":
        # 1. ЧЕЛОВЕК-ПАУК: Маска-капюшон вокруг головы с паутиной и белыми глазами паука (лицо открыто!)
        pts_hood = []
        n_seg = 18
        r_out_x, r_out_y = hr * 1.25, hr * 1.45
        r_in_x, r_in_y = hr * 0.88, hr * 1.05
        for i_s in range(n_seg + 1):
            ang = math.pi * (0.85 + 1.30 * (i_s / float(n_seg)))
            pts_hood.append([nx + math.cos(ang) * r_out_x, ny + math.sin(ang) * r_out_y])
        for i_s in range(n_seg + 1):
            ang = math.pi * (2.15 - 1.30 * (i_s / float(n_seg)))
            pts_hood.append([nx + math.cos(ang) * r_in_x, ny + math.sin(ang) * r_in_y])
        pts_hood_arr = np.array(pts_hood, np.int32)
        cv2.fillPoly(img, [pts_hood_arr], (25, 25, 220), AA)
        cv2.polylines(img, [pts_hood_arr], True, (20, 20, 25), 3, AA)

        for w_ang in (-0.45, -0.22, 0.0, 0.22, 0.45):
            p_top = (int(nx + math.sin(w_ang) * hr * 1.25), int(ny - math.cos(w_ang) * hr * 1.45))
            p_bot = (int(nx + math.sin(w_ang) * hr * 0.88), int(ny - math.cos(w_ang) * hr * 1.05))
            cv2.line(img, p_top, p_bot, (20, 20, 25), 2, AA)
        for r_arc in (hr * 1.02, hr * 1.20, hr * 1.38):
            cv2.ellipse(img, (int(nx), int(ny)), (int(r_arc * 0.88), int(r_arc)), 0, -135, -45, (20, 20, 25), 2, AA)

        # Белые паучьи глаза на висках с чёрным контуром
        ey_y = ny - hr * 0.62
        for s_k in (-1, 1):
            eye_pts = np.array([
                [int(nx + s_k * hr * 0.42), int(ey_y + hr * 0.08)],
                [int(nx + s_k * hr * 1.05), int(ey_y - hr * 0.32)],
                [int(nx + s_k * hr * 0.90), int(ey_y + hr * 0.22)],
            ], np.int32)
            cv2.fillPoly(img, [eye_pts], (255, 255, 255), AA)
            cv2.polylines(img, [eye_pts], True, (20, 20, 25), 3, AA)

    elif hid == "iron":
        # 2. ТЕМІР БАТЫР (IRON MAN): Шлем с открытым забралом + голографический HUD
        helm_pts = np.array([
            [int(nx - hr * 1.08), int(ny - hr * 0.45)],
            [int(nx - hr * 0.95), int(ny - hr * 1.35)],
            [int(nx - hr * 0.45), int(ny - hr * 1.55)],
            [int(nx + hr * 0.45), int(ny - hr * 1.55)],
            [int(nx + hr * 0.95), int(ny - hr * 1.35)],
            [int(nx + hr * 1.08), int(ny - hr * 0.45)],
            [int(nx + hr * 0.82), int(ny - hr * 0.70)],
            [int(nx - hr * 0.82), int(ny - hr * 0.70)],
        ], np.int32)
        cv2.fillPoly(img, [helm_pts], (30, 40, 195), AA)
        cv2.polylines(img, [helm_pts], True, (50, 215, 255), 3, AA)

        brow_pts = np.array([
            [int(nx - hr * 0.65), int(ny - hr * 0.70)],
            [int(nx - hr * 0.45), int(ny - hr * 1.15)],
            [int(nx + hr * 0.45), int(ny - hr * 1.15)],
            [int(nx + hr * 0.65), int(ny - hr * 0.70)],
            [int(nx), int(ny - hr * 0.78)],
        ], np.int32)
        cv2.fillPoly(img, [brow_pts], (50, 215, 255), AA)
        cv2.polylines(img, [brow_pts], True, (255, 255, 255), 2, AA)

        for s_k in (-1, 1):
            chk_pts = np.array([
                [int(nx + s_k * hr * 0.95), int(ny - hr * 0.50)],
                [int(nx + s_k * hr * 1.18), int(ny + hr * 0.20)],
                [int(nx + s_k * hr * 0.92), int(ny + hr * 0.55)],
                [int(nx + s_k * hr * 0.78), int(ny - hr * 0.10)],
            ], np.int32)
            cv2.fillPoly(img, [chk_pts], (30, 40, 195), AA)
            cv2.polylines(img, [chk_pts], True, (50, 215, 255), 2, AA)

        hud_c = (255, 240, 60)
        for s_k in (-1, 1):
            hx = int(nx + s_k * hr * 0.75)
            hy = int(ny - hr * 0.35)
            cv2.ellipse(img, (hx, hy), (int(hr * 0.32), int(hr * 0.32)), 0, -50 if s_k == -1 else 130, 50 if s_k == -1 else 230, hud_c, 2, AA)
            cv2.line(img, (hx - int(hr * 0.15), hy), (hx + int(hr * 0.15), hy), hud_c, 1, AA)

    elif hid == "alpamys":
        # 3. АЛПАМЫС БАТЫР: Богатырский шлем «Дулыға» с перьями «Үкі» и золотым налобником
        top_y = int(ny - hr * 2.30)
        brow_y = int(ny - hr * 0.72)
        helm_pts = np.array([
            [int(nx - hr * 1.05), brow_y],
            [int(nx - hr * 0.85), int(ny - hr * 1.45)],
            [int(nx - hr * 0.35), int(ny - hr * 2.05)],
            [int(nx), top_y],
            [int(nx + hr * 0.35), int(ny - hr * 2.05)],
            [int(nx + hr * 0.85), int(ny - hr * 1.45)],
            [int(nx + hr * 1.05), brow_y],
            [int(nx), int(brow_y - hr * 0.15)],
        ], np.int32)
        cv2.fillPoly(img, [helm_pts], (180, 195, 205), AA)
        cv2.polylines(img, [helm_pts], True, (45, 175, 235), 4, AA)

        brow_pts = np.array([
            [int(nx - hr * 1.02), brow_y],
            [int(nx + hr * 1.02), brow_y],
            [int(nx + hr * 0.85), int(brow_y - hr * 0.30)],
            [int(nx), int(brow_y - hr * 0.42)],
            [int(nx - hr * 0.85), int(brow_y - hr * 0.30)],
        ], np.int32)
        cv2.fillPoly(img, [brow_pts], (45, 175, 235), AA)
        cv2.polylines(img, [brow_pts], True, (255, 255, 255), 2, AA)
        cv2.circle(img, (int(nx), int(brow_y - hr * 0.20)), max(4, int(hr * 0.12)), (25, 25, 200), -1, AA)

        sway = math.sin(t * 4.5) * hr * 0.18
        for f_i in (-2, -1, 0, 1, 2):
            ang_f = f_i * 0.22 + sway * 0.08
            f_len = hr * (0.85 + 0.15 * math.cos(f_i))
            p_base = (int(nx), top_y)
            p_tip = (int(nx + math.sin(ang_f) * f_len + sway), int(top_y - math.cos(ang_f) * f_len))
            cv2.line(img, p_base, p_tip, (255, 255, 255), max(4, int(hr * 0.14)), AA)
            cv2.circle(img, p_tip, max(3, int(hr * 0.10)), (255, 245, 220), -1, AA)

        for s_k in (-1, 1):
            mail_pts = np.array([
                [int(nx + s_k * hr * 0.95), brow_y],
                [int(nx + s_k * hr * 1.15), int(ny + hr * 0.40)],
                [int(nx + s_k * hr * 0.82), int(ny + hr * 0.50)],
                [int(nx + s_k * hr * 0.72), int(ny - hr * 0.10)],
            ], np.int32)
            cv2.fillPoly(img, [mail_pts], (150, 160, 175), AA)
            cv2.polylines(img, [mail_pts], True, (220, 230, 240), 2, AA)

    elif hid == "tomiris":
        # 4. ТОМИРИС: Золотая сакская царская тиара-венец с крупным рубином
        brow_y = int(ny - hr * 0.70)
        top_y = int(ny - hr * 1.85)
        tiara_pts = np.array([
            [int(nx - hr * 1.10), brow_y],
            [int(nx - hr * 0.85), int(ny - hr * 1.35)],
            [int(nx - hr * 0.45), int(ny - hr * 1.65)],
            [int(nx), top_y],
            [int(nx + hr * 0.45), int(ny - hr * 1.65)],
            [int(nx + hr * 0.85), int(ny - hr * 1.35)],
            [int(nx + hr * 1.10), brow_y],
            [int(nx), int(brow_y - hr * 0.22)],
        ], np.int32)
        cv2.fillPoly(img, [tiara_pts], (45, 185, 245), AA)
        cv2.polylines(img, [tiara_pts], True, (255, 255, 255), 3, AA)

        ruby_c = (int(nx), int(ny - hr * 1.10))
        cv2.circle(img, ruby_c, max(6, int(hr * 0.22)), (30, 20, 230), -1, AA)
        cv2.circle(img, ruby_c, max(6, int(hr * 0.22)), (255, 220, 110), 3, AA)
        cv2.circle(img, (ruby_c[0] - 2, ruby_c[1] - 2), max(2, int(hr * 0.08)), (255, 255, 255), -1, AA)

        for s_k in (-1, 1):
            px0 = int(nx + s_k * hr * 1.05)
            py0 = brow_y
            for p_i in range(3):
                py_d = py0 + int(p_i * hr * 0.25)
                cv2.circle(img, (px0, py_d), max(3, int(hr * 0.08)), (45, 185, 245), -1, AA)
                cv2.circle(img, (px0, py_d), max(3, int(hr * 0.08)), (255, 255, 255), 1, AA)

    elif hid == "elsa":
        # 5. ЭЛЬЗА: Хрустальная ледяная диадема с 5 кристаллами и снежинками
        brow_y = int(ny - hr * 0.65)
        ice_col = (255, 240, 185)
        spikes = [
            (0.0, 1.95, hr * 0.18),
            (-0.42, 1.65, hr * 0.15),
            (0.42, 1.65, hr * 0.15),
            (-0.85, 1.35, hr * 0.12),
            (0.85, 1.35, hr * 0.12),
        ]
        for dx_k, h_k, w_k in spikes:
            sp_pts = np.array([
                [int(nx + dx_k * hr - w_k), brow_y],
                [int(nx + dx_k * hr), int(ny - hr * h_k)],
                [int(nx + dx_k * hr + w_k), brow_y],
            ], np.int32)
            cv2.fillPoly(img, [sp_pts], ice_col, AA)
            cv2.polylines(img, [sp_pts], True, (255, 255, 255), 2, AA)

        cv2.ellipse(img, (int(nx), brow_y), (int(hr * 0.95), int(hr * 0.18)), 0, -180, 0, (255, 255, 255), 4, AA)

        sn_c = (int(nx), int(ny - hr * 1.15))
        for i_sn in range(6):
            ang_sn = i_sn * math.pi / 3.0
            p_sn = (int(sn_c[0] + math.cos(ang_sn) * hr * 0.22), int(sn_c[1] + math.sin(ang_sn) * hr * 0.22))
            cv2.line(img, sn_c, p_sn, (255, 255, 255), 2, AA)

        for i_st in range(6):
            a_st = t * 2.8 + i_st * (2.0 * math.pi / 6.0)
            st_x = int(nx + math.cos(a_st) * hr * 1.15)
            st_y = int(ny - hr * 1.45 + math.sin(a_st * 1.4) * hr * 0.35)
            cv2.circle(img, (st_x, st_y), 3, (255, 255, 255), -1, AA)
            cv2.circle(img, (st_x, st_y), 6, ice_col, 1, AA)

    elif hid == "ladybug":
        # 6. ЛЕДИБАГ: Маска супергероини с точками + усики на ободке
        mask_y = int(ny - hr * 0.35)
        mw, mh = int(hr * 0.95), int(hr * 0.36)
        for s_k in (-1, 1):
            m_cx = int(nx + s_k * hr * 0.45)
            cv2.ellipse(img, (m_cx, mask_y), (mw // 2 + 4, mh), -10 * s_k, 0, 360, (35, 35, 235), -1, AA)
            cv2.ellipse(img, (m_cx, mask_y), (mw // 2 + 4, mh), -10 * s_k, 0, 360, (20, 20, 25), 2, AA)
            cv2.circle(img, (int(m_cx + s_k * hr * 0.28), mask_y - 6), max(3, int(hr * 0.08)), (20, 20, 25), -1, AA)
            cv2.circle(img, (int(m_cx - s_k * hr * 0.08), mask_y + 8), max(2, int(hr * 0.06)), (20, 20, 25), -1, AA)
        cv2.line(img, (int(nx - hr * 0.18), mask_y), (int(nx + hr * 0.18), mask_y), (35, 35, 235), 5, AA)

        top_h = int(ny - hr * 1.30)
        for s_k in (-1, 1):
            bob_ant = math.sin(t * 6.5 + s_k) * hr * 0.14
            p_b = (int(nx + s_k * hr * 0.35), top_h)
            p_m = (int(nx + s_k * hr * 0.65), int(top_h - hr * 0.45))
            p_t = (int(nx + s_k * hr * 0.75 + bob_ant), int(top_h - hr * 0.75))
            cv2.line(img, p_b, p_m, (20, 20, 25), 3, AA)
            cv2.line(img, p_m, p_t, (20, 20, 25), 3, AA)
            cv2.circle(img, p_t, max(4, int(hr * 0.12)), (35, 35, 235), -1, AA)
            cv2.circle(img, p_t, max(4, int(hr * 0.12)), (255, 255, 255), 1, AA)

    elif hid == "astronaut":
        # 7. ҒАРЫШКЕР (ASTRONAUT): Шлем скафандра с открытым золотистым стеклом и прожекторами
        h_cx, h_cy = int(nx), int(ny - hr * 0.20)
        rx_out, ry_out = int(hr * 1.35), int(hr * 1.45)
        cv2.ellipse(img, (h_cx, h_cy), (rx_out, ry_out), 0, -165, -15, (225, 230, 240), 18, AA)
        cv2.ellipse(img, (h_cx, h_cy), (rx_out, ry_out), 0, -165, -15, (120, 130, 145), 2, AA)
        cv2.ellipse(img, (h_cx, h_cy), (int(hr * 1.15), int(hr * 1.25)), 0, 0, 360, (50, 215, 255), 4, AA)

        for s_k in (-1, 1):
            lx = int(h_cx + s_k * hr * 1.35)
            ly = int(h_cy - hr * 0.10)
            cv2.circle(img, (lx, ly), max(6, int(hr * 0.18)), (240, 245, 255), -1, AA)
            cv2.circle(img, (lx, ly), max(6, int(hr * 0.18)), (50, 215, 255), 2, AA)
            cv2.circle(img, (lx, ly), max(3, int(hr * 0.09)), (255, 255, 255), -1, AA)
            cone_pts = np.array([
                [lx, ly],
                [int(lx + s_k * hr * 0.85), int(ly - hr * 0.45)],
                [int(lx + s_k * hr * 0.85), int(ly + hr * 0.45)],
            ], np.int32)
            apoly(img, cone_pts, (255, 255, 220), 0.22)

    elif hid == "fairy":
        # 8. ПЕРІ (FAIRY): Волшебный венок из цветов, эльфийские ушки и пыльца
        brow_y = int(ny - hr * 0.70)
        for s_k in (-1, 1):
            ear_pts = np.array([
                [int(nx + s_k * hr * 0.95), int(ny - hr * 0.25)],
                [int(nx + s_k * hr * 1.55), int(ny - hr * 0.75)],
                [int(nx + s_k * hr * 1.05), int(ny + hr * 0.20)],
            ], np.int32)
            cv2.fillPoly(img, [ear_pts], (215, 175, 235), AA)
            cv2.polylines(img, [ear_pts], True, (255, 255, 255), 2, AA)
            cv2.circle(img, (int(nx + s_k * hr * 1.05), int(ny + hr * 0.22)), 3, (255, 255, 255), -1, AA)

        fl_cols = [(225, 160, 255), (80, 240, 255), (130, 255, 120), (255, 180, 200), (255, 245, 130)]
        for i_fl in range(7):
            frac = (i_fl - 3) / 3.0
            fx = int(nx + frac * hr * 0.92)
            fy = int(brow_y - math.cos(frac * 1.2) * hr * 0.25)
            c_fl = fl_cols[i_fl % len(fl_cols)]
            cv2.circle(img, (fx, fy), max(5, int(hr * 0.16)), c_fl, -1, AA)
            cv2.circle(img, (fx, fy), max(2, int(hr * 0.07)), (255, 255, 255), -1, AA)

        cv2.ellipse(img, (int(nx), brow_y), (int(hr * 0.95), int(hr * 0.18)), 0, -180, 0, (80, 240, 255), 2, AA)

        for i_p in range(8):
            ang_p = t * 3.2 + i_p * (2.0 * math.pi / 8.0)
            px_f = int(nx + math.cos(ang_p) * hr * 1.25)
            py_f = int(ny - hr * 0.85 + math.sin(ang_p * 1.5) * hr * 0.45)
            cv2.circle(img, (px_f, py_f), 3, (255, 255, 255), -1, AA)
            cv2.circle(img, (px_f, py_f), 5, (120, 245, 255), 1, AA)


def draw_front(img, g, hero, t, ai_active=False):
    """На теле (ниже лица): броня, эмблема, перчатки, шлемы/маски НАД и ВОКРУГ головы. Лицо всегда открыто!"""
    sw, th = g["sw"], g["th"]
    ls, rs, lh, rh = g["ls"], g["rs"], g["lh"], g["rh"]
    glow, acc = hero["glow"], hero["accent"]
    procedural = getattr(config, "PROCEDURAL_ARMOR", False)

    armor_png = load_hero_png(hero["id"], "armor")
    if armor_png is not None:
        cx, cy = (g["neck"][0] + g["hip"][0]) * 0.5, (g["neck"][1] + g["hip"][1]) * 0.5
        ang = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
        overlay_rgba_sprite(img, armor_png, cx, cy, sw * 1.45, th * 1.15, -ang)
    elif not ai_active and procedural and hero["armor"]:
        plate = [ls + [sw * .06, sw * .07], rs + [-sw * .06, sw * .07], lerp(rs, rh, .72) + [-sw * .04, 0], lerp(ls, lh, .72) + [sw * .04, 0]]
        apoly(img, plate, acc, 0.88, (255, 255, 255), 2)

    # Узнаваемые фирменные аксессуары каждого персонажа (рисуются всегда поверх костюма/ИИ)
    if hero["id"] == "spider":
        c_ch = lerp(g["neck"], g["hip"], 0.35)
        for a_i in (-0.55, -0.28, 0.0, 0.28, 0.55):
            cv2.line(img, ip(c_ch), ip(c_ch + [math.sin(a_i) * sw * 0.52, math.cos(a_i) * th * 0.38]), (30, 30, 210), 2, AA)
        for r_web in (int(sw * 0.18), int(sw * 0.32)):
            cv2.ellipse(img, ip(c_ch), (r_web, int(r_web * 0.72)), 0, -60, 240, (30, 30, 210), 2, AA)
    elif hero["id"] == "ladybug":
        for dx, dy in ((-0.25, 0.18), (0.25, 0.18), (-0.16, 0.48), (0.16, 0.48), (0.0, 0.62), (-0.32, 0.35), (0.32, 0.35)):
            sp = g["neck"] + np.array([dx * sw, dy * th])
            cv2.circle(img, ip(sp), max(4, int(sw * 0.055)), (20, 20, 25), -1, AA)
    elif hero["id"] in ("alpamys", "tomiris", "iron"):
        # Золотой или титановый пояс с пряжкой
        belt_col = hero["cape"] if hero["cape"] is not None else glow
        a, b = lerp(ls, lh, .65) + [sw * .04, 0], lerp(rs, rh, .65) + [-sw * .04, 0]
        cv2.line(img, ip(a), ip(b), (35, 145, 205), max(3, int(sw * .05)), AA)
        cv2.circle(img, ip((a + b) / 2), max(6, int(sw * .065)), (255, 235, 120), -1, AA)
        cv2.circle(img, ip((a + b) / 2), max(6, int(sw * .065)), (40, 40, 50), 2, AA)

    if hero["pauldron"] and armor_png is None:
        for s in (ls, rs):
            cv2.ellipse(img, ip(s), (int(sw * .24), int(sw * .17)), 0, 0, 360, acc, -1, AA)
            cv2.ellipse(img, ip(s), (int(sw * .24), int(sw * .17)), 0, 0, 360, (255, 255, 255), 2, AA)

    # Фирменная светящаяся эмблема на груди (реактор, паук, снежинка, батырский медальон)
    if armor_png is None:
        emblem(img, lerp(g["neck"], g["hip"], .33 if hero["armor"] else .4), sw * .2, hero["emblem"], hero, t)

    if hero["id"] == "astronaut" and armor_png is None:
        cv2.ellipse(img, ip(g["neck"] + [0, sw * .04]), (int(sw * .4), int(sw * .12)), 0, 0, 360, (235, 235, 240), int(sw * .07), AA)
        cv2.ellipse(img, ip(g["neck"] + [0, sw * .04]), (int(sw * .4), int(sw * .12)), 0, 0, 360, glow, 2, AA)

    # Фирменные маски, шлемы и головные уборы (лицо всегда полностью открыто!)
    draw_hero_headpiece(img, g, hero, t)

    # Перчатки и наручи на запястьях
    glove_png = load_hero_png(hero["id"], "glove")
    for w, w_vis in ((g["lw"], g.get("lw_vis", 1.0)), (g["rw"], g.get("rw_vis", 1.0))):
        if w_vis < 0.45:
            continue
        r = sw * .15 * (1 + .12 * math.sin(t * 8))
        if glove_png is not None:
            overlay_rgba_sprite(img, glove_png, w[0], w[1], r * 2.4, r * 2.4)
        else:
            cv2.circle(img, ip(w), int(r), glow, -1, AA)
            cv2.circle(img, ip(w), int(r), (255, 255, 255), 2, AA)
