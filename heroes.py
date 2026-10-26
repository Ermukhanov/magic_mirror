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


def draw_front(img, g, hero, t, ai_active=False):
    """На теле (ниже лица): броня, эмблема, перчатки, корона/нимб НАД головой. Лицо всегда открыто!"""
    sw, th = g["sw"], g["th"]
    ls, rs, lh, rh = g["ls"], g["rs"], g["lh"], g["rh"]
    glow, acc = hero["glow"], hero["accent"]

    # Если активно реальное ИИ-превращение (ai_active=True), ИИ сам детально отрисовал броню, эмблему и наручи на теле;
    # поверх тела не перекрываем их плоским полигоном (рисуем только корону/нимб над головой и PNG-арт при наличии).
    if not ai_active:
        armor_png = load_hero_png(hero["id"], "armor")
        if armor_png is not None:
            cx, cy = (g["neck"][0] + g["hip"][0]) * 0.5, (g["neck"][1] + g["hip"][1]) * 0.5
            ang = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
            overlay_rgba_sprite(img, armor_png, cx, cy, sw * 1.45, th * 1.15, -ang)
        elif hero["armor"]:
            plate = [ls + [sw * .06, sw * .07], rs + [-sw * .06, sw * .07], lerp(rs, rh, .72) + [-sw * .04, 0], lerp(ls, lh, .72) + [sw * .04, 0]]
            apoly(img, plate, acc, 0.88, (255, 255, 255), 2)
            if hero["id"] == "spider":
                c_ch = lerp(g["neck"], g["hip"], 0.35)
                for a_i in (-0.5, 0.0, 0.5):
                    cv2.line(img, ip(c_ch), ip(c_ch + [math.sin(a_i) * sw * 0.45, math.cos(a_i) * th * 0.35]), (40, 40, 190), 2, AA)
            elif hero["id"] == "ladybug":
                for dx, dy in ((-0.22, 0.22), (0.22, 0.22), (-0.15, 0.52), (0.15, 0.52), (0.0, 0.65)):
                    sp = g["neck"] + np.array([dx * sw, dy * th])
                    cv2.circle(img, ip(sp), max(4, int(sw * 0.055)), (25, 25, 30), -1, AA)
            else:
                belt_col = hero["cape"] if hero["cape"] is not None else glow
                a, b = lerp(ls, lh, .62) + [sw * .04, 0], lerp(rs, rh, .62) + [-sw * .04, 0]
                cv2.line(img, ip(a), ip(b), belt_col, max(3, int(sw * .05)), AA)
                a, b = lerp(ls, lh, .78), lerp(rs, rh, .78)
                cv2.line(img, ip(a), ip(b), (30, 30, 40), max(4, int(sw * .07)), AA)
                cv2.circle(img, ip((a + b) / 2), int(sw * .06), (255, 255, 255), -1, AA)

        if hero["pauldron"] and armor_png is None:
            for s in (ls, rs):
                cv2.ellipse(img, ip(s), (int(sw * .24), int(sw * .17)), 0, 0, 360, acc, -1, AA)
                cv2.ellipse(img, ip(s), (int(sw * .24), int(sw * .17)), 0, 0, 360, (255, 255, 255), 2, AA)

        if armor_png is None:
            emblem(img, lerp(g["neck"], g["hip"], .33 if hero["armor"] else .4), sw * .2, hero["emblem"], hero, t)

        if hero["id"] == "astronaut":
            cv2.ellipse(img, ip(g["neck"] + [0, sw * .04]), (int(sw * .4), int(sw * .12)), 0, 0, 360, (235, 235, 240), int(sw * .07), AA)
            cv2.ellipse(img, ip(g["neck"] + [0, sw * .04]), (int(sw * .4), int(sw * .12)), 0, 0, 360, glow, 2, AA)

    hr = g["hr"]
    crown_png = load_hero_png(hero["id"], "crown")
    if crown_png is not None:
        overlay_rgba_sprite(img, crown_png, g["nose"][0], g["nose"][1] - hr * 1.75, hr * 1.8, hr * 1.1)
    elif hero["crown"]:   # корона/диадема парит НАД головой, лицо открыто
        b, wd = g["nose"] + [0, -hr * 1.55], hr * .75
        pts = [b + [-wd, 0], b + [-wd, -hr * .55], b + [-wd / 2, -hr * .2], b + [0, -hr * .75], b + [wd / 2, -hr * .2], b + [wd, -hr * .55], b + [wd, 0]]
        apoly(img, pts, acc, 1.0, (255, 255, 255), 2)
        for q in (pts[1], pts[3], pts[5]):
            cv2.circle(img, ip(q), max(3, int(hr * .1)), (255, 255, 255), -1, AA)
    if hero["halo"]:
        c = g["nose"] + [0, -hr * 1.75 + math.sin(t * 3) * 4]
        cv2.ellipse(img, ip(c), (int(hr * .85), int(hr * .22)), 0, 0, 360, hero["glow"], 6, AA)
        cv2.ellipse(img, ip(c), (int(hr * .85), int(hr * .22)), 0, 0, 360, (255, 255, 255), 2, AA)

    glove_png = load_hero_png(hero["id"], "glove")
    for w, w_vis in ((g["lw"], g.get("lw_vis", 1.0)), (g["rw"], g.get("rw_vis", 1.0))):
        if w_vis < 0.45:
            continue
        r = sw * .15 * (1 + .12 * math.sin(t * 8))
        if glove_png is not None:
            overlay_rgba_sprite(img, glove_png, w[0], w[1], r * 2.4, r * 2.4)
        elif not ai_active:
            cv2.circle(img, ip(w), int(r), glow, -1, AA)
            cv2.circle(img, ip(w), int(r), (255, 255, 255), 3, AA)
        else:
            cv2.circle(img, ip(w), int(r * 0.9), glow, 2, AA)
