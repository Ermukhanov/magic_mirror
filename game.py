"""
Модуль игрового процесса (game.py) для «Сиқырлы айна»:
- Распознавание базовых и специальных жестов 8 героев (полёт, луч, щит, бег/Тұлпар, лук Томирис, паутина, ледяная волна, хлопок Ледибаг, репульсоры, прыжок/невесомость, волшебная пыль)
- Монстры («Құбыжық»), звёзды («Жұлдыз») и тематические мини-боссы
- Комбо, множитель очков, шкала энергии и автоподбор сложности (3 уровня)
"""
import time
import math
import random
import os
import cv2
import numpy as np
import config
from config import W, H, HOLD_ACTION, AA
from tracking import ip
from heroes import HEROES, apoly, star_pts
from render import spawn_firework, update_and_draw_fireworks
from i18n import tr, txt, get_lang
from audio import get_audio

_fx_cache = {}
_fx_missing = set()


class FXClipManager:
    """
    Менеджер видеоэффектов атак (Этап 3, fx_clips: true):
    - Загружает кадры из assets/<hero_id>/fx_<action>.mp4.
    - Кэширует в памяти для мгновенного наложения через аддитивный бленд (0.05 мс).
    - При отсутствии файла безопасно возвращает None (фоллбэк на процедурные эффекты).
    """

    def __init__(self):
        self._cache = _fx_cache
        self._missing = _fx_missing

    def get_frame(self, hero_id, action, age_sec):
        if not bool(config.FX_CLIPS):
            return None
        key = (hero_id, action)
        if key in self._missing:
            return None
        if key not in self._cache:
            p = os.path.join(config.BASE_DIR, "assets", hero_id, f"fx_{action}.mp4")
            if not os.path.exists(p):
                self._missing.add(key)
                return None
            frames = []
            cap = cv2.VideoCapture(p)
            while cap.isOpened():
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                frames.append(fr)
            cap.release()
            if not frames:
                self._missing.add(key)
                return None
            self._cache[key] = frames

        frames = self._cache[key]
        idx = int(age_sec * 30.0) % len(frames)
        return frames[idx]


_FX_MGR = None


def get_fx_manager():
    global _FX_MGR
    if _FX_MGR is None:
        _FX_MGR = FXClipManager()
    return _FX_MGR


def overlay_fx_clip(out, fx_img, center_pt, scale_w, scale_h):
    """Накладывает кадр FX-клипа с аддитивным смешиванием (чёрный фон видео становится 100% прозрачным)."""
    if fx_img is None:
        return
    fh, fw = fx_img.shape[:2]
    tw, th = int(scale_w), int(scale_h)
    if tw <= 4 or th <= 4:
        return
    if tw != fw or th != fh:
        fx_resized = cv2.resize(fx_img, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        fx_resized = fx_img
    cx, cy = int(center_pt[0]), int(center_pt[1])
    x0, y0 = cx - tw // 2, cy - th // 2
    x1, y1 = x0 + tw, y0 + th
    H_img, W_img = out.shape[:2]
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(W_img, x1), min(H_img, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    fx_roi = fx_resized[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    out_roi = out[iy0:iy1, ix0:ix1]
    out[iy0:iy1, ix0:ix1] = cv2.add(out_roi, fx_roi)


BOSS_THEMES = {
    "alpamys":   dict(name_kk="Дәу Құбыжық", name_ru="Степной Великан", col=(50, 95, 220), eye=(0, 235, 255), r=78),
    "tomiris":   dict(name_kk="Алтын Айдаһар", name_ru="Золотой Дракон", col=(140, 50, 210), eye=(80, 255, 255), r=80),
    "spider":    dict(name_kk="Меха-Шаян", name_ru="Меха-Скорпион", col=(70, 180, 60), eye=(60, 60, 255), r=76),
    "elsa":      dict(name_kk="Мұз Алыбы", name_ru="Ледяной Голем", col=(245, 195, 90), eye=(255, 255, 200), r=82),
    "ladybug":   dict(name_kk="Түн Көбелегі", name_ru="Ночной Мотылёк", col=(175, 70, 145), eye=(80, 230, 255), r=74),
    "iron":      dict(name_kk="Мега-Робот", name_ru="Мега-Робот", col=(130, 130, 145), eye=(60, 90, 255), r=80),
    "astronaut": dict(name_kk="Ғарыш Метеориті", name_ru="Космический Босс", col=(90, 110, 180), eye=(120, 255, 180), r=82),
    "fairy":     dict(name_kk="Көлеңке Бұлты", name_ru="Тучка-Тень", col=(155, 110, 140), eye=(255, 220, 120), r=76),
}


def draw_cute_monster(img, cx, cy, r, col, frozen=False, t=0.0):
    cx, cy, r = int(cx), int(cy), int(r)
    bob = int(math.sin(t * 7.0 + cx * 0.05) * 4)
    cy += bob
    c_body = (245, 225, 145) if frozen else col
    cv2.circle(img, (cx - int(r * 0.65), cy - int(r * 0.65)), max(5, int(r * 0.28)), c_body, -1, AA)
    cv2.circle(img, (cx + int(r * 0.65), cy - int(r * 0.65)), max(5, int(r * 0.28)), c_body, -1, AA)
    cv2.circle(img, (cx, cy), r, c_body, -1, AA)
    cv2.circle(img, (cx, cy), r, (255, 255, 255), 3, AA)
    er = max(5, int(r * 0.28))
    for s in (-1, 1):
        ex, ey = cx + int(s * r * 0.34), cy - int(r * 0.12)
        cv2.circle(img, (ex, ey), er, (255, 255, 255), -1, AA)
        cv2.circle(img, (ex, ey), max(2, int(er * 0.52)), (25, 25, 35), -1, AA)
    cv2.ellipse(img, (cx, cy + int(r * 0.25)), (int(r * 0.35), int(r * 0.18)), 0, 0, 180, (255, 255, 255), 3, AA)
    if frozen:
        cv2.rectangle(img, (cx - r - 4, cy - r - 4), (cx + r + 4, cy + r + 4), (255, 245, 190), 2, AA)


def draw_mini_boss(img, boss, hero, t):
    if not boss or not boss.get("alive"):
        return
    cfg = BOSS_THEMES.get(hero["id"], BOSS_THEMES["alpamys"])
    cx, cy = int(boss["pos"][0]), int(boss["pos"][1] + math.sin(t * 4.2) * 10)
    r = cfg["r"]
    col = cfg["col"]
    pulse = int(8 * math.sin(t * 8.0))
    cv2.circle(img, (cx, cy), r + 14 + pulse, cfg["eye"], 3, AA)
    draw_cute_monster(img, cx, cy, r, col, frozen=False, t=t)
    pts = np.array([[cx - 42, cy - r + 6], [cx - 32, cy - r - 28], [cx, cy - r - 10],
                    [cx + 32, cy - r - 28], [cx + 42, cy - r + 6]], np.int32)
    cv2.fillPoly(img, [pts], (40, 215, 255), AA)
    cv2.polylines(img, [pts], True, (255, 255, 255), 2, AA)
    bw, bh = 180, 18
    bx0, by0 = cx - bw // 2, max(25, cy - r - 58)
    hp_ratio = max(0.0, boss["hp"] / max(1.0, boss["max_hp"]))
    cv2.rectangle(img, (bx0, by0), (bx0 + bw, by0 + bh), (30, 30, 45), -1, AA)
    cv2.rectangle(img, (bx0, by0), (bx0 + int(bw * hp_ratio), by0 + bh), (60, 240, 110) if hp_ratio > 0.35 else (60, 90, 255), -1, AA)
    cv2.rectangle(img, (bx0, by0), (bx0 + bw, by0 + bh), (255, 255, 255), 2, AA)
    bname = cfg["name_kk"] if get_lang() == "kk" else cfg["name_ru"]
    txt(img, bname, cx, by0 - 28, 24, (255, 240, 150), center=True)


def add_score_and_combo(S, base_pts, t, pos=None):
    if t - S.get("last_hit_t", -10.0) < 3.2:
        S["combo"] = S.get("combo", 0) + 1
    else:
        S["combo"] = 1
    S["last_hit_t"] = t
    mult = 3 if S["combo"] >= 8 else (2 if S["combo"] >= 4 else 1)
    S["mult"] = mult
    S["score"] += base_pts * mult
    S["energy"] = min(100.0, S.get("energy", 0.0) + 14.0)
    if pos is not None:
        S["pops"].append([list(pos), 0.0])


def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


def evaluate_pose_scores(p, S=None, dt=0.016):
    """
    Вычисляет плавную оценку точности выполнения каждого жеста (0.0 .. 1.0, т.е. 0%..100%).
    Использует относительные пороги от ширины плеч (sw) и высоты торса (th) с допуском,
    чтобы ребёнку 4-6 лет было легко попадать в позу и видеть шкалу прогресса («Қимыл дәлдігі: 60%»).
    """
    if p is None:
        return {k: 0.0 for k in ("fly", "beam", "shield", "tulpar", "arrows", "web", "ice_wave", "lucky_charm", "repulsor", "zero_g", "magic_dust")}

    P = p[:, :2]
    ls, rs, le, re, nose, lw, rw, lh, rh = P[11], P[12], P[13], P[14], P[0], P[15], P[16], P[23], P[24]
    lk, rk = P[25], P[26]
    sw = max(40.0, float(abs(ls[0] - rs[0])) + 1.0)
    th = max(55.0, float(np.linalg.norm((lh + rh) * 0.5 - (ls + rs) * 0.5)) + 1.0)
    shoulder_y = float((ls[1] + rs[1]) * 0.5)
    hip_y = float((lh[1] + rh[1]) * 0.5)

    # 1. Полёт (fly): обе руки поднимаются вверх к уровню плеч/головы
    l_up = _clamp01((shoulder_y + sw * 0.35 - float(lw[1])) / (sw * 0.85)) if p[15][2] > 0.35 else 0.0
    r_up = _clamp01((shoulder_y + sw * 0.35 - float(rw[1])) / (sw * 0.85)) if p[16][2] > 0.35 else 0.0
    sc_fly = min(l_up, r_up) * 0.75 + max(l_up, r_up) * 0.25

    # 2. Щит (shield) и Хлопок (lucky_charm): сведение рук перед грудью
    wrist_dist = float(np.linalg.norm(lw - rw))
    chest_ok_l = _clamp01(1.0 - abs(float(lw[1]) - (shoulder_y + th * 0.35)) / (th * 0.65))
    chest_ok_r = _clamp01(1.0 - abs(float(rw[1]) - (shoulder_y + th * 0.35)) / (th * 0.65))
    close_hands = _clamp01((sw * 1.25 - wrist_dist) / (sw * 0.72))
    sc_shield = close_hands * min(chest_ok_l, chest_ok_r)
    sc_lucky = _clamp01((sw * 1.15 - wrist_dist) / (sw * 0.68)) * max(chest_ok_l, chest_ok_r)

    # 3. Луч (beam): рука вытянута в сторону
    l_Horiz = _clamp01((abs(float(lw[0] - ls[0])) - sw * 0.35) / (sw * 0.62)) * _clamp01(1.0 - abs(float(lw[1] - ls[1])) / (sw * 0.85))
    r_Horiz = _clamp01((abs(float(rw[0] - rs[0])) - sw * 0.35) / (sw * 0.62)) * _clamp01(1.0 - abs(float(rw[1] - rs[1])) / (sw * 0.85))
    sc_beam = max(l_Horiz, r_Horiz)

    # 4. Стрелы Томирис (arrows): одна рука вытянута, другая у плеча/груди
    r_pull = _clamp01((sw * 1.25 - float(min(np.linalg.norm(rw - rs), np.linalg.norm(rw - ls)))) / (sw * 0.55))
    l_pull = _clamp01((sw * 1.25 - float(min(np.linalg.norm(lw - ls), np.linalg.norm(lw - rs)))) / (sw * 0.55))
    sc_arrows = max(min(r_Horiz, l_pull), min(l_Horiz, r_pull), sc_beam * 0.85)

    # 5. Паутина (web): вытянутая рука или приседание
    hip_knee_dist = float(((lk[1] + rk[1]) * 0.5) - hip_y)
    sc_squat = _clamp01((th * 0.78 - hip_knee_dist) / (th * 0.28)) if hip_knee_dist > 0 else 0.0
    sc_web = max(sc_beam, sc_squat)

    # 6. Ледяная волна Эльзы (ice_wave) и Репульсоры (repulsor): обе руки в стороны
    sc_ice = min(l_Horiz, r_Horiz) * 0.7 + max(l_Horiz, r_Horiz) * 0.3
    spread_w = _clamp01((wrist_dist - sw * 0.35) / (sw * 0.45))
    sc_repulsor = max(sc_ice, spread_w * min(chest_ok_l, chest_ok_r))

    # 7. Волшебная пыль Феи (magic_dust): взмах одной или двумя руками над головой
    sc_dust = max(l_up, r_up)

    # 8. Бег Тұлпар (tulpar) и Прыжок Космонавта (zero_g)
    run_e = float(S.get("run_energy", 0.0)) if S is not None else 0.0
    sc_tulpar = max(_clamp01(run_e / 0.38), sc_fly * 0.85)
    sc_zerog = max(_clamp01(run_e / 0.42), sc_fly * 0.92)

    raw_scores = {
        "fly": sc_fly,
        "beam": sc_beam,
        "shield": sc_shield,
        "tulpar": sc_tulpar,
        "arrows": sc_arrows,
        "web": sc_web,
        "ice_wave": sc_ice,
        "lucky_charm": sc_lucky,
        "repulsor": sc_repulsor,
        "zero_g": sc_zerog,
        "magic_dust": sc_dust,
    }

    # Сглаживание шкалы точности, чтобы проценты не прыгали резко
    if S is not None:
        sm = S.setdefault("pose_scores", {})
        for k, v in raw_scores.items():
            prev_v = sm.get(k, v)
            # Быстро растёт вверх, плавно убывает вниз (гистерезис)
            alpha = 0.38 if v >= prev_v else 0.18
            sm[k] = float(prev_v * (1.0 - alpha) + v * alpha)
        return dict(sm)
    return raw_scores


def detect_actions(p, S=None, dt=0.016):
    P = p[:, :2]
    ls, rs, le, re, nose, lw, rw, lh, rh = P[11], P[12], P[13], P[14], P[0], P[15], P[16], P[23], P[24]
    lk, rk = P[25], P[26]
    sw = abs(ls[0] - rs[0]) + 1.0
    th = float(np.linalg.norm((lh + rh) * 0.5 - (ls + rs) * 0.5)) + 1.0
    a = {"fly": False, "shield": False, "beams": [], "punch": [], "jump": False, "squat": False, "clap": False, "run": False, "special": set(), "scores": {}}

    if S is not None:
        neck_y = float((ls[1] + rs[1]) * 0.5)
        knee_diff = float(abs(lk[1] - rk[1]))
        prev_neck_y = S.get("prev_neck_y", neck_y)
        prev_kd = S.get("prev_kd", knee_diff)
        dy_neck = prev_neck_y - neck_y
        dkd = abs(knee_diff - prev_kd)
        S["prev_neck_y"] = neck_y
        S["prev_kd"] = knee_diff
        run_energy = S.get("run_energy", 0.0) * (1.0 - 2.2 * dt) + (abs(dy_neck) + dkd) * 0.08
        S["run_energy"] = min(2.5, run_energy)

    scores = evaluate_pose_scores(p, S, dt)
    a["scores"] = scores

    # Гистерезис включения/выключения поз (вход 0.68, удержание до 0.52), чтобы поза не мигала на границе
    hyst = S.setdefault("pose_hyst", set()) if S is not None else set()

    def check_hyst(key, enter_th=0.68, exit_th=0.52):
        sc = scores.get(key, 0.0)
        if key in hyst:
            if sc < exit_th:
                hyst.discard(key)
                return False
            return True
        else:
            if sc >= enter_th:
                hyst.add(key)
                return True
            return False

    if (lw[1] < nose[1] + sw * 0.12 and rw[1] < nose[1] + sw * 0.12 and p[15][2] > .40 and p[16][2] > .40) or check_hyst("fly", 0.70, 0.52):
        a["fly"] = True
    elif (np.linalg.norm(lw - rw) < sw * 0.72 and ls[1] - sw * 0.15 < lw[1] < lh[1] and ls[1] - sw * 0.15 < rw[1] < rh[1]) or check_hyst("shield", 0.72, 0.55):
        a["shield"] = True
    else:
        for s, w in ((ls, lw), (rs, rw)):
            if abs(w[1] - s[1]) < sw * 0.65 and abs(w[0] - s[0]) > sw * 0.92:
                a["beams"].append((w, (w - s) / (np.linalg.norm(w - s) + 1e-6)))
        if not a["beams"] and check_hyst("beam", 0.70, 0.52):
            # Выбираем более вытянутую руку для луча
            if abs(rw[0] - rs[0]) >= abs(lw[0] - ls[0]):
                a["beams"].append((rw, (rw - rs) / (np.linalg.norm(rw - rs) + 1e-6)))
            else:
                a["beams"].append((lw, (lw - ls) / (np.linalg.norm(lw - ls) + 1e-6)))

    wrist_dist = float(np.linalg.norm(lw - rw))
    if (wrist_dist < sw * 0.62 and min(lw[1], rw[1]) < lh[1]) or check_hyst("lucky_charm", 0.70, 0.52):
        a["clap"] = True

    for s, w in ((ls, lw), (rs, rw)):
        if abs(w[1] - s[1]) < sw * 0.52 and abs(w[0] - s[0]) > sw * 0.75:
            a["punch"].append(w)

    hip_knee_dist = float(((lk[1] + rk[1]) * 0.5) - ((lh[1] + rh[1]) * 0.5))
    if 0 < hip_knee_dist < th * 0.60:
        a["squat"] = True

    for sp_k in ("arrows", "web", "ice_wave", "lucky_charm", "repulsor", "magic_dust", "tulpar", "zero_g"):
        if check_hyst(sp_k, 0.66, 0.50):
            a["special"].add(sp_k)

    if S is not None and S.get("run_energy", 0.0) > 0.40:
        a["run"] = True
        a["special"].add("tulpar")
    if a["fly"]:
        a["jump"] = True
        a["special"].add("zero_g")

    return a


def draw_tulpar_horse(img, cx, cy, scale, col, t):
    gallop = math.sin(t * 14.0) * 14
    c = np.array([cx, cy + gallop * 0.4], np.float32)
    body = np.array([
        c + [-55 * scale, -10 * scale],
        c + [45 * scale, -18 * scale],
        c + [75 * scale, -62 * scale],
        c + [98 * scale, -52 * scale],
        c + [70 * scale, -15 * scale],
        c + [55 * scale, 22 * scale],
        c + [-50 * scale, 22 * scale],
    ], np.int32)
    apoly(img, body, col, 0.90, (255, 255, 255), 3)
    for lx0, ly0, dx, dy in (
        (-45, 20, -28 - gallop, 48),
        (-20, 20, 15 + gallop, 48),
        (35, 20, 35 + gallop, 45),
        (50, 20, -10 - gallop, 46),
    ):
        p0 = c + np.array([lx0 * scale, ly0 * scale])
        p1 = p0 + np.array([dx * scale, dy * scale])
        cv2.line(img, ip(p0), ip(p1), (255, 255, 255), max(3, int(6 * scale)), AA)
    for i in range(4):
        tx = int(c[0] - (60 + i * 18) * scale)
        ty = int(c[1] + (-5 + i * 6 + math.sin(t * 12 + i) * 8) * scale)
        cv2.circle(img, (tx, ty), max(3, int((8 - i) * scale)), (80, 235, 255), -1, AA)


def trigger_hero_special(out, p, g, hero, acts, t, dt, S):
    spec = hero.get("special", "")
    special_set = acts.get("special", set())

    last_sp = S.get("last_special_t", -10.0)
    if spec in special_set and (t - last_sp) > 1.8:
        S["last_special_t"] = t
        S["special_active_until"] = t + 2.6
        S["score"] += 3
        get_audio().play_sfx("special")
        get_audio().play_voice(f"spec_{hero['id']}")

        if spec == "ice_wave":
            S["freeze_until"] = t + 3.0
        elif spec == "zero_g":
            S["zerog_until"] = t + 3.2

        if spec == "tulpar":
            S["special_fx"].append({"kind": "tulpar", "x": -120.0, "y": float(H * 0.62), "t0": t})
        elif spec == "arrows":
            for ang_deg in (-22, -11, 0, 11, 22):
                rad = math.radians(ang_deg)
                S["special_fx"].append({
                    "kind": "arrow", "x": float(g["neck"][0]), "y": float(g["neck"][1]),
                    "vx": math.cos(rad) * 980.0, "vy": math.sin(rad) * 980.0 - 180.0, "t0": t
                })
        elif spec == "lucky_charm":
            for i in range(18):
                ang = i * (2 * math.pi / 18)
                S["special_fx"].append({
                    "kind": "ladybug", "x": float(g["neck"][0]), "y": float(g["neck"][1]),
                    "ang": ang, "r": 25.0, "t0": t
                })
        elif spec == "magic_dust":
            for s in S["stars"]:
                if s["alive"]:
                    s["alive"] = False
                    S["score"] += 5
                    S["special_fx"].append({"kind": "butterfly", "x": float(s["pos"][0]), "y": float(s["pos"][1]), "t0": t})

    is_sp_active = t < S.get("special_active_until", 0.0)

    if spec == "web" and ("web" in special_set) and S["stars"]:
        hand_w = g["rw"] if g["rw"][1] < g["lw"][1] else g["lw"]
        alive_stars = [s for s in S["stars"] if s["alive"]]
        if alive_stars:
            target = min(alive_stars, key=lambda s: math.hypot(s["pos"][0] - hand_w[0], s["pos"][1] - hand_w[1]))
            tp = np.array(target["pos"], np.float32)
            cv2.line(out, ip(hand_w), ip(tp), (245, 245, 255), 4, AA)
            cv2.circle(out, ip(tp), 36, (245, 245, 255), 2, AA)
            cv2.line(out, (int(tp[0] - 28), int(tp[1] - 28)), (int(tp[0] + 28), int(tp[1] + 28)), (245, 245, 255), 2, AA)
            cv2.line(out, (int(tp[0] + 28), int(tp[1] - 28)), (int(tp[0] - 28), int(tp[1] + 28)), (245, 245, 255), 2, AA)
            vec = hand_w - tp
            dist = float(np.linalg.norm(vec))
            if dist < 65:
                target["alive"] = False
                S["score"] += 6
                spawn_firework(S["fireworks"], tp[0], tp[1], count=22)
            else:
                target["pos"][0] += float(vec[0] / dist) * 650.0 * dt
                target["pos"][1] += float(vec[1] / dist) * 650.0 * dt

    if spec == "repulsor" and ("repulsor" in special_set or is_sp_active):
        for w_pt in (g["lw"], g["rw"]):
            cv2.circle(out, ip(w_pt), 38, (80, 230, 255), -1, AA)
            cv2.circle(out, ip(w_pt), 22, (255, 255, 255), -1, AA)
            end_pt = (int(w_pt[0] + (w_pt[0] - g["neck"][0]) * 3.5), int(w_pt[1] - 520))
            cv2.line(out, ip(w_pt), end_pt, (60, 210, 255), 26, AA)
            cv2.line(out, ip(w_pt), end_pt, (255, 255, 255), 10, AA)
        for s in S["stars"]:
            if s["alive"] and (abs(s["pos"][0] - g["lw"][0]) < 110 or abs(s["pos"][0] - g["rw"][0]) < 110):
                s["alive"] = False
                S["score"] += 5
                spawn_firework(S["fireworks"], s["pos"][0], s["pos"][1], count=20)

    if t < S.get("freeze_until", 0.0):
        r_wave = int(((3.0 - (S["freeze_until"] - t)) * 680) % 950)
        cv2.circle(out, ip(g["neck"]), r_wave, (255, 240, 170), 6, AA)
        cv2.circle(out, ip(g["neck"]), max(10, r_wave - 25), (255, 255, 255), 2, AA)

    if t < S.get("zerog_until", 0.0):
        for s in S["stars"]:
            if s["alive"]:
                cv2.circle(out, ip(s["pos"]), 52, (255, 235, 130), 3, AA)
                vec = g["neck"] - np.array(s["pos"], np.float32)
                d = float(np.linalg.norm(vec)) + 1e-5
                s["pos"][0] += float(vec[0] / d) * 240.0 * dt
                s["pos"][1] += float(vec[1] / d) * 240.0 * dt
                if d < 85:
                    s["alive"] = False
                    S["score"] += 5
                    spawn_firework(S["fireworks"], s["pos"][0], s["pos"][1], count=18)

    alive_fx = []
    for fx in S.get("special_fx", []):
        age = t - fx["t0"]
        fk = fx["kind"]
        if fk == "tulpar" and age < 2.2:
            fx["x"] += 780.0 * dt
            draw_tulpar_horse(out, fx["x"], fx["y"], 1.35, (40, 205, 255), t)
            for s in S["stars"]:
                if s["alive"] and abs(s["pos"][0] - fx["x"]) < 160:
                    s["alive"] = False
                    S["score"] += 5
                    spawn_firework(S["fireworks"], s["pos"][0], s["pos"][1], count=24)
            alive_fx.append(fx)
        elif fk == "arrow" and age < 1.4:
            fx["x"] += fx["vx"] * dt
            fx["y"] += fx["vy"] * dt
            p0 = (int(fx["x"]), int(fx["y"]))
            p1 = (int(fx["x"] - fx["vx"] * 0.045), int(fx["y"] - fx["vy"] * 0.045))
            cv2.arrowedLine(out, p1, p0, (60, 230, 255), 6, AA, tipLength=0.35)
            for s in S["stars"]:
                if s["alive"] and math.hypot(s["pos"][0] - fx["x"], s["pos"][1] - fx["y"]) < 75:
                    s["alive"] = False
                    S["score"] += 5
                    spawn_firework(S["fireworks"], s["pos"][0], s["pos"][1], count=18)
            alive_fx.append(fx)
        elif fk == "ladybug" and age < 2.0:
            fx["ang"] += 5.5 * dt
            fx["r"] += 290.0 * dt
            lx = int(fx["x"] + math.cos(fx["ang"]) * fx["r"])
            ly = int(fx["y"] + math.sin(fx["ang"]) * fx["r"])
            cv2.circle(out, (lx, ly), 14, (45, 45, 245), -1, AA)
            cv2.circle(out, (lx, ly), 14, (255, 255, 255), 2, AA)
            cv2.circle(out, (lx - 4, ly - 3), 3, (20, 20, 20), -1, AA)
            cv2.circle(out, (lx + 4, ly + 3), 3, (20, 20, 20), -1, AA)
            for s in S["stars"]:
                if s["alive"]:
                    s["alive"] = False
                    S["score"] += 5
                    spawn_firework(S["fireworks"], s["pos"][0], s["pos"][1], count=16)
            alive_fx.append(fx)
        elif fk == "butterfly" and age < 2.2:
            fx["y"] -= 140.0 * dt
            fx["x"] += math.sin(t * 8.0 + fx["t0"]) * 90.0 * dt
            bx, by = int(fx["x"]), int(fx["y"])
            w_flap = int(12 + 10 * abs(math.sin(t * 14.0)))
            cv2.ellipse(out, (bx - 10, by), (w_flap, 18), -20, 0, 360, (255, 160, 245), -1, AA)
            cv2.ellipse(out, (bx + 10, by), (w_flap, 18), 20, 0, 360, (255, 160, 245), -1, AA)
            cv2.ellipse(out, (bx - 10, by), (w_flap, 18), -20, 0, 360, (255, 255, 255), 2, AA)
            cv2.ellipse(out, (bx + 10, by), (w_flap, 18), 20, 0, 360, (255, 255, 255), 2, AA)
            alive_fx.append(fx)
    S["special_fx"] = alive_fx

    if is_sp_active:
        sp_age = max(0.0, 2.5 - (S.get("special_active_until", t) - t))
        fx_spec = get_fx_manager().get_frame(hero["id"], "special", sp_age)
        if fx_spec is not None:
            overlay_fx_clip(out, fx_spec, (int(g["neck"][0]), int(g["neck"][1] + g["th"] * 0.35)), g["sw"] * 2.8, g["sw"] * 2.8)
        txt(out, tr(f"spec_{hero['id']}"), W // 2, 132, 42, (80, 255, 140), center=True)


def damage_boss(S, dmg, t):
    boss = S.get("boss")
    if boss and boss.get("alive"):
        boss["hp"] -= dmg
        if boss["hp"] <= 0:
            boss["alive"] = False
            get_audio().play_sfx("victory")
            get_audio().play_voice("awesome", force=True)
            add_score_and_combo(S, 30, t, boss["pos"])
            spawn_firework(S["fireworks"], boss["pos"][0], boss["pos"][1], count=55)
            spawn_firework(S["fireworks"], boss["pos"][0] - 90, boss["pos"][1] - 40, count=40)
            spawn_firework(S["fireworks"], boss["pos"][0] + 90, boss["pos"][1] - 40, count=40)


def game_fx(out, p, g, hero, t, dt, S):
    acts = detect_actions(p, S, dt)
    names = [n for n, v in (("fly", acts["fly"]), ("shield", acts["shield"]), ("beam", bool(acts["beams"]))) if v]
    for n in list(S["act"]):
        if n not in names:
            del S["act"][n]
    for n in names:
        if n not in S["act"]:
            S["act"][n] = t
            # Мгновенная обратная связь (< 0.1 сек): вспышка и звук при начале любого жеста
            if t - S.get("last_act_sfx_t", 0.0) > 0.45:
                S["last_act_sfx_t"] = t
                get_audio().play_sfx("select")
                S["pops"].append([list(g["neck"]), 0.0])
    active = {n for n in names if t - S["act"][n] >= HOLD_ACTION}
    glow = hero["glow"]
    if "fly" in active:
        c = ip(g["hip"])
        fx_fly = get_fx_manager().get_frame(hero["id"], "fly", t - S["act"]["fly"])
        if fx_fly is not None:
            overlay_fx_clip(out, fx_fly, (c[0], c[1] + int(g["th"] * 0.45)), g["sw"] * 2.2, g["sw"] * 2.2)
        for i in range(3):
            r = (t * 320 + i * 110) % 330
            cv2.circle(out, c, int(r + 60), tuple(int(q * (1 - r / 330)) for q in glow), 4, AA)
        for i in range(16):
            x, y0 = (i * 83) % W, (t * 900 + i * 137) % H
            cv2.line(out, (x, int(y0)), (x, int(y0 + 90)), (255, 255, 255), 2, AA)
        txt(out, tr("fly_bang"), W // 2, 70, 76, glow, center=True)
        if t - S["last_fly"] > 0.5:
            add_score_and_combo(S, 1, t)
            S["last_fly"] = t
    if "shield" in active:
        c, R = g["hip"] * .4 + g["neck"] * .6, g["th"] * 1.35
        fx_shield = get_fx_manager().get_frame(hero["id"], "shield", t - S["act"]["shield"])
        if fx_shield is not None:
            overlay_fx_clip(out, fx_shield, c, R * 2.2, R * 2.2)
        hexa = [(c[0] + R * math.cos(math.pi / 3 * i + t), c[1] + R * math.sin(math.pi / 3 * i + t)) for i in range(6)]
        apoly(out, hexa, glow, 0.22, (255, 255, 255), 5)
        cv2.circle(out, ip(c), int(R * .85), glow, 3, AA)
        txt(out, tr("shield_bang"), W // 2, 70, 76, glow, center=True)
        for s in S["stars"]:
            if s["alive"] and math.hypot(s["pos"][0] - c[0], s["pos"][1] - c[1]) < R * 1.15:
                s["alive"] = False
                S["monsters_defeated"] = S.get("monsters_defeated", 0) + 1
                add_score_and_combo(S, 5, t, s["pos"])
                get_audio().play_sfx("hit")
                spawn_firework(S["fireworks"], s["pos"][0], s["pos"][1], count=18)

    for w_pt in (g["lw"], g["rw"]):
        for s in S["stars"]:
            if s["alive"] and math.hypot(s["pos"][0] - w_pt[0], s["pos"][1] - w_pt[1]) < 68:
                s["alive"] = False
                if s.get("kind") == "star":
                    S["stars_collected"] = S.get("stars_collected", 0) + 1
                else:
                    S["monsters_defeated"] = S.get("monsters_defeated", 0) + 1
                add_score_and_combo(S, 5, t, s["pos"])
                get_audio().play_sfx("hit")
                spawn_firework(S["fireworks"], s["pos"][0], s["pos"][1], count=20)

    if "beam" in active:
        txt(out, tr("beam_bang"), W // 2, 70, 76, glow, center=True)
        for w, d in acts["beams"]:
            fx_beam = get_fx_manager().get_frame(hero["id"], "beam", t - S["act"]["beam"])
            if fx_beam is not None:
                overlay_fx_clip(out, fx_beam, w + d * 160, g["sw"] * 1.9, g["sw"] * 1.9)
            e = w + d * 1700
            cv2.line(out, ip(w), ip(e), glow, 28, AA)
            cv2.line(out, ip(w), ip(e), (255, 255, 255), 9, AA)
            for s in S["stars"]:
                v = np.array(s["pos"]) - w
                if s["alive"] and np.dot(v, d) > 0 and abs(v[0] * d[1] - v[1] * d[0]) < 62:
                    s["alive"] = False
                    if s.get("kind") == "star":
                        S["stars_collected"] = S.get("stars_collected", 0) + 1
                    else:
                        S["monsters_defeated"] = S.get("monsters_defeated", 0) + 1
                    add_score_and_combo(S, 5, t, s["pos"])
                    get_audio().play_sfx("hit")
                    spawn_firework(S["fireworks"], s["pos"][0], s["pos"][1], count=20)
            boss = S.get("boss")
            if boss and boss.get("alive"):
                vb = np.array(boss["pos"]) - w
                if np.dot(vb, d) > 0 and abs(vb[0] * d[1] - vb[1] * d[0]) < 90:
                    damage_boss(S, 28.0 * dt, t)

    trigger_hero_special(out, p, g, hero, acts, t, dt, S)
    if t < S.get("special_active_until", 0.0):
        damage_boss(S, 38.0 * dt, t)


def update_stars(out, S, dt):
    now = time.perf_counter()
    frozen = now < S.get("freeze_until", 0.0)
    zerog = now < S.get("zerog_until", 0.0)
    hero = S.get("hero") or HEROES[0]

    score = S.get("score", 0)
    missed = S.get("missed", 0)
    if S.get("diff_mode", "auto") == "auto":
        if score > 90 and missed < 4:
            diff = 3
        elif score > 35 and missed < 6:
            diff = 2
        else:
            diff = 1
    else:
        diff = int(S.get("diff_level", 1))
    S["current_diff"] = diff
    spd_mult = 0.78 if diff == 1 else (1.0 if diff == 2 else 1.25)

    max_targets = 5 + diff
    if len(S["stars"]) < max_targets and random.random() < (2.4 + 0.5 * diff) * dt:
        is_star = random.random() < 0.38
        m_cols = [(75, 95, 245), (185, 80, 225), (70, 205, 125), (245, 145, 70)]
        S["stars"].append(dict(
            pos=[float(random.randint(85, W - 85)), -35.0],
            alive=True,
            vy=float(random.randint(105, 215)) * spd_mult,
            vx=float(random.choice([-45, -25, 0, 25, 45])) * spd_mult,
            kind="star" if is_star else "monster",
            col=random.choice(m_cols),
        ))

    for s in S["stars"]:
        if not frozen and not zerog:
            s["pos"][1] += s["vy"] * dt
            s["pos"][0] = float(np.clip(s["pos"][0] + s.get("vx", 0.0) * dt, 65, W - 65))
        if s["alive"]:
            if s.get("kind") == "star":
                col_s = (255, 235, 160) if frozen else (0, 230, 255)
                apoly(out, star_pts(s["pos"], 38, rot=now * 2.0), col_s, 1.0, (255, 255, 255), 2)
            else:
                draw_cute_monster(out, s["pos"][0], s["pos"][1], 36, s.get("col", (80, 95, 245)), frozen=frozen, t=now)

    rem = []
    for s in S["stars"]:
        if not s["alive"]:
            continue
        if s["pos"][1] >= H + 45:
            S["missed"] = S.get("missed", 0) + 1
        else:
            rem.append(s)
    S["stars"] = rem

    elapsed = now - S.get("start", now)
    if elapsed > config.GAME_TIME * 0.45 and S.get("boss") is None:
        S["boss"] = dict(
            pos=[float(W // 2), 145.0],
            vx=145.0 * spd_mult,
            hp=100.0,
            max_hp=100.0,
            alive=True,
        )
        get_audio().play_voice("mission_boss", force=True)
    boss = S.get("boss")
    if boss and boss.get("alive"):
        if not frozen:
            boss["pos"][0] += boss["vx"] * dt
            if boss["pos"][0] < 180 or boss["pos"][0] > W - 180:
                boss["vx"] *= -1
        draw_mini_boss(out, boss, hero, now)

    for pp in S["pops"]:
        pp[1] += dt
        cv2.circle(out, ip(pp[0]), int(30 + pp[1] * 300), (255, 255, 255), 4, AA)
    S["pops"] = [pp for pp in S["pops"] if pp[1] < 0.3]
    update_and_draw_fireworks(out, S["fireworks"], dt)
