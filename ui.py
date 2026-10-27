"""
Модуль пользовательского интерфейса (ui.py) для «Сиқырлы айна»:
- Экран выбора 8 героев (сетка 4x2) с живой анимацией дыхания/свечения
- Игровой HUD (очки, время, комбо, шкала энергии, голосовые/текстовые миссии)
- Финальный экран с золотой медалью, салютом и жест-кнопками
- Режим воспитателя по клавише T и локальная статистика (stats.json, без фото/видео)
- Индикатор освещённости и силуэт калибровки по росту
"""
import os
import time
import json
import math
import random
import logging
import cv2
import numpy as np
import config
from config import W, H, STATS_PATH, AA
from tracking import geom, ip, CURSOR, draw_hand_skeleton_feedback, draw_cursor_with_ring, update_height_calibration
from heroes import HEROES, apoly, star_pts, draw_back, draw_front, hero_display_name
from render import (
    stage_bg, draw_sky_parallax, compose, spawn_firework, update_and_draw_fireworks, get_stage_or_video_bg,
)
from tutorial import gesture_button
from i18n import tr, txt, get_lang
from audio import get_audio

DEBUG_POSE_PATH = os.path.join(config.BASE_DIR, "debug_pose.json")

CW, CH = 276, 240
_card_bg = {}
_card_anim_cache = {}


def load_stats():
    default_st = {"games_played": 0, "total_score": 0, "best_score": 0, "hero_counts": {}}
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            default_st.update(d)
        except Exception as e:
            logging.warning(f"Ошибка чтения stats.json: {e}")
    return default_st


def record_game_stat(hero_id, score):
    st = load_stats()
    st["games_played"] = int(st.get("games_played", 0)) + 1
    st["total_score"] = int(st.get("total_score", 0)) + int(score)
    st["best_score"] = max(int(st.get("best_score", 0)), int(score))
    hc = st.get("hero_counts", {})
    hc[hero_id] = int(hc.get(hero_id, 0)) + 1
    st["hero_counts"] = hc
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Ошибка записи stats.json: {e}")
    return st


def card_rect(i):
    r, c = divmod(i, 4)
    return 44 + c * (CW + 28), 108 + r * (CH + 36), CW, CH


def fake_pose(cx, cy, s, t=0.0):
    p = np.zeros((33, 3), np.float32)
    p[:, 2] = 1
    breathe = math.sin(t * 3.2) * 0.04
    turn = math.sin(t * 2.0) * 0.07
    arm_wave = math.sin(t * 3.8) * 0.12
    pts = {
        0: (0, -1.1 + breathe),
        7: (-.17, -1.1 + breathe),
        8: (.17, -1.1 + breathe),
        11: (-.35 + turn, -.65 + breathe),
        12: (.35 + turn, -.65 + breathe),
        13: (-.50, -.20 - arm_wave),
        14: (.50, -.20 + arm_wave),
        15: (-.58, .12 - arm_wave * 1.4),
        16: (.58, .12 + arm_wave * 1.4),
        23: (-.20, .10),
        24: (.20, .10),
        25: (-.20, .55),
        26: (.20, .55),
        27: (-.20, 1.0),
        28: (.20, 1.0),
    }
    for k, (x, y) in pts.items():
        p[k, 0], p[k, 1] = cx + x * s, cy + y * s
    return p


def card_canvas(hero, t):
    k = hero["id"]
    h_off = hash(k) % 8
    tick = int(t * 14.0 + h_off * 0.125)
    cached = _card_anim_cache.get(k)
    if cached is not None and cached[0] == tick:
        return cached[1]
    if k not in _card_bg:
        bg = cv2.resize(stage_bg(hero), (CW, CH), interpolation=cv2.INTER_AREA)
        bg = (bg * 0.72).astype(np.uint8)
        cv2.ellipse(bg, (CW // 2, 184), (int(CW * .36), 16), 0, 0, 360, tuple(int(c * .7) for c in hero["glow"]), -1, AA)
        _card_bg[k] = bg
    cv = _card_bg[k].copy()
    p = fake_pose(CW // 2, 110, 62, t + h_off)
    g = geom(p)
    draw_back(cv, g, hero, t)
    P, col = p[:, :2], (145, 138, 158)
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)):
        cv2.line(cv, ip(P[a]), ip(P[b]), col, int(g["sw"] * .22), AA)
    cv2.fillPoly(cv, [np.array([P[11], P[12], P[24], P[23]], np.int32)], col, AA)
    cv2.circle(cv, ip(P[0]), int(g["hr"]), (215, 200, 195), -1, AA)
    cv2.circle(cv, ip(P[0]), int(g["hr"]), hero["glow"], 2, AA)
    draw_front(cv, g, hero, t)
    _card_anim_cache[k] = (tick, cv)
    return cv


def draw_come_closer_overlay(out, p, t):
    """
    Если ребёнок не виден или стоит слишком далеко (ширина плеч < 48 px),
    показывает крупную анимацию «Камераға жақындай түс! / Кадрға тұр!» со стрелками и контуром тела.
    """
    too_far = False
    if p is not None:
        sw = float(np.linalg.norm(p[11][:2] - p[12][:2]))
        if sw < 48.0:
            too_far = True
    if p is not None and not too_far:
        return

    cx, cy = W // 2, H - 150
    pulse = int(8 * math.sin(t * 6.0))
    col = (0, 235, 255)

    # Полупрозрачная плашка и контур силуэта, куда встать
    apoly(out, [(W // 2 - 440, H - 92), (W // 2 + 440, H - 92), (W // 2 + 440, H - 16), (W // 2 - 440, H - 16)],
          (18, 26, 44), 0.85, col, 3)
    msg = tr("come_closer") if too_far else tr("stand_in_frame")
    txt(out, msg, W // 2, H - 76, 38, (0, 255, 255), center=True)

    # Светящийся круг-платформа на полу и пульсирующие стрелки к центру
    cv2.ellipse(out, (cx, H - 102), (115 + pulse, 24), 0, 0, 360, col, 3, AA)
    cv2.circle(out, (cx, cy - 110), 28 + pulse // 2, col, 2, AA)
    cv2.rectangle(out, (cx - 48, cy - 75), (cx + 48, cy + 20), col, 2, AA)

    arr_off = int(14 * math.sin(t * 7.0))
    cv2.arrowedLine(out, (cx - 210 - arr_off, H - 115), (cx - 135, H - 115), col, 6, AA, tipLength=0.4)
    cv2.arrowedLine(out, (cx + 210 + arr_off, H - 115), (cx + 135, H - 115), col, 6, AA, tipLength=0.4)


def render_select(frame, hand, hover, prog, t, p=None):
    base_sky = stage_bg(HEROES[hover if hover >= 0 else 0])
    draw_sky_parallax(base_sky, HEROES[hover if hover >= 0 else 0], t, is_flying=False)
    out = cv2.addWeighted(base_sky, 0.40, frame, 0.65, 12)

    # Приветствие и заголовок на казахском
    txt(out, f"{tr('hello')} {tr('choose')}", W // 2, 18, 52, (255, 255, 255), center=True)

    # Сначала рисуем обычные карточки, а наведённую карточку рисуем увеличенной и светящейся поверх
    for i, hero in enumerate(HEROES):
        if i == hover:
            continue
        x, y, w, h = card_rect(i)
        out[y:y + h, x:x + w] = card_canvas(hero, t)
        cv2.rectangle(out, (x, y), (x + w, y + h), (155, 165, 180), 2, AA)
        txt(out, hero_display_name(hero), x + w // 2, y + h - 50, 30, (255, 255, 255), center=True)

    if 0 <= hover < len(HEROES):
        hero = HEROES[hover]
        x, y, w, h = card_rect(hover)
        # Карточка при наведении увеличивается (~1.08x) и ярко светится
        pad_x, pad_y = 11, 10
        nx0, ny0 = max(0, x - pad_x), max(0, y - pad_y)
        nx1, ny1 = min(W, x + w + pad_x), min(H, y + h + pad_y)
        nw, nh = nx1 - nx0, ny1 - ny0
        card_img = cv2.resize(card_canvas(hero, t), (nw, nh), interpolation=cv2.INTER_LINEAR)
        out[ny0:ny1, nx0:nx1] = card_img
        glow = hero["glow"]
        cv2.rectangle(out, (nx0 - 3, ny0 - 3), (nx1 + 3, ny1 + 3), (255, 255, 255), 10, AA)
        cv2.rectangle(out, (nx0, ny0), (nx1, ny1), glow, 6, AA)
        if prog > 0.01:
            cv2.rectangle(out, (nx0, ny1 - 12), (nx0 + int(nw * prog), ny1), (80, 255, 120), -1)
        txt(out, hero_display_name(hero), nx0 + nw // 2, ny1 - 54, 34, (255, 255, 255), center=True)

    # Светящийся неоновый скелет рук и плеч ребёнка (обратная связь: система видит ребёнка!)
    if p is not None:
        draw_hand_skeleton_feedback(out, p, active_side=CURSOR.active_side, thin=False, t=t)

    # Если ребёнок виден — подсказка удерживать руку; иначе анимация «Подойди ближе / Встань в кадр»
    if p is not None and float(np.linalg.norm(p[11][:2] - p[12][:2])) >= 48.0:
        txt(out, tr("hold_hero"), W // 2, H - 52, 32, (80, 255, 160), center=True)
    else:
        draw_come_closer_overlay(out, p, t)

    # Кольцо прогресса рисуется ПРЯМО ВОКРУГ КУРСОРА руки (задержка 1.0 сек)
    if hand is not None:
        draw_cursor_with_ring(out, hand, prog=prog, t=t)
        if prog > 0.02:
            cnt = max(1, int(math.ceil((1.0 - prog) * 3)))
            cx, cy = ip(hand)
            txt(out, str(cnt), cx, max(20, cy - 68), 40, (80, 255, 120), center=True)

    return out


def render_calib(frame, p, mask_s, hero, t, dt, parts, S):
    """
    Экран 3-секундной калибровки в силуэте после выбора героя:
    Ребёнок встаёт в светящийся силуэт в костюме выбранного героя, система калибрует
    рост и ширину плеч (для зоны досягаемости и порогов жестов) и переходит к обучению.
    """
    if p is not None and mask_s is not None:
        out, g = compose(frame, p, mask_s, hero, t, dt, parts, is_flying=False)
        draw_hand_skeleton_feedback(out, p, active_side=CURSOR.active_side, thin=False, t=t)
    else:
        out = get_stage_or_video_bg(hero, t, is_flying=False)
        g = None

    cx, cy = W // 2, int(H * 0.54)
    sw_ok = (g is not None and g["sw"] >= 42.0)

    if sw_ok:
        update_height_calibration(frame, p, S, dt)
        S["calib_prog"] = min(1.0, S.get("calib_prog", 0.0) + dt / 2.6)
        col = (80, 255, 120)
    else:
        # Даже если ребёнок стоит далеко, после 3.5 сек плавно продвигаем, чтобы не блокировать игру
        elapsed_c = t - S.get("calib_t0", t)
        if elapsed_c > 3.5 and p is not None:
            S["calib_prog"] = min(1.0, S.get("calib_prog", 0.0) + dt / 2.5)
        else:
            S["calib_prog"] = max(0.0, S.get("calib_prog", 0.0) - dt * 0.4)
        col = (0, 230, 255)

    prog = float(S.get("calib_prog", 0.0))
    pulse = int(6 * math.sin(t * 6.0))

    # Контур силуэта героя в центре экрана
    cv2.circle(out, (cx, cy - 155), 52 + pulse // 2, col, 4, AA)
    body_pts = np.array([
        [cx - 95, cy - 90], [cx + 95, cy - 90],
        [cx + 135, cy + 20], [cx + 80, cy + 25],
        [cx + 75, cy + 165], [cx - 75, cy + 165],
        [cx - 80, cy + 25], [cx - 135, cy + 20],
    ], np.int32)
    cv2.polylines(out, [body_pts], True, (255, 255, 255), 6, AA)
    cv2.polylines(out, [body_pts], True, col, 3, AA)
    cv2.ellipse(out, (cx, cy + 175), (125 + pulse, 26), 0, 0, 360, col, 4, AA)

    # Кольцо обратного отсчёта 3-2-1
    r_ring = 58
    rcx, rcy = cx, cy + 15
    cv2.circle(out, (rcx, rcy), r_ring, (25, 35, 52), -1, AA)
    cv2.circle(out, (rcx, rcy), r_ring, (255, 255, 255), 3, AA)
    if prog > 0.01:
        cv2.ellipse(out, (rcx, rcy), (r_ring, r_ring), -90, 0, int(360 * prog), (80, 255, 120), 10, AA)
    cnt = max(1, int(math.ceil((1.0 - prog) * 3)))
    txt(out, str(cnt), rcx, rcy - 30, 62, (255, 255, 255), center=True)

    apoly(out, [(W // 2 - 420, 20), (W // 2 + 420, 20), (W // 2 + 420, 96), (W // 2 - 420, 96)],
          (20, 28, 44), 0.80, col, 3)
    txt(out, tr("calib_title"), W // 2, 36, 44, (255, 255, 255), center=True)

    if not sw_ok:
        draw_come_closer_overlay(out, p, t)

    if prog >= 1.0:
        get_audio().play_sfx("victory")
        get_audio().play_voice("calib_done", force=True)
        spawn_firework(S["fireworks"], W // 2, H // 3, count=42)
        S.update(
            state="tutorial",
            tut_step=0,
            tut_step_t=t,
            tut_prog=0.0,
            tut_voiced_step=-1,
            ready_voiced=False,
            tut_is_doing=False,
        )

    update_and_draw_fireworks(out, S["fireworks"], dt)
    return out


def render_end(hero, score, t, dt, p, S, btn_state):
    out = get_stage_or_video_bg(hero, t, is_flying=False)
    if random.random() < 2.2 * dt:
        spawn_firework(S["fireworks"], random.randint(200, W - 200), random.randint(120, H // 2), count=38)
    update_and_draw_fireworks(out, S["fireworks"], dt)

    c = (W // 2, 300)
    for r, col in ((180, tuple(int(q * .4) for q in hero["glow"])), (140, (0, 190, 255))):
        cv2.circle(out, c, r, col, -1, AA)
    cv2.circle(out, c, 140, (255, 255, 255), 6, AA)
    apoly(out, star_pts(c, 110, math.sin(t) * .15 - math.pi / 2), (0, 230, 255), 1.0, (255, 255, 255), 4)
    txt(out, tr("victory"), W // 2, 35, 58, (80, 255, 120), center=True)
    txt(out, f"{hero_display_name(hero)} — {tr('medal_praise')}", W // 2, 102, 40, (255, 255, 255), center=True)
    txt(out, str(score), W // 2, 255, 92, (30, 20, 20), center=True)

    if t - S["end_t"] > 1.5:
        if gesture_button(out, p, btn_state, "btn_again", W // 2 - 220, 560, 74, tr("play_again"), t):
            S["state"], S["hover"] = "select", -1
        if gesture_button(out, p, btn_state, "btn_next", W // 2 + 220, 560, 74, tr("next_level"), t, color=(255, 190, 40)):
            S.update(state="play", score=0, start=t, stars=[], pops=[], special_fx=[], act={},
                     boss=None, combo=0, mult=1, energy=0.0, stars_collected=0, monsters_defeated=0, missed=0)
    return out


def draw_hud(out, S, left_sec, t):
    txt(out, f"{tr('score')}: {S['score']}", 30, 18, 46, (0, 255, 255))
    txt(out, f"{tr('time')}: {int(max(left_sec, 0))} {tr('sec')}", W - 340, 18, 46)

    combo = S.get("combo", 0)
    mult = S.get("mult", 1)
    if combo >= 2 and (t - S.get("last_hit_t", -10.0)) < 3.2:
        txt(out, f"{tr('combo')} x{mult} ({combo})", 30, 74, 34, (80, 255, 130))

    energy = S.get("energy", 0.0)
    ex0, ey0, ew, eh = 30, H - 96, 220, 22
    cv2.rectangle(out, (ex0, ey0), (ex0 + ew, ey0 + eh), (25, 35, 50), -1, AA)
    e_col = (60, 255, 140) if energy >= 80 else (0, 215, 255)
    cv2.rectangle(out, (ex0, ey0), (ex0 + int(ew * min(1.0, energy / 100.0)), ey0 + eh), e_col, -1, AA)
    cv2.rectangle(out, (ex0, ey0), (ex0 + ew, ey0 + eh), (255, 255, 255), 2, AA)
    txt(out, f"{tr('energy')}: {int(energy)}%", ex0, ey0 - 30, 24, (255, 255, 255))

    boss = S.get("boss")
    if boss and boss.get("alive"):
        m_text = tr("mission_boss")
        m_col = (80, 230, 255)
    elif S.get("stars_collected", 0) < 5:
        m_text = f"{tr('mission_stars')} ({S.get('stars_collected', 0)}/5)"
        m_col = (120, 255, 160)
    else:
        m_text = f"{tr('mission_monsters')} ({S.get('monsters_defeated', 0)})"
        m_col = (255, 235, 120)
    txt(out, m_text, W // 2, 22, 34, m_col, center=True)
    txt(out, tr("hint"), W // 2, H - 48, 26, (255, 255, 255), center=True)


def check_lighting_and_warn(out, frame, S, t):
    if t - S.get("light_check_t", 0.0) > 1.5:
        S["light_check_t"] = t
        mean_luma = float(np.mean(frame[::16, ::16]))
        S["low_light"] = (mean_luma < 38.0)
    if S.get("low_light", False):
        apoly(out, [(W // 2 - 280, 68), (W // 2 + 280, 68), (W // 2 + 280, 114), (W // 2 - 280, 114)],
              (20, 30, 160), 0.85, (0, 235, 255), 2)
        txt(out, tr("low_light"), W // 2, 74, 28, (0, 255, 255), center=True)


def draw_calibration_silhouette(out, p, t):
    cx, cy = W // 2, H - 105
    pulse = int(6 * math.sin(t * 5.0))
    cv2.ellipse(out, (cx, cy + 45), (95 + pulse, 22), 0, 0, 360, (0, 255, 255), 3, AA)


def draw_teacher_panel(out, S):
    if not S.get("teacher_mode", False):
        return
    x0, y0, x1, y1 = 120, 48, W - 120, H - 48
    roi = out[y0:y1, x0:x1]
    ov = np.full_like(roi, (28, 34, 48))
    out[y0:y1, x0:x1] = cv2.addWeighted(ov, 0.92, roi, 0.08, 0)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 230, 255), 4, AA)

    title_s = "ТӘРБИЕШІ ПАНЕЛІ / ПАНЕЛЬ ВОСПИТАТЕЛЯ [T — жабу/закрыть]"
    txt(out, title_s, W // 2, y0 + 14, 28, (0, 255, 255), center=True)

    diff_str = "АВТО" if S.get("diff_mode", "auto") == "auto" else f"Деңгей / Уровень {S.get('diff_level', 1)}"
    paused_str = "ИӘ / ДА" if S.get("paused", False) else "ЖОҚ / НЕТ"
    vol_pct = int(get_audio().volume * 100) if not get_audio().muted else 0

    ai_str = str(config.AI_TRANSFORM).upper()
    ai_live_info = ""
    try:
        from ai_transform import get_ai_manager
        mgr = get_ai_manager()
        if mgr.ready and mgr.ai_fps > 0:
            ai_live_info = f" [{mgr.ai_fps:.1f} FPS | {mgr.vram_gb:.2f} ГБ VRAM]"
        elif mgr.ready:
            ai_live_info = " [ГОТОВ / ДАЙЫН]"
        elif not mgr.failed and ai_str != "OFF":
            ai_live_info = " [ЖҮКТЕЛУДЕ / ЗАГРУЗКА...]"
    except Exception:
        pass
    vw_str = "ҚОСУЛЫ / ВКЛ" if config.VIDEO_WORLDS else "ӨШІРУЛІ / ВЫКЛ"
    fx_str = "ҚОСУЛЫ / ВКЛ" if config.FX_CLIPS else "ӨШІРУЛІ / ВЫКЛ"
    ha_str = "ҚОСУЛЫ / ВКЛ" if config.HERO_ART else "ӨШІРУЛІ / ВЫКЛ"
    gl_str = "ҚОСУЛЫ / ВКЛ" if config.GAME_LAYER else "ӨШІРУЛІ / ВЫКЛ"

    lines = [
        "[1..8] Кейіпкерді бірден таңдау (Выбрать героя): 1-Алпамыс .. 8-Пері",
        f"[D] Қиындық (Сложность): {diff_str}   |   [G] Ойын уақыты (Время): {config.GAME_TIME} сек   |   [L] Тіл: {'KK' if get_lang() == 'kk' else 'RU'}",
        f"[P] Пауза: {paused_str}   |   [S] Жаттығуды өткізу (Пропуск)   |   [R] Сброс   |   [+ / - / M] Дыбыс: {vol_pct}%",
        f"[A] ИИ-превращение (ai_transform): {ai_str}{ai_live_info} (auto / on / off)",
        f"[W] Видео-әлемдер (video_worlds): {vw_str}   |   [X] FX-клиптер (fx_clips): {fx_str}",
        f"[H] Кейіпкер графикасы / PNG-ассеты героев (hero_art): {ha_str}",
        f"[Y] Ойын қабаты / Игровой слой (game_layer — звёзды/монстры/босс): {gl_str}",
    ]
    for idx_l, ln in enumerate(lines):
        col_ln = (80, 255, 160) if idx_l >= 3 else (255, 255, 255)
        txt(out, ln, x0 + 32, y0 + 64 + idx_l * 44, 22, col_ln)

    st = S.get("stats_cache") or load_stats()
    gp = int(st.get("games_played", 0))
    avg_sc = int(round(st.get("total_score", 0) / max(1, gp))) if gp > 0 else 0
    best_sc = int(st.get("best_score", 0))
    hc = st.get("hero_counts", {})
    fav_hero = max(hc, key=hc.get) if hc else "—"

    cv2.line(out, (x0 + 30, y0 + 382), (x1 - 30, y0 + 382), (100, 160, 200), 2, AA)
    stat_title = "ЖЕРГІЛІКТІ СТАТИСТИКА (ЛОКАЛЬНАЯ СТАТИСТИКА — ОФЛАЙН, БЕЗ ВИДЕО):"
    txt(out, stat_title, x0 + 32, y0 + 396, 22, (80, 255, 140))
    stat_line = f"Ойнаған балалар (Сыграло детей): {gp}   |   Орташа ұпай (Средний балл): {avg_sc}   |   Рекорд: {best_sc}   |   Топ кейіпкер: {fav_hero}"
    txt(out, stat_line, x0 + 32, y0 + 436, 21, (235, 245, 255))


def record_debug_pose(p_raw, S):
    """
    Сохраняет ТОЛЬКО числовые координаты скелета (33x3) и метрики в debug_pose.json
    (без видео и без фотографий), чтобы запись можно было прогнать через tools/replay.py.
    """
    if p_raw is None:
        return 0
    records = []
    if os.path.exists(DEBUG_POSE_PATH):
        try:
            with open(DEBUG_POSE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "frames" in data:
                    records = data["frames"]
                elif isinstance(data, list):
                    records = data
        except Exception:
            records = []

    ls, rs = p_raw[11, :2], p_raw[12, :2]
    lh, rh = p_raw[23, :2], p_raw[24, :2]
    sw = float(np.linalg.norm(ls - rs))
    th = float(np.linalg.norm((lh + rh) * 0.5 - (ls + rs) * 0.5))

    entry = {
        "ts": round(time.time(), 3),
        "state": S.get("state", "select"),
        "active_hand": CURSOR.active_side,
        "sw": round(sw, 2),
        "th": round(th, 2),
        "calib_sw": round(float(S.get("calib_sw", sw)), 2),
        "calib_th": round(float(S.get("calib_th", th)), 2),
        "reach_box": list(CURSOR.reach_box),
        "cursor_raw": [round(float(x), 2) for x in CURSOR.raw_cursor] if CURSOR.raw_cursor is not None else None,
        "cursor_mapped": [round(float(x), 2) for x in CURSOR.mapped_unfiltered] if CURSOR.mapped_unfiltered is not None else None,
        "cursor_smooth": [round(float(x), 2) for x in CURSOR.smoothed_cursor] if CURSOR.smoothed_cursor is not None else None,
        "cursor_final": [round(float(x), 2) for x in CURSOR.final_cursor] if CURSOR.final_cursor is not None else None,
        "pose_scores": {k: round(float(v), 3) for k, v in S.get("pose_scores", {}).items()},
        "landmarks": [[round(float(c), 2) for c in pt] for pt in p_raw.tolist()],
    }
    records.append(entry)
    # Ограничиваем 500 записями, чтобы файл оставался компактным
    if len(records) > 500:
        records = records[-500:]
    try:
        with open(DEBUG_POSE_PATH, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "resolution": [W, H], "frames": records}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Ошибка записи debug_pose.json: {e}")
    S["debug_saved_count"] = len(records)
    S["debug_rec_flash_t"] = time.perf_counter()
    return len(records)


def draw_debug_overlay(out, p_raw, p_lead, S, fps, pose_latency_ms, magnet_targets, t, btn_state=None):
    """
    Режим отладки для разработчика (клавиша B):
    - Полный скелет (сырой p_raw и с предсказанием p_lead)
    - Hitbox карточек и кнопок (видимый и увеличенный на +25%)
    - Зона досягаемости рук (reach_box вокруг плеч ребёнка)
    - Позиция курсора ДО и ПОСЛЕ маппинга/сглаживания/магнита
    - Значения порогов жестов в реальном времени (0%..100%)
    - FPS и задержка позы (pose_latency_ms)
    - Запись координат в debug_pose.json по клавише C или экранной кнопке REC POSE
    """
    if not S.get("debug_mode", False):
        return

    # 1. Зона досягаемости (Reach Zone)
    rx0, ry0, rx1, ry1 = CURSOR.reach_box
    cv2.rectangle(out, (rx0, ry0), (rx1, ry1), (255, 80, 255), 2, AA)
    cv2.putText(out, f"REACH ZONE (+-1.6 sw): {rx1-rx0}x{ry1-ry0}px", (max(10, rx0 + 6), max(24, ry0 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 120, 255), 2, AA)

    # 2. Hitbox карточек и кнопок (+25%)
    if magnet_targets:
        for tgt in magnet_targets:
            if "rect" in tgt:
                x, y, w, h = tgt["rect"]
                tcx, tcy = x + w * 0.5, y + h * 0.5
                hw, hh = w * 0.5 * 1.25, h * 0.5 * 1.25
                cv2.rectangle(out, (int(tcx - hw), int(tcy - hh)), (int(tcx + hw), int(tcy + hh)), (60, 255, 120), 1, AA)
            elif "circle" in tgt:
                tcx, tcy, r = tgt["circle"]
                cv2.circle(out, (int(tcx), int(tcy)), int(r * 1.25), (60, 255, 120), 1, AA)

    # 3. Полный скелет (оранжевый = без lead, голубой = с lead)
    bones_all = (
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (15, 19), (15, 17),
        (12, 14), (14, 16), (16, 20), (16, 18),
        (23, 25), (25, 27), (24, 26), (26, 28),
    )
    if p_raw is not None:
        for a_i, b_i in bones_all:
            if p_raw[a_i, 2] > 0.3 and p_raw[b_i, 2] > 0.3:
                cv2.line(out, ip(p_raw[a_i, :2]), ip(p_raw[b_i, :2]), (0, 165, 255), 2, AA)
        for idx_pt in (0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26):
            if p_raw[idx_pt, 2] > 0.3:
                pt_i = ip(p_raw[idx_pt, :2])
                cv2.circle(out, pt_i, 4, (0, 165, 255), -1, AA)
                cv2.putText(out, str(idx_pt), (pt_i[0] + 5, pt_i[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, AA)

    if p_lead is not None:
        for a_i, b_i in bones_all:
            if p_lead[a_i, 2] > 0.3 and p_lead[b_i, 2] > 0.3:
                cv2.line(out, ip(p_lead[a_i, :2]), ip(p_lead[b_i, :2]), (255, 255, 0), 1, AA)

    # 4. Курсор ДО и ПОСЛЕ сглаживания/маппинга
    if CURSOR.raw_cursor is not None:
        rc = ip(CURSOR.raw_cursor)
        cv2.circle(out, rc, 7, (60, 60, 255), -1, AA)
        cv2.putText(out, "RAW_TIP", (rc[0] + 8, rc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 255), 1, AA)
    if CURSOR.mapped_unfiltered is not None:
        mc = ip(CURSOR.mapped_unfiltered)
        cv2.circle(out, mc, 8, (0, 230, 255), 2, AA)
        cv2.putText(out, "PRE_SMOOTH", (mc[0] + 10, mc[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, AA)
    if CURSOR.smoothed_cursor is not None and CURSOR.final_cursor is not None:
        sc = ip(CURSOR.smoothed_cursor)
        fc = ip(CURSOR.final_cursor)
        cv2.circle(out, sc, 10, (255, 200, 0), 2, AA)
        if CURSOR.mapped_unfiltered is not None:
            cv2.line(out, ip(CURSOR.mapped_unfiltered), fc, (0, 255, 180), 2, AA)
        cv2.putText(out, f"FINAL({CURSOR.active_side})", (fc[0] + 14, fc[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 255, 120), 2, AA)

    # 5. Панель телеметрии и порогов жестов в реальном времени
    px0, py0, pw, ph = 14, 14, 370, 345
    roi = out[py0:py0 + ph, px0:px0 + pw]
    ov = np.full_like(roi, (18, 22, 32))
    out[py0:py0 + ph, px0:px0 + pw] = cv2.addWeighted(ov, 0.84, roi, 0.16, 0)
    cv2.rectangle(out, (px0, py0), (px0 + pw, py0 + ph), (0, 255, 180), 2, AA)

    sw_v = float(S.get("calib_sw", 0.0))
    th_v = float(S.get("calib_th", 0.0))
    saved_n = int(S.get("debug_saved_count", 0))
    header_lines = [
        f"DEBUG MODE [B] | FPS: {fps:.1f} | Latency: {pose_latency_ms:.1f} ms",
        f"Hand: {CURSOR.active_side} | Calib SW: {sw_v:.0f}px TH: {th_v:.0f}px",
        f"[C] Record Pose -> debug_pose.json (Saved: {saved_n})",
    ]
    for i_l, hl in enumerate(header_lines):
        col_h = (80, 255, 140) if i_l == 0 else ((0, 235, 255) if i_l == 2 else (230, 230, 240))
        cv2.putText(out, hl, (px0 + 10, py0 + 22 + i_l * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_h, 1, AA)

    scores = S.get("pose_scores", {})
    hyst = S.get("pose_hyst", set())
    y_bar = py0 + 84
    for gk in ("fly", "beam", "shield", "arrows", "web", "ice_wave", "lucky_charm", "repulsor", "tulpar", "zero_g", "magic_dust"):
        val = float(scores.get(gk, 0.0))
        is_on = gk in hyst
        cv2.putText(out, f"{gk:11s}: {int(val*100):3d}%", (px0 + 10, y_bar + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 255, 120) if is_on else (210, 215, 225), 1, AA)
        bx = px0 + 145
        bw_g = 195
        cv2.rectangle(out, (bx, y_bar + 2), (bx + bw_g, y_bar + 13), (45, 52, 68), -1)
        cv2.rectangle(out, (bx, y_bar + 2), (bx + int(bw_g * min(1.0, val)), y_bar + 13),
                      (60, 255, 110) if is_on else (0, 200, 255), -1)
        # Порог срабатывания 68%
        th_px = bx + int(bw_g * 0.68)
        cv2.line(out, (th_px, y_bar), (th_px, y_bar + 15), (255, 255, 255), 1)
        y_bar += 22

    # 6. Экранная жест-кнопка записи позы в debug_pose.json
    if btn_state is not None and p_raw is not None:
        if gesture_button(out, p_raw, btn_state, "dbg_rec_pose", W - 110, H - 125, 48, "REC [C]", t, hold_sec=0.8, color=(255, 120, 60)):
            record_debug_pose(p_raw, S)
            get_audio().play_sfx("select")

    if t - S.get("debug_rec_flash_t", 0.0) < 0.8:
        txt(out, f"SAVED TO debug_pose.json (#{saved_n})", W // 2, H - 140, 32, (80, 255, 120), center=True)


def draw_ai_corner_indicator(out, ai_mgr, debug_mode=False, is_play_state=False, t=None):
    """
    Отрисовывает аккуратный индикатор статуса ИИ в углу экрана:
    - Обычный режим: тонкий аккуратный бейдж «ИИ: загрузка… / ИИ: готов / ИИ: ошибка (причина)».
    - Режим отладки (B): подробные метрики (FPS ИИ, задержка мс, расход VRAM).
    """
    if ai_mgr is None:
        return

    if t is None:
        t = time.perf_counter()

    try:
        st = ai_mgr.get_status_info()
    except Exception:
        st = {
            "status": "error",
            "label": "ИИ: ошибка",
            "color": (50, 50, 240),
            "fps": 0.0,
            "ms": 0.0,
            "vram": 0.0,
            "details": "ИИ: ошибка",
        }

    status_name = st.get("status", "ready")
    base_col = st.get("color", (60, 230, 90))

    if debug_mode:
        text = st.get("details", st.get("label", "ИИ"))
        font_scale = 0.42
        thickness = 1
        h_pill = 26
    else:
        text = st.get("label", "ИИ")
        font_scale = 0.44
        thickness = 1
        h_pill = 24

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    w_pill = tw + 34

    # Позиционирование в правом верхнем углу (в режиме игры смещаем под таймер)
    x1 = W - 20
    x0 = max(10, x1 - w_pill)
    y0 = 68 if is_play_state else 16
    y1 = y0 + h_pill

    if y1 > H or x1 > W:
        return

    # Полупрозрачная тёмная подложка
    roi = out[y0:y1, x0:x1]
    bg_pill = np.full_like(roi, (14, 18, 26))
    out[y0:y1, x0:x1] = cv2.addWeighted(bg_pill, 0.85, roi, 0.15, 0)

    # Тонкая рамка с цветом статуса
    border_col = tuple(int(c * 0.75) for c in base_col)
    cv2.rectangle(out, (x0, y0), (x1, y1), border_col, 1, AA)

    # Пульсирующий индикатор-светодиод
    dot_cx = x0 + 13
    dot_cy = (y0 + y1) // 2
    if status_name == "loading":
        pulse = 0.55 + 0.45 * abs(math.sin(t * 3.8))
        dot_col = tuple(int(c * pulse) for c in base_col)
    else:
        dot_col = base_col

    cv2.circle(out, (dot_cx, dot_cy), 4, dot_col, -1, AA)
    cv2.circle(out, (dot_cx, dot_cy), 5, (255, 255, 255), 1, AA)

    # Текст статуса
    ty = (y0 + y1 + th) // 2 - 1
    cv2.putText(out, text, (x0 + 24, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 248, 255), thickness, AA)

