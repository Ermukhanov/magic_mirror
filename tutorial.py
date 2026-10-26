"""
Модуль обучения через анимацию (tutorial.py) для «Сиқырлы айна»:
- Полупрозрачный силуэт-«призрак» (скелет), показывающий нужное движение
- Огромные пульсирующие стрелки направления
- Проверка каждого действия в реальном времени (мигает жёлтым -> зеленеет, заполняется кольцо прогресса, звучит «Жарайсың!» и взрывается салют)
- Анимированная жест-кнопка с ладошкой и отсчётом 3-2-1 без клавиатуры
"""
import math
import cv2
import numpy as np
from config import W, H, HOLD_TUTORIAL, HOLD_BUTTON, AA
from tracking import lerp, ip, CURSOR, draw_hand_skeleton_feedback, draw_cursor_with_ring
from heroes import apoly
from render import (
    stage_bg, draw_sky_parallax, compose, spawn_firework, update_and_draw_fireworks, get_stage_or_video_bg,
)
from game import detect_actions
from i18n import tr, txt
from audio import get_audio


def draw_animated_hand_icon(img, cx, cy, t, col=(0, 255, 255)):
    dy = int(math.sin(t * 5.0) * 10)
    hx, hy = int(cx), int(cy + dy)
    cv2.circle(img, (hx, hy), 18, col, -1, AA)
    cv2.circle(img, (hx, hy), 18, (255, 255, 255), 2, AA)
    for ang, flen in ((-0.55, 24), (-0.25, 28), (0.05, 30), (0.35, 26), (0.75, 20)):
        fx = int(hx + math.sin(ang) * flen)
        fy = int(hy - math.cos(ang) * flen)
        cv2.line(img, (hx, hy), (fx, fy), col, 6, AA)
        cv2.line(img, (hx, hy), (fx, fy), (255, 255, 255), 2, AA)


def gesture_button(img, p, btn_state, key_id, cx, cy, r, label, t, hold_sec=HOLD_BUTTON, color=(0, 230, 120)):
    """
    Кнопка выбора жестом:
    - Hitbox увеличен на +25% (r * 1.25)
    - Использует умный курсор руки CURSOR.final_cursor (и прямые координаты кистей как резерв)
    - Если курсор вышел менее чем на 0.3 сек, прогресс не сбрасывается, а плавно убывает
    - Кольцо прогресса рисуется прямо вокруг курсора руки
    """
    candidates = []
    if CURSOR.final_cursor is not None:
        candidates.append(CURSOR.final_cursor)
    if p is not None:
        if p[15][2] > 0.38:
            candidates.append(p[15][:2])
        if p[16][2] > 0.38:
            candidates.append(p[16][:2])

    inside = False
    active_h = None
    hitbox_r = r * 1.25  # Hitbox на +25% больше видимой кнопки
    for h in candidates:
        if math.hypot(float(h[0]) - cx, float(h[1]) - cy) <= hitbox_r:
            inside = True
            active_h = h
            break

    dt_btn = max(0.0, min(0.1, t - btn_state.get(f"last_t_{key_id}", t)))
    btn_state[f"last_t_{key_id}"] = t
    prog = float(btn_state.get(f"prog_{key_id}", 0.0))

    if inside:
        if btn_state.get("id") != key_id:
            btn_state["id"] = key_id
            get_audio().play_sfx("select")
        btn_state[f"leave_t_{key_id}"] = 0.0
        prog = min(1.0, prog + dt_btn / max(0.2, hold_sec))
    else:
        leave_t = btn_state.get(f"leave_t_{key_id}", 0.0)
        if leave_t == 0.0:
            leave_t = t
            btn_state[f"leave_t_{key_id}"] = leave_t
        # Если ушёл менее чем на 0.3 сек — плавно убывает, иначе быстрее
        decay_rate = 0.85 if (t - leave_t) < 0.30 else 2.6
        prog = max(0.0, prog - dt_btn * decay_rate)
        if prog <= 0.0 and btn_state.get("id") == key_id:
            btn_state["id"] = None

    btn_state[f"prog_{key_id}"] = prog

    scale_k = 1.08 if inside else 1.0
    pulse = scale_k + 0.05 * math.sin(t * 6.0)
    rr = int(r * pulse)

    x0, y0 = max(0, cx - rr - 16), max(0, cy - rr - 16)
    x1, y1 = min(W, cx + rr + 16), min(H, cy + rr + 16)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        ov = roi.copy()
        rcx, rcy = cx - x0, cy - y0
        cv2.circle(ov, (rcx, rcy), rr, (25, 35, 50), -1, AA)
        cv2.circle(ov, (rcx, rcy), int(rr * 0.86), color if inside else (70, 130, 190), -1, AA)
        w_a = 0.68 if inside else 0.42
        img[y0:y1, x0:x1] = cv2.addWeighted(ov, w_a, roi, 1.0 - w_a, 0)

    cv2.circle(img, (cx, cy), rr, (80, 255, 140) if inside else (255, 255, 255), 5 if inside else 3, AA)
    if prog > 0.01:
        ang = int(360 * prog)
        cv2.ellipse(img, (cx, cy), (rr + 8, rr + 8), -90, 0, ang, (60, 255, 120), 8, AA)
        countdown = max(1, int(math.ceil((1.0 - prog) * 3)))
        txt(img, str(countdown), cx, cy - 28, 64, (255, 255, 255), center=True)
    else:
        draw_animated_hand_icon(img, cx, cy - 12, t, col=(0, 255, 255))
        ay = int(cy + rr + 28 + math.sin(t * 6.0) * 8)
        cv2.arrowedLine(img, (cx, ay + 24), (cx, ay - 6), (0, 255, 255), 5, AA, tipLength=0.45)

    txt(img, label, cx, cy + rr + 42, 32, (255, 255, 255), center=True)

    # Кольцо прогресса прямо вокруг курсора руки!
    cur_draw = active_h if active_h is not None else CURSOR.final_cursor
    if cur_draw is not None:
        draw_cursor_with_ring(img, cur_draw, prog=prog, t=t)

    if prog >= 1.0:
        btn_state["id"] = None
        btn_state[f"prog_{key_id}"] = 0.0
        return True
    return False


def draw_pulsing_arrow(img, p0, p1, col, t, scale=1.0):
    pulse = 0.85 + 0.25 * math.sin(t * 7.0)
    th = max(4, int(10 * scale * pulse))
    p0i, p1i = ip(p0), ip(p1)
    cv2.arrowedLine(img, p0i, p1i, (255, 255, 255), th + 4, AA, tipLength=0.35)
    cv2.arrowedLine(img, p0i, p1i, col, th, AA, tipLength=0.35)


def draw_ghost_tutor(img, cx, cy, scale, step_kind, is_correct, t):
    # Плавная медленная повторяющаяся анимация (чтобы ребёнку 4-6 лет было легко повторять)
    phase = 0.5 + 0.5 * math.sin(t * 2.5)

    if is_correct:
        col = (80, 255, 100)
        alpha = 0.78
    else:
        blink = 0.55 + 0.45 * math.sin(t * 8.0)
        col = (0, int(190 + 65 * blink), 255)
        alpha = 0.52 + 0.22 * blink

    pad_r = int(scale * 1.48)
    x0, y0 = max(0, int(cx - pad_r)), max(0, int(cy - pad_r))
    x1, y1 = min(W, int(cx + pad_r)), min(H, int(cy + pad_r))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    ov = roi.copy()
    rcx, rcy = cx - x0, cy - y0

    head = np.array([rcx, rcy - 1.05 * scale], np.float32)
    ls = np.array([rcx - 0.36 * scale, rcy - 0.62 * scale], np.float32)
    rs = np.array([rcx + 0.36 * scale, rcy - 0.62 * scale], np.float32)
    lh = np.array([rcx - 0.20 * scale, rcy + 0.12 * scale], np.float32)
    rh = np.array([rcx + 0.20 * scale, rcy + 0.12 * scale], np.float32)
    lk = np.array([rcx - 0.22 * scale, rcy + 0.58 * scale], np.float32)
    rk = np.array([rcx + 0.22 * scale, rcy + 0.58 * scale], np.float32)
    la = np.array([rcx - 0.22 * scale, rcy + 1.02 * scale], np.float32)
    ra = np.array([rcx + 0.22 * scale, rcy + 1.02 * scale], np.float32)

    if step_kind == "fly":
        ang = lerp(0.35, 2.65, phase)
        le = ls + np.array([-math.sin(ang) * 0.42 * scale, math.cos(ang) * 0.42 * scale])
        lw = le + np.array([-math.sin(ang) * 0.42 * scale, math.cos(ang) * 0.42 * scale])
        re = rs + np.array([math.sin(ang) * 0.42 * scale, math.cos(ang) * 0.42 * scale])
        rw = re + np.array([math.sin(ang) * 0.42 * scale, math.cos(ang) * 0.42 * scale])
    elif step_kind == "beam":
        ang = lerp(0.25, 1.58, phase)
        le = ls + np.array([-0.18 * scale, 0.38 * scale])
        lw = le + np.array([-0.15 * scale, 0.35 * scale])
        re = rs + np.array([math.sin(ang) * 0.46 * scale, math.cos(ang) * 0.46 * scale])
        rw = re + np.array([math.sin(ang) * 0.48 * scale, math.cos(ang) * 0.48 * scale])
    elif step_kind in ("tulpar", "zero_g"):
        bob = math.sin(t * 7.5) * 0.18 * scale
        lk = lk + np.array([0, -abs(bob)])
        rk = rk + np.array([0, -abs(math.cos(t * 7.5) * 0.18 * scale)])
        le = ls + np.array([-0.28 * scale, 0.25 * scale])
        lw = le + np.array([0.08 * scale, -0.22 * scale + bob * 0.5])
        re = rs + np.array([0.28 * scale, 0.25 * scale])
        rw = re + np.array([-0.08 * scale, -0.22 * scale - bob * 0.5])
    elif step_kind == "arrows":
        le = ls + np.array([-0.25 * scale, 0.05 * scale])
        lw = ls + np.array([-0.05 * scale, 0.02 * scale])
        re = rs + np.array([0.45 * scale * phase, 0.0])
        rw = re + np.array([0.45 * scale * phase, 0.0])
    elif step_kind in ("ice_wave", "repulsor", "web"):
        ang = lerp(0.45, 1.45, phase)
        le = ls + np.array([-math.sin(ang) * 0.44 * scale, math.cos(ang) * 0.28 * scale])
        lw = le + np.array([-math.sin(ang) * 0.44 * scale, 0.0])
        re = rs + np.array([math.sin(ang) * 0.44 * scale, math.cos(ang) * 0.28 * scale])
        rw = re + np.array([math.sin(ang) * 0.44 * scale, 0.0])
    elif step_kind == "magic_dust":
        sw_ang = math.sin(t * 6.0) * 0.65
        le = ls + np.array([-0.20 * scale, 0.35 * scale])
        lw = le + np.array([-0.15 * scale, 0.30 * scale])
        re = rs + np.array([0.25 * scale, -0.45 * scale])
        rw = re + np.array([math.sin(sw_ang) * 0.48 * scale, -0.45 * scale])
    elif step_kind == "lucky_charm":
        clap_d = (1.0 - phase) * 0.55 * scale
        le = ls + np.array([-0.15 * scale, 0.28 * scale])
        re = rs + np.array([0.15 * scale, 0.28 * scale])
        lw = np.array([rcx - clap_d, rcy - 0.25 * scale])
        rw = np.array([rcx + clap_d, rcy - 0.25 * scale])
    else:
        le = lerp(ls + np.array([-0.35 * scale, 0.25 * scale]), ls + np.array([-0.10 * scale, 0.28 * scale]), phase)
        re = lerp(rs + np.array([0.35 * scale, 0.25 * scale]), rs + np.array([0.10 * scale, 0.28 * scale]), phase)
        lw = lerp(ls + np.array([-0.45 * scale, 0.50 * scale]), np.array([rcx + 0.14 * scale, rcy - 0.22 * scale]), phase)
        rw = lerp(rs + np.array([0.45 * scale, 0.50 * scale]), np.array([rcx - 0.14 * scale, rcy - 0.22 * scale]), phase)

    cv2.circle(ov, (int(rcx), int(rcy - 0.05 * scale)), int(scale * 1.35), (22, 28, 42), -1, AA)
    cv2.circle(ov, (int(rcx), int(rcy - 0.05 * scale)), int(scale * 1.35), col, 5, AA)

    torso = np.array([ls, rs, rh, lh], np.int32)
    cv2.fillPoly(ov, [torso], col, AA)
    cv2.polylines(ov, [torso], True, (255, 255, 255), 3, AA)

    limb_th = max(6, int(scale * 0.16))
    for a, b in ((ls, le), (le, lw), (rs, re), (re, rw), (lh, lk), (lk, la), (rh, rk), (rk, ra)):
        cv2.line(ov, ip(a), ip(b), (255, 255, 255), limb_th + 4, AA)
        cv2.line(ov, ip(a), ip(b), col, limb_th, AA)

    for jt in (ls, rs, le, re, lh, rh, lk, rk):
        cv2.circle(ov, ip(jt), max(4, int(scale * 0.08)), (255, 255, 255), -1, AA)
    for w_pt in (lw, rw):
        cv2.circle(ov, ip(w_pt), max(8, int(scale * 0.15)), (255, 255, 255), -1, AA)
        cv2.circle(ov, ip(w_pt), max(6, int(scale * 0.11)), col, -1, AA)

    cv2.circle(ov, ip(head), int(scale * 0.28), col, -1, AA)
    cv2.circle(ov, ip(head), int(scale * 0.28), (255, 255, 255), 3, AA)

    img[y0:y1, x0:x1] = cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0)

    s_arrow = scale / 110.0
    if step_kind == "fly":
        draw_pulsing_arrow(img, (cx - 0.78 * scale, cy - 0.15 * scale), (cx - 0.78 * scale, cy - 1.15 * scale), col, t, s_arrow)
        draw_pulsing_arrow(img, (cx + 0.78 * scale, cy - 0.15 * scale), (cx + 0.78 * scale, cy - 1.15 * scale), col, t, s_arrow)
    elif step_kind in ("beam", "arrows", "web"):
        draw_pulsing_arrow(img, (cx + 0.42 * scale, cy - 0.88 * scale), (cx + 1.28 * scale, cy - 0.88 * scale), col, t, s_arrow)
    elif step_kind in ("tulpar", "zero_g"):
        draw_pulsing_arrow(img, (cx, cy + 0.35 * scale), (cx, cy - 0.95 * scale), col, t, s_arrow)
    elif step_kind in ("ice_wave", "repulsor"):
        draw_pulsing_arrow(img, (cx - 0.35 * scale, cy - 0.65 * scale), (cx - 1.25 * scale, cy - 0.65 * scale), col, t, s_arrow)
        draw_pulsing_arrow(img, (cx + 0.35 * scale, cy - 0.65 * scale), (cx + 1.25 * scale, cy - 0.65 * scale), col, t, s_arrow)
    elif step_kind == "magic_dust":
        draw_pulsing_arrow(img, (cx - 0.65 * scale, cy - 1.28 * scale), (cx + 0.65 * scale, cy - 1.28 * scale), col, t, s_arrow)
    else:
        draw_pulsing_arrow(img, (cx - 0.95 * scale, cy - 0.25 * scale), (cx - 0.22 * scale, cy - 0.25 * scale), col, t, s_arrow)
        draw_pulsing_arrow(img, (cx + 0.95 * scale, cy - 0.25 * scale), (cx + 0.22 * scale, cy - 0.25 * scale), col, t, s_arrow)


def get_hero_tutorial_steps(hero):
    spec_kind = hero.get("special", "shield")
    spec_label = f"spec_{hero['id']}"
    return [
        dict(kind="fly", label_key="tut_fly", voice_key="raise_both_hands"),
        dict(kind="beam", label_key="tut_beam", voice_key="raise_hand"),
        dict(kind=spec_kind, label_key=spec_label, voice_key="tut_special"),
    ]


def render_tutorial(frame, p, mask_s, hero, t, dt, parts, S, btn_state):
    steps = get_hero_tutorial_steps(hero)
    if p is not None and mask_s is not None:
        acts = detect_actions(p, S, dt)
        out, g = compose(frame, p, mask_s, hero, t, dt, parts, is_flying=acts["fly"])
        # Живой неоновый скелет рук и плеч ребёнка для сравнения с анимацией-призраком
        draw_hand_skeleton_feedback(out, p, active_side=CURSOR.active_side, thin=False, t=t)
    else:
        out = get_stage_or_video_bg(hero, t, is_flying=False)
        g = None
        acts = {"fly": False, "shield": False, "beams": [], "special": set(), "scores": {}}

    step_idx = S["tut_step"]

    if step_idx >= len(steps):
        update_and_draw_fireworks(out, S["fireworks"], dt)
        txt(out, tr("ready_q"), W // 2, 70, 62, (0, 255, 255), center=True)
        txt(out, tr("well_done"), W // 2, 145, 48, (80, 255, 120), center=True)
        if not S.get("ready_voiced"):
            get_audio().play_voice("ready_q", force=True)
            S["ready_voiced"] = True
        if gesture_button(out, p, btn_state, "start_game", W // 2, H // 2 + 20, 95, tr("lets_start"), t):
            get_audio().play_voice("lets_start", force=True)
            get_audio().play_sfx("select")
            S.update(state="play", score=0, start=t, stars=[], pops=[], act={}, special_fx=[],
                     boss=None, combo=0, mult=1, energy=0.0, stars_collected=0, monsters_defeated=0, missed=0)
        return out

    step = steps[step_idx]
    kind = step["kind"]
    step_elapsed = t - S["tut_step_t"]

    if S.get("tut_voiced_step") != step_idx:
        v_key = step["label_key"] if step["label_key"].startswith("spec_") else step["voice_key"]
        get_audio().play_voice(v_key, force=True)
        S["tut_voiced_step"] = step_idx
        S["tut_bigger_voiced"] = False
    elif step_elapsed > 10.0 and not S.get("tut_bigger_voiced"):
        get_audio().play_voice("tut_try_bigger", force=True, louder=True)
        S["tut_bigger_voiced"] = True

    # Непрерывная оценка точности позы (0.0 .. 1.0)
    scores = acts.get("scores", {})
    acc = float(scores.get(kind, 0.0))
    if step_idx == 2:
        acc = max(acc, float(scores.get("fly", 0.0)) * 0.85, float(scores.get("beam", 0.0)) * 0.85, float(scores.get("shield", 0.0)) * 0.85)

    # Пороги с гистерезисом: через 10 секунд упрощаем порог с 68% до 45%
    bigger = step_elapsed > 10.0
    enter_th = 0.45 if bigger else 0.68
    exit_th = 0.32 if bigger else 0.52

    was_doing = S.get("tut_is_doing", False)
    is_doing = False
    if kind == "fly" and acts["fly"]:
        is_doing = True
    elif kind == "beam" and bool(acts["beams"]):
        is_doing = True
    elif kind == "shield" and acts["shield"]:
        is_doing = True
    elif kind in acts.get("special", set()):
        is_doing = True
    elif acc >= (exit_th if was_doing else enter_th):
        is_doing = True
    elif step_idx == 2 and (acts["fly"] or bool(acts["beams"]) or acts["shield"]):
        is_doing = True

    # Мгновенная обратная связь при входе в правильную позу
    if is_doing and not was_doing:
        get_audio().play_sfx("select")
    S["tut_is_doing"] = is_doing

    if is_doing:
        S["tut_prog"] = min(1.0, S["tut_prog"] + dt / HOLD_TUTORIAL)
        border_col = (80, 255, 100)
    else:
        # Даже если поза чуть ниже порога, при acc > 0.35 прогресс не сбрасывается резко
        decay_k = 0.35 if acc > 0.35 else 1.1
        S["tut_prog"] = max(0.0, S["tut_prog"] - dt * decay_k)
        blink = 0.5 + 0.5 * math.sin(t * 7.5)
        border_col = (0, int(190 + 65 * blink), 255)

    cv2.rectangle(out, (10, 10), (W - 10, H - 10), border_col, 8 if is_doing else 4, AA)

    ghost_scale = 138.0 if bigger else 108.0
    ghost_cx = int(W * 0.23) if (g is None or g["neck"][0] > W * 0.45) else int(W * 0.77)
    ghost_cy = int(H * 0.46)

    draw_ghost_tutor(out, ghost_cx, ghost_cy, ghost_scale, kind, is_doing, t)

    # Шаги 1 - 2 - 3 наверху
    for i in range(len(steps)):
        cx_dot = W // 2 + (i - 1) * 70
        cy_dot = 42
        c_dot = (80, 255, 100) if i < step_idx else (border_col if i == step_idx else (120, 120, 140))
        cv2.circle(out, (cx_dot, cy_dot), 18, c_dot, -1, AA)
        cv2.circle(out, (cx_dot, cy_dot), 18, (255, 255, 255), 3, AA)
        txt(out, str(i + 1), cx_dot, cy_dot - 14, 24, (20, 20, 30), center=True)

    # Кольцо удержания под призраком
    prog = S["tut_prog"]
    prog_pct = int(round(prog * 20)) * 5
    ring_cx, ring_cy, ring_r = ghost_cx, int(ghost_cy + ghost_scale * 1.48), 40
    cv2.circle(out, (ring_cx, ring_cy), ring_r, (35, 45, 65), -1, AA)
    cv2.circle(out, (ring_cx, ring_cy), ring_r, (255, 255, 255), 3, AA)
    if prog > 0:
        cv2.ellipse(out, (ring_cx, ring_cy), (ring_r, ring_r), -90, 0, int(360 * prog), (80, 255, 100), 10, AA)
    txt(out, f"{prog_pct}%", ring_cx, ring_cy - 14, 25, (255, 255, 255), center=True)

    # Шкала точности позы («Қимыл дәлдігі: 60%»)
    acc_pct = int(round(max(acc, prog) * 20)) * 5
    bx0, by0, bw, bh = W // 2 - 210, H - 154, 420, 26
    cv2.rectangle(out, (bx0, by0), (bx0 + bw, by0 + bh), (25, 32, 48), -1, AA)
    bar_col = (80, 255, 110) if is_doing else ((0, 225, 255) if acc_pct >= 40 else (0, 165, 255))
    fill_w = int(bw * min(1.0, max(acc, prog)))
    if fill_w > 0:
        cv2.rectangle(out, (bx0, by0), (bx0 + fill_w, by0 + bh), bar_col, -1, AA)
    # Отметка порога срабатывания на шкале
    th_x = bx0 + int(bw * enter_th)
    cv2.line(out, (th_x, by0 - 4), (th_x, by0 + bh + 4), (255, 255, 255), 3, AA)
    cv2.rectangle(out, (bx0, by0), (bx0 + bw, by0 + bh), (255, 255, 255), 2, AA)
    txt(out, f"{tr('pose_accuracy')}: {acc_pct}%", W // 2, by0 - 30, 26, (255, 255, 255), center=True)

    caption = tr(step["label_key"])
    if is_doing:
        txt(out, tr("well_done"), W // 2, 76, 62, (80, 255, 100), center=True)
    elif bigger:
        txt(out, tr("tut_try_bigger"), W // 2, 76, 42, (0, 235, 255), center=True)

    apoly(out, [(W // 2 - 420, H - 102), (W // 2 + 420, H - 102), (W // 2 + 420, H - 22), (W // 2 - 420, H - 22)],
          (20, 26, 40), 0.75, border_col, 3)
    txt(out, caption, W // 2, H - 82, 46 if not bigger else 52, (255, 255, 255), center=True)

    if prog >= 1.0:
        get_audio().play_voice("well_done", force=True)
        get_audio().play_sfx("victory")
        spawn_firework(S["fireworks"], W // 2, H // 3, count=48)
        spawn_firework(S["fireworks"], ghost_cx, ghost_cy - 60, count=36)
        S["tut_step"] += 1
        S["tut_step_t"] = t
        S["tut_prog"] = 0.0
        S["tut_is_doing"] = False

    if step_elapsed > 20.0:
        txt(out, tr("tut_skip_hint"), W // 2, 130, 26, (200, 230, 255), center=True)
        if gesture_button(out, p, btn_state, f"skip_{step_idx}", W - 130, 130, 52, tr("btn_skip"), t, hold_sec=1.0, color=(0, 180, 255)):
            S["tut_step"] += 1
            S["tut_step_t"] = t
            S["tut_prog"] = 0.0
            S["tut_is_doing"] = False

    update_and_draw_fireworks(out, S["fireworks"], dt)
    return out
