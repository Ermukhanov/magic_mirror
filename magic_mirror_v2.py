"""
СИҚЫРЛЫ АЙНА / ВОЛШЕБНОЕ ЗЕРКАЛО v2
Интерактивная игра с компьютерным зрением для детского сада (дети 4–6 лет).
Ребёнок встаёт перед камерой, выбирает героя рукой и превращается в голограмму-героя в костюме.
Лицо ребёнка всегда остаётся настоящим. Всё работает полностью офлайн (30+ FPS при 1280x720).

Установка:  pip install -r requirements.txt
Озвучка:    python tools/gen_voice.py
Запуск:     python magic_mirror_v2.py
Клавиши:
    q — выход
    r — к выбору героя
    l — язык (қазақша / русский, по умолчанию всегда қазақша)
    f — показать/скрыть FPS
    t — режим воспитателя (выбор героя 1..8, сложность D, время G, пауза P, пропуск обучения S, статистика)
    b — режим отладки разработчика (Debug Overlay: скелет, hitbox +25%, зона досягаемости, пороги жестов, задержка мс)
    c — сохранить координаты скелета в debug_pose.json (ТОЛЬКО цифры, без фото/видео)
    + / - / m — громкость звука и выключение звука
"""
import math
import time
import logging
import traceback
import cv2
import numpy as np

import config
from config import W, H, SW, SH, HOLD_SELECT, LOG_PATH, AA
import i18n
from i18n import tr, txt, verify_kazakh_glyphs
from tracking import (
    Cam, OneEuro, PoseWorker, lerp, geom, ip, update_height_calibration,
    CURSOR, get_fingertip_pos, draw_hand_skeleton_feedback, draw_cursor_with_ring,
)
from heroes import HEROES, apoly, star_pts, draw_back, draw_front, emblem, hero_display_name, load_hero_png, overlay_rgba_sprite
from render import (
    stage_bg, draw_sky_parallax, stage_fx, holo_person, aura, compose,
    spawn_firework, update_and_draw_fireworks, get_stage_or_video_bg,
)
from tutorial import gesture_button, draw_ghost_tutor, get_hero_tutorial_steps, render_tutorial
from game import evaluate_pose_scores, detect_actions, trigger_hero_special, game_fx, update_stars, damage_boss, draw_cute_monster, draw_mini_boss
from audio import get_audio, SoundManager
from ai_transform import get_ai_manager
from ui import (
    CW, CH, card_rect, fake_pose, card_canvas, render_select, render_calib, render_end,
    draw_hud, check_lighting_and_warn, draw_calibration_silhouette, draw_come_closer_overlay,
    draw_teacher_panel, draw_debug_overlay, record_debug_pose, load_stats, record_game_stat,
    draw_ai_corner_indicator,
)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)

LANG = config.LANG
SHOW_FPS = config.SHOW_FPS
GAME_TIME = config.GAME_TIME


def main():
    global LANG, SHOW_FPS, GAME_TIME
    verify_kazakh_glyphs()
    aud = get_audio()
    aud.play_voice("hold_hero", force=True)
    cam = Cam()
    cam.start()
    worker = PoseWorker(cam)
    worker.start()
    ai_mgr = None
    if str(config.AI_TRANSFORM).lower() != "off":
        try:
            ai_mgr = get_ai_manager()
        except Exception as e_ai:
            logging.warning(f"Не удалось запустить фоновый поток ИИ: {e_ai}")
    win = "Magic Mirror"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if config.FULLSCREEN:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    now_init = time.perf_counter()
    S = dict(
        state="select", hover=-1, hover_t=0.0, hover_prog=0.0, hover_leave_t=0.0,
        last_hold_voice_t=now_init, hero=None, score=0, start=0.0,
        stars=[], pops=[], fireworks=[], special_fx=[], act={}, last_fly=0.0, end_t=0.0,
        calib_t0=0.0, calib_prog=0.0,
        tut_step=0, tut_step_t=0.0, tut_prog=0.0,
        teacher_mode=False, debug_mode=False, paused=False, diff_mode="auto", diff_level=1,
        stats_cache=load_stats(), last_good_p=None, last_good_p_raw=None, last_good_mask=None, last_good_ts=0.0,
    )
    btn_state = {"id": None, "t0": 0.0}
    parts = np.random.rand(70, 3).astype(np.float32)
    parts[:, 0] = parts[:, 0] * 2 - 1
    parts[:, 2] = 0.4 + parts[:, 2]
    last_idx, prev, fps = -1, time.perf_counter(), 30.0
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            S["state"], S["hover"], S["hover_prog"], S["paused"] = "select", -1, 0.0, False
            S["last_hold_voice_t"] = time.perf_counter()
            aud.stop_bgm()
            aud.play_voice("hold_hero", force=True)
        if key == ord('l'):
            LANG = i18n.toggle_lang()
            aud.play_voice("hold_hero", force=True)
        if key == ord('f'):
            SHOW_FPS = not SHOW_FPS
            config.SHOW_FPS = SHOW_FPS
        if key == ord('t'):
            S["teacher_mode"] = not S.get("teacher_mode", False)
            S["stats_cache"] = load_stats()
        if key == ord('b'):
            S["debug_mode"] = not S.get("debug_mode", False)
        if key == ord('c'):
            if S.get("last_good_p_raw") is not None:
                record_debug_pose(S["last_good_p_raw"], S)
                aud.play_sfx("select")
        if key == ord('p'):
            S["paused"] = not S.get("paused", False)
        if key == ord('d'):
            if S.get("diff_mode", "auto") == "auto":
                S["diff_mode"], S["diff_level"] = "manual", 1
            elif S["diff_level"] < 3:
                S["diff_level"] += 1
            else:
                S["diff_mode"] = "auto"
        if key == ord('g'):
            GAME_TIME = 90 if GAME_TIME == 60 else (45 if GAME_TIME == 90 else 60)
            config.GAME_TIME = GAME_TIME
        if key in (ord('+'), ord('=')):
            aud.change_volume(+0.10)
        if key in (ord('-'), ord('_')):
            aud.change_volume(-0.10)
        if key == ord('m'):
            aud.toggle_mute()
        if key == ord('a'):
            modes_ai = ("auto", "on", "off")
            cur_ai = str(config.AI_TRANSFORM).lower()
            config.AI_TRANSFORM = modes_ai[(modes_ai.index(cur_ai) + 1) % len(modes_ai)] if cur_ai in modes_ai else "auto"
            config.save_config_flags()
            if config.AI_TRANSFORM != "off" and ai_mgr is None:
                try:
                    ai_mgr = get_ai_manager()
                except Exception as e_ai:
                    logging.warning(f"Не удалось запустить фоновый поток ИИ: {e_ai}")
        if key == ord('w'):
            config.VIDEO_WORLDS = not config.VIDEO_WORLDS
            config.RICH_WORLDS = config.VIDEO_WORLDS
            config.save_config_flags()
        if key == ord('x'):
            config.FX_CLIPS = not config.FX_CLIPS
            config.save_config_flags()
        if key == ord('h'):
            config.HERO_ART = not config.HERO_ART
            config.save_config_flags()
        if key == ord('y'):
            config.GAME_LAYER = not config.GAME_LAYER
            config.save_config_flags()
        if key == ord('s') and S["state"] in ("calib", "tutorial"):
            if S["state"] == "calib":
                S.update(state="tutorial", tut_step=0, tut_step_t=time.perf_counter(), tut_prog=0.0, tut_voiced_step=-1, ready_voiced=False)
            else:
                S["tut_step"] = 99
                S["tut_step_t"] = time.perf_counter()
                S["tut_prog"] = 0.0
        if ord('1') <= key <= ord('8'):
            h_idx = key - ord('1')
            if h_idx < len(HEROES):
                chosen = HEROES[h_idx]
                aud.play_sfx("select")
                aud.start_bgm(chosen["id"])
                now_k = time.perf_counter()
                S.update(
                    state="play", hero=chosen, score=0, start=now_k,
                    stars=[], pops=[], fireworks=[], special_fx=[], act={},
                    boss=None, combo=0, mult=1, energy=0.0, stars_collected=0, monsters_defeated=0, missed=0,
                    teacher_mode=False, paused=False,
                )

        frame, fts, idx = cam.latest()
        if frame is None or idx == last_idx:
            time.sleep(0.001)
            continue
        last_idx = idx
        now = time.perf_counter()
        dt = min(now - prev, 0.1)
        prev = now
        fps = fps * 0.92 + 0.08 / max(dt, 1e-3)

        try:
            with worker.lock:
                r = worker.result
            p = mask_s = p_no_lead = None
            pose_latency_ms = 0.0
            if r is not None and now - r["ts"] < 0.5:
                pose_latency_ms = (now - r["ts"]) * 1000.0
                lead = min(max(now - r["ts"], 0.0), 0.1)
                p_cand = r["p"].copy()
                p_cand[:, :2] += r["vel"] * lead * 0.9
                sw_cand = float(np.linalg.norm(p_cand[11][:2] - p_cand[12][:2]))
                if p_cand[11][2] >= 0.48 and p_cand[12][2] >= 0.48 and sw_cand >= 32.0:
                    p = p_cand
                    # Для курсора используем позу БЕЗ предсказания вперёд (lead=0), чтобы не было перелёта!
                    p_no_lead = r.get("raw_p", r["p"]).copy()
                    v = r["vel"][[11, 12, 23, 24]].mean(0) * lead * 0.9
                    M = np.float32([[1, 0, v[0] * SW / W], [0, 1, v[1] * SH / H]])
                    m = cv2.warpAffine(r["mask"], M, (SW, SH))
                    mask_s = np.clip((cv2.GaussianBlur(m, (5, 5), 0) - 0.35) * 4, 0, 1)
                    S["last_good_p"] = p.copy()
                    S["last_good_p_raw"] = p_no_lead.copy()
                    S["last_good_mask"] = mask_s.copy()
                    S["last_good_ts"] = now

            if p is None and S.get("last_good_p") is not None and (now - S.get("last_good_ts", 0.0)) < 0.65:
                fade_k = max(0.15, 1.0 - (now - S["last_good_ts"]) / 0.65)
                p = S["last_good_p"]
                p_no_lead = S.get("last_good_p_raw", p)
                mask_s = S["last_good_mask"] * fade_k

            update_height_calibration(frame, p, S, dt)

            # Цели магнитного притяжения для умного курсора (+25% hitbox)
            magnet_targets = []
            if S["state"] == "select":
                magnet_targets = [{"id": i, "rect": card_rect(i)} for i in range(len(HEROES))]
            elif S["state"] == "tutorial":
                if S.get("tut_step", 0) >= 3:
                    magnet_targets.append({"id": "start_game", "circle": (W // 2, H // 2 + 20, 95)})
                elif (now - S.get("tut_step_t", now)) > 20.0:
                    magnet_targets.append({"id": "skip", "circle": (W - 130, 130, 52)})
            elif S["state"] == "end":
                magnet_targets = [
                    {"id": "btn_again", "circle": (W // 2 - 220, 560, 74)},
                    {"id": "btn_next", "circle": (W // 2 + 220, 560, 74)},
                ]
            if S.get("debug_mode", False):
                magnet_targets.append({"id": "dbg_rec_pose", "circle": (W - 110, H - 125, 48)})

            hand_cursor, _ = CURSOR.update(p_no_lead, S, now, magnet_targets=magnet_targets)

            if S.get("paused", False):
                S["start"] += dt
                out = cv2.convertScaleAbs(frame, alpha=0.5)
                txt(out, "ПАУЗА [P]", W // 2, H // 2 - 40, 68, (0, 255, 255), center=True)
            elif S["state"] == "select":
                # Голосовая подсказка «Қолыңды кейіпкердің үстінде ұста!», если ребёнок не двигается 8 секунд
                if (now - CURSOR.last_move_t) > 8.0 and (now - S.get("last_hold_voice_t", 0.0)) > 8.0:
                    S["last_hold_voice_t"] = now
                    aud.play_voice("hold_hero")

                # Проверка попадания курсора в карточки с увеличенным на +25% hitbox'ом
                cur = -1
                if hand_cursor is not None:
                    best_d = 1e9
                    for i in range(len(HEROES)):
                        x, y, w, h = card_rect(i)
                        tcx, tcy = x + w * 0.5, y + h * 0.5
                        hw, hh = w * 0.5 * 1.25, h * 0.5 * 1.25
                        if abs(float(hand_cursor[0]) - tcx) <= hw and abs(float(hand_cursor[1]) - tcy) <= hh:
                            d = math.hypot(float(hand_cursor[0]) - tcx, float(hand_cursor[1]) - tcy)
                            if d < best_d:
                                best_d = d
                                cur = i

                prev_hover = S.get("hover", -1)
                prog = float(S.get("hover_prog", 0.0))

                if cur >= 0:
                    if cur != prev_hover:
                        S["hover"] = cur
                        prog = 0.0
                        aud.play_sfx("select")
                        aud.play_voice(f"hero_{HEROES[cur]['id']}", force=True)
                    S["hover_leave_t"] = 0.0
                    prog = min(1.0, prog + dt / HOLD_SELECT)
                else:
                    # Если курсор ушёл с карточки менее чем на 0.3 сек — прогресс не сбрасывается, а плавно убывает!
                    if prev_hover >= 0 and prog > 0.0:
                        leave_t = S.get("hover_leave_t", 0.0)
                        if leave_t == 0.0:
                            leave_t = now
                            S["hover_leave_t"] = leave_t
                        decay = 0.55 if (now - leave_t) < 0.30 else 2.2
                        prog = max(0.0, prog - dt * decay)
                        if prog <= 0.0:
                            S["hover"] = -1
                    else:
                        S["hover"] = -1
                        prog = 0.0

                S["hover_prog"] = prog
                disp_hover = S["hover"]
                out = render_select(frame, hand_cursor, disp_hover, prog, now, p=p)

                if disp_hover >= 0 and prog >= 1.0:
                    chosen = HEROES[disp_hover]
                    aud.play_sfx("victory")
                    aud.start_bgm(chosen["id"])
                    aud.play_voice("calib_title", force=True)
                    S.update(
                        state="calib", hero=chosen, calib_t0=now, calib_prog=0.0,
                        hover=-1, hover_prog=0.0, score=0, start=now,
                        stars=[], pops=[], fireworks=[], special_fx=[], act={},
                    )
                    spawn_firework(S["fireworks"], W // 2, H // 2, count=36)
            elif S["state"] == "calib":
                out = render_calib(frame, p, mask_s, S["hero"], now, dt, parts, S)
            elif S["state"] == "tutorial":
                out = render_tutorial(frame, p, mask_s, S["hero"], now, dt, parts, S, btn_state)
            elif S["state"] == "play":
                hero = S["hero"]
                left = GAME_TIME - (now - S["start"])
                if config.GAME_LAYER and left <= 0:
                    S["state"], S["end_t"] = "end", now
                    S["stats_cache"] = record_game_stat(hero["id"], S["score"])
                    aud.play_sfx("victory")
                    aud.play_voice("medal_praise", force=True)
                    spawn_firework(S["fireworks"], W // 2, H // 3, count=50)
                    continue
                if p is None:
                    out = get_stage_or_video_bg(hero, now, is_flying=False)
                    txt(out, tr("pause"), W // 2, H // 2 - 50, 70, (0, 255, 255), center=True)
                    draw_come_closer_overlay(out, None, now)
                    S["start"] += dt
                else:
                    acts_now = detect_actions(p, S, dt)
                    out, g = compose(frame, p, mask_s, hero, now, dt, parts, is_flying=acts_now["fly"])
                    # В игре рисуем тонкий неоновый скелет рук, чтобы ребёнок чувствовал отклик
                    draw_hand_skeleton_feedback(out, p, active_side=CURSOR.active_side, thin=True, t=now)
                    game_fx(out, p, g, hero, now, dt, S)
                if config.GAME_LAYER:
                    update_stars(out, S, dt)
                    draw_hud(out, S, left, now)
                else:
                    update_and_draw_fireworks(out, S["fireworks"], dt)
                    txt(out, tr("hint"), W // 2, H - 48, 26, (255, 255, 255), center=True)
            else:
                out = render_end(S["hero"], S["score"], now, dt, p, S, btn_state)

            if S.get("debug_mode", False) and p is not None and S["state"] in ("select", "calib", "end"):
                evaluate_pose_scores(p, S, dt)
            draw_debug_overlay(out, p_no_lead, p, S, fps, pose_latency_ms, magnet_targets, now, btn_state=btn_state)
            check_lighting_and_warn(out, frame, S, now)
            draw_teacher_panel(out, S)
        except Exception as err:
            logging.error(f"Ошибка рендера кадра: {err}\n{traceback.format_exc()}")
            out = frame.copy()

        if SHOW_FPS:
            cv2.putText(out, f"{fps:.0f} FPS [{i18n.get_lang().upper()}]", (W - 185, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        draw_ai_corner_indicator(out, ai_mgr, debug_mode=S.get("debug_mode", False), is_play_state=(S.get("state") == "play"), t=now)
        cv2.imshow(win, out)
    aud.stop_bgm()
    try:
        get_ai_manager().stop()
    except Exception:
        pass
    cam.running = worker.running = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
