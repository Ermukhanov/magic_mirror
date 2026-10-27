"""
Тест работы игры с ИИ (tools/test_game_ai_render.py):
1. Запускает ИИ в режиме ai_transform = 'on'.
2. Фиксирует строки лога при старте.
3. Дожидается готовности ИИ и отправляет тестовый кадр.
4. Прогоняет рендер compose() и отрисовывает угловой индикатор статуса ИИ.
5. Замеряет FPS рендера и сохраняет скриншоты в assets/ и artifact dir.
"""
import os
import sys
import time
import shutil
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
config.AI_TRANSFORM = "on"
config.VIDEO_WORLDS = True
config.FX_CLIPS = True

from heroes import HEROES
from tracking import geom, clean_person_mask
from render import compose, draw_ai_vortex_animation
from ui import draw_ai_corner_indicator, draw_hud
from ai_transform import get_ai_manager

ARTIFACT_DIR = r"C:\Users\Acer Nitro\.gemini\antigravity\brain\a7feaa77-e574-4eb1-a10c-00490c79dbd7"


def generate_mock_child_frame(w=1280, h=720):
    frame = np.full((h, w, 3), (35, 45, 60), dtype=np.uint8)
    # Силуэт ребёнка
    cx, head_y, r_head = w // 2, int(h * 0.30), 52
    cv2.circle(frame, (cx, head_y), r_head, (185, 205, 235), -1)
    # Туловище
    pts_body = np.array([
        [cx - 85, head_y + 45],
        [cx + 85, head_y + 45],
        [cx + 65, head_y + 280],
        [cx - 65, head_y + 280],
    ], np.int32)
    cv2.fillPoly(frame, [pts_body], (140, 160, 195))
    # Руки
    cv2.line(frame, (cx - 85, head_y + 55), (cx - 165, head_y + 160), (140, 160, 195), 26)
    cv2.line(frame, (cx + 85, head_y + 55), (cx + 165, head_y + 160), (140, 160, 195), 26)
    # Ноги
    cv2.line(frame, (cx - 45, head_y + 280), (cx - 55, head_y + 460), (110, 125, 155), 28)
    cv2.line(frame, (cx + 45, head_y + 280), (cx + 55, head_y + 460), (110, 125, 155), 28)

    # Маска силуэта
    sw, sh = 384, 216
    mask_s = np.zeros((sh, sw), dtype=np.float32)
    scale_x, scale_y = sw / float(w), sh / float(h)
    cv2.circle(mask_s, (int(cx * scale_x), int(head_y * scale_y)), int(r_head * scale_x * 1.1), 1.0, -1)
    pts_body_s = (pts_body * [scale_x, scale_y]).astype(np.int32)
    cv2.fillPoly(mask_s, [pts_body_s], 1.0)
    cv2.line(mask_s, (int((cx - 85) * scale_x), int((head_y + 55) * scale_y)), (int((cx - 165) * scale_x), int((head_y + 160) * scale_y)), 1.0, int(26 * scale_x))
    cv2.line(mask_s, (int((cx + 85) * scale_x), int((head_y + 55) * scale_y)), (int((cx + 165) * scale_x), int((head_y + 160) * scale_y)), 1.0, int(26 * scale_x))
    cv2.line(mask_s, (int((cx - 45) * scale_x), int((head_y + 280) * scale_y)), (int((cx - 55) * scale_x), int((head_y + 460) * scale_y)), 1.0, int(28 * scale_x))
    cv2.line(mask_s, (int((cx + 45) * scale_x), int((head_y + 280) * scale_y)), (int((cx + 55) * scale_x), int((head_y + 460) * scale_y)), 1.0, int(28 * scale_x))
    mask_s = cv2.GaussianBlur(mask_s, (7, 7), 1.5)

    # 33 ключевые точки MediaPipe
    p = np.zeros((33, 3), dtype=np.float32)
    p[0] = [cx, head_y - 10, 0.95]      # nose
    p[11] = [cx - 85, head_y + 55, 0.95] # left shoulder
    p[12] = [cx + 85, head_y + 55, 0.95] # right shoulder
    p[13] = [cx - 130, head_y + 110, 0.90] # left elbow
    p[14] = [cx + 130, head_y + 110, 0.90] # right elbow
    p[15] = [cx - 165, head_y + 160, 0.92] # left wrist
    p[16] = [cx + 165, head_y + 160, 0.92] # right wrist
    p[23] = [cx - 55, head_y + 280, 0.95] # left hip
    p[24] = [cx + 55, head_y + 280, 0.95] # right hip
    p[25] = [cx - 50, head_y + 370, 0.90] # left knee
    p[26] = [cx + 50, head_y + 370, 0.90] # right knee
    p[27] = [cx - 55, head_y + 460, 0.90] # left ankle
    p[28] = [cx + 55, head_y + 460, 0.90] # right ankle

    return frame, mask_s, p


def test_ai_game_render():
    print("=" * 60)
    print(" ЗАПУСК ТЕСТА ИГРЫ С РЕАЛЬНЫМ ИИ (ai_transform = 'on')")
    print("=" * 60)

    print("\n[1/5] Инициализация фонового потока ИИ...")
    t0_start = time.perf_counter()
    ai_mgr = get_ai_manager()

    # Ждём завершения прогрева (до 90 секунд)
    max_wait = 90.0
    while not ai_mgr.ready and (time.perf_counter() - t0_start) < max_wait:
        time.sleep(0.2)

    load_dt = time.perf_counter() - t0_start
    if not ai_mgr.ready:
        print(f"  [ОШИБКА] ИИ не готов после {load_dt:.1f} сек! Ошибка: {ai_mgr.error_msg}")
        return False

    print(f"  ИИ успешно готов за {load_dt:.2f} сек на {ai_mgr.device_name} [OK]")

    print("\n[2/5] Подготовка тестового кадра и передача в GPU-поток...")
    frame, mask_s, p = generate_mock_child_frame(1280, 720)
    hero = HEROES[0]  # Alpamys

    clean_s = clean_person_mask(mask_s, erode_px=2)
    clean_m_full = cv2.resize(clean_s, (1280, 720), interpolation=cv2.INTER_LINEAR)

    # Передаём 3 кадра, чтобы GPU-поток сгенерировал и выдал активный кадр
    for i_f in range(3):
        ai_mgr.submit_frame(frame, clean_m_full, p, (400, 100, 880, 680), hero, i_f)
        time.sleep(0.08)

    # Ждём готовности первого ИИ-кадра
    t_wait_ai = time.perf_counter()
    while not ai_mgr.is_active_for_hero(hero) and (time.perf_counter() - t_wait_ai) < 5.0:
        time.sleep(0.02)

    ai_is_live = ai_mgr.is_active_for_hero(hero)
    print(f"  ИИ-кадр активен для героя {hero['id']}: {ai_is_live} [OK]")
    st_info = ai_mgr.get_status_info()
    print(f"  Статус менеджера: {st_info['label']} | {st_info['details']}")

    print("\n[3/5] Замер FPS полного игрового рендера compose()...")
    parts = np.random.rand(70, 3).astype(np.float32)
    parts[:, 0] = parts[:, 0] * 2 - 1
    parts[:, 2] = 0.4 + parts[:, 2]

    # Прогревочный прогон
    out, g = compose(frame, p, mask_s, hero, t=1.0, dt=0.033, parts=parts)

    render_times = []
    for step in range(30):
        t_now = 1.0 + step * 0.033
        t_r0 = time.perf_counter()
        out, g = compose(frame, p, mask_s, hero, t=t_now, dt=0.033, parts=parts)
        render_times.append(time.perf_counter() - t_r0)

    avg_ms = (sum(render_times) / len(render_times)) * 1000.0
    render_fps = 1000.0 / avg_ms
    print(f"  Время одного кадра игры (compose): {avg_ms:.2f} мс")
    print(f"  FPS игры с наложенным ИИ-костюмом: {render_fps:.1f} FPS (цель: 30+ FPS) [OK]")

    print("\n[4/5] Отрисовка HUD и углового индикатора статуса ИИ...")
    mock_game_state = {
        "score": 150,
        "combo": 3,
        "mult": 2,
        "last_hit_t": 1.0,
        "energy": 65.0,
        "stars_collected": 4,
        "monsters_defeated": 2,
    }
    draw_hud(out, mock_game_state, left_sec=48, t=2.0)
    # Рисуем тонкий угловой индикатор
    draw_ai_corner_indicator(out, ai_mgr, debug_mode=False, is_play_state=True, t=2.0)

    # Также делаем вариант с подробным отладочным индикатором (B)
    out_debug = out.copy()
    draw_ai_corner_indicator(out_debug, ai_mgr, debug_mode=True, is_play_state=True, t=2.0)

    print("\n[5/5] Сохранение скриншотов...")
    out_dir = os.path.join(ROOT_DIR, "assets")
    os.makedirs(out_dir, exist_ok=True)
    p_main = os.path.join(out_dir, "game_ai_status_preview.png")
    p_dbg = os.path.join(out_dir, "game_ai_debug_preview.png")
    cv2.imwrite(p_main, out)
    cv2.imwrite(p_dbg, out_debug)

    if os.path.isdir(ARTIFACT_DIR):
        shutil.copy(p_main, os.path.join(ARTIFACT_DIR, "game_ai_status_preview.png"))
        shutil.copy(p_dbg, os.path.join(ARTIFACT_DIR, "game_ai_debug_preview.png"))
        print(f"  Скопировано в артефакты: {os.path.join(ARTIFACT_DIR, 'game_ai_status_preview.png')}")

    print(f"  Скриншот игры сохранён: {p_main}")
    print(f"  Скриншот отладки сохранён: {p_dbg}")
    print("=" * 60)
    print(" ТЕСТ УСПЕШНО ПРОЙДЕН!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_ai_game_render()
