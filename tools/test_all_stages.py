"""
Комплексный тест всех 4 этапов «Сиқырлы айна»:
- Этап 1: Камера 30 FPS (DSHOW, MJPG, AE=0.75, Backlight=0).
- Этап 2: ИИ-превращение (ai_transform.py, отдельный поток, SD-Turbo fp16 + TAESD).
- Этап 3: Живые видео-миры (video_worlds: true) и видеоэффекты атак (fx_clips: true).
- Этап 4: Высококачественная графика героев (hero_art: true, PNG-ассеты) и игровой слой (game_layer: true).
- Сохранение итогового кадра в артефакты для комиссии.
"""
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import W, H, SW, SH
from tracking import Cam, PoseWorker, geom
from heroes import HEROES
from render import compose, get_stage_or_video_bg
from ai_transform import get_ai_manager
from game import detect_actions, game_fx, update_stars
from ui import draw_hud


def main():
    print("=" * 65)
    print("ТЕСТ ВСЕХ 4 ЭТАПОВ СИҚЫРЛЫ АЙНА (КОМИССИЯ)")
    print("=" * 65)

    # 1. Запуск потока камеры
    cam = Cam()
    cam.start()
    t_wait_c = time.perf_counter()
    while cam.latest()[0] is None and (time.perf_counter() - t_wait_c) < 5.0:
        time.sleep(0.01)

    print(f"[ЭТАП 1] Камера запущена: бэкенд={cam.backend_name}, стартовый FPS={cam.init_fps:.1f} FPS")

    # 2. Запуск потока поз MediaPipe
    pw = PoseWorker(cam)
    pw.start()

    # 3. Запуск фонового GPU-потока ИИ (Этап 2)
    print("[ЭТАП 2] Инициализация фонового GPU-потока ИИ (SD-Turbo + TAESD)...")
    ai_mgr = get_ai_manager()
    t_wait_ai = time.perf_counter()
    while not ai_mgr.ready and not ai_mgr.failed and (time.perf_counter() - t_wait_ai) < 120.0:
        time.sleep(0.1)

    if ai_mgr.ready:
        print(f"[ЭТАП 2] ИИ готов: backend=CUDA Graph, steps={ai_mgr.steps}, str={ai_mgr.strength}")
    else:
        print(f"[ЭТАП 2] Внимание: ИИ в режиме отката ({ai_mgr.error_msg})")

    # 4. Проверка видео-миров и FX-клипов (Этап 3)
    hero = HEROES[0]
    vw_path = os.path.join(config.BASE_DIR, "assets", hero["id"], "world.mp4")
    print(f"[ЭТАП 3] Проверка видео-мира: {vw_path} (существует: {os.path.exists(vw_path)})")
    for act in ("fly", "beam", "shield", "special"):
        fx_p = os.path.join(config.BASE_DIR, "assets", hero["id"], f"fx_{act}.mp4")
        print(f"  FX-клип {act:7s}: {fx_p} (существует: {os.path.exists(fx_p)})")

    # 5. Проверка PNG-ассетов героев (Этап 4)
    for part in ("armor", "back", "crown", "glove"):
        png_p = os.path.join(config.BASE_DIR, "assets", hero["id"], f"{part}.png")
        print(f"  PNG-ассет {part:7s}: {png_p} (существует: {os.path.exists(png_p)})")

    # 6. Замер игрового цикла
    parts = np.random.rand(70, 3).astype(np.float32)
    parts[:, 0] = parts[:, 0] * 2 - 1
    parts[:, 2] = 0.4 + parts[:, 2]

    S = dict(
        score=15, combo=2, mult=2, energy=65.0, start=time.perf_counter(),
        stars=[
            {"pos": [380, 220], "alive": True, "kind": "star", "vy": 55.0, "vx": 0.0},
            {"pos": [880, 260], "alive": True, "kind": "monster", "col": (50, 95, 220), "r": 28, "vy": 45.0, "vx": 0.0},
        ],
        pops=[], fireworks=[], special_fx=[], act={"shield": time.perf_counter() - 0.5},
        last_fly=0.0, stars_collected=3, monsters_defeated=1, boss=None,
    )

    print("\n[ТЕСТ] Прогон игрового конвейера 3.5 секунды...")
    t_start = time.perf_counter()
    n_frames = 0
    final_canvas = None

    while (time.perf_counter() - t_start) < 3.5:
        fr, fts, idx = cam.latest()
        if fr is None:
            time.sleep(0.002)
            continue
        now = time.perf_counter()
        dt = 0.016

        with pw.lock:
            r = pw.result
        if r is not None:
            p = r["p"].copy()
            mask_s = np.clip((cv2.GaussianBlur(r["mask"], (5, 5), 0) - 0.35) * 4, 0, 1)
        else:
            p = np.zeros((33, 3), dtype=np.float32)
            p[:, 2] = 0.99
            p[0, :2] = (640, 205); p[7, :2] = (595, 205); p[8, :2] = (685, 205)
            p[11, :2] = (530, 330); p[12, :2] = (750, 330)
            p[13, :2] = (480, 440); p[14, :2] = (800, 440)
            p[15, :2] = (460, 540); p[16, :2] = (820, 540)
            p[23, :2] = (560, 560); p[24, :2] = (720, 560)
            mask_s = np.zeros((SH, SW), dtype=np.float32)
            cv2.circle(mask_s, (SW // 2, 65), 24, 1.0, -1)
            cv2.ellipse(mask_s, (SW // 2, 145), (52, 68), 0, 0, 360, 1.0, -1)

        out, g = compose(fr, p, mask_s, hero, now, dt, parts, is_flying=False)
        game_fx(out, p, g, hero, now, dt, S)
        if config.GAME_LAYER:
            update_stars(out, S, dt)
            draw_hud(out, S, 45.0, now)

        n_frames += 1
        final_canvas = out

    total_time = time.perf_counter() - t_start
    game_fps = n_frames / total_time
    print(f"\n=================================================================")
    print(f"РЕЗУЛЬТАТЫ ЗАМЕРА ВСЕХ 4 ЭТАПОВ:")
    print(f"  Камера (DSHOW, 1280x720)       : {cam.cam_fps:.1f} FPS")
    print(f"  Рендер игры compose + game_fx  : {game_fps:.1f} FPS (цель: 30+ FPS)")
    print(f"  ИИ-поток (SD-Turbo + TAESD)    : {ai_mgr.ai_fps:.2f} FPS ({ai_mgr.ai_ms:.1f} мс)")
    print(f"  Использование VRAM GPU         : {ai_mgr.vram_gb:.2f} ГБ (лимит: 5.0 ГБ)")
    print(f"=================================================================")

    preview_path = r"C:\Users\Acer Nitro\.gemini\antigravity\brain\a7feaa77-e574-4eb1-a10c-00490c79dbd7\full_game_preview.png"
    if final_canvas is not None:
        cv2.imwrite(preview_path, final_canvas)
        print(f"[ИТОГ] Скриншот игры сохранён в {preview_path}")

    ai_mgr.stop()
    cam.running = pw.running = False


if __name__ == "__main__":
    main()
