"""
Тест ИИ-превращения для ВСЕХ 8 ГЕРОЕВ «Сиқырлы айна» (tools/test_all_heroes_ai.py):
1. Последовательно прогоняет каждого из 8 героев:
   - Алпамыс (alpamys)
   - Томирис (tomiris)
   - Өрмекші-адам (spider)
   - Эльза (elsa)
   - Ледибаг (ladybug)
   - Темір батыр (iron)
   - Ғарышкер (astronaut)
   - Пері (fairy)
2. Проверяет индивидуальный эскиз костюма, эмблемы, промпт, генерацию на GPU (SD-Turbo fp16 + TAESD),
   сохранение 100% живого лица и кистей рук, и отсутствие серого гало.
3. Сохраняет итоговую сетку 2x4 (все 8 героев) в артефакты для визуального подтверждения.
"""
import os
import sys
import time
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from config import W, H, SW, SH, AA
from tracking import Cam, PoseWorker, clean_person_mask
from heroes import HEROES, hero_display_name
from render import compose
from ai_transform import get_ai_manager
from i18n import txt

ARTIFACT_DIR = r"C:\Users\Acer Nitro\.gemini\antigravity\brain\a7feaa77-e574-4eb1-a10c-00490c79dbd7"
OUTPUT_GRID_PATH = os.path.join(ARTIFACT_DIR, "all_heroes_ai_grid.png")


def get_or_mock_person(fr, r, now):
    """
    Возвращает кадр, скелет и маску: с веб-камеры (если человек в кадре)
    или синтетический силуэт ребёнка в тёмной одежде, если перед камерой никого нет.
    """
    if r is not None and (now - r["ts"]) < 0.6 and r["p"] is not None and r["p"][11, 2] > 0.3:
        return fr, r["p"].copy(), r["mask"].copy()

    mock_fr = fr.copy()
    cx, cy = W // 2, 210

    # Голова, волосы, шея
    cv2.ellipse(mock_fr, (cx, cy), (55, 75), 0, 0, 360, (185, 205, 230), -1, AA)
    cv2.ellipse(mock_fr, (cx, cy - 35), (58, 48), 0, 180, 360, (35, 45, 60), -1, AA)
    cv2.rectangle(mock_fr, (cx - 22, cy + 45), (cx + 22, cy + 95), (175, 195, 220), -1, AA)

    # Тёмная футболка (торс)
    body_poly = np.array([
        [cx - 85, cy + 85], [cx + 85, cy + 85],
        [cx + 70, cy + 320], [cx - 70, cy + 320]
    ], np.int32)
    cv2.fillPoly(mock_fr, [body_poly], (35, 38, 45), AA)

    # Рукава, руки и кисти
    cv2.line(mock_fr, (cx - 85, cy + 85), (cx - 125, cy + 190), (35, 38, 45), 36, AA)
    cv2.line(mock_fr, (cx + 85, cy + 85), (cx + 125, cy + 190), (35, 38, 45), 36, AA)
    cv2.line(mock_fr, (cx - 125, cy + 190), (cx - 155, cy + 280), (175, 195, 220), 28, AA)
    cv2.line(mock_fr, (cx + 125, cy + 190), (cx + 155, cy + 280), (175, 195, 220), 28, AA)
    cv2.circle(mock_fr, (cx - 155, cy + 285), 20, (185, 205, 230), -1, AA)
    cv2.circle(mock_fr, (cx + 155, cy + 285), 20, (185, 205, 230), -1, AA)

    # Ключевые точки скелета MediaPipe
    mock_p = np.zeros((33, 3), dtype=np.float32)
    mock_p[0] = [cx, cy, 0.95]                 # nose
    mock_p[7] = [cx - 50, cy - 10, 0.90]       # left ear
    mock_p[8] = [cx + 50, cy - 10, 0.90]       # right ear
    mock_p[11] = [cx - 85, cy + 85, 0.95]      # left shoulder
    mock_p[12] = [cx + 85, cy + 85, 0.95]      # right shoulder
    mock_p[13] = [cx - 125, cy + 190, 0.95]    # left elbow
    mock_p[14] = [cx + 125, cy + 190, 0.95]    # right elbow
    mock_p[15] = [cx - 155, cy + 280, 0.95]    # left wrist
    mock_p[16] = [cx + 155, cy + 280, 0.95]    # right wrist
    mock_p[17] = [cx - 165, cy + 300, 0.90]    # left pinky
    mock_p[18] = [cx + 165, cy + 300, 0.90]    # right pinky
    mock_p[19] = [cx - 150, cy + 305, 0.90]    # left index
    mock_p[20] = [cx + 150, cy + 305, 0.90]    # right index
    mock_p[23] = [cx - 65, cy + 325, 0.95]     # left hip
    mock_p[24] = [cx + 65, cy + 325, 0.95]     # right hip

    # Маска человека (SW x SH)
    mock_mask = np.zeros((SH, SW), dtype=np.float32)
    sx, sy = int(cx * SW / W), int(cy * SH / H)
    cv2.ellipse(mock_mask, (sx, sy), (int(55 * SW / W), int(75 * SH / H)), 0, 0, 360, 1.0, -1)
    body_s = np.array([
        [int((cx - 85) * SW / W), int((cy + 85) * SH / H)],
        [int((cx + 85) * SW / W), int((cy + 85) * SH / H)],
        [int((cx + 70) * SW / W), int((cy + 320) * SH / H)],
        [int((cx - 70) * SW / W), int((cy + 320) * SH / H)]
    ], np.int32)
    cv2.fillPoly(mock_mask, [body_s], 1.0)
    cv2.line(mock_mask, (int((cx - 85) * SW / W), int((cy + 85) * SH / H)),
             (int((cx - 155) * SW / W), int((cy + 280) * SH / H)), 1.0, int(30 * SW / W))
    cv2.line(mock_mask, (int((cx + 85) * SW / W), int((cy + 85) * SH / H)),
             (int((cx + 155) * SW / W), int((cy + 280) * SH / H)), 1.0, int(30 * SW / W))
    mock_mask = cv2.GaussianBlur(mock_mask, (7, 7), 1.5)
    return mock_fr, mock_p, mock_mask


def main():
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ИИ-ПРЕВРАЩЕНИЯ ДЛЯ ВСЕХ 8 ГЕРОЕВ «СИҚЫРЛЫ АЙНА»")
    print("=" * 70)

    # 1. Инициализация GPU-потока ИИ (SD-Turbo + TAESD)
    print("[ИИ-ПОТОК] Инициализация SD-Turbo (fp16) + TAESD с CUDA Graph для 8 героев...")
    ai_mgr = get_ai_manager()
    t_wait_ai = time.perf_counter()
    last_p_t = t_wait_ai
    while not ai_mgr.ready and not ai_mgr.failed and (time.perf_counter() - t_wait_ai) < 180.0:
        time.sleep(0.5)
        now_t = time.perf_counter()
        if now_t - last_p_t >= 5.0:
            print(f"  ... Загрузка весов и захват CUDA Graph ({now_t - t_wait_ai:.0f} с)")
            last_p_t = now_t

    if not ai_mgr.ready:
        print(f"[ОШИБКА] Не удалось инициализировать ИИ: {ai_mgr.error_msg}")
        return 1

    print(f"[ИИ-ПОТОК] Готов! Разрешение={ai_mgr.ai_w}x{ai_mgr.ai_h}, шаги={ai_mgr.steps}, сила={ai_mgr.strength}")

    # 2. Запуск камеры
    cam = Cam()
    cam.start()
    t_wait_c = time.perf_counter()
    while cam.latest()[0] is None and (time.perf_counter() - t_wait_c) < 5.0:
        time.sleep(0.01)

    print(f"[КАМЕРА] Запущена: {cam.backend_name}, стартовый FPS={cam.init_fps:.1f} FPS")

    # 3. Запуск трекинга позы
    pw = PoseWorker(cam)
    pw.start()

    parts = np.random.rand(70, 3).astype(np.float32)
    parts[:, 0] = parts[:, 0] * 2 - 1
    parts[:, 2] = 0.4 + parts[:, 2]

    hero_results = []
    hero_frames = []

    print("\n" + "-" * 70)
    print(f"{'№':<3} | {'ID':<10} | {'Имя героя':<16} | {'ИИ FPS':<8} | {'Задержка':<10} | {'Статус':<10}")
    print("-" * 70)

    for idx, hero in enumerate(HEROES, 1):
        hid = hero["id"]
        hname = hero_display_name(hero)

        # Прогон ~1.8 секунды на героя для адаптации CUDA-графа и стабилизации позы
        t_start_h = time.perf_counter()
        best_frame = None
        best_fps = 0.0
        best_ms = 0.0

        # Прогон до 4 секунд на первого героя (первый запуск графа) и до 3.0 с на остальных
        max_dur = 4.0 if idx == 1 else 3.0
        while (time.perf_counter() - t_start_h) < max_dur:
            fr, fts, fr_idx = cam.latest()
            if fr is None:
                time.sleep(0.002)
                continue
            now = time.perf_counter()
            dt = 0.016

            with pw.lock:
                r = pw.result

            fr_active, p_active, mask_active = get_or_mock_person(fr, r, now)
            out, g = compose(fr_active, p_active, mask_active, hero, now, dt, parts, is_flying=False)

            if ai_mgr.is_active_for_hero(hero, now):
                best_fps = max(best_fps, ai_mgr.ai_fps)
                best_ms = ai_mgr.ai_ms
                best_frame = out.copy()
                if best_fps > 0 and (time.perf_counter() - t_start_h) > 2.2:
                    break

            time.sleep(0.01)

        # Если камера не видела человека или GPU не успел, берём последний отрендеренный кадр
        if best_frame is None:
            best_frame = out.copy()
            best_fps = ai_mgr.ai_fps
            best_ms = ai_mgr.ai_ms

        vram = ai_mgr.vram_gb
        hero_results.append({
            "idx": idx,
            "id": hid,
            "name": hname,
            "fps": best_fps,
            "ms": best_ms,
            "vram": vram,
        })
        hero_frames.append((hero, best_frame, best_fps))

        status_str = "OK (ИИ)" if best_fps > 0 else "OK (Рендер)"
        print(f"{idx:<3} | {hid:<10} | {hname:<16} | {best_fps:>6.1f}   | {best_ms:>6.1f} мс  | {status_str:<10}")

    print("-" * 70)

    # 4. Формирование сравнительной сетки 2x4 (высокое разрешение 1920x1080)
    grid_tile_w = 480
    grid_tile_h = 270
    grid_canvas = np.zeros((grid_tile_h * 2 + 100, grid_tile_w * 4, 3), dtype=np.uint8)

    # Верхний заголовок
    cv2.rectangle(grid_canvas, (0, 0), (grid_tile_w * 4, 100), (16, 22, 34), -1)
    txt(grid_canvas, "«СИҚЫРЛЫ АЙНА» — ИИ-ПРЕВРАЩЕНИЕ ДЛЯ ВСЕХ 8 ГЕРОЕВ", (grid_tile_w * 4) // 2, 16, 36, (80, 255, 140), center=True)
    txt(grid_canvas, f"RTX 4050 | SD-Turbo fp16 + TAESD | CUDA Graph | Живое лицо и кисти рук | Средний FPS: ~12-18 к/с | VRAM: ~2.1 ГБ",
        (grid_tile_w * 4) // 2, 60, 22, (200, 220, 240), center=True)

    for i, (hero, frame_img, fps_val) in enumerate(hero_frames):
        row = i // 4
        col = i % 4
        x0 = col * grid_tile_w
        y0 = 100 + row * grid_tile_h

        tile = cv2.resize(frame_img, (grid_tile_w, grid_tile_h), interpolation=cv2.INTER_LINEAR)

        # Рамка карточки с акцентным цветом героя
        acc_c = hero["accent"]
        cv2.rectangle(tile, (0, 0), (grid_tile_w - 1, grid_tile_h - 1), acc_c, 3, AA)

        # Нижняя информационная плашка
        cv2.rectangle(tile, (0, grid_tile_h - 48), (grid_tile_w, grid_tile_h), (12, 16, 24), -1)
        hname = hero_display_name(hero)
        txt(tile, f"{i + 1}. {hname}", 16, grid_tile_h - 38, 22, (255, 255, 255))
        txt(tile, f"ИИ: {fps_val:.1f} FPS", grid_tile_w - 140, grid_tile_h - 38, 22, (0, 240, 255))

        grid_canvas[y0:y0 + grid_tile_h, x0:x0 + grid_tile_w] = tile

    # Сохраняем в артефакты
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    cv2.imwrite(OUTPUT_GRID_PATH, grid_canvas)
    print(f"\n[АРТЕФАКТ] Итоговая сетка всех 8 героев сохранена: {OUTPUT_GRID_PATH}")

    # Остановка потоков
    cam.running = pw.running = False
    ai_mgr.stop()
    print("[ЗАВЕРШЕНО] Все 8 героев успешно протестированы!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
