"""
tools/test_new_assets.py: Тест и замер FPS нового конвейера импортированных ассетов:
1. Бесшовный цикл вперёд-назад (ping-pong loop) видео-мира assets/<hero_id>/world.mp4.
2. Общие FX-клипы из assets/fx/ (beam, shield, burst, fireworks) с перекраской HSV и cv2.add.
3. Проверка hero_art и отключённых по умолчанию плоских процедурных доспехов.
4. Детализированный эскиз ИИ со складками плаща, градиентами кирасы и орнаментом «қошқар мүйіз».
5. Замер FPS (цель: 30+ FPS) и сохранение скриншота.
"""
import os
import sys
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from config import W, H, AA
import heroes
from heroes import HEROES
import render
from render import get_stage_or_video_bg, compose
import game
from game import get_fx_manager, overlay_fx_clip
import ai_transform
from ai_transform import build_costume_sketch, HERO_SKETCH_PALETTES


def make_synthetic_pose():
    """Синтетический скелет ребёнка по центру кадра."""
    p = np.zeros((33, 4), dtype=np.float32)
    # Нос
    p[0] = [640.0, 200.0, 0.95, 0.95]
    # Глаза и уши
    p[1] = [625.0, 190.0, 0.95, 0.95]
    p[2] = [615.0, 190.0, 0.95, 0.95]
    p[3] = [605.0, 190.0, 0.95, 0.95]
    p[4] = [655.0, 190.0, 0.95, 0.95]
    p[5] = [665.0, 190.0, 0.95, 0.95]
    p[6] = [675.0, 190.0, 0.95, 0.95]
    p[7] = [585.0, 200.0, 0.95, 0.95]
    p[8] = [695.0, 200.0, 0.95, 0.95]
    # Плечи
    p[11] = [530.0, 290.0, 0.95, 0.95]
    p[12] = [750.0, 290.0, 0.95, 0.95]
    # Локти
    p[13] = [460.0, 390.0, 0.90, 0.90]
    p[14] = [820.0, 390.0, 0.90, 0.90]
    # Запястья
    p[15] = [410.0, 350.0, 0.90, 0.90]
    p[16] = [870.0, 350.0, 0.90, 0.90]
    # Бёдра
    p[23] = [560.0, 520.0, 0.92, 0.92]
    p[24] = [720.0, 520.0, 0.92, 0.92]
    # Колени
    p[25] = [570.0, 640.0, 0.90, 0.90]
    p[26] = [710.0, 640.0, 0.90, 0.90]
    return p


def main():
    print("=" * 70)
    print("  ТЕСТИРОВАНИЕ НОВОГО КОНВЕЙЕРА АССЕТОВ «СИҚЫРЛЫ АЙНА»")
    print("=" * 70)

    hero = HEROES[0]  # alpamys
    print(f"  Выбран герой: {hero['name'][0]} ({hero['id']})")
    print(f"  Флаги: VIDEO_WORLDS={config.VIDEO_WORLDS}, FX_CLIPS={config.FX_CLIPS}, HERO_ART={config.HERO_ART}, PROCEDURAL_ARMOR={config.PROCEDURAL_ARMOR}")

    p = make_synthetic_pose()
    g = render.geom(p)

    # Синтетический кадр с веб-камеры и маска человека
    frame_cam = np.full((H, W, 3), (35, 30, 40), dtype=np.uint8)
    cv2.circle(frame_cam, (640, 210), 65, (190, 160, 140), -1, AA)  # лицо
    cv2.rectangle(frame_cam, (520, 280), (760, 560), (30, 25, 35), -1, AA)  # тёмная одежда тела

    mask_person = np.zeros((H, W), dtype=np.float32)
    cv2.circle(mask_person, (640, 210), 75, 1.0, -1)
    cv2.rectangle(mask_person, (510, 280), (770, 660), 1.0, -1)
    mask_s = cv2.resize(mask_person, (config.SW, config.SH), interpolation=cv2.INTER_LINEAR)

    # 1. Проверяем генерацию детализированного эскиза ИИ
    t_start_sk = time.perf_counter()
    sketch, face_hands_mask = build_costume_sketch(frame_cam, mask_person, p, hero)
    t_sketch_ms = (time.perf_counter() - t_start_sk) * 1000.0
    print(f"  [SKETCH OK] Построен детализированный эскиз за {t_sketch_ms:.2f} мс")

    # 2. Проверяем загрузку видео-мира (ping-pong loop)
    mgr = render.get_world_manager()
    # Даём фоновому потоку декодировать кадры
    time.sleep(0.3)
    bg_frame = get_stage_or_video_bg(hero, 0.5)
    print(f"  [WORLD OK] Видео-мир получен, форма кадра: {bg_frame.shape}, тип: {bg_frame.dtype}")

    # 3. Проверяем загрузку и перекраску общих FX
    fx_mgr = get_fx_manager()
    fx_shield = fx_mgr.get_frame(hero, "shield", 0.5)
    fx_beam = fx_mgr.get_frame(hero, "beam", 0.5)
    fx_burst = fx_mgr.get_frame(hero, "burst", 0.5)
    fx_fw = fx_mgr.get_frame(hero, "fireworks", 0.5)
    print(f"  [FX LOAD OK] shield: {fx_shield is not None}, beam: {fx_beam is not None}, burst: {fx_burst is not None}, fireworks: {fx_fw is not None}")

    # 4. Замер FPS полного цикла рендеринга (120 кадров)
    print("  [BENCH] Запуск замера FPS рендера (120 кадров, 1280x720)...")
    config.AI_TRANSFORM = "off"  # В тесте проверяем основной рендер-конвейер без ожидания фонового воркера
    parts = np.random.rand(40, 3).astype(np.float32)

    # Прогрев кэшей (5 кадров)
    for w_i in range(5):
        out_w, _ = compose(frame_cam, p, mask_s, hero, w_i / 30.0, 0.033, parts, is_flying=False)
        c_sh = (int(g["hip"][0] * 0.4 + g["neck"][0] * 0.6), int(g["hip"][1] * 0.4 + g["neck"][1] * 0.6))
        overlay_fx_clip(out_w, fx_shield, c_sh, g["th"] * 2.2, g["th"] * 2.2)
        p_bm = (int(g["rw"][0] + 160), int(g["rw"][1]))
        overlay_fx_clip(out_w, fx_beam, p_bm, g["sw"] * 2.4, g["sw"] * 1.6)

    times = []

    for i in range(120):
        t_sim = i / 30.0
        dt_sim = 1.0 / 30.0
        t0 = time.perf_counter()

        # Полный композит игры: видео-мир + проектор + голограмма героя
        out, _ = compose(frame_cam, p, mask_s, hero, t_sim, dt_sim, parts, is_flying=False)

        # Наложение активного щита и лазерного луча по скелету с аддитивным смешиванием
        c_shield = (int(g["hip"][0] * 0.4 + g["neck"][0] * 0.6), int(g["hip"][1] * 0.4 + g["neck"][1] * 0.6))
        overlay_fx_clip(out, fx_shield, c_shield, g["th"] * 2.2, g["th"] * 2.2)

        p_beam = (int(g["rw"][0] + 160), int(g["rw"][1]))
        overlay_fx_clip(out, fx_beam, p_beam, g["sw"] * 2.4, g["sw"] * 1.6)

        times.append(time.perf_counter() - t0)

    avg_time_s = sum(times) / len(times)
    fps_render = 1.0 / avg_time_s if avg_time_s > 0 else 0
    p95_time_ms = float(np.percentile(times, 95)) * 1000.0

    print("=" * 70)
    print(f"  РЕЗУЛЬТАТЫ ЗАМЕРА ПРОИЗВОДИТЕЛЬНОСТИ:")
    print(f"  Среднее время кадра : {avg_time_s * 1000.0:.2f} мс")
    print(f"  95-й процентиль (P95): {p95_time_ms:.2f} мс")
    print(f"  Рендер FPS          : {fps_render:.1f} FPS (цель: 30+ FPS) -> {'[В НОРМЕ]' if fps_render >= 30.0 else '[НИЖЕ НОРМЫ]'}")
    print("=" * 70)

    # 5. Отрисовываем информационную панель на финальном кадре
    hud_bg = out.copy()
    cv2.rectangle(hud_bg, (20, 20), (460, 160), (20, 20, 25), -1)
    out = cv2.addWeighted(out, 0.35, hud_bg, 0.65, 0)
    cv2.rectangle(out, (20, 20), (460, 160), (40, 215, 255), 2, AA)
    cv2.putText(out, f"HERO: {hero['name'][0]} | NEW ASSETS V2", (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, AA)
    cv2.putText(out, f"RENDER FPS: {fps_render:.1f} FPS (Target: 30+)", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 255, 120), 2, AA)
    cv2.putText(out, f"FRAME LATENCY: {avg_time_s * 1000.0:.1f} ms (P95: {p95_time_ms:.1f} ms)", (35, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 220, 100), 2, AA)
    cv2.putText(out, "WORLD: 1280x720 30fps ping-pong | FX: cv2.add HSV", (35, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, AA)

    # Сохраняем скриншот финального композита и эскиза
    artifact_dir = r"C:\Users\Acer Nitro\.gemini\antigravity\brain\a7feaa77-e574-4eb1-a10c-00490c79dbd7"
    out_path = os.path.join(artifact_dir, "new_assets_pipeline_preview.png")
    cv2.imwrite(out_path, out)
    print(f"  [SAVED PREVIEW] Скриншот сохранён: {out_path}")

    # Сохраняем также сравнение: Эскиз vs Финальный рендер
    sk_thumb = cv2.resize(sketch, (640, 360))
    rn_thumb = cv2.resize(out, (640, 360))
    grid = np.hstack([sk_thumb, rn_thumb])
    grid_path = os.path.join(artifact_dir, "sketch_and_render_comparison.png")
    cv2.imwrite(grid_path, grid)
    print(f"  [SAVED COMPARISON] Сравнение сохранено: {grid_path}")


if __name__ == "__main__":
    main()
