"""
Скрипт воспроизведения и отладки скелета без камеры (tools/replay.py) для «Сиқырлы айна».
Читает записанные координаты скелета (ТОЛЬКО цифры 33x3, без фото и видео) из debug_pose.json
и прогоняет их через умный курсор руки (CURSOR), маппинг зоны досягаемости, магнитные карточки (+25% hitbox)
и детектор жестов (evaluate_pose_scores / detect_actions).

Запуск:
    python tools/replay.py                  # интерактивное окно OpenCV (Space - пауза, A/D - шаг, Q - выход)
    python tools/replay.py --headless       # автоматическая проверка и отчёт в консоль без окна
"""
import os
import sys
import json
import time
import math
import argparse
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from config import W, H, AA
from tracking import CURSOR, update_height_calibration, draw_hand_skeleton_feedback, draw_cursor_with_ring
from heroes import HEROES
from game import detect_actions
from ui import card_rect, render_select, draw_debug_overlay, DEBUG_POSE_PATH, fake_pose


def generate_sample_poses(num_frames=60):
    """Генерирует тестовую последовательность поз (движение рукой по карточкам, полёт, луч, щит), если debug_pose.json ещё пуст."""
    frames = []
    for i in range(num_frames):
        t = i * 0.05
        p = fake_pose(W // 2, int(H * 0.56), 135.0, t)
        # Добавляем пальцы (17..20)
        p[19, :2] = p[15, :2] + np.array([-14.0, -22.0], np.float32)
        p[20, :2] = p[16, :2] + np.array([14.0, -22.0], np.float32)
        p[17, :2] = p[15, :2] + np.array([-8.0, -18.0], np.float32)
        p[18, :2] = p[16, :2] + np.array([8.0, -18.0], np.float32)

        if i < 20:
            # Наведение правой руки на карточки героев
            ang = (i / 20.0) * math.pi
            p[14, :2] = p[12, :2] + np.array([math.cos(ang) * 75.0, -55.0], np.float32)
            p[16, :2] = p[14, :2] + np.array([math.cos(ang) * 85.0, -65.0 - math.sin(ang) * 45.0], np.float32)
            p[20, :2] = p[16, :2] + np.array([math.cos(ang) * 26.0, -24.0], np.float32)
        elif i < 40:
            # Полёт: обе руки вверх над головой
            p[13, :2] = p[11, :2] + np.array([-45.0, -75.0], np.float32)
            p[15, :2] = p[13, :2] + np.array([-25.0, -85.0], np.float32)
            p[19, :2] = p[15, :2] + np.array([-10.0, -25.0], np.float32)
            p[14, :2] = p[12, :2] + np.array([45.0, -75.0], np.float32)
            p[16, :2] = p[14, :2] + np.array([25.0, -85.0], np.float32)
            p[20, :2] = p[16, :2] + np.array([10.0, -25.0], np.float32)
        else:
            # Луч: правая рука вытянута в сторону
            p[14, :2] = p[12, :2] + np.array([90.0, 0.0], np.float32)
            p[16, :2] = p[14, :2] + np.array([95.0, 0.0], np.float32)
            p[20, :2] = p[16, :2] + np.array([28.0, 0.0], np.float32)

        frames.append({
            "ts": round(t, 3),
            "state": "select" if i < 20 else "play",
            "landmarks": [[round(float(c), 2) for c in pt] for pt in p.tolist()],
        })
    return frames


def load_or_create_debug_poses(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            frames = data.get("frames", data) if isinstance(data, dict) else data
            if frames:
                return frames
        except Exception as e:
            print(f"[WARN] Ошибка чтения {path}: {e}")
    frames = generate_sample_poses(60)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "resolution": [W, H], "frames": frames}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Создан демонстрационный файл {path} ({len(frames)} кадров скелета без видео).")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Проигрыватель записанных поз debug_pose.json")
    parser.add_argument("--file", default=DEBUG_POSE_PATH, help="Путь к debug_pose.json")
    parser.add_argument("--headless", action="store_true", help="Прогнать все кадры без GUI и вывести отчёт")
    args = parser.parse_args()

    frames = load_or_create_debug_poses(args.file)
    print(f"[REPLAY] Загружено кадров скелета: {len(frames)} из {args.file}")

    S = dict(
        state="select", hover=-1, hover_prog=0.0, debug_mode=True,
        debug_saved_count=len(frames), pose_scores={}, pose_hyst=set(),
    )
    magnet_targets = [{"id": i, "rect": card_rect(i)} for i in range(len(HEROES))]
    blank_bg = np.full((H, W, 3), (28, 34, 48), dtype=np.uint8)

    if args.headless:
        t_sim = 1.0
        for idx, fr in enumerate(frames):
            p = np.array(fr["landmarks"], dtype=np.float32)
            update_height_calibration(blank_bg, p, S, 0.033)
            cur_pt, side = CURSOR.update(p, S, t_sim, magnet_targets=magnet_targets)
            acts = detect_actions(p, S, 0.033)
            out = render_select(blank_bg, cur_pt, 0, 0.5, t_sim, p=p)
            draw_debug_overlay(out, p, p, S, 60.0, 8.5, magnet_targets, t_sim)
            t_sim += 0.033
        print(f"[REPLAY OK] Прогнано {len(frames)} кадров. Последний курсор: {CURSOR.final_cursor}, Пороги: {S.get('pose_scores')}")
        return

    win = "Magic Mirror - Pose Replay [Space: Pause, A/D: Step, Q: Quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    idx = 0
    paused = False
    while True:
        fr = frames[idx]
        p = np.array(fr["landmarks"], dtype=np.float32)
        now = time.perf_counter()
        update_height_calibration(blank_bg, p, S, 0.033)
        cur_pt, side = CURSOR.update(p, S, now, magnet_targets=magnet_targets)
        acts = detect_actions(p, S, 0.033)

        out = render_select(blank_bg, cur_pt, S.get("hover", -1), 0.35, now, p=p)
        draw_debug_overlay(out, p, p, S, 60.0, 0.0, magnet_targets, now)
        cv2.putText(out, f"REPLAY FRAME {idx + 1}/{len(frames)} [Space: Pause | A/D: Step | Q: Quit]",
                    (W // 2 - 260, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 255, 140), 2, AA)
        cv2.imshow(win, out)

        key = cv2.waitKey(30 if not paused else 50) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('d'), 83):
            idx = (idx + 1) % len(frames)
        elif key in (ord('a'), 81):
            idx = (idx - 1) % len(frames)
        elif not paused:
            idx = (idx + 1) % len(frames)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
