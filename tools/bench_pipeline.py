"""
Замер конвейера камеры, трекинга и рендера (tools/bench_pipeline.py) для «Сиқырлы айна».
Выводит выбранный бэкенд камеры (DSHOW / MSMF), реальный аппаратный FPS камеры (цель: 30 FPS),
FPS потока MediaPipe Pose и FPS рендера.
"""
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tracking import Cam, PoseWorker


def main():
    print("[BENCH_PIPELINE] Инициализация камеры (автовыбор DSHOW vs MSMF, 1280x720, MJPG, 30 FPS)...")
    cam = Cam()
    cam.start()
    pose_worker = PoseWorker(cam)
    pose_worker.start()

    # Ждём 1-й кадр
    t_wait = time.perf_counter()
    while cam.latest()[0] is None and (time.perf_counter() - t_wait) < 5.0:
        time.sleep(0.01)

    print(f"[BENCH_PIPELINE] Выбран бэкенд камеры: {cam.backend_name} (стартовый замер: {cam.init_fps:.1f} FPS)")
    print("[BENCH_PIPELINE] Замер реального FPS камеры и трекинга в течение 3.0 секунд...")

    t0 = time.perf_counter()
    _, _, idx0 = cam.latest()
    with pose_worker.lock:
        p0 = pose_worker.processed_count

    while (time.perf_counter() - t0) < 3.0:
        time.sleep(0.01)

    elapsed = time.perf_counter() - t0
    _, _, idx1 = cam.latest()
    with pose_worker.lock:
        p1 = pose_worker.processed_count
    real_cam_fps = (idx1 - idx0) / elapsed
    pose_fps = (p1 - p0) / elapsed

    cam.running = False
    pose_worker.running = False
    print("=" * 68)
    print(f"Бэкенд камеры      : {cam.backend_name} (1280x720, 30 FPS target)")
    print(f"Реальный FPS камеры: {real_cam_fps:.1f} FPS (сглаженный cam.cam_fps = {cam.cam_fps:.1f} FPS)")
    print(f"FPS MediaPipe Pose : {pose_fps:.1f} FPS")
    print("=" * 68)


if __name__ == "__main__":
    main()
