"""
tools/import_assets.py: Импорт и стандартизация видео-ассетов для «Сиқырлы айна» (Magic Mirror v2).

1. Берёт сырые видео из:
   - raw/worlds/<hero_id>.mp4 (или .mov, .avi, .webm)
   - raw/fx/<name>.mp4 (fireworks, beam, shield, burst)
2. Конвертирует:
   - Миры: разрешение 1280x720, 30 FPS, кодек H.264/mp4v, обрезка ровно до 8 секунд (240 кадров),
     сохранение в assets/<hero_id>/world.mp4 (с сохранением резервных процедурных копий).
   - Эффекты: 30 FPS, гарантированный чёрный фон (для идеального cv2.add), сохранение в assets/fx/<name>.mp4.
3. При отсутствии файлов в raw/ с флагом --gen-samples генерирует кинематографичные исходники
   для всех 8 миров и 4 общих эффектов.
"""
import os
import sys
import math
import argparse
import cv2
import numpy as np

# Добавляем корневую директорию проекта в sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

HERO_IDS = [
    "alpamys", "tomiris", "spider", "elsa",
    "ladybug", "iron", "astronaut", "fairy"
]

FX_NAMES = ["fireworks", "beam", "shield", "burst"]

W_WORLD = 1280
H_WORLD = 720
FPS_TARGET = 30.0
MAX_WORLD_SEC = 8.0
MAX_WORLD_FRAMES = int(FPS_TARGET * MAX_WORLD_SEC)  # 240 кадров

FOURCC_MP4V = cv2.VideoWriter_fourcc(*'mp4v')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_raw_file(folder, base_name):
    """Ищет файл base_name с популярными видео-расширениями."""
    extensions = [".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"]
    for ext in extensions:
        candidate = os.path.join(folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def convert_video_world(src_path, dst_path, max_frames=MAX_WORLD_FRAMES):
    """Конвертирует сырой видео-мир в 1280x720, 30 FPS, до 8 секунд."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"  [ERROR] Не удалось открыть: {src_path}")
        return False

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ensure_dir(os.path.dirname(dst_path))
    writer = cv2.VideoWriter(dst_path, FOURCC_MP4V, FPS_TARGET, (W_WORLD, H_WORLD))
    if not writer.isOpened():
        print(f"  [ERROR] Не удалось создать writer: {dst_path}")
        cap.release()
        return False

    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame.shape[1] != W_WORLD or frame.shape[0] != H_WORLD:
            frame = cv2.resize(frame, (W_WORLD, H_WORLD), interpolation=cv2.INTER_LINEAR)

        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    sz_kb = os.path.getsize(dst_path) / 1024.0 if os.path.exists(dst_path) else 0
    duration_s = written / FPS_TARGET
    print(f"  [WORLD OK] {os.path.basename(src_path)} -> {dst_path} | {written} кадров ({duration_s:.1f}с, {sz_kb:.1f} KB)")
    return written > 0


def convert_video_fx(src_path, dst_path, target_size=(480, 480), max_sec=3.0):
    """Конвертирует сырой эффект FX на чёрном фоне в 30 FPS assets/fx/<name>.mp4."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"  [ERROR] Не удалось открыть FX: {src_path}")
        return False

    max_frames = int(max_sec * FPS_TARGET)
    ensure_dir(os.path.dirname(dst_path))
    tw, th = target_size
    writer = cv2.VideoWriter(dst_path, FOURCC_MP4V, FPS_TARGET, (tw, th))
    if not writer.isOpened():
        print(f"  [ERROR] Не удалось создать writer: {dst_path}")
        cap.release()
        return False

    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame.shape[2] == 4:
            # Альфа-канал накладываем на чистый чёрный фон
            b, g, r, a = cv2.split(frame)
            alpha_f = a.astype(np.float32) / 255.0
            rgb = cv2.merge([b, g, r]).astype(np.float32)
            frame = (rgb * alpha_f[:, :, None]).astype(np.uint8)

        if frame.shape[1] != tw or frame.shape[0] != th:
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    sz_kb = os.path.getsize(dst_path) / 1024.0 if os.path.exists(dst_path) else 0
    print(f"  [FX OK] {os.path.basename(src_path)} -> {dst_path} | {written} кадров ({written/FPS_TARGET:.1f}с, {sz_kb:.1f} KB)")
    return written > 0


# ==============================================================================
# ГЕНЕРАЦИЯ КИНЕМАТОГРАФИЧНЫХ СЫРЫХ ОБРАЗЦОВ ДЛЯ RAW/ (ПРИ ИХ ОТСУТСТВИИ)
# ==============================================================================

def create_sample_raw_world(hero_id, out_path, num_frames=240):
    """Создаёт богатый 8-секундный сырой видео-мир 1280x720 30 FPS."""
    ensure_dir(os.path.dirname(out_path))
    writer = cv2.VideoWriter(out_path, FOURCC_MP4V, FPS_TARGET, (W_WORLD, H_WORLD))
    AA = cv2.LINE_AA

    for i in range(num_frames):
        t = i / FPS_TARGET
        frame = np.zeros((H_WORLD, W_WORLD, 3), dtype=np.uint8)

        if hero_id == "alpamys":
            # Золотая казахская степь, синее небо, восходящие лучи солнца, парящий беркут
            # Градиент неба
            for y in range(H_WORLD // 2):
                alpha = y / (H_WORLD * 0.5)
                col = [int(240 * (1 - alpha) + 200 * alpha), int(180 * (1 - alpha) + 140 * alpha), int(90 * (1 - alpha) + 40 * alpha)]
                frame[y, :] = col
            # Горы на горизонте
            pts_m = np.array([[0, 360], [240, 220], [480, 310], [780, 190], [1050, 290], [1280, 240], [1280, 360], [0, 360]], np.int32)
            cv2.fillPoly(frame, [pts_m], (130, 100, 90), AA)
            # Зелёно-золотая степь
            for y in range(H_WORLD // 2, H_WORLD):
                alpha = (y - H_WORLD // 2) / (H_WORLD * 0.5)
                col = [int(40 * (1 - alpha) + 20 * alpha), int(150 * (1 - alpha) + 190 * alpha), int(210 * (1 - alpha) + 160 * alpha)]
                frame[y, :] = col
            # Солнце и лучи
            sun_x = int(W_WORLD * 0.78)
            sun_y = int(H_WORLD * 0.28)
            cv2.circle(frame, (sun_x, sun_y), 55, (160, 235, 255), -1, AA)
            # Парящий беркут (плавно летит слева направо)
            bx = int((t * 90) % (W_WORLD + 160)) - 80
            by = int(140 + math.sin(t * 2.2) * 25)
            wing = math.sin(t * 5.0) * 16
            cv2.line(frame, (bx - 32, int(by - wing)), (bx, by), (25, 30, 45), 3, AA)
            cv2.line(frame, (bx + 32, int(by - wing)), (bx, by), (25, 30, 45), 3, AA)

        elif hero_id == "tomiris":
            # Закатная сакская цитадель, пурпурно-золотые облака, пылающие чаши
            for y in range(H_WORLD):
                alpha = y / float(H_WORLD)
                col = [int(150 * (1 - alpha) + 30 * alpha), int(60 * (1 - alpha) + 30 * alpha), int(180 * (1 - alpha) + 90 * alpha)]
                frame[y, :] = col
            # Каменные сакские арки
            cv2.rectangle(frame, (120, 200), (220, H_WORLD), (60, 50, 70), -1, AA)
            cv2.rectangle(frame, (W_WORLD - 220, 200), (W_WORLD - 120, H_WORLD), (60, 50, 70), -1, AA)
            cv2.line(frame, (120, 200), (W_WORLD - 120, 200), (80, 70, 95), 24, AA)
            # Огонь в чашах
            for fx in (170, W_WORLD - 170):
                f_h = int(35 + math.sin(t * 12 + fx) * 10)
                cv2.ellipse(frame, (fx, 190), (18, f_h), 0, 0, 360, (40, 160, 255), -1, AA)
                cv2.ellipse(frame, (fx, 190), (10, int(f_h * 0.65)), 0, 0, 360, (120, 240, 255), -1, AA)

        elif hero_id == "spider":
            # Ночной мегаполис, освещённые небоскрёбы, лучи прожекторов, луна
            frame[:] = (35, 20, 15)
            # Небоскрёбы
            for idx, bx in enumerate(range(0, W_WORLD, 110)):
                h_b = int(320 + math.sin(idx * 1.7) * 140)
                cv2.rectangle(frame, (bx, H_WORLD - h_b), (bx + 95, H_WORLD), (50, 35, 25), -1, AA)
                # Окна
                for wy in range(H_WORLD - h_b + 20, H_WORLD - 20, 32):
                    for wx in range(bx + 12, bx + 85, 22):
                        lit = (idx * 7 + wx + wy + int(t * 0.5)) % 5 != 0
                        cv2.rectangle(frame, (wx, wy), (wx + 10, wy + 16), (120, 220, 255) if lit else (20, 15, 10), -1)
            # Луч прожектора
            ang_s = math.sin(t * 1.4) * 0.4
            p_top = (int(W_WORLD * 0.5 + math.sin(ang_s) * 450), 0)
            cv2.line(frame, (W_WORLD // 2, H_WORLD), p_top, (180, 180, 220), 12, AA)

        elif hero_id == "elsa":
            # Ледяная пещера, полярное сияние (Aurora Borealis), кристаллы и падающий снег
            for y in range(H_WORLD):
                alpha = y / float(H_WORLD)
                frame[y, :] = [int(180 * (1 - alpha) + 90 * alpha), int(120 * (1 - alpha) + 40 * alpha), int(45 * (1 - alpha) + 15 * alpha)]
            # Аврора
            for k in range(5):
                wave_y = int(180 + math.sin(t * 1.8 + k * 1.2) * 55)
                cv2.ellipse(frame, (W_WORLD // 2, wave_y), (W_WORLD // 2, 45), 0, 0, 360, (230, 255, 120), 4, AA)
            # Ледяные кристаллы
            for ix in range(80, W_WORLD, 180):
                cr_h = int(160 + math.sin(ix * 0.3) * 60)
                pts_cr = np.array([[ix, H_WORLD], [ix + 25, H_WORLD - cr_h], [ix + 50, H_WORLD]], np.int32)
                cv2.fillPoly(frame, [pts_cr], (245, 225, 175), AA)
                cv2.polylines(frame, [pts_cr], True, (255, 255, 255), 2, AA)
            # Снежинки
            for s_i in range(35):
                sx = int((s_i * 97 + math.sin(t * 2.0 + s_i) * 30) % W_WORLD)
                sy = int((s_i * 61 + t * 90) % H_WORLD)
                cv2.circle(frame, (sx, sy), 3, (255, 255, 255), -1, AA)

        elif hero_id == "ladybug":
            # Парижский закат, силуэт Эйфелевой башни, летящие лепестки
            for y in range(H_WORLD):
                alpha = y / float(H_WORLD)
                frame[y, :] = [int(190 * (1 - alpha) + 80 * alpha), int(130 * (1 - alpha) + 40 * alpha), int(230 * (1 - alpha) + 180 * alpha)]
            # Башня
            tx = W_WORLD - 320
            pts_t = np.array([[tx - 65, H_WORLD], [tx - 16, 260], [tx - 4, 140], [tx + 4, 140], [tx + 16, 260], [tx + 65, H_WORLD]], np.int32)
            cv2.fillPoly(frame, [pts_t], (55, 40, 50), AA)
            # Лепестки
            for p_i in range(25):
                px = int((p_i * 123 + t * 75) % W_WORLD)
                py = int((p_i * 81 + math.sin(t * 3.0 + p_i) * 45 + t * 40) % H_WORLD)
                cv2.ellipse(frame, (px, py), (8, 4), int(t * 45 + p_i * 20), 0, 360, (160, 140, 245), -1, AA)

        elif hero_id == "iron":
            # Киберпанк-лаборатория, вращающиеся кольца реактора, неоновые сетки
            frame[:] = (25, 20, 25)
            # Сетка перспективы пола
            for gy in range(420, H_WORLD, 28):
                cv2.line(frame, (0, gy), (W_WORLD, gy), (80, 60, 40), 1, AA)
            for gx in range(0, W_WORLD + 100, 90):
                cv2.line(frame, (W_WORLD // 2, 380), (gx, H_WORLD), (80, 60, 40), 1, AA)
            # Голографический реактор
            cx, cy = W_WORLD // 2, 280
            for r_ring in (70, 115, 160):
                rot = t * 2.5 * (1 if r_ring % 2 == 0 else -1)
                cv2.ellipse(frame, (cx, cy), (r_ring, int(r_ring * 0.45)), math.degrees(rot), 0, 360, (255, 210, 80), 3, AA)
            cv2.circle(frame, (cx, cy), 32, (255, 240, 180), -1, AA)

        elif hero_id == "astronaut":
            # Глубокий космос, планета Земля, звёздные туманности, вращающиеся спутники
            frame[:] = (18, 12, 10)
            # Звёзды
            for st_i in range(60):
                st_x = int((st_i * 137) % W_WORLD)
                st_y = int((st_i * 89) % H_WORLD)
                shimmer = 0.5 + 0.5 * math.sin(t * 4.0 + st_i)
                cv2.circle(frame, (st_x, st_y), 2 if shimmer > 0.8 else 1, (int(255 * shimmer), int(255 * shimmer), 255), -1)
            # Земля в углу
            e_x, e_y, e_r = W_WORLD - 180, 160, 140
            cv2.circle(frame, (e_x, e_y), e_r, (210, 150, 40), -1, AA)
            cv2.circle(frame, (e_x, e_y), e_r + 8, (255, 220, 120), 4, AA)
            # Пролетающий астероид/спутник
            sat_x = int((t * 110) % (W_WORLD + 200)) - 100
            sat_y = int(H_WORLD * 0.65 + math.sin(t * 1.5) * 40)
            cv2.rectangle(frame, (sat_x - 18, sat_y - 8), (sat_x + 18, sat_y + 8), (220, 220, 225), -1, AA)
            cv2.line(frame, (sat_x - 45, sat_y), (sat_x + 45, sat_y), (255, 180, 60), 4, AA)

        else:  # fairy
            # Сказочный волшебный лес, биолюминесцентные грибы, мерцающие светлячки
            for y in range(H_WORLD):
                alpha = y / float(H_WORLD)
                frame[y, :] = [int(45 * (1 - alpha) + 20 * alpha), int(95 * (1 - alpha) + 60 * alpha), int(30 * (1 - alpha) + 15 * alpha)]
            # Гигантские светящиеся грибы
            for mx in (180, W_WORLD // 2, W_WORLD - 220):
                my = int(H_WORLD - 120 + math.sin(mx) * 30)
                cv2.ellipse(frame, (mx, my), (85, 45), 0, 180, 360, (220, 140, 255), -1, AA)
                cv2.rectangle(frame, (mx - 14, my), (mx + 14, H_WORLD), (160, 210, 180), -1, AA)
            # Волшебная пыльца
            for f_i in range(40):
                fx = int((f_i * 91 + math.cos(t * 2.2 + f_i) * 50) % W_WORLD)
                fy = int((f_i * 67 + math.sin(t * 2.8 + f_i) * 60) % H_WORLD)
                glow = 0.5 + 0.5 * math.sin(t * 5.0 + f_i)
                cv2.circle(frame, (fx, fy), max(2, int(4 * glow)), (180, 255, 255), -1, AA)

        writer.write(frame)

    writer.release()
    print(f"  [GEN RAW WORLD] {hero_id} -> {out_path} ({num_frames} кадров, 8.0с)")


def create_sample_raw_fx(fx_name, out_path, num_frames=90):
    """Создаёт богатый сырой видеоэффект на чистом чёрном фоне (3.0 сек, 30 FPS)."""
    ensure_dir(os.path.dirname(out_path))
    fw, fh = 480, 480
    writer = cv2.VideoWriter(out_path, FOURCC_MP4V, FPS_TARGET, (fw, fh))
    AA = cv2.LINE_AA

    for i in range(num_frames):
        t = i / FPS_TARGET
        frame = np.zeros((fh, fw, 3), dtype=np.uint8)

        if fx_name == "fireworks":
            # Взрыв салюта с разлетающимися искрами и затухающими хвостами
            phase = (t % 1.5) / 1.5
            cx, cy = fw // 2, fh // 2
            n_sparks = 48
            for s_idx in range(n_sparks):
                ang = s_idx * (2 * math.pi / n_sparks)
                spd = 160.0 + (s_idx % 7) * 25.0
                r = phase * spd
                px = int(cx + math.cos(ang) * r)
                py = int(cy + math.sin(ang) * r + phase * phase * 65.0)  # гравитация
                alpha = max(0.0, 1.0 - phase)
                col = (int(80 * alpha), int(220 * alpha), int(255 * alpha)) if s_idx % 2 == 0 else (int(255 * alpha), int(120 * alpha), int(230 * alpha))
                cv2.circle(frame, (px, py), max(2, int(6 * alpha)), col, -1, AA)
                cv2.circle(frame, (px, py), max(1, int(3 * alpha)), (255, 255, 255), -1, AA)

        elif fx_name == "beam":
            # Концентрированный плазменный лазерный луч с пульсацией и искрами
            cy = fh // 2
            pulse = int(math.sin(t * 18.0) * 8)
            # Центральный луч
            cv2.line(frame, (10, cy), (fw - 10, cy), (120, 240, 255), 24 + pulse, AA)
            cv2.line(frame, (10, cy), (fw - 10, cy), (255, 255, 255), 10, AA)
            # Спиральные кольца вокруг луча
            for sp_i in range(12):
                sx = int((sp_i * 40 + t * 450) % (fw - 40)) + 20
                sy = int(cy + math.sin(t * 14.0 + sp_i) * 26)
                cv2.circle(frame, (sx, sy), 5, (255, 230, 120), -1, AA)

        elif fx_name == "shield":
            # Вращающийся силовой кристаллический гексагональный щит с волновой рябью
            cx, cy = fw // 2, fh // 2
            r_sh = int(140 + math.sin(t * 6.0) * 14)
            rot = t * 1.5
            hexa = [(int(cx + r_sh * math.cos(math.pi / 3 * k + rot)),
                     int(cy + r_sh * math.sin(math.pi / 3 * k + rot))) for k in range(6)]
            cv2.polylines(frame, [np.array(hexa, np.int32)], True, (120, 220, 255), 6, AA)
            cv2.polylines(frame, [np.array(hexa, np.int32)], True, (255, 255, 255), 2, AA)
            # Внутренняя волна
            r_in = int((t * 120) % r_sh)
            cv2.circle(frame, (cx, cy), r_in, (80, 160, 255), 3, AA)

        else:  # burst
            # Радиальная сверхновая ударная волна суперспособности
            cx, cy = fw // 2, fh // 2
            phase = (t % 1.2) / 1.2
            r_wave = int(phase * 210)
            alpha = max(0.0, 1.0 - phase)
            col = (int(90 * alpha), int(220 * alpha), int(255 * alpha))
            cv2.circle(frame, (cx, cy), r_wave, col, max(2, int(18 * alpha)), AA)
            cv2.circle(frame, (cx, cy), max(2, int(r_wave * 0.75)), (255, 255, 255), 3, AA)
            for p_i in range(24):
                ang = p_i * (2 * math.pi / 24) + t * 4.0
                px = int(cx + math.cos(ang) * r_wave * 1.15)
                py = int(cy + math.sin(ang) * r_wave * 1.15)
                cv2.circle(frame, (px, py), 4, (255, 240, 180), -1, AA)

        writer.write(frame)

    writer.release()
    print(f"  [GEN RAW FX] {fx_name} -> {out_path} ({num_frames} кадров, 3.0с)")


def main():
    parser = argparse.ArgumentParser(description="Импорт сырых видео-ассетов миров и FX для «Сиқырлы айна».")
    parser.add_argument("--raw-dir", default=os.path.join(BASE_DIR, "raw"), help="Папка с сырыми видео")
    parser.add_argument("--assets-dir", default=os.path.join(BASE_DIR, "assets"), help="Папка ассетов игры")
    parser.add_argument("--gen-samples", action="store_true", help="Сгенерировать сырые образцы при их отсутствии")
    parser.add_argument("--hero", default=None, help="Импортировать только конкретного героя")
    parser.add_argument("--force", action="store_true", help="Перезаписывать существующие файлы")
    args = parser.parse_args()

    raw_worlds = os.path.join(args.raw_dir, "worlds")
    raw_fx = os.path.join(args.raw_dir, "fx")
    assets_fx = os.path.join(args.assets_dir, "fx")

    ensure_dir(raw_worlds)
    ensure_dir(raw_fx)
    ensure_dir(assets_fx)

    heroes = [args.hero] if args.hero else HERO_IDS

    print("=" * 70)
    print("  ИМПОРТ И СТАНДАРТИЗАЦИЯ АССЕТОВ «СИҚЫРЛЫ АЙНА»")
    print(f"  Сырая папка:   {args.raw_dir}")
    print(f"  Папка ассетов: {args.assets_dir}")
    print("=" * 70)

    # 1. Проверяем / создаём сырые миры
    for hid in heroes:
        raw_file = find_raw_file(raw_worlds, hid)
        if not raw_file and args.gen_samples:
            sample_path = os.path.join(raw_worlds, f"{hid}.mp4")
            create_sample_raw_world(hid, sample_path)
            raw_file = sample_path

        if raw_file:
            dst_world = os.path.join(args.assets_dir, hid, "world.mp4")
            if not os.path.exists(dst_world) or args.force or args.gen_samples:
                convert_video_world(raw_file, dst_world)
            else:
                print(f"  [WORLD SKIP] {dst_world} уже существует (используйте --force для перезаписи)")
        else:
            print(f"  [WORLD MISSING] Не найден сырой файл для '{hid}' в {raw_worlds} (добавьте {hid}.mp4 или укажите --gen-samples)")

    # 2. Проверяем / создаём сырые FX
    for fx_name in FX_NAMES:
        raw_file = find_raw_file(raw_fx, fx_name)
        if not raw_file and args.gen_samples:
            sample_path = os.path.join(raw_fx, f"{fx_name}.mp4")
            create_sample_raw_fx(fx_name, sample_path)
            raw_file = sample_path

        if raw_file:
            dst_fx = os.path.join(assets_fx, f"{fx_name}.mp4")
            if not os.path.exists(dst_fx) or args.force or args.gen_samples:
                convert_video_fx(raw_file, dst_fx)
            else:
                print(f"  [FX SKIP] {dst_fx} уже существует (используйте --force для перезаписи)")
        else:
            print(f"  [FX MISSING] Не найден сырой FX '{fx_name}' в {raw_fx} (добавьте {fx_name}.mp4 или укажите --gen-samples)")

    print("=" * 70)
    print("  Импорт завершён!")
    print("=" * 70)


if __name__ == "__main__":
    main()
