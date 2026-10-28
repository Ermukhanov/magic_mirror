"""
Модуль компьютерного зрения и трекинга «Сиқырлы айна».
Неприкосновенная архитектура:
- Поток Cam (1280x720, MJPG, BUFFERSIZE=1)
- Поток PoseWorker (MediaPipe Pose на 384x216)
- Фильтр OneEuro без задержки
- Предсказание позы вперёд (lead) для костюма/голограммы
+ Умный курсор руки (tracking.HandCursorTracker):
  - Точка на кончике указательного пальца (19/20) или экстраполяция вдоль предплечья (13->15 / 14->16)
  - БЕЗ предсказания вперёд (lead=0), чтобы не было перелёта
  - Отдельный фильтр OneEuro для курсора (меньше дрожи в покое, без задержки в движении)
  - Маппинг зоны досягаемости ребёнка (±1.6 ширины плеч, от макушки до бёдер) на весь экран 1280x720
  - Выбор рабочей руки с гистерезисом (выше плеча / двигалась последней, без перескоков каждый кадр)
  - Магнитное притяжение к карточкам/кнопкам с увеличенным на +25% hitbox'ом
"""
import time
import math
import threading
import cv2
import numpy as np
import mediapipe as mp
from config import W, H, SW, SH, POSE_COMPLEXITY, AA


class Cam(threading.Thread):
    def __init__(self, device_index=0):
        super().__init__(daemon=True)
        self.cap, self.backend_name, self.init_fps = self._open_best_camera(device_index)
        self.cam_fps = self.init_fps if self.init_fps > 0 else 30.0
        self.lock = threading.Lock()
        self.frame, self.ts, self.idx, self.running = None, 0.0, 0, True
        if self.cap is None or not self.cap.isOpened():
            print("НЕ УДАЛОСЬ ОТКРЫТЬ КАМЕРУ")

    @staticmethod
    def _configure_cap(cap, ae_val=0.75):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            # Включаем аппаратную автоэкспозицию и автоусиление (AGC), чтобы картинка была светлой,
            # и отключаем Low-Light Compensation (CAP_PROP_BACKLIGHT=0), которая резала FPS вдвое (до 15 к/с)
            if ae_val is not None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae_val)
            cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
        except Exception:
            pass

    @classmethod
    def _open_best_camera(cls, device_index=0):
        """
        Пробует DSHOW и MSMF (1280x720, MJPG, 30 к/с, светлая автоэкспозиция AE=0.75 + Backlight=0 без падения FPS)
        и автоматически выбирает бэкенд с максимальным реальным FPS.
        """
        candidates = [
            ("DSHOW", cv2.CAP_DSHOW, 0.75),
            ("MSMF", cv2.CAP_MSMF, 0.75),
        ]
        best_mode = None
        best_fps = -1.0

        for name, api, ae in candidates:
            cap = cv2.VideoCapture(device_index, api)
            if not cap.isOpened():
                cap.release()
                continue
            cls._configure_cap(cap, ae_val=ae)
            # Короткий прогрев 5 кадров
            ok_any = False
            for _ in range(5):
                ok, _ = cap.read()
                if ok:
                    ok_any = True
            if not ok_any:
                cap.release()
                continue
            t0 = time.perf_counter()
            cnt = 0
            for _ in range(10):
                ok, _ = cap.read()
                if ok:
                    cnt += 1
            dt = max(1e-3, time.perf_counter() - t0)
            meas_fps = cnt / dt
            if meas_fps > best_fps:
                best_fps = meas_fps
                best_mode = (name, api, ae)
            cap.release()
            # Если DSHOW уже даёт честные >= 27 FPS, сразу выбираем его без ожидания медленного MSMF
            if name == "DSHOW" and meas_fps >= 27.0:
                break

        if best_mode is None:
            cap = cv2.VideoCapture(device_index)
            cls._configure_cap(cap, ae_val=None)
            return cap, "DEFAULT", 30.0

        name, api, ae = best_mode
        cap = cv2.VideoCapture(device_index, api)
        cls._configure_cap(cap, ae_val=ae)
        return cap, name, best_fps

    def run(self):
        prev_t = time.perf_counter()
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            f = cv2.flip(f, 1)
            if f.shape[1] != W or f.shape[0] != H:
                f = cv2.resize(f, (W, H))

            # Автоматическая адаптивная яркость камеры: гарантирует яркое солнечное освещение даже в тёмной комнате
            # Берём центральную область кадра (где стоит ребёнок), чтобы окна сзади не затемняли лицо
            center_crop = f[H // 4: 3 * H // 4, W // 4: 3 * W // 4: 4]
            mean_b = float(cv2.mean(center_crop)[:3][0] * 0.114 + cv2.mean(center_crop)[:3][1] * 0.587 + cv2.mean(center_crop)[:3][2] * 0.299)
            if mean_b < 155.0:
                scale_cam = min(2.5, 158.0 / max(25.0, mean_b))
                beta_cam = int(26 * (scale_cam - 0.5))
                f = cv2.convertScaleAbs(f, alpha=scale_cam, beta=beta_cam)
            now = time.perf_counter()
            dt = max(1e-3, now - prev_t)
            prev_t = now
            with self.lock:
                self.cam_fps = self.cam_fps * 0.88 + (1.0 / dt) * 0.12
                self.frame, self.ts, self.idx = f, now, self.idx + 1

    def latest(self):
        with self.lock:
            return self.frame, self.ts, self.idx


class OneEuro:
    """Сглаживание без задержки: дрожь убирает, быстрые движения не тормозит."""

    def __init__(self, mincut=1.2, beta=0.03, dcut=1.0):
        self.mincut, self.beta, self.dcut = mincut, beta, dcut
        self.x = None

    def reset(self):
        self.x = None

    @staticmethod
    def _a(te, cut):
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cut)) / te)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy(), self.dx.copy()
        te = max(t - self.t, 1e-3)
        self.t = t
        ad = self._a(te, self.dcut)
        self.dx = ad * (x - self.x) / te + (1 - ad) * self.dx
        a = self._a(te, self.mincut + self.beta * np.abs(self.dx))
        self.x = a * x + (1 - a) * self.x
        return self.x.copy(), self.dx.copy()


class PoseWorker(threading.Thread):
    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam, self.lock, self.result, self.running = cam, threading.Lock(), None, True
        self.pose = mp.solutions.pose.Pose(
            model_complexity=POSE_COMPLEXITY,
            enable_segmentation=True,
            smooth_landmarks=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.filt = OneEuro()
        self.processed_count = 0
        self.pose_fps = 30.0

    def run(self):
        last = -1
        prev_t = time.perf_counter()
        while self.running:
            frame, ts, idx = self.cam.latest()
            if frame is None or idx == last:
                time.sleep(0.002)
                continue
            last = idx
            res = self.pose.process(cv2.cvtColor(cv2.resize(frame, (SW, SH)), cv2.COLOR_BGR2RGB))
            now_p = time.perf_counter()
            dt_p = max(1e-3, now_p - prev_t)
            prev_t = now_p
            r = None
            if res.pose_landmarks and res.segmentation_mask is not None:
                lm = res.pose_landmarks.landmark
                a = np.array([[q.x * W, q.y * H, q.visibility] for q in lm], np.float32)
                xy, vel = self.filt(a[:, :2], ts)
                r = dict(
                    p=np.concatenate([xy, a[:, 2:3]], 1),
                    raw_p=a.copy(),
                    vel=vel,
                    mask=res.segmentation_mask.copy(),
                    ts=ts,
                )
            else:
                self.filt.reset()
            with self.lock:
                self.result = r
                self.processed_count += 1
                self.pose_fps = self.pose_fps * 0.88 + (1.0 / dt_p) * 0.12


def lerp(a, b, k):
    return a + (b - a) * k


def ip(a):
    return int(a[0]), int(a[1])


def geom(p):
    P = p[:, :2]
    ls, rs, lh, rh = P[11], P[12], P[23], P[24]
    neck = (ls + rs) / 2
    er = float(np.linalg.norm(P[7] - P[8]))
    sw_raw = float(np.linalg.norm(ls - rs))
    sw = max(sw_raw, er * 2.1, 110.0) + 1.0
    hr = er * 0.55 if er > 12 else sw * 0.38
    # Если человек сидит близко к веб-камере и бёдра (23, 24) не видны — опускаем виртуальные бёдра вниз под плечи
    if p[23, 2] > 0.35 and p[24, 2] > 0.35 and min(lh[1], rh[1]) > neck[1] + sw * 0.45:
        hip = (lh + rh) / 2
        th = max(float(np.linalg.norm(hip - neck)), sw * 0.95) + 1.0
    else:
        th = sw * 1.25
        hip = np.array([neck[0], min(float(H - 12), neck[1] + th)], dtype=np.float32)
        lh = np.array([neck[0] - sw * 0.36, hip[1]], dtype=np.float32)
        rh = np.array([neck[0] + sw * 0.36, hip[1]], dtype=np.float32)
    return dict(
        ls=ls, rs=rs, lh=lh, rh=rh, neck=neck, hip=hip, sw=sw, th=th,
        nose=P[0], hr=hr, lw=P[15], rw=P[16],
        lw_vis=float(p[15, 2]), rw_vis=float(p[16, 2]),
        hips_vis=float(min(p[23, 2], p[24, 2])),
    )


def get_fingertip_pos(p, side="R"):
    """
    Возвращает координаты кончика указательного пальца (19/20) или экстраполяцию вдоль предплечья (13->15 / 14->16).
    Если палец не виден, использует запястье + вектор предплечья.
    """
    if side == "L":
        el_idx, wr_idx, ind_idx, pnk_idx = 13, 15, 19, 17
    else:
        el_idx, wr_idx, ind_idx, pnk_idx = 14, 16, 20, 18

    el = p[el_idx, :2]
    wr = p[wr_idx, :2]
    ind = p[ind_idx, :2]
    pnk = p[pnk_idx, :2]
    wr_vis = float(p[wr_idx, 2])
    ind_vis = float(p[ind_idx, 2])
    el_vis = float(p[el_idx, 2])

    if wr_vis < 0.25 and ind_vis < 0.25:
        return None, 0.0

    forearm = (wr - el) if el_vis >= 0.30 else np.zeros(2, np.float32)

    # Центр ладони: между запястьем и основаниями указательного/мизинца (landmarks 17/19, 18/20)
    pnk_vis = float(p[pnk_idx, 2])
    if ind_vis >= 0.40 and float(np.linalg.norm(ind)) > 5.0 and float(np.linalg.norm(ind - wr)) < 160.0:
        vec = ind - wr
        if float(np.dot(vec, forearm)) >= 0.0 or el_vis < 0.30:
            if pnk_vis >= 0.35 and float(np.linalg.norm(pnk - wr)) < 160.0:
                tip = 0.30 * wr + 0.40 * ind + 0.30 * pnk
            else:
                tip = wr + 0.50 * vec
        else:
            tip = wr + 0.20 * forearm
        return tip.astype(np.float32), max(wr_vis, ind_vis)

    # Иначе экстраполируем от локтя через запястье вдоль предплечья (до центра ладони)
    if el_vis >= 0.35:
        forearm = wr - el
        tip = wr + 0.20 * forearm
        return tip.astype(np.float32), wr_vis

    return wr.astype(np.float32), wr_vis


class HandCursorTracker:
    """
    Управляет курсором руки ребёнка:
    - Без предсказания вперёд (lead=0), чтобы исключить перелёт
    - Отдельный OneEuro фильтр (плавный в покое, быстрый в движении)
    - Маппинг зоны досягаемости вокруг плеч (±1.6 sw по ширине, от макушки до бёдер по высоте) на весь экран 1280x720
    - Гистерезис выбора рабочей руки (та, что выше плеча / двигалась последней; не прыгает каждый кадр)
    - Магнитное притяжение к карточкам/кнопкам с увеличенным на +25% hitbox'ом
    """
    def __init__(self):
        self.filt = OneEuro(mincut=0.80, beta=0.048, dcut=1.0)
        self.active_side = "R"
        self.switch_candidate = None
        self.switch_t0 = 0.0
        self.prev_l = None
        self.prev_r = None
        self.vel_l = 0.0
        self.vel_r = 0.0
        self.last_t = 0.0
        self.raw_cursor = None
        self.mapped_unfiltered = None
        self.smoothed_cursor = None
        self.final_cursor = None
        self.reach_box = (0, 0, W, H)
        self.last_move_t = time.perf_counter()

    def reset(self):
        self.filt.reset()
        self.raw_cursor = None
        self.mapped_unfiltered = None
        self.smoothed_cursor = None
        self.final_cursor = None

    def update(self, p_no_lead, S, t, magnet_targets=None):
        """
        Обновляет позицию курсора по позе БЕЗ предсказания вперёд (p_no_lead).
        magnet_targets: список словарей {'id': ..., 'rect': (x, y, w, h)} или {'id': ..., 'circle': (cx, cy, r)}
        Возвращает (final_cursor, active_side).
        """
        if p_no_lead is None:
            self.reset()
            return None, self.active_side

        dt = max(1e-3, min(0.1, t - self.last_t)) if self.last_t > 0 else 0.016
        self.last_t = t

        tip_l, vis_l = get_fingertip_pos(p_no_lead, "L")
        tip_r, vis_r = get_fingertip_pos(p_no_lead, "R")
        if tip_l is None and tip_r is None:
            self.reset()
            return None, self.active_side

        # Оцениваем скорость движения обеих рук (какая рука двигалась последней)
        if tip_l is not None and self.prev_l is not None:
            inst_vl = float(np.linalg.norm(tip_l - self.prev_l)) / dt
            self.vel_l = self.vel_l * 0.78 + inst_vl * 0.22
        else:
            self.vel_l *= 0.8
        if tip_r is not None and self.prev_r is not None:
            inst_vr = float(np.linalg.norm(tip_r - self.prev_r)) / dt
            self.vel_r = self.vel_r * 0.78 + inst_vr * 0.22
        else:
            self.vel_r *= 0.8
        self.prev_l = tip_l.copy() if tip_l is not None else None
        self.prev_r = tip_r.copy() if tip_r is not None else None

        ls, rs = p_no_lead[11, :2], p_no_lead[12, :2]
        lh, rh = p_no_lead[23, :2], p_no_lead[24, :2]
        shoulder_y = float(min(ls[1], rs[1]))
        hip_y = float((lh[1] + rh[1]) * 0.5)

        # Оценка активности левой и правой руки (выше плеча + скорость движения)
        def hand_score(tip, vis, vel):
            if tip is None or vis < 0.30:
                return -9999.0
            # Чем выше рука (меньше Y), тем выше приоритет
            height_bonus = (hip_y - float(tip[1])) * 1.8
            above_shoulder_bonus = 180.0 if float(tip[1]) < shoulder_y + 40 else 0.0
            vel_bonus = min(220.0, vel * 0.28)
            return height_bonus + above_shoulder_bonus + vel_bonus

        sc_l = hand_score(tip_l, vis_l, self.vel_l)
        sc_r = hand_score(tip_r, vis_r, self.vel_r)

        # Гистерезис переключения рабочей руки: не переключаемся мгновенно на 1 кадре
        desired = self.active_side
        if self.active_side == "R":
            if tip_r is None or (sc_l > sc_r + 85.0):
                desired = "L"
        else:
            if tip_l is None or (sc_r > sc_l + 85.0):
                desired = "R"

        if desired != self.active_side:
            if self.switch_candidate != desired:
                self.switch_candidate = desired
                self.switch_t0 = t
            elif (t - self.switch_t0) >= 0.26 or (tip_r is None if self.active_side == "R" else tip_l is None):
                self.active_side = desired
                self.switch_candidate = None
                self.filt.reset()
        else:
            self.switch_candidate = None

        raw_pt = tip_l if self.active_side == "L" else tip_r
        if raw_pt is None:
            raw_pt = tip_r if tip_r is not None else tip_l
        if raw_pt is None:
            self.reset()
            return None, self.active_side

        self.raw_cursor = raw_pt.copy()

        # Зона досягаемости ребёнка вокруг плеч (калиброванная по росту и ширине плеч)
        neck = (ls + rs) * 0.5
        hip = (lh + rh) * 0.5
        sw = float(S.get("calib_sw") or np.linalg.norm(ls - rs))
        th = float(S.get("calib_th") or np.linalg.norm(hip - neck))
        sw = max(48.0, sw)
        th = max(65.0, th)

        # Прямоугольник зоны досягаемости: ±1.6 ширины плеч по X, от макушки до бёдер по Y
        cx_body = float(neck[0])
        rx0 = cx_body - 1.60 * sw
        rx1 = cx_body + 1.60 * sw
        ry0 = float(neck[1]) - 0.95 * th   # уровень макушки / поднятой руки
        ry1 = float(hip[1]) + 0.12 * th    # уровень бёдер
        self.reach_box = (int(rx0), int(ry0), int(rx1), int(ry1))

        # Маппинг зоны досягаемости на весь экран 1280x720
        nx = (float(raw_pt[0]) - rx0) / max(40.0, (rx1 - rx0))
        ny = (float(raw_pt[1]) - ry0) / max(40.0, (ry1 - ry0))
        mapped_x = float(np.clip(nx * W, 18.0, W - 18.0))
        mapped_y = float(np.clip(ny * H, 18.0, H - 18.0))
        mapped_pt = np.array([mapped_x, mapped_y], dtype=np.float32)
        self.mapped_unfiltered = mapped_pt.copy()

        # Отдельный фильтр OneEuro для курсора (без предсказания вперёд!)
        smooth_pt, c_vel = self.filt(mapped_pt, t)
        if float(np.linalg.norm(c_vel)) > 35.0:
            self.last_move_t = t
        self.smoothed_cursor = smooth_pt.copy()

        # Магнитное притяжение к ближайшей карточке или кнопке (с hitbox +25%)
        final_pt = smooth_pt.copy()
        if magnet_targets:
            best_dist = 1e9
            best_center = None
            for tgt in magnet_targets:
                if "rect" in tgt:
                    x, y, w, h = tgt["rect"]
                    tcx, tcy = x + w * 0.5, y + h * 0.5
                    # Hitbox на 25% больше видимой карточки (полуширина * 1.25)
                    hw, hh = w * 0.5 * 1.25, h * 0.5 * 1.25
                    if abs(smooth_pt[0] - tcx) <= hw and abs(smooth_pt[1] - tcy) <= hh:
                        d = math.hypot(smooth_pt[0] - tcx, smooth_pt[1] - tcy)
                        if d < best_dist:
                            best_dist = d
                            best_center = np.array([tcx, tcy], dtype=np.float32)
                elif "circle" in tgt:
                    tcx, tcy, r = tgt["circle"]
                    if math.hypot(smooth_pt[0] - tcx, smooth_pt[1] - tcy) <= r * 1.25:
                        d = math.hypot(smooth_pt[0] - tcx, smooth_pt[1] - tcy)
                        if d < best_dist:
                            best_dist = d
                            best_center = np.array([tcx, tcy], dtype=np.float32)
            if best_center is not None:
                # Плавно притягиваем курсор на 35% к центру карточки/кнопки
                final_pt = lerp(smooth_pt, best_center, 0.35)

        self.final_cursor = final_pt
        return final_pt, self.active_side


CURSOR = HandCursorTracker()


def update_height_calibration(frame, p, S, dt):
    """
    Калибровка по росту при входе ребёнка в кадр:
    замеряет базовую высоту плеч и размах плеч, чтобы жесты и зона досягаемости курсора
    адаптировались под любой рост ребёнка (4–6 лет).
    """
    if p is None:
        return
    ls, rs, lh, rh = p[11][:2], p[12][:2], p[23][:2], p[24][:2]
    sw = float(np.linalg.norm(ls - rs))
    th = float(np.linalg.norm((lh + rh) * 0.5 - (ls + rs) * 0.5))
    neck_y = float((ls[1] + rs[1]) * 0.5)
    if sw > 35 and th > 45:
        if "calib_sw" not in S or S["calib_sw"] <= 0:
            S["calib_sw"] = sw
            S["calib_th"] = th
            S["calib_neck_y"] = neck_y
        else:
            S["calib_sw"] = S["calib_sw"] * 0.96 + sw * 0.04
            S["calib_th"] = S["calib_th"] * 0.96 + th * 0.04
            S["calib_neck_y"] = S["calib_neck_y"] * 0.98 + neck_y * 0.02


def draw_hand_skeleton_feedback(img, p, active_side="R", thin=False, t=0.0):
    """
    Рисует светящийся неоновый скелет рук и плеч ребёнка прямо на экране
    (тонкие неоновые линии и яркие точки на пальцах), чтобы он видел, что система его видит.
    Всегда ярко на экране выбора и в обучении (thin=False), а в игре тоньше (thin=True).
    """
    if p is None:
        return
    P = p[:, :2]
    outer_th = 3 if thin else 7
    inner_th = 1 if thin else 3
    jt_r = 4 if thin else 7

    # Линии плеч и рук
    bones = (
        (11, 12, "M"),
        (11, 13, "L"), (13, 15, "L"), (15, 19, "L"), (15, 17, "L"),
        (12, 14, "R"), (14, 16, "R"), (16, 20, "R"), (16, 18, "R"),
    )
    for a_i, b_i, side in bones:
        if p[a_i, 2] < 0.28 or p[b_i, 2] < 0.28:
            continue
        is_act = (side == active_side)
        col_glow = (60, 255, 160) if is_act else (255, 220, 60)
        cv2.line(img, ip(P[a_i]), ip(P[b_i]), col_glow, outer_th, AA)
        cv2.line(img, ip(P[a_i]), ip(P[b_i]), (255, 255, 255), inner_th, AA)

    # Суставы плеч, локтей, запястий
    for idx_j, side in ((11, "L"), (12, "R"), (13, "L"), (14, "R"), (15, "L"), (16, "R")):
        if p[idx_j, 2] >= 0.30:
            col_j = (60, 255, 160) if side == active_side else (0, 235, 255)
            cv2.circle(img, ip(P[idx_j]), jt_r, col_j, -1, AA)
            if not thin:
                cv2.circle(img, ip(P[idx_j]), jt_r + 2, (255, 255, 255), 1, AA)

    # Яркие точки на кончиках пальцев обеих рук
    pulse = int(3 * math.sin(t * 8.0)) if not thin else 0
    for side in ("L", "R"):
        tip, vis = get_fingertip_pos(p, side)
        if tip is not None and vis >= 0.30:
            is_act = (side == active_side)
            r_tip = (6 if thin else (13 + pulse if is_act else 9))
            col_t = (50, 255, 120) if is_act else (0, 240, 255)
            cv2.circle(img, ip(tip), r_tip + 3, col_t, 2, AA)
            cv2.circle(img, ip(tip), r_tip, col_t, -1, AA)
            cv2.circle(img, ip(tip), max(2, r_tip // 2), (255, 255, 255), -1, AA)


def draw_cursor_with_ring(img, cursor_pt, prog=0.0, t=0.0):
    """
    Рисует умный курсор руки и кольцо прогресса ПРЯМО ВОКРУГ КУРСОРА,
    чтобы ребёнок видел заполнение рядом со своей ладошкой/пальцем.
    """
    if cursor_pt is None:
        return
    cx, cy = ip(cursor_pt)
    pulse = int(3 * math.sin(t * 7.5))
    r_base = 26 + pulse

    # Внешнее неоновое свечение курсора
    col_c = (60, 255, 120) if prog > 0.02 else (0, 255, 255)
    cv2.circle(img, (cx, cy), r_base + 4, (20, 30, 45), 5, AA)
    cv2.circle(img, (cx, cy), r_base, col_c, 4, AA)
    cv2.circle(img, (cx, cy), 9, (255, 255, 255), -1, AA)
    cv2.circle(img, (cx, cy), 5, col_c, -1, AA)

    # Кольцо удержания прямо вокруг курсора (1.0 сек)
    if prog > 0.01:
        r_ring = 38
        ang = int(360 * min(1.0, prog))
        cv2.circle(img, (cx, cy), r_ring, (35, 45, 65), 10, AA)
        cv2.ellipse(img, (cx, cy), (r_ring, r_ring), -90, 0, ang, (255, 255, 255), 13, AA)
        cv2.ellipse(img, (cx, cy), (r_ring, r_ring), -90, 0, ang, (60, 255, 120), 9, AA)


_K_OPEN_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K_ERODE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def clean_person_mask(raw_mask, out_w=None, out_h=None, erode_px=3):
    """
    Очищает маску сегментации человека от артефактов (Пункт 5):
    1. Морфологическое открытие для подавления мелких отрывных пятен.
    2. Фильтр связных компонент по площади — оставляет ТОЛЬКО самую крупную компоненту человека.
    3. Сужение маски на 2-3 px (erode), чтобы исключить рваный край и захват фона.
    4. Мягкое размытие края (GaussianBlur).
    """
    if raw_mask is None:
        return None
    m_bin = (raw_mask > 0.25).astype(np.uint8) * 255
    m_open = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, _K_OPEN_3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_open, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        if areas[largest_label - 1] >= 120:
            m_clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        else:
            m_clean = np.zeros_like(m_open)
    else:
        m_clean = m_open

    if erode_px >= 2:
        m_clean = cv2.erode(m_clean, _K_OPEN_3, iterations=1)

    m_soft_u8 = cv2.GaussianBlur(m_clean, (5, 5), 1.4)
    if out_w is not None and out_h is not None and (m_soft_u8.shape[1] != out_w or m_soft_u8.shape[0] != out_h):
        m_soft_u8 = cv2.resize(m_soft_u8, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    return m_soft_u8.astype(np.float32) * (1.0 / 255.0)
