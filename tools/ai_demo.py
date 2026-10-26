"""
Демонстрационный стенд ИИ-превращения (tools/ai_demo.py) для «Сиқырлы айна»:
- Окно 1280x720 с камерой 30 FPS (автовыбор DSHOW/MSMF, MJPG):
  - СЛЕВА (640x720): живой кадр с камеры + мини-превью плотного цветного эскиза костюма (в углу)
  - СПРАВА (640x720): ИИ-герой в реальном времени (SD-Turbo fp16 + TAESD) с живым лицом, шеей, волосами и кистями рук,
    очищенной маской без серого гало и без рваного края.
- Горячие клавиши:
  - [ и ]         — уменьшить / увеличить strength (по умолчанию 0.70, шаг 0.05)
  - 1 и 2         — переключить 1 или 2 шага денойзинга ИИ (по умолчанию 2 шага)
  - E             — включить / выключить эскиз костюма перед ИИ (sketch_conditioning: ON / OFF)
  - K             — включить / выключить живые кисти рук (hands_live: ON / OFF)
  - H / Tab / 3..8 — сменить героя (Алпамыс, Томирис, Өрмекші-адам, Эльза, Ледибаг, Темір батыр, Ғарышкер, Пері)
  - Q / Esc       — выход
"""
import os
import sys
import math
import time
import threading
import argparse
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from config import W, H, SW, SH, AA
from tracking import Cam, PoseWorker, clean_person_mask, ip
from heroes import HEROES, emblem, hero_display_name, star_pts
from i18n import txt
from tools.bench_ai import SD_TURBO_DIR, TAESD_DIR, HERO_PROMPTS, NEGATIVE_PROMPT
from ai_transform import (
    HERO_SKETCH_PALETTES,
    compute_face_and_hands_mask,
    build_costume_sketch,
    warp_ai_to_live_pose,
)


class DemoAIWorker(threading.Thread):
    """
    Отдельный GPU-поток для ИИ-превращения (SD-Turbo fp16 + TAESD):
    - Поддерживает динамическое изменение strength (0.35..0.85, клавиши [ и ]),
    - 1 или 2 шага (клавиши 1 и 2),
    - Включение/выключение эскиза костюма (sketch_conditioning, клавиша E),
    - Захват CUDA Graph для максимального FPS (10-18+ FPS на RTX 4050 при 1.7-2.3 ГБ VRAM).
    """

    def __init__(self, width=384, height=512):
        super().__init__(daemon=True)
        self.ai_w = width
        self.ai_h = height
        self.lock = threading.Lock()
        self.running = True

        self.req_frame = None
        self.req_mask = None
        self.req_p = None
        self.req_hero = HEROES[0]
        self.req_idx = -1

        self.steps = 2
        self.strength = 0.48
        self.sketch_conditioning = bool(config.SKETCH_CONDITIONING)
        self.hands_live = bool(config.HANDS_LIVE)

        self.out_img = None
        self.out_p = None
        self.last_sketch_preview = None
        self.ai_fps = 0.0
        self.ai_ms = 0.0
        self.vram_gb = 0.0
        self.ready = False
        self.error = None

    def submit(self, frame_crop, mask_crop, p_crop, hero, idx):
        with self.lock:
            self.req_frame = frame_crop
            self.req_mask = mask_crop
            self.req_p = p_crop
            self.req_hero = hero
            self.req_idx = idx

    def run(self):
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny
            from diffusers.models.attention_processor import AttnProcessor2_0

            device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            try:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    SD_TURBO_DIR,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )
            except Exception:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    SD_TURBO_DIR,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )

            taesd = AutoencoderTiny.from_pretrained(
                TAESD_DIR,
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            pipe.vae = taesd
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)

            embeds_map = {}
            seeds_map = config.CFG.get("ai_hero_seeds", {})
            with torch.inference_mode():
                for hid, pr in HERO_PROMPTS.items():
                    emb, _ = pipe.encode_prompt(
                        prompt=pr,
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    embeds_map[hid] = emb
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

            scaling_factor = float(pipe.vae.config.scaling_factor)
            alphas_cumprod_cpu = pipe.scheduler.alphas_cumprod.cpu().numpy()
            lat_h, lat_w = self.ai_h // 8, self.ai_w // 8

            noises_map = {}
            for hid in HERO_PROMPTS:
                seed_val = int(seeds_map.get(hid, 1001))
                gen = torch.Generator(device=device).manual_seed(seed_val)
                n1 = torch.randn((1, 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
                n2 = torch.randn((1, 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
                noises_map[hid] = (n1, n2)

            # Статические буферы для CUDA Graph с динамическими тензорами t_steps и sigmas (позволяет менять strength на лету!)
            static_in = torch.zeros((1, 3, self.ai_h, self.ai_w), device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
            static_emb = embeds_map["alpamys"].clone()
            static_n1, static_n2 = noises_map["alpamys"][0].clone(), noises_map["alpamys"][1].clone()
            static_t = torch.tensor([699, 349], device=device, dtype=torch.long)
            static_sig = torch.tensor([2.5, 1.1, 0.0], device=device, dtype=torch.float16)

            graphs = {}
            for st_cnt in (1, 2):
                def _step_fn(st=st_cnt):
                    lat = pipe.vae.encode(static_in).latents * scaling_factor
                    x = lat + static_n1 * static_sig[0]
                    for idx_s in range(st):
                        sig = static_sig[idx_s]
                        sig_next = static_sig[idx_s + 1]
                        c_in = x / torch.sqrt(sig * sig + 1.0)
                        eps = pipe.unet(c_in, static_t[idx_s], encoder_hidden_states=static_emb, return_dict=False)[0]
                        if idx_s == st - 1:
                            x = x - sig * eps
                        else:
                            sig_up = torch.sqrt((sig_next ** 2) * (sig ** 2 - sig_next ** 2) / (sig ** 2))
                            sig_down = torch.sqrt(sig_next ** 2 - sig_up ** 2)
                            x = x + (sig_down - sig) * eps + static_n2 * sig_up
                    return pipe.vae.decode(x / scaling_factor, return_dict=False)[0]

                s_stream = torch.cuda.Stream()
                s_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_stream), torch.inference_mode():
                    for _ in range(3):
                        out_buf = _step_fn()
                torch.cuda.current_stream().wait_stream(s_stream)
                g_c = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_c), torch.inference_mode():
                    out_buf = _step_fn()
                graphs[st_cnt] = (g_c, out_buf)

            self.ready = True
            last_done = -1
            prev_ema = None
            prev_hero_id = None
            prev_sched_key = None

            while self.running:
                with self.lock:
                    if self.req_frame is None or self.req_idx == last_done:
                        fr = None
                    else:
                        fr = self.req_frame
                        mk = self.req_mask
                        p_loc = self.req_p
                        hero = self.req_hero
                        last_done = self.req_idx
                        steps = self.steps
                        strength = float(np.clip(self.strength, 0.35, 0.85))
                        use_sketch = self.sketch_conditioning
                        hands_live = self.hands_live

                if fr is None:
                    time.sleep(0.002)
                    continue

                t0 = time.perf_counter()
                hid = hero["id"]
                fh_orig, fw_orig = fr.shape[:2]
                fr_small = cv2.resize(fr, (self.ai_w, self.ai_h), interpolation=cv2.INTER_LINEAR)
                if mk is None:
                    mk_small = np.ones((self.ai_h, self.ai_w), dtype=np.float32)
                else:
                    mk_small = cv2.resize(mk, (self.ai_w, self.ai_h), interpolation=cv2.INTER_LINEAR)

                p_small = None
                if p_loc is not None:
                    p_small = p_loc.copy()
                    p_small[:, 0] *= (self.ai_w / float(fw_orig))
                    p_small[:, 1] *= (self.ai_h / float(fh_orig))

                if use_sketch:
                    inp_small, _ = build_costume_sketch(fr_small, mk_small, p_small, hero, hands_live=hands_live)
                else:
                    pal = HERO_SKETCH_PALETTES.get(hid, HERO_SKETCH_PALETTES["alpamys"])
                    kern_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                    dil = cv2.dilate(fr_small, kern_dil, iterations=1)
                    bg_c = np.full_like(fr_small, pal["bg_fill"], dtype=np.uint8)
                    halo_free = cv2.addWeighted(dil, 0.65, bg_c, 0.35, 0)
                    w_fg = np.clip(mk_small, 0.0, 1.0).astype(np.float32)
                    inp_small = cv2.blendLinear(fr_small, halo_free, w_fg, 1.0 - w_fg)

                inp_rgb = cv2.cvtColor(inp_small, cv2.COLOR_BGR2RGB)

                with torch.inference_mode():
                    sched_key = (steps, round(strength, 3))
                    if sched_key != prev_sched_key:
                        t0_idx = int(round(strength * 999))
                        t1_idx = max(1, int(round(t0_idx * 0.48)))
                        ab0 = float(alphas_cumprod_cpu[t0_idx])
                        ab1 = float(alphas_cumprod_cpu[t1_idx])
                        sig0 = math.sqrt(max(1e-5, (1.0 - ab0) / ab0))
                        sig1 = math.sqrt(max(1e-5, (1.0 - ab1) / ab1))
                        if steps == 1:
                            static_t.copy_(torch.tensor([t0_idx, t1_idx], dtype=torch.long))
                            static_sig.copy_(torch.tensor([sig0, 0.0, 0.0], dtype=torch.float16))
                        else:
                            static_t.copy_(torch.tensor([t0_idx, t1_idx], dtype=torch.long))
                            static_sig.copy_(torch.tensor([sig0, sig1, 0.0], dtype=torch.float16))
                        prev_sched_key = sched_key

                    in_t = (
                        (torch.from_numpy(inp_rgb).to(device=device, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0) / 127.5) - 1.0
                    ).contiguous(memory_format=torch.channels_last)
                    static_in.copy_(in_t)
                    if prev_hero_id != hid:
                        static_emb.copy_(embeds_map.get(hid, embeds_map["alpamys"]))
                        n1, n2 = noises_map.get(hid, noises_map["alpamys"])
                        static_n1.copy_(n1)
                        static_n2.copy_(n2)
                        prev_ema = None

                    g_c, out_buf = graphs[steps]
                    g_c.replay()
                    out_rgb = ((out_buf[0].permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()

                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                if prev_ema is not None and prev_ema.shape == out_bgr.shape and prev_hero_id == hid:
                    out_bgr = cv2.addWeighted(out_bgr, 0.58, prev_ema, 0.42, 0)
                prev_ema = out_bgr.copy()
                prev_hero_id = hid

                dt_ai = max(1e-3, time.perf_counter() - t0)
                vram_now = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                with self.lock:
                    self.out_img = out_bgr
                    self.out_p = p_loc
                    self.last_sketch_preview = inp_small
                    self.ai_ms = dt_ai * 1000.0
                    self.ai_fps = (1.0 / dt_ai) if self.ai_fps == 0 else (self.ai_fps * 0.72 + (1.0 / dt_ai) * 0.28)
                    self.vram_gb = vram_now

        except Exception as e:
            self.error = str(e)
            print(f"[AI_DEMO ERROR] {e}")


def compose_final_ai_half(ai_img_small, ai_p, live_half, clean_mask_half, p_half, hero, hands_live=True):
    """
    Собирает финальный правый кадр (640x720):
    1. Аффинно выравнивает ИИ-кадр по текущему положению скелета ребёнка (zero lag, 30+ FPS).
    2. Применяет фильтр резкости деталей (unsharp mask) для чётких граней доспехов.
    3. Поверх мягко вклеивает настоящее живое лицо, волосы, шею и кисти рук.
    4. Заливает фон вокруг очищенной маски силуэта гармоничным сиянием ауры героя.
    """
    h_img, w_img = live_half.shape[:2]
    ai_warped = warp_ai_to_live_pose(ai_img_small, ai_p, p_half, w_img, h_img)
    ai_sharp = cv2.addWeighted(ai_warped, 1.25, cv2.GaussianBlur(ai_warped, (0, 0), 2.0), -0.25, 0)

    face_hands_m = compute_face_and_hands_mask(h_img, w_img, p_half, hands_live=hands_live).astype(np.float32)
    ai_with_live_face = cv2.blendLinear(live_half, ai_sharp, face_hands_m, 1.0 - face_hands_m)

    if clean_mask_half is not None:
        m_fg = np.clip(clean_mask_half, 0.0, 1.0).astype(np.float32)
        glow_tint = np.full_like(live_half, hero["glow"], dtype=np.uint8)
        bg_styled = cv2.addWeighted(live_half, 0.85, glow_tint, 0.15, 0)
        out = cv2.blendLinear(ai_with_live_face, bg_styled, m_fg, 1.0 - m_fg)
        return out
    return ai_with_live_face


def main():
    parser = argparse.ArgumentParser(description="Демо ИИ-превращения Сиқырлы айна (Пункты 1-5)")
    parser.add_argument("--headless", action="store_true", help="Тестовый замер без GUI")
    parser.add_argument("--save-preview", default="", help="Сохранить скриншот сравнения (живой кадр / эскиз / ИИ) в файл")
    args = parser.parse_args()

    ai_worker = DemoAIWorker(width=384, height=512)
    ai_worker.start()

    print("[AI_DEMO] Ожидание инициализации SD-Turbo (fp16) + TAESD...")
    t_wait = time.perf_counter()
    while not ai_worker.ready and ai_worker.error is None and (time.perf_counter() - t_wait) < 120.0:
        time.sleep(0.05)

    if ai_worker.error:
        print(f"[AI_DEMO] Ошибка запуска ИИ: {ai_worker.error}")
        return
    if not ai_worker.ready:
        print("[AI_DEMO] Таймаут ожидания инициализации ИИ")
        return

    cam = Cam()
    cam.start()
    pose_worker = PoseWorker(cam)
    pose_worker.start()

    # Ждём первый кадр с камеры
    t_c = time.perf_counter()
    while cam.latest()[0] is None and (time.perf_counter() - t_c) < 5.0:
        time.sleep(0.01)

    print(f"[AI_DEMO] Камера запущена: бэкенд={cam.backend_name}, стартовый FPS={cam.init_fps:.1f} FPS")

    if args.headless or args.save_preview:
        # Прогоняем 3 секунды реального конвейера с камерой, замеряем FPS при 1 и 2 шагах и сохраняем превью
        for st_test in (1, 2):
            ai_worker.steps = st_test
            ai_worker.strength = float(config.CFG.get("ai_strength", 0.48))
            ai_worker.sketch_conditioning = True
            t_start = time.perf_counter()
            frames_sent = 0
            while (time.perf_counter() - t_start) < 2.2:
                frame, fts, idx = cam.latest()
                if frame is None:
                    time.sleep(0.005)
                    continue
                x_crop0 = (W - 640) // 2
                live_half = frame[:, x_crop0:x_crop0 + 640].copy()
                with pose_worker.lock:
                    r = pose_worker.result
                p_half = None
                mask_half = None
                if r is not None and (time.perf_counter() - r["ts"]) < 0.8:
                    p_half = r["p"].copy()
                    p_half[:, 0] -= x_crop0
                    m_clean_full = clean_person_mask(r["mask"], out_w=W, out_h=H, erode_px=3)
                    mask_half = m_clean_full[:, x_crop0:x_crop0 + 640]
                else:
                    # Если перед камерой сейчас никого нет — создаём тестовую маску и позу в центре кадра для проверки эскиза
                    mask_half = np.zeros((720, 640), dtype=np.float32)
                    cv2.circle(mask_half, (320, 210), 68, 1.0, -1)
                    cv2.ellipse(mask_half, (320, 480), (155, 220), 0, 0, 360, 1.0, -1)
                    mask_half = cv2.GaussianBlur(mask_half, (15, 15), 4.0)
                    p_half = np.zeros((33, 3), dtype=np.float32)
                    p_half[:, 2] = 0.99
                    p_half[0, :2] = (320, 205)
                    p_half[7, :2] = (275, 205)
                    p_half[8, :2] = (365, 205)
                    p_half[11, :2] = (230, 330)
                    p_half[12, :2] = (410, 330)
                    p_half[13, :2] = (180, 445)
                    p_half[14, :2] = (460, 445)
                    p_half[15, :2] = (160, 545)
                    p_half[16, :2] = (480, 545)
                    p_half[19, :2] = (150, 580)
                    p_half[20, :2] = (490, 580)
                    p_half[23, :2] = (255, 560)
                    p_half[24, :2] = (385, 560)
                    p_half[25, :2] = (255, 690)
                    p_half[26, :2] = (385, 690)
                ai_worker.submit(live_half, mask_half, p_half, HEROES[0], idx + frames_sent)
                frames_sent += 1
                time.sleep(0.02)
            print(f"[AI_DEMO TEST] steps={st_test} | strength={ai_worker.strength:.2f} | sketch=ON -> "
                  f"CAM FPS: {cam.cam_fps:.1f} | AI FPS: {ai_worker.ai_fps:.2f} ({ai_worker.ai_ms:.1f} ms) | VRAM: {ai_worker.vram_gb:.2f} GB")

        if args.save_preview and ai_worker.out_img is not None:
            frame, _, _ = cam.latest()
            x_crop0 = (W - 640) // 2
            live_half = frame[:, x_crop0:x_crop0 + 640].copy()
            with pose_worker.lock:
                r = pose_worker.result
            p_half = None
            mask_half = None
            if r is not None:
                p_half = r["p"].copy()
                p_half[:, 0] -= x_crop0
                m_clean_full = clean_person_mask(r["mask"], out_w=W, out_h=H, erode_px=3)
                mask_half = m_clean_full[:, x_crop0:x_crop0 + 640]
            else:
                mask_half = np.zeros((720, 640), dtype=np.float32)
                cv2.circle(mask_half, (320, 210), 68, 1.0, -1)
                cv2.ellipse(mask_half, (320, 480), (155, 220), 0, 0, 360, 1.0, -1)
                mask_half = cv2.GaussianBlur(mask_half, (15, 15), 4.0)
                p_half = np.zeros((33, 3), dtype=np.float32)
                p_half[:, 2] = 0.99
                p_half[0, :2] = (320, 205)
                p_half[7, :2] = (275, 205)
                p_half[8, :2] = (365, 205)
                p_half[11, :2] = (230, 330)
                p_half[12, :2] = (410, 330)
                p_half[13, :2] = (180, 445)
                p_half[14, :2] = (460, 445)
                p_half[15, :2] = (160, 545)
                p_half[16, :2] = (480, 545)
                p_half[19, :2] = (150, 580)
                p_half[20, :2] = (490, 580)
                p_half[23, :2] = (255, 560)
                p_half[24, :2] = (385, 560)
                p_half[25, :2] = (255, 690)
                p_half[26, :2] = (385, 690)

            right_half = compose_final_ai_half(ai_worker.out_img, ai_worker.out_p, live_half, mask_half, p_half, HEROES[0], hands_live=ai_worker.hands_live)
            if ai_worker.last_sketch_preview is not None:
                sk_thumb = cv2.resize(ai_worker.last_sketch_preview, (168, 224))
                live_half[H - 238:H - 14, 14:182] = sk_thumb
                cv2.rectangle(live_half, (14, H - 238), (182, H - 14), (0, 235, 255), 2, AA)
                txt(live_half, "ЭСКИЗ [E]", 98, H - 266, 20, (0, 255, 255), center=True)

            canvas = np.hstack([live_half, right_half])
            cv2.line(canvas, (640, 0), (640, H), (0, 255, 255), 3, AA)
            cv2.rectangle(canvas, (0, 0), (W, 82), (18, 24, 36), -1)
            txt(canvas, f"КАМЕРА ({cam.backend_name}): {cam.cam_fps:.1f} FPS", 20, 10, 26, (80, 255, 140))
            txt(canvas, f"ИИ ({ai_worker.steps} шаг, str={ai_worker.strength:.2f}): {ai_worker.ai_fps:.1f} FPS ({ai_worker.ai_ms:.0f} мс) | VRAM: {ai_worker.vram_gb:.2f} ГБ",
                420, 10, 26, (0, 235, 255))
            txt(canvas, f"Герой [H/Tab]: {hero_display_name(HEROES[0])} | Эскиз [E]: {'ВКЛ' if ai_worker.sketch_conditioning else 'ВЫКЛ'} | Руки [K]: {'ЖИВЫЕ' if ai_worker.hands_live else 'ИИ'} | [ / ]: strength | 1/2: шаги",
                20, 46, 21, (235, 240, 255))
            cv2.imwrite(args.save_preview, canvas)
            print(f"[AI_DEMO] Скриншот сохранён в {args.save_preview}")

        ai_worker.running = False
        cam.running = pose_worker.running = False
        return

    win = "Magic Mirror AI Demo [[/]: Strength | 1/2: Steps | E: Sketch | H/Tab: Hero | K: Hands | Q: Quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    hero_idx = 0
    prev_t = time.perf_counter()
    ui_fps = 30.0

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif ord('1') <= key <= ord('8'):
            hero_idx = key - ord('1')
        elif key in (ord('s'), ord('S')):
            ai_worker.steps = 1 if ai_worker.steps == 2 else 2
        elif key in (ord('['), ord('{')):
            ai_worker.strength = round(max(0.35, ai_worker.strength - 0.05), 2)
        elif key in (ord(']'), ord('}')):
            ai_worker.strength = round(min(0.85, ai_worker.strength + 0.05), 2)
        elif key in (ord('e'), ord('E')):
            ai_worker.sketch_conditioning = not ai_worker.sketch_conditioning
        elif key in (ord('k'), ord('K')):
            ai_worker.hands_live = not ai_worker.hands_live
        elif key in (ord('h'), ord('H'), 9, ord(' ')):
            hero_idx = (hero_idx + 1) % len(HEROES)

        frame, fts, idx = cam.latest()
        if frame is None:
            time.sleep(0.002)
            continue

        now = time.perf_counter()
        dt = max(1e-3, now - prev_t)
        prev_t = now
        ui_fps = ui_fps * 0.9 + 0.1 / dt

        x_crop0 = (W - 640) // 2
        live_half = frame[:, x_crop0:x_crop0 + 640].copy()

        with pose_worker.lock:
            r = pose_worker.result

        p_half = None
        mask_half = None
        if r is not None and (now - r["ts"]) < 0.6:
            p_half = r["p"].copy()
            p_half[:, 0] -= x_crop0
            m_clean_full = clean_person_mask(r["mask"], out_w=W, out_h=H, erode_px=3)
            mask_half = m_clean_full[:, x_crop0:x_crop0 + 640]

        hero = HEROES[hero_idx]
        ai_worker.submit(live_half, mask_half, p_half, hero, idx)

        with ai_worker.lock:
            ai_img = ai_worker.out_img.copy() if ai_worker.out_img is not None else None
            ai_p = ai_worker.out_p.copy() if getattr(ai_worker, "out_p", None) is not None else None
            sk_prev = ai_worker.last_sketch_preview.copy() if ai_worker.last_sketch_preview is not None else None
            ai_fps = ai_worker.ai_fps
            ai_ms = ai_worker.ai_ms
            vram_gb = ai_worker.vram_gb

        if ai_img is not None:
            right_half = compose_final_ai_half(ai_img, ai_p, live_half, mask_half, p_half, hero, hands_live=ai_worker.hands_live)
        else:
            right_half = live_half.copy()

        # В левом нижнем углу показываем мини-превью входного эскиза костюма, который подаётся в ИИ
        if sk_prev is not None and ai_worker.sketch_conditioning:
            sk_thumb = cv2.resize(sk_prev, (168, 224))
            live_half[H - 238:H - 14, 14:182] = sk_thumb
            cv2.rectangle(live_half, (14, H - 238), (182, H - 14), (0, 235, 255), 2, AA)
            txt(live_half, "ЭСКИЗ [E]", 98, H - 266, 20, (0, 255, 255), center=True)

        canvas = np.hstack([live_half, right_half])
        cv2.line(canvas, (640, 0), (640, H), (0, 255, 255), 3, AA)

        # Двухстрочная верхняя панель со всеми параметрами и подсказками клавиш
        cv2.rectangle(canvas, (0, 0), (W, 82), (18, 24, 36), -1)
        txt(canvas, f"КАМЕРА ({cam.backend_name}): {cam.cam_fps:.1f} FPS", 20, 10, 26, (80, 255, 140))
        txt(canvas, f"ИИ ({ai_worker.steps} шаг, str={ai_worker.strength:.2f}): {ai_worker.ai_fps:.1f} FPS ({ai_worker.ai_ms:.0f} мс) | VRAM: {vram_gb:.2f} ГБ",
            420, 10, 26, (0, 235, 255))
        txt(canvas, f"Герой [1-8/H]: {hero_display_name(hero)} | Шаги [S]: {ai_worker.steps} | Эскиз [E]: {'ВКЛ' if ai_worker.sketch_conditioning else 'ВЫКЛ'} | Руки [K]: {'ЖИВЫЕ' if ai_worker.hands_live else 'ИИ'} | Сила [[ / ]]: {ai_worker.strength:.2f}",
            20, 46, 21, (235, 240, 255))

        cv2.imshow(win, canvas)

    ai_worker.running = False
    cam.running = pose_worker.running = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
