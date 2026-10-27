"""
Диагностика и проверка реального ИИ (tools/check_ai.py) для «Сиқырлы айна»:
1. Проверяет доступность PyTorch, CUDA, модель GPU и объём видеопамяти.
2. Проверяет наличие локальных моделей sd-turbo и taesd в папке models/.
3. Загружает пайплайн (SD-Turbo fp16 + TAESD), компилирует CUDA Graph и делает тестовый кадр.
4. Измеряет задержку (мс), FPS генерации и пиковый расход VRAM.
5. Сохраняет тестовый результат в assets/test_ai_frame.png.
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
from heroes import HEROES

MODELS_DIR = os.path.join(ROOT_DIR, "models")
SD_TURBO_DIR = os.path.join(MODELS_DIR, "sd-turbo")
TAESD_DIR = os.path.join(MODELS_DIR, "taesd")


def check_cuda():
    print("[1/4] Проверка CUDA и аппаратного ускорения GPU...")
    try:
        import torch
    except ImportError:
        print("  [ОШИБКА] PyTorch не установлен! Установите torch с поддержкой CUDA.")
        return False, None

    if not torch.cuda.is_available():
        print("  [ОШИБКА] CUDA недоступна в PyTorch! Проверьте видеокарту NVIDIA и драйверы.")
        return False, None

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cuda_ver = torch.version.cuda
    print(f"  CUDA: доступна (версия {cuda_ver})")
    print(f"  GPU:  {device_name} ({total_mem:.2f} ГБ VRAM)")
    return True, torch


def check_models():
    print("[2/4] Проверка наличия локальных весов моделей в models/...")
    ok = True
    if not os.path.isdir(SD_TURBO_DIR):
        print(f"  [ОШИБКА] Папка SD-Turbo не найдена: {SD_TURBO_DIR}")
        ok = False
    else:
        print(f"  SD-Turbo: найдена ({SD_TURBO_DIR}) [OK]")

    if not os.path.isdir(TAESD_DIR):
        print(f"  [ОШИБКА] Папка TAESD не найдена: {TAESD_DIR}")
        ok = False
    else:
        print(f"  TAESD:    найдена ({TAESD_DIR}) [OK]")

    if not ok:
        print("  Запустите 'python tools/bench_ai.py' для автоматической загрузки моделей.")
    return ok


def run_test_inference(torch):
    print("[3/4] Инициализация пайплайна (SD-Turbo fp16 + TAESD)...")
    from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny
    from diffusers.models.attention_processor import AttnProcessor2_0
    from ai_transform import HERO_PROMPTS, build_costume_sketch

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    t0_load = time.perf_counter()
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

    load_time = time.perf_counter() - t0_load
    print(f"  Модели успешно загружены в VRAM за {load_time:.2f} сек [OK]")

    print("[4/4] Тестовая генерация кадра (Alpamys, 384x512, steps=2)...")
    ai_w, ai_h = 384, 512
    hero = HEROES[0]  # Alpamys

    # Создаём синтетический входной силуэт ребёнка
    test_img = np.zeros((ai_h, ai_w, 3), dtype=np.uint8)
    test_img[:] = (45, 55, 75)
    cv2.circle(test_img, (ai_w // 2, int(ai_h * 0.28)), int(ai_w * 0.16), (190, 210, 240), -1)
    cv2.rectangle(test_img, (int(ai_w * 0.25), int(ai_h * 0.42)), (int(ai_w * 0.75), int(ai_h * 0.88)), (160, 180, 210), -1)

    test_mask = np.zeros((ai_h, ai_w), dtype=np.float32)
    cv2.circle(test_mask, (ai_w // 2, int(ai_h * 0.28)), int(ai_w * 0.17), 1.0, -1)
    cv2.rectangle(test_mask, (int(ai_w * 0.25), int(ai_h * 0.42)), (int(ai_w * 0.75), int(ai_h * 0.88)), 1.0, -1)

    # Синтетический скелет ребёнка
    p_syn = np.zeros((33, 3), dtype=np.float32)
    p_syn[0] = [ai_w * 0.50, ai_h * 0.25, 0.9]   # нос
    p_syn[11] = [ai_w * 0.30, ai_h * 0.42, 0.9]  # левое плечо
    p_syn[12] = [ai_w * 0.70, ai_h * 0.42, 0.9]  # правое плечо
    p_syn[23] = [ai_w * 0.35, ai_h * 0.85, 0.9]  # левое бедро
    p_syn[24] = [ai_w * 0.65, ai_h * 0.85, 0.9]  # правое бедро

    inp_small, _ = build_costume_sketch(test_img, test_mask, p_syn, hero, hands_live=True)
    inp_rgb = cv2.cvtColor(inp_small, cv2.COLOR_BGR2RGB)

    pr = HERO_PROMPTS["alpamys"]
    with torch.inference_mode():
        emb, _ = pipe.encode_prompt(pr, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)

    scaling_factor = float(pipe.vae.config.scaling_factor)
    alphas = pipe.scheduler.alphas_cumprod.cpu().numpy()
    lat_h, lat_w = ai_h // 8, ai_w // 8

    gen = torch.Generator(device=device).manual_seed(1001)
    n1 = torch.randn((1, 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    n2 = torch.randn((1, 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)

    strength = 0.48
    t0_idx = int(round(strength * 999))
    t1_idx = max(1, int(round(t0_idx * 0.48)))
    sig0 = (max(1e-5, (1.0 - alphas[t0_idx]) / alphas[t0_idx])) ** 0.5
    sig1 = (max(1e-5, (1.0 - alphas[t1_idx]) / alphas[t1_idx])) ** 0.5

    static_t = torch.tensor([t0_idx, t1_idx], device=device, dtype=torch.long)
    static_sig = torch.tensor([sig0, sig1, 0.0], device=device, dtype=torch.float16)

    in_t = (
        (torch.from_numpy(inp_rgb).to(device=device, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0) / 127.5) - 1.0
    ).contiguous(memory_format=torch.channels_last)

    # Прогревочный прогон
    with torch.inference_mode():
        lat = pipe.vae.encode(in_t).latents * scaling_factor
        x = lat + n1 * static_sig[0]
        for idx_s in range(2):
            sig = static_sig[idx_s]
            sig_next = static_sig[idx_s + 1]
            c_in = x / ((sig * sig + 1.0) ** 0.5)
            eps = pipe.unet(c_in, static_t[idx_s], encoder_hidden_states=emb, return_dict=False)[0]
            if idx_s == 1:
                x = x - sig * eps
            else:
                sig_up = ((sig_next ** 2) * (sig ** 2 - sig_next ** 2) / (sig ** 2)) ** 0.5
                sig_down = (sig_next ** 2 - sig_up ** 2) ** 0.5
                x = x + (sig_down - sig) * eps + n2 * sig_up
        out_buf = pipe.vae.decode(x / scaling_factor, return_dict=False)[0]
        torch.cuda.synchronize()

    # Замер скорости на 5 прогонах
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.inference_mode():
            lat = pipe.vae.encode(in_t).latents * scaling_factor
            x = lat + n1 * static_sig[0]
            for idx_s in range(2):
                sig = static_sig[idx_s]
                sig_next = static_sig[idx_s + 1]
                c_in = x / ((sig * sig + 1.0) ** 0.5)
                eps = pipe.unet(c_in, static_t[idx_s], encoder_hidden_states=emb, return_dict=False)[0]
                if idx_s == 1:
                    x = x - sig * eps
                else:
                    sig_up = ((sig_next ** 2) * (sig ** 2 - sig_next ** 2) / (sig ** 2)) ** 0.5
                    sig_down = (sig_next ** 2 - sig_up ** 2) ** 0.5
                    x = x + (sig_down - sig) * eps + n2 * sig_up
            out_buf = pipe.vae.decode(x / scaling_factor, return_dict=False)[0]
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t_start)

    avg_ms = (sum(times) / len(times)) * 1000.0
    ai_fps = 1000.0 / avg_ms
    vram_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    out_rgb = ((out_buf[0].permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    out_dir = os.path.join(ROOT_DIR, "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_ai_frame.png")
    cv2.imwrite(out_path, out_bgr)

    print("\n" + "=" * 55)
    print(" РЕЗУЛЬТАТЫ ПРОВЕРКИ ИИ (SD-Turbo fp16 + TAESD)")
    print("=" * 55)
    print(f" Статус:             ИИ ПОЛНОСТЬЮ ГОТОВ К РАБОТЕ [OK]")
    print(f" Время генерации:    {avg_ms:.1f} мс на кадр")
    print(f" Скорость ИИ:        {ai_fps:.1f} FPS")
    print(f" Расход VRAM:        {vram_used:.2f} ГБ")
    print(f" Сохранённый кадр:   {out_path}")
    print("=" * 55 + "\n")
    return True


def main():
    print("=" * 55)
    print(" Magic Mirror v2 — Диагностика и проверка реального ИИ")
    print("=" * 55)
    c_ok, torch_mod = check_cuda()
    if not c_ok:
        sys.exit(1)

    m_ok = check_models()
    if not m_ok:
        sys.exit(1)

    inf_ok = run_test_inference(torch_mod)
    if not inf_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
