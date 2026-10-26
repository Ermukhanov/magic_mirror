"""
ЭТАП 1 — Замер производительности ИИ (tools/bench_ai.py) для «Сиқырлы айна».
1. Проверяет зависимости (PyTorch с CUDA, diffusers, transformers, accelerate, safetensors, onnxruntime).
2. Скачивает локально в папку models/ модели:
   - stabilityai/sd-turbo (fp16) -> models/sd-turbo
   - madebyollin/taesd (fp16)     -> models/taesd
   После скачивания всё работает полностью ОФЛАЙН.
3. Замеряет скорость img2img (ИИ-кадров/с, задержка в мс, пиковый VRAM в ГБ) при разрешениях:
   - 384x384 и 384x512 (при 1 и 2 шагах)
   - Без компиляции графа (Standard Diffusers fp16 + TAESD)
   - С оптимизацией графа (Direct-Stream + CUDA Graph / torch.compile + TAESD fp16)
4. Сохраняет рекомендуемые настройки, флаги и промты всех 8 героев в config.json и выводит таблицу.
"""
import os
import sys
import json
import time
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
SD_TURBO_DIR = os.path.join(MODELS_DIR, "sd-turbo")
TAESD_DIR = os.path.join(MODELS_DIR, "taesd")
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

HERO_PROMPTS = {
    "alpamys": "wearing a long red cloak and shining golden Kazakh armor, steppe ornaments, wide belt and golden bracers, child-friendly cartoon style",
    "tomiris": "wearing royal purple and shining golden Saka warrior queen armor, golden crown, ornate belt and purple cape, child-friendly cartoon style",
    "spider": "wearing a bright red and blue superhero suit with black web pattern on chest and shoulders, child-friendly cartoon style",
    "elsa": "wearing a sparkling light blue crystal ice princess gown with shimmering snowflake cape and ice bodice, child-friendly cartoon style",
    "ladybug": "wearing a bright red superheroine suit with black polka dots and high collar, child-friendly cartoon style",
    "iron": "wearing shining red and gold high-tech armor with a bright glowing blue chest reactor and golden shoulder pauldrons, child-friendly cartoon style",
    "astronaut": "wearing a bright white space suit with chest control panel, blue-red stripes and shoulder patches, helmet-less open face, child-friendly cartoon style",
    "fairy": "wearing a glowing pink and emerald green fairy dress with shimmering butterfly wings and golden floral belt, child-friendly cartoon style",
}
NEGATIVE_PROMPT = "bad anatomy, extra limbs, deformed hands, text, logo"


def ensure_dependencies():
    """Проверяет наличие CUDA PyTorch, diffusers, transformers, accelerate, safetensors, onnxruntime."""
    chk = subprocess.run(
        [sys.executable, "-c", "import torch, torchvision; assert torch.cuda.is_available()"],
        capture_output=True,
    )
    if chk.returncode != 0:
        print("[BENCH_AI] Устанавливаем PyTorch с поддержкой CUDA (cu124)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch==2.6.0+cu124", "torchvision==0.21.0+cu124",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ])

    for pkg in ("diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub", "onnxruntime"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"[BENCH_AI] Установка пакета {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def download_models_offline():
    """Скачивает локально fp16-веса SD-Turbo и TAESD в папку models/."""
    from huggingface_hub import snapshot_download

    os.makedirs(MODELS_DIR, exist_ok=True)
    unet_safetensors = os.path.join(SD_TURBO_DIR, "unet", "diffusion_pytorch_model.fp16.safetensors")
    unet_std = os.path.join(SD_TURBO_DIR, "unet", "diffusion_pytorch_model.safetensors")
    if not (os.path.exists(unet_safetensors) or os.path.exists(unet_std)):
        print("[BENCH_AI] Скачивание stabilityai/sd-turbo (fp16) в models/sd-turbo ...")
        snapshot_download(
            repo_id="stabilityai/sd-turbo",
            local_dir=SD_TURBO_DIR,
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "text_encoder/*.json",
                "text_encoder/model.fp16.safetensors",
                "unet/*.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/*.json",
                "vae/diffusion_pytorch_model.fp16.safetensors",
            ],
        )
        print("[BENCH_AI] SD-Turbo (fp16) сохранён в models/sd-turbo")
    else:
        print("[BENCH_AI] Локальная модель models/sd-turbo готова (офлайн).")

    taesd_cfg = os.path.join(TAESD_DIR, "config.json")
    if not os.path.exists(taesd_cfg):
        print("[BENCH_AI] Скачивание madebyollin/taesd в models/taesd ...")
        snapshot_download(
            repo_id="madebyollin/taesd",
            local_dir=TAESD_DIR,
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )
        print("[BENCH_AI] TAESD сохранён в models/taesd")
    else:
        print("[BENCH_AI] Локальная модель models/taesd готова (офлайн).")


def run_benchmark():
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny
    from diffusers.models.attention_processor import AttnProcessor2_0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна!")

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n[BENCH_AI] GPU: {gpu_name} ({total_vram_gb:.2f} GB VRAM)")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("[BENCH_AI] Загрузка локальных моделей models/sd-turbo и models/taesd (fp16, local_files_only=True)...")
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

    # Предвычисляем эмбеддинги промтов и выгружаем text_encoder на CPU (экономит ~1.2 ГБ VRAM)
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=HERO_PROMPTS["alpamys"],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    has_trt = False
    try:
        import tensorrt  # noqa: F401
        has_trt = True
    except ImportError:
        has_trt = False

    resolutions = [(384, 384), (384, 512)]
    step_counts = [1, 2]
    results = []

    print("\n[BENCH_AI] 1) Замер БЕЗ компиляции графа (Standard Diffusers fp16 + TAESD)...")
    for (w, h) in resolutions:
        dummy_np = np.full((h, w, 3), 128, dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_np)
        for steps in step_counts:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            n_infer = 2 if steps == 1 else 4
            strength = 0.50

            with torch.inference_mode():
                for _ in range(3):
                    _ = pipe(
                        image=dummy_pil,
                        prompt_embeds=prompt_embeds,
                        num_inference_steps=n_infer,
                        strength=strength,
                        guidance_scale=0.0,
                        output_type="np",
                    ).images[0]
            torch.cuda.synchronize()

            N_RUNS = 14
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(N_RUNS):
                    _ = pipe(
                        image=dummy_pil,
                        prompt_embeds=prompt_embeds,
                        num_inference_steps=n_infer,
                        strength=strength,
                        guidance_scale=0.0,
                        output_type="np",
                    ).images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            fps = N_RUNS / elapsed
            ms = (elapsed / N_RUNS) * 1000.0
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            row = {
                "backend": "Без TRT/Compile (Diffusers fp16 + TAESD)",
                "backend_key": "diffusers",
                "resolution": f"{w}x{h}",
                "width": w,
                "height": h,
                "steps": steps,
                "fps": round(fps, 2),
                "ms": round(ms, 1),
                "vram_gb": round(vram_gb, 2),
            }
            results.append(row)
            print(f"  [Без TRT/Compile] {w}x{h} | шаги={steps} -> {fps:.2f} FPS ({ms:.1f} мс) | VRAM: {vram_gb:.2f} ГБ")

    print("\n[BENCH_AI] 2) Замер С оптимизацией графа (Direct-Euler + CUDA Graph + TAESD fp16)...")
    scaling_factor = float(pipe.vae.config.scaling_factor)

    for (w, h) in resolutions:
        lat_h, lat_w = h // 8, w // 8
        dummy_np = np.full((h, w, 3), 128, dtype=np.uint8)
        static_in_img = (
            (torch.from_numpy(dummy_np).to(device=device, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0) / 127.5) - 1.0
        ).contiguous(memory_format=torch.channels_last)

        gen = torch.Generator(device=device).manual_seed(1001)
        static_noise = torch.randn((1, 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        static_noise_2 = torch.randn((1, 4, lat_h, lat_w), generator=gen, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)

        for steps in step_counts:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            n_sched = 2 if steps == 1 else 4
            pipe.scheduler.set_timesteps(n_sched, device=device)
            t_steps = pipe.scheduler.timesteps[-steps:]
            sigmas = pipe.scheduler.sigmas[-(steps + 1):].to(device=device, dtype=torch.float16)

            def direct_step():
                lat = pipe.vae.encode(static_in_img).latents * scaling_factor
                x = lat + static_noise * sigmas[0]
                for idx_s in range(steps):
                    sig = sigmas[idx_s]
                    sig_next = sigmas[idx_s + 1]
                    c_in = x / torch.sqrt(sig * sig + 1.0)
                    eps = pipe.unet(c_in, t_steps[idx_s], encoder_hidden_states=prompt_embeds, return_dict=False)[0]
                    if idx_s == steps - 1:
                        x = x - sig * eps
                    else:
                        sig_up = torch.sqrt((sig_next ** 2) * (sig ** 2 - sig_next ** 2) / (sig ** 2))
                        sig_down = torch.sqrt(sig_next ** 2 - sig_up ** 2)
                        x = x + (sig_down - sig) * eps + static_noise_2 * sig_up
                out_dec = pipe.vae.decode(x / scaling_factor, return_dict=False)[0]
                return out_dec

            # Пробуем захват CUDA Graph для нулевого оверхеда CPU
            use_graph = False
            g_cuda = None
            static_out = None
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s), torch.inference_mode():
                    for _ in range(3):
                        static_out = direct_step()
                torch.cuda.current_stream().wait_stream(s)
                g_cuda = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_cuda), torch.inference_mode():
                    static_out = direct_step()
                use_graph = True
            except Exception:
                use_graph = False
                with torch.inference_mode():
                    for _ in range(3):
                        static_out = direct_step()
            torch.cuda.synchronize()

            N_RUNS = 20
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(N_RUNS):
                    if use_graph and g_cuda is not None:
                        g_cuda.replay()
                        res_t = static_out
                    else:
                        res_t = direct_step()
                    _ = ((res_t[0].permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            fps = N_RUNS / elapsed
            ms = (elapsed / N_RUNS) * 1000.0
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            b_label = f"С оптимизацией ({'CUDA Graph' if use_graph else 'Direct-Euler'} + TAESD fp16)"
            row = {
                "backend": b_label,
                "backend_key": "cuda_graph" if use_graph else "direct_euler",
                "resolution": f"{w}x{h}",
                "width": w,
                "height": h,
                "steps": steps,
                "fps": round(fps, 2),
                "ms": round(ms, 1),
                "vram_gb": round(vram_gb, 2),
            }
            results.append(row)
            print(f"  [{b_label}] {w}x{h} | шаги={steps} -> {fps:.2f} FPS ({ms:.1f} мс) | VRAM: {vram_gb:.2f} ГБ")

    print("\n" + "=" * 96)
    print(f"{'Бэкенд / Режим':<52} | {'Разрешение':<10} | {'Шаги':<4} | {'ИИ FPS':<8} | {'Задержка':<9} | {'VRAM':<8}")
    print("-" * 96)
    for r in results:
        print(f"{r['backend']:<52} | {r['resolution']:<10} | {r['steps']:<4} | {r['fps']:<8.2f} | {r['ms']:<6.1f} мс | {r['vram_gb']:<5.2f} ГБ")
    print("=" * 96)

    best_384x512 = max([r for r in results if r["width"] == 384 and r["height"] == 512], key=lambda x: x["fps"])
    best_1step_384x512 = max([r for r in results if r["width"] == 384 and r["height"] == 512 and r["steps"] == 1], key=lambda x: x["fps"])
    best_2step_384x512 = max([r for r in results if r["width"] == 384 and r["height"] == 512 and r["steps"] == 2], key=lambda x: x["fps"])

    chosen_regime = "A" if best_384x512["fps"] >= 8.0 else "B"
    rec_row = best_1step_384x512

    print(f"\n[ИТОГ ЗАМЕРА]: При 384x512 (1 шаг) = {best_1step_384x512['fps']:.2f} FPS ({best_1step_384x512['ms']:.1f} мс), при 2 шагах = {best_2step_384x512['fps']:.2f} FPS, VRAM = {rec_row['vram_gb']:.2f} ГБ.")
    if chosen_regime == "A":
        print(">>> РЕКОМЕНДУЕМЫЙ РЕЖИМ: РЕЖИМ А (потоковое ИИ-превращение на каждом кадре, т.к. FPS >= 8.0) <<<")
    else:
        print(">>> РЕКОМЕНДУЕМЫЙ РЕЖИМ: РЕЖИМ Б (генерация ИИ-костюма под ребёнка при калибровке, т.к. FPS < 8.0) <<<")

    cfg = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    cfg["ai_transform"] = cfg.get("ai_transform", "auto")
    cfg["video_worlds"] = cfg.get("video_worlds", True)
    cfg["fx_clips"] = cfg.get("fx_clips", True)
    cfg["hero_art"] = cfg.get("hero_art", True)
    cfg["game_layer"] = cfg.get("game_layer", True)
    cfg["ai_regime"] = chosen_regime
    cfg["ai_recommended"] = {
        "gpu": gpu_name,
        "tensorrt_available": has_trt,
        "backend": rec_row["backend_key"],
        "width": 384,
        "height": 512,
        "steps": 1,
        "strength": 0.50,
        "ema_alpha": 0.50,
        "measured_fps_1step_384x512": best_1step_384x512["fps"],
        "measured_fps_2step_384x512": best_2step_384x512["fps"],
        "measured_latency_ms": best_1step_384x512["ms"],
        "measured_vram_gb": rec_row["vram_gb"],
        "fallback_min_fps": 8.0,
        "fallback_timeout_sec": 3.0,
        "sd_turbo_license": "Stability AI Non-Commercial Research Community License (некоммерческое / образовательное использование бесплатно)",
        "benchmark_table": results,
    }
    cfg["ai_prompts"] = HERO_PROMPTS
    cfg["ai_negative_prompt"] = NEGATIVE_PROMPT
    cfg["ai_hero_seeds"] = {
        "alpamys": 1001,
        "tomiris": 2002,
        "spider": 3003,
        "elsa": 4004,
        "ladybug": 5005,
        "iron": 6006,
        "astronaut": 7007,
        "fairy": 8008,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[BENCH_AI] Рекомендуемые настройки сохранены в {CONFIG_PATH}")


if __name__ == "__main__":
    ensure_dependencies()
    download_models_offline()
    run_benchmark()
