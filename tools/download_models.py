"""Скачивает модели ИИ (SD-Turbo fp16 + TAESD, ~2.4 ГБ) в models/ — нужно один раз, дальше игра работает офлайн."""
import os
from huggingface_hub import snapshot_download

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")

if not os.path.exists(os.path.join(MODELS, "sd-turbo", "unet", "diffusion_pytorch_model.fp16.safetensors")):
    print("Скачивание stabilityai/sd-turbo ...")
    snapshot_download(
        repo_id="stabilityai/sd-turbo",
        local_dir=os.path.join(MODELS, "sd-turbo"),
        allow_patterns=[
            "model_index.json", "scheduler/*", "tokenizer/*",
            "text_encoder/*.json", "text_encoder/model.fp16.safetensors",
            "unet/*.json", "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/*.json", "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
    )
if not os.path.exists(os.path.join(MODELS, "taesd", "config.json")):
    print("Скачивание madebyollin/taesd ...")
    snapshot_download(repo_id="madebyollin/taesd", local_dir=os.path.join(MODELS, "taesd"))
print("Модели готовы.")
