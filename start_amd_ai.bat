@echo off
chcp 65001 >nul
cd /d "%~dp0"
set MM_PROFILE=amd
if not exist "models\sd-turbo\unet\diffusion_pytorch_model.fp16.safetensors" (
  echo Models not found. Downloading ^(about 2.4 GB, needs internet^)...
  py -3.11 -m pip install huggingface_hub
  py -3.11 tools\download_models.py
)
if not exist "models\taesd\config.json" (
  py -3.11 tools\download_models.py
)
if not exist "models\sd-turbo\unet\diffusion_pytorch_model.fp16.safetensors" (
  echo.
  echo [ERROR] Models were not downloaded. Check internet and send a screenshot of the text above.
  pause
  exit /b 1
)
py -3.11 --version >nul 2>nul && (py -3.11 magic_mirror_v2.py & goto :end)
py -3.10 --version >nul 2>nul && (py -3.10 magic_mirror_v2.py & goto :end)
echo Python 3.11 not found. Run install_amd_ai.bat first.
:end
echo.
echo Game closed. If it crashed, send a screenshot of the text above.
pause

