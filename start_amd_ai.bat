@echo off
chcp 65001 >nul
cd /d "%~dp0"
set MM_PROFILE=amd

echo ===== Сиқырлы Айна (AMD AI Mode) =====

REM --- Определяем Python ---
set PY=
py -3.11 --version >nul 2>nul && set PY=py -3.11
if "%PY%"=="" (py -3.10 --version >nul 2>nul && set PY=py -3.10)
if "%PY%"=="" (python --version >nul 2>nul && set PY=python)
if "%PY%"=="" (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python311") do (
        if exist "%%D\python.exe" set PY=%%D\python.exe
    )
)
if "%PY%"=="" (
    echo [ERROR] Python не найден. Установите Python 3.11 и добавьте в PATH.
    pause
    exit /b 1
)
echo Python: %PY%

REM --- Проверяем модели (ищем любой safetensors файл в unet) ---
set MODELS_OK=0
if exist "models\sd-turbo\unet\diffusion_pytorch_model.fp16.safetensors" set MODELS_OK=1
if exist "models\sd-turbo\unet\diffusion_pytorch_model.safetensors" set MODELS_OK=1
if "%MODELS_OK%"=="0" (
    if not exist "models\taesd\config.json" (
        echo Models not found. Downloading ^(about 2.4 GB, needs internet^)...
        %PY% -m pip install huggingface_hub
        %PY% tools\download_models.py
    )
)

echo Starting game...
%PY% magic_mirror_v2.py

echo.
echo Game closed. If it crashed, send a screenshot of the text above.
pause
