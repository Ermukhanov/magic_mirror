@echo off
chcp 65001 >nul
cd /d "%~dp0"
set MM_PROFILE=amd

echo ==========================================================
echo   MAGIC MIRROR - AMD AI Mode
echo ==========================================================

REM --- Find Python ---
set "PY="
py -3.11 --version >nul 2>nul
if not errorlevel 1 set "PY=py -3.11"
if not defined PY (
    py -3.10 --version >nul 2>nul
    if not errorlevel 1 set "PY=py -3.10"
)
if not defined PY (
    python --version >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)
if not defined PY (
    echo [ERROR] Python not found! Install Python 3.11 and add to PATH.
    pause
    exit /b 1
)
echo Python: %PY%

REM --- Check if cv2 is installed, if not run install ---
"%PY%" -c "import cv2" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Dependencies not installed. Installing now...
    echo This may take a few minutes...
    echo.
    "%PY%" -m pip install --upgrade pip
    if exist "requirements_light.txt" (
        "%PY%" -m pip install -r requirements_light.txt
    ) else (
        "%PY%" -m pip install -r requirements.txt
    )
    "%PY%" -m pip install torch-directml
    "%PY%" -m pip install "diffusers==0.30.3" "transformers==4.44.2" "accelerate==0.34.2" "huggingface_hub==0.25.2" "numpy<2" "safetensors<0.5"
    echo.
    echo Dependencies installed!
)

REM --- Check models ---
set "MODELS_OK=0"
if exist "models\sd-turbo\unet\diffusion_pytorch_model.fp16.safetensors" set "MODELS_OK=1"
if exist "models\sd-turbo\unet\diffusion_pytorch_model.safetensors" set "MODELS_OK=1"
if "%MODELS_OK%"=="0" (
    echo.
    echo Models not found. Downloading - about 2.4 GB, needs internet...
    "%PY%" -m pip install huggingface_hub >nul 2>nul
    "%PY%" tools\download_models.py
)
if not exist "models\taesd\config.json" (
    echo Downloading TAESD model...
    "%PY%" tools\download_models.py
)

echo.
echo Starting game...
"%PY%" magic_mirror_v2.py

echo.
echo Game closed. If it crashed, send a screenshot of the text above.
pause
