@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=
rem Нужен именно Python 3.11 (mediapipe 0.10.14 и torch-directml не поддерживают 3.13+)
py -3.11 --version >nul 2>nul && set PY=py -3.11
if "%PY%"=="" py -3.10 --version >nul 2>nul && set PY=py -3.10
if "%PY%"=="" (
  echo Python 3.11 not found. Installing Python 3.11 via winget...
  winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
  py -3.11 --version >nul 2>nul && set PY=py -3.11
)
if "%PY%"=="" (
  echo.
  echo [ERROR] Could not install Python 3.11 automatically.
  echo Download manually: https://www.python.org/downloads/release/python-3119/  ^(Windows installer 64-bit^)
  echo Then run this file again.
  pause
  exit /b 1
)
echo Using: %PY%
%PY% --version
echo === Installing. Log: install_log.txt ===
(
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements_light.txt
%PY% -m pip install torch-directml
%PY% -m pip install diffusers transformers accelerate safetensors huggingface_hub
%PY% tools\download_models.py
) > install_log.txt 2>&1
type install_log.txt | more
echo.
echo === Finished. If errors above, send the LAST 15 lines. Then run start_amd_ai.bat ===
pause
