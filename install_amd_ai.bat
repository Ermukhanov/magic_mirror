@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=
where python >nul 2>nul && set PY=python
if "%PY%"=="" where py >nul 2>nul && set PY=py
if "%PY%"=="" for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do set PY="%%D\python.exe"
if "%PY%"=="" (
  echo [ERROR] Python not found. Install Python 3.10-3.12 and tick "Add python.exe to PATH".
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
echo === Finished. If you see errors above, send a screenshot of the LAST 15 lines. ===
pause
