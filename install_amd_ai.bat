@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Installing AI for AMD (DirectML). Needs Python 3.10-3.12 and internet. This takes several minutes.
python -m pip install --upgrade pip
python -m pip install -r requirements_light.txt
python -m pip install torch-directml diffusers transformers accelerate safetensors huggingface_hub
python tools\download_models.py
echo.
echo Done. Now run start_amd_ai.bat
pause
