@echo off
chcp 65001 >nul
cd /d "%~dp0"
set MM_PROFILE=amd
py -3.11 magic_mirror_v2.py && goto :eof
py -3.10 magic_mirror_v2.py && goto :eof
echo Python 3.11 not found. Run install_amd_ai.bat first.
pause
