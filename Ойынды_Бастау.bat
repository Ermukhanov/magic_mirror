@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" pythonw magic_mirror_v2.py
exit
