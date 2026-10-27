@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "P="
where pythonw >nul 2>&1 && set "P=pythonw"
if not defined P where pyw >nul 2>&1 && set "P=pyw"
if not defined P where python >nul 2>&1 && set "P=python"
if not defined P where py >nul 2>&1 && set "P=py"

if not defined P (
    for %%D in (Python313 Python312 Python311 Python310) do (
        if not defined P if exist "%LOCALAPPDATA%\Programs\Python\%%D\pythonw.exe" set "P=%LOCALAPPDATA%\Programs\Python\%%D\pythonw.exe"
        if not defined P if exist "%LOCALAPPDATA%\Programs\Python\%%D\python.exe" set "P=%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
    )
)

if not defined P (
    echo Python табылмады! install_on_new_pc.bat іске қосыңыз
    pause
    exit /b 1
)

start "" "%P%" magic_mirror_v2.py
exit
