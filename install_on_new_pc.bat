@echo off
chcp 65001 >nul
title Орнату және іске қосу / Установка и запуск Magic Mirror
cd /d "%~dp0"

echo ==========================================================
echo   СИҚЫРЛЫ АЙНА (MAGIC MIRROR) — ЖАҢА НОУТБУККЕ ОРНАТУ
echo ==========================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ҚАТЕ / ОШИБКА] Python табылмады! / Python не найден!
    echo Python 3.10 - 3.12 нұсқасын орнатыңыз және "Add Python to PATH" белгілеңіз.
    echo Пожалуйста, установите Python с python.org и отметьте галочку "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo [1/3] Қажетті кітапханаларды орнату... / Установка библиотек...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo [2/3] Жұмыс үстеліне таңбаша (ярлык) жасау... / Создание ярлыка на Рабочий стол...
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Desk = [Environment]::GetFolderPath('Desktop'); $Shortcut = $WshShell.CreateShortcut(\"$Desk\Сиқырлы Айна.lnk\"); $Shortcut.TargetPath = 'pythonw.exe'; $Shortcut.Arguments = 'magic_mirror_v2.py'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Description = 'Сиқырлы Айна'; $Shortcut.Save()"

echo.
echo [3/3] Орнату сәтті аяқталды! Ойын іске қосылуда...
echo       Установка успешно завершена! Запуск игры...
start "" pythonw magic_mirror_v2.py
exit
