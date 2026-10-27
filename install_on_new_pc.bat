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
    echo [Python табылмады / Python не найден!]
    if exist "python-3.12-installer.exe" (
        echo Python орнатушысы іске қосылуда... / Запуск установщика Python...
        echo.
        echo ====================================================================
        echo  МАҢЫЗДЫ / ВАЖНО: 
        echo  ТӨМЕНДЕ "Add python.exe to PATH" ҚҰСБЕЛГІСІН МІНДЕТТІ ТҮРДЕ ҚОЙЫҢЫЗ!
        echo  ОБЯЗАТЕЛЬНО ПОСТАВЬТЕ ГАЛОЧКУ "Add python.exe to PATH" ВНИЗУ ОКНА!
        echo ====================================================================
        echo.
        start /wait python-3.12-installer.exe
    ) else (
        echo Python 3.10 - 3.12 нұсқасын орнатыңыз.
        pause
        exit /b 1
    )
)

echo [1/3] Қажетті кітапханаларды орнату... / Установка библиотек...
python -m pip install --upgrade pip
if exist "requirements_light.txt" (
    python -m pip install -r requirements_light.txt
) else (
    python -m pip install -r requirements.txt
)

echo.
echo [2/3] Жұмыс үстеліне таңбаша жасау... / Создание ярлыка на Рабочий стол...
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Desk = [Environment]::GetFolderPath('Desktop'); $Shortcut = $WshShell.CreateShortcut(\"$Desk\Сиқырлы Айна.lnk\"); $Shortcut.TargetPath = 'pythonw.exe'; $Shortcut.Arguments = 'magic_mirror_v2.py'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Description = 'Сиқырлы Айна'; $Shortcut.Save()"

echo.
echo [3/3] Орнату сәтті аяқталды! Ойын іске қосылуда...
echo       Установка успешно завершена! Запуск игры...
start "" pythonw magic_mirror_v2.py
exit
