@echo off
chcp 65001 >nul
title Орнату және іске қосу / Установка и запуск Magic Mirror
cd /d "%~dp0"

echo ==========================================================
echo   СИҚЫРЛЫ АЙНА (MAGIC MIRROR) — ОРНАТУ ЖӘНЕ ІСКЕ ҚОСУ
echo ==========================================================
echo.

:: 1. Ищем рабочий Python
set "PY_CMD="
set "PYW_CMD="

:: Проверяем pythonw в PATH
where pythonw >nul 2>&1 && set "PYW_CMD=pythonw"
:: Проверяем pyw в PATH
if not defined PYW_CMD where pyw >nul 2>&1 && set "PYW_CMD=pyw"

:: Проверяем python в PATH
where python >nul 2>&1 && set "PY_CMD=python"
:: Проверяем py в PATH
if not defined PY_CMD where py >nul 2>&1 && set "PY_CMD=py"

:: Если в PATH не найден, ищем в стандартных папках установки
if not defined PY_CMD (
    for %%D in (Python313 Python312 Python311 Python310) do (
        if not defined PY_CMD if exist "%LOCALAPPDATA%\Programs\Python\%%D\python.exe" (
            set "PY_CMD=%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
            set "PYW_CMD=%LOCALAPPDATA%\Programs\Python\%%D\pythonw.exe"
        )
        if not defined PY_CMD if exist "C:\Program Files\Python\%%D\python.exe" (
            set "PY_CMD=C:\Program Files\Python\%%D\python.exe"
            set "PYW_CMD=C:\Program Files\Python\%%D\pythonw.exe"
        )
    )
)

:: Если pythonw не найден, используем обычный python
if not defined PYW_CMD set "PYW_CMD=%PY_CMD%"

if not defined PY_CMD (
    echo [ҚАТЕ / ОШИБКА] Python бұл ноутбукте орнатылмаған!
    echo [ОШИБКА] Python на этом ноутбуке ещё не установлен!
    echo.
    echo 1. Браузерден Python жүктеп орнатыңыз / Скачайте Python:
    echo    https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe
    echo.
    echo 2. МАҢЫЗДЫ: Орнату кезінде "Add python.exe to PATH" құсбелгісін қойыңыз!
    echo    ВАЖНО: При установке ОБЯЗАТЕЛЬНО отметьте "Add python.exe to PATH"!
    echo.
    pause
    exit /b 1
)

echo [OK] Python табылды: %PY_CMD%
echo.
echo [1/3] Қажетті кітапханаларды орнату... / Установка библиотек...
"%PY_CMD%" -m pip install --upgrade pip
if exist "requirements_light.txt" (
    "%PY_CMD%" -m pip install -r requirements_light.txt
) else (
    "%PY_CMD%" -m pip install -r requirements.txt
)

echo.
echo [2/3] Жұмыс үстеліне таңбаша жасау... / Создание ярлыка на Рабочий стол...
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Desk = [Environment]::GetFolderPath('Desktop'); $Shortcut = $WshShell.CreateShortcut(\"$Desk\Сиқырлы Айна.lnk\"); $Shortcut.TargetPath = '%PYW_CMD%'; $Shortcut.Arguments = 'magic_mirror_v2.py'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Description = 'Сиқырлы Айна'; $Shortcut.Save()"

echo.
echo [3/3] Орнату сәтті аяқталды! Ойын іске қосылуда...
echo       Установка успешно завершена! Запуск игры...
start "" "%PYW_CMD%" magic_mirror_v2.py
exit
