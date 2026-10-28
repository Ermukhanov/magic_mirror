@echo off
chcp 65001 >nul
cd /d "%~dp0"
set MM_PROFILE=amd
where python >nul 2>nul && (python magic_mirror_v2.py & goto :eof)
where py >nul 2>nul && (py magic_mirror_v2.py & goto :eof)
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
  "%%D\python.exe" magic_mirror_v2.py
  goto :eof
)
echo Python not found.
pause
