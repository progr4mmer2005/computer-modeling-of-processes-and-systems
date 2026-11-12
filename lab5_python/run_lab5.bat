@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m pip install -q -r requirements.txt
%PY% -X utf8 lab5.py
if errorlevel 1 goto end
%PY% -X utf8 ..\tools\build_lab5_report.py
:end
pause
