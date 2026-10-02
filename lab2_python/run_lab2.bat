@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 -X utf8 lab2.py
goto done
:use_python
python -X utf8 lab2.py
:done
if errorlevel 1 echo Install matplotlib: python -m pip install -r requirements.txt
pause
