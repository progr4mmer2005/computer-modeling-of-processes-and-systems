@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto install
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto install
echo Python not found. Install Python 3.10+ from https://www.python.org and tick "Add to PATH".
pause
exit /b 1
:install
echo [1/2] Installing libraries: numpy, scipy, matplotlib ...
%PY% -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
    echo Failed to install libraries. Check the internet connection.
    pause
    exit /b 1
)
echo [2/2] Running lab3.py ...
echo.
%PY% -X utf8 lab3.py
if errorlevel 1 echo Program finished with an error.
echo.
echo Results: %~dp0results
pause
