@echo off
cd /d "%~dp0"
set "PY_RUN=python"
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY_RUN=py -3.11"
py -3.10 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY_RUN=py -3.10"
%PY_RUN% setup_env.py --torch auto
if errorlevel 1 (
 echo Installation failed. Read the error above.
 pause
 exit /b 1
)
pause
