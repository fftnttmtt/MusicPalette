@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run install_cpu.bat or install_gpu.bat first.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" download_models.py
pause
