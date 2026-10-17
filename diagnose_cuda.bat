@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run install_gpu.bat first.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" repair_cuda.py --check
if errorlevel 1 (
 echo CUDA check or repair failed. See output and cuda_diagnostic.json.
 pause
 exit /b 1
)
pause
