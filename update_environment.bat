@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run install_cpu.bat or install_gpu.bat first.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
 echo Update failed. Read the error above.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -c "import mir_eval, pyarrow, beat_this, scipy; mir_eval.chord.encode('C'); print('Dependencies ready. Run start.bat.')"
pause
