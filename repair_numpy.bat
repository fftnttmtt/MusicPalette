@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Existing environment not found. Run install_cpu.bat first.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade mir_eval==0.8.2
if errorlevel 1 (
 echo Repair failed. Read the error above.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -c "import mir_eval; print(mir_eval.chord.encode('C')); print('Chord encoding OK. Restart start.bat.')"
if errorlevel 1 (
 echo Verification failed. Read the error above.
 pause
 exit /b 1
)
pause
