@echo off
cd /d "%~dp0"
echo SPECTRA LOCAL FIX R32 / EPIC GAP 3px
echo Project: %CD%
if not exist ".venv\Scripts\python.exe" (
 echo Extract this ZIP directly into your existing music2emo_test folder.
 echo start_spectra_fixed.bat must be beside .venv and spectra_server.py.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" spectra_server.py
pause
