@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Dependencies are ready. You can launch the application now.
pause
exit /b 0
:failed
echo Setup failed. Check Python 3.11 and the network connection.
pause
exit /b 1
