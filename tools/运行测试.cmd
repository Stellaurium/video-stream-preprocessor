@echo off
setlocal
cd /d "%~dp0.."
if not defined QT_QPA_PLATFORM set "QT_QPA_PLATFORM=offscreen"
set "PYTHONDONTWRITEBYTECODE=1"
".venv\Scripts\python.exe" -m pytest -q %*
if errorlevel 1 (
    echo Tests failed. See the output above.
    pause
    exit /b 1
)
exit /b 0
