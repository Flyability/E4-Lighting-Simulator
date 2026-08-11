@echo off
title LED Lighting Simulator
echo.
echo ============================================================
echo   LED Lighting Simulation - Desktop App
echo ============================================================
echo.
echo Starting server...
echo The browser will open automatically.
echo.
echo Close this window to stop the server.
echo ============================================================
echo.

cd /d "%~dp0"

REM Try venv python first, then system python
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe interactive_lighting.py
) else if exist "..\\.venv\\Scripts\\python.exe" (
    ..\\.venv\\Scripts\\python.exe interactive_lighting.py
) else if exist "dist\LightingSim\LightingSim.exe" (
    dist\LightingSim\LightingSim.exe
) else (
    python interactive_lighting.py
)

pause
