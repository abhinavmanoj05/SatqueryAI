@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
title SatQuery AI Launcher
cd /d "%~dp0"

echo ============================================
echo   SatQuery AI - Interactive Launcher
echo ============================================
echo.

if exist "ben_venv\Scripts\activate.bat" (
    call "ben_venv\Scripts\activate.bat"
) else (
    echo [WARNING] ben_venv not found. Using system python.
)

echo [1] Launch Gradio Web UI (ui.py)
echo [2] Run benchmark tests (benchmarks\test_all_models.py)
echo [3] Quick ResNet-18 inference (scripts\run_real_inference.py)
echo.

set /p choice="Select option (1/2/3): "

if "%choice%"=="1" (
    echo Launching Web UI...
    echo Access the Web UI in your browser at: http://127.0.0.1:7860
    start "" "http://127.0.0.1:7860"
    python ui.py
)
if "%choice%"=="2" (
    echo Running benchmark suite...
    python benchmarks\test_all_models.py
)
if "%choice%"=="3" (
    echo Running ResNet-18 inference...
    python scripts\run_real_inference.py
)

pause
