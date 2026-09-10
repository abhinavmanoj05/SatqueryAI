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

echo [0] Launch SatQuery AI Unified App (http://127.0.0.1:8000) - Recommended
echo [1] Launch in Development Mode (Vite Dev Server + API)
echo [2] Run benchmark tests (benchmarks\test_all_models.py)
echo [3] Quick ViT-Base inference (scripts\run_vit_inference.py)
echo.

set /p choice="Select option (0/1/2/3) [Default: 0]: "
if "%choice%"=="" set choice=0

if "%choice%"=="0" (
    echo Launching SatQuery AI Unified Server...
    python run.py
)
if "%choice%"=="1" (
    echo Launching SatQuery AI in Development Mode...
    python run.py --dev
)
if "%choice%"=="2" (
    echo Running benchmark suite...
    python benchmarks\test_all_models.py
)
if "%choice%"=="3" (
    echo Running ViT-Base inference...
    python scripts\run_vit_inference.py
)

pause
