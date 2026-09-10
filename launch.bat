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

echo [0] Launch Everything (Backend + Production React UI) - Recommended
echo [1] Launch Gradio Web UI only (ui.py)
echo [2] Run benchmark tests (benchmarks\test_all_models.py)
echo [3] Quick ResNet-18 inference (scripts\run_real_inference.py)
echo [4] Launch Production React Frontend only (Vite UI)
echo.

set /p choice="Select option (0/1/2/3/4) [Default: 0]: "
if "%choice%"=="" set choice=0

if "%choice%"=="0" (
    echo Launching complete SatQuery AI system...
    python run.py
)

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
if "%choice%"=="4" (
    echo Launching Production React Frontend...
    cd frontend
    if not exist "node_modules\" (
        echo [INFO] Installing frontend dependencies (first-time setup)...
        call npm install
    )
    start "" "http://localhost:5173"
    call npm run dev
)

pause
