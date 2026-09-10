"""
SatQuery AI - Unified Single-Command Application Runner
======================================================
Launches both the SatQuery AI specialist model backend (Gradio on port 7860)
and the Production React + Three.js 3D Frontend (Vite on port 5173) simultaneously.

Usage:
    python run.py
    python run.py --backend-only
    python run.py --frontend-only
    python run.py --no-browser
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"


def get_python_executable() -> str:
    """Return Python executable inside ben_venv if available, else current."""
    if sys.platform == "win32":
        venv_py = ROOT_DIR / "ben_venv" / "Scripts" / "python.exe"
    else:
        venv_py = ROOT_DIR / "ben_venv" / "bin" / "python"

    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def ensure_frontend_deps() -> bool:
    """Check if node_modules exists in frontend/; install if missing."""
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        return True

    print("=" * 60)
    print("  [Setup] Installing frontend dependencies (first-time run)...")
    print("=" * 60)
    shell_cmd = "npm install"
    res = subprocess.run(shell_cmd, cwd=str(FRONTEND_DIR), shell=True)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Unified Runner")
    parser.add_argument("--backend-only", action="store_true", help="Launch only the Gradio backend UI (port 7860)")
    parser.add_argument("--frontend-only", action="store_true", help="Launch only the React frontend (port 5173)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = parser.parse_args()

    run_backend = not args.frontend_only
    run_frontend = not args.backend_only

    procs = []

    print("=" * 65)
    print("  [SatQuery AI] - Unified Launcher")
    print("=" * 65)

    py_exe = get_python_executable()
    print(f"[*] Python Interpreter : {py_exe}")
    print(f"[*] Workspace Root     : {ROOT_DIR}")

    if run_frontend:
        if not ensure_frontend_deps():
            print("[!] Warning: npm install had issues. Attempting to start frontend anyway.")

    try:
        # 1. Start Agentic Orchestrator FastAPI Backend (port 8000)
        if run_backend:
            print("\n[+] Launching SatQuery AI Orchestrator API (backend_api.py on port 8000)...")
            api_cmd = [py_exe, "-m", "uvicorn", "backend_api:app", "--host", "127.0.0.1", "--port", "8000"]
            api_proc = subprocess.Popen(
                api_cmd,
                cwd=str(ROOT_DIR),
                env=os.environ.copy(),
            )
            procs.append(("Orchestrator API (FastAPI)", api_proc))
            print("    -> Orchestrator API    : http://127.0.0.1:8000")

            # 2. Start Interactive Gradio UI (port 7860)
            print("\n[+] Launching SatQuery AI Interactive Assistant (ui.py on port 7860)...")
            backend_cmd = [py_exe, "ui.py"]
            backend_proc = subprocess.Popen(
                backend_cmd,
                cwd=str(ROOT_DIR),
                env=os.environ.copy(),
            )
            procs.append(("Backend (Gradio)", backend_proc))
            print("    -> Gradio Assistant    : http://127.0.0.1:7860")

        # 3. Start Production Frontend (Vite on port 5173)
        if run_frontend:
            print("\n[+] Launching SatQuery AI Production Frontend (npm run dev)...")
            frontend_proc = subprocess.Popen(
                "npm run dev",
                cwd=str(FRONTEND_DIR),
                shell=True,
                env=os.environ.copy(),
            )
            procs.append(("Frontend (Vite)", frontend_proc))
            print("    -> Frontend target URL: http://localhost:5173")

        print("\n" + "=" * 65)
        print("  [SUCCESS] All services started!")
        print("  - Production Frontend : http://localhost:5173")
        print("  - Orchestrator API    : http://127.0.0.1:8000")
        print("  - Gradio Assistant    : http://127.0.0.1:7860")
        print("  Press Ctrl+C at any time to gracefully terminate all services.")
        print("=" * 65 + "\n")

        # Give servers a few seconds to initialize before opening browser
        if not args.no_browser:
            time.sleep(3)
            target_url = "http://localhost:5173" if run_frontend else "http://127.0.0.1:7860"
            print(f"[*] Opening {target_url} in your default browser...")
            webbrowser.open(target_url)

        # Keep parent alive and monitor child processes
        while True:
            for name, proc in procs:
                ret = proc.poll()
                if ret is not None:
                    print(f"\n[!] Process '{name}' exited with code {ret}")
                    return ret
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[*] Shutting down SatQuery AI services...")
    finally:
        for name, proc in procs:
            if proc.poll() is None:
                print(f"[*] Terminating {name}...")
                if sys.platform == "win32":
                    subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    proc.terminate()
        print("[*] All services stopped cleanly.")


if __name__ == "__main__":
    main()
