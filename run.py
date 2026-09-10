"""
SatQuery AI - Unified Single-Command Application Runner
======================================================
Launches the SatQuery AI unified application on http://127.0.0.1:8000.
FastAPI serves both the production React SPA (from frontend/dist)
and the Agentic Orchestrator REST endpoints (/api/analyze, /api/models, /api/health).

Usage:
    python run.py                   # Production mode: single server on port 8000 (No Node.js needed)
    python run.py --dev             # Development mode: Vite on 5173 + FastAPI on 8000
    python run.py --no-browser      # Run without opening browser
    python run.py --port 8000       # Specify custom port
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
FRONTEND_DIST = FRONTEND_DIR / "dist"


def get_python_executable() -> str:
    """Return Python executable inside ben_venv if available, else current."""
    if sys.platform == "win32":
        venv_py = ROOT_DIR / "ben_venv" / "Scripts" / "python.exe"
    else:
        venv_py = ROOT_DIR / "ben_venv" / "bin" / "python"

    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def ensure_frontend_built() -> bool:
    """Ensure frontend/dist exists. If missing, runs npm run build."""
    if (FRONTEND_DIST / "index.html").exists():
        return True

    print("=" * 60)
    print("  [Setup] Building production frontend (first-time build)...")
    print("=" * 60)
    shell_cmd = "npm run build"
    res = subprocess.run(shell_cmd, cwd=str(FRONTEND_DIR), shell=True)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Unified Application Runner")
    parser.add_argument("--dev", action="store_true", help="Launch in development mode (Vite dev server on 5173 + API on 8000)")
    parser.add_argument("--port", type=int, default=8000, help="Port to host SatQuery AI (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = parser.parse_args()

    procs = []

    print("=" * 65)
    print("  [SatQuery AI v2] - Unified Application Server")
    print("=" * 65)

    py_exe = get_python_executable()
    print(f"[*] Python Interpreter : {py_exe}")
    print(f"[*] Workspace Root     : {ROOT_DIR}")

    if not args.dev:
        if not ensure_frontend_built():
            print("[!] Warning: Frontend build had issues. Falling back to dev mode.")
            args.dev = True

    try:
        # 1. Start Agentic Orchestrator FastAPI Backend
        print(f"\n[+] Launching SatQuery AI Unified Server (backend_api.py on port {args.port})...")
        api_cmd = [py_exe, "-m", "uvicorn", "backend_api:app", "--host", "127.0.0.1", "--port", str(args.port)]
        api_proc = subprocess.Popen(
            api_cmd,
            cwd=str(ROOT_DIR),
            env=os.environ.copy(),
        )
        procs.append(("SatQuery AI Server (FastAPI)", api_proc))

        target_url = f"http://127.0.0.1:{args.port}"

        # 2. If development mode requested, also launch Vite hot-reload server
        if args.dev:
            print("\n[+] Launching Vite Development Server (npm run dev on port 5173)...")
            dev_proc = subprocess.Popen(
                "npm run dev",
                cwd=str(FRONTEND_DIR),
                shell=True,
                env=os.environ.copy(),
            )
            procs.append(("Frontend Dev Server (Vite)", dev_proc))
            target_url = "http://localhost:5173"

        print("\n" + "=" * 65)
        print("  [SUCCESS] SatQuery AI is running!")
        if args.dev:
            print("  - Mode                : Development (Hot Reloading)")
            print(f"  - Application UI      : {target_url}")
            print(f"  - Orchestrator API    : http://127.0.0.1:{args.port}")
        else:
            print("  - Mode                : Production (Unified Single Server)")
            print(f"  - Application URL     : {target_url}")
            print("  - Active Models       : ViT-Base, Google Gemini 3.8 Flash, CDVQA")
        print("  Press Ctrl+C at any time to gracefully terminate.")
        print("=" * 65 + "\n")

        # Give server time to bind before opening browser
        if not args.no_browser:
            time.sleep(2)
            print(f"[*] Opening {target_url} in your default browser...")
            webbrowser.open(target_url)

        # Monitor child processes
        while True:
            for name, proc in procs:
                ret = proc.poll()
                if ret is not None:
                    print(f"\n[!] Process '{name}' exited with code {ret}")
                    return ret
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[*] Shutting down SatQuery AI...")
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
