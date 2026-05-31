"""Build script: npm build for Vue frontend + PyInstaller for the app."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def step(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def main():
    # 1. Build Vue frontend
    frontend_dir = ROOT / "frontend"
    if (frontend_dir / "package.json").exists():
        step("Building Vue frontend...")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=True)
        subprocess.run(["npm", "run", "build"], cwd=str(frontend_dir), check=True)
        print(f"  Frontend built -> {frontend_dir / 'dist'}")
    else:
        print("WARNING: frontend/package.json not found, skipping npm build")

    # 2. PyInstaller
    step("Running PyInstaller...")
    spec = ROOT / "build" / "eht_analytica.spec"
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec), "--clean", "--noconfirm"],
        cwd=str(ROOT),
        check=True,
    )

    # 3. Copy output
    dist_dir = ROOT / "dist" / "EHT_Analytica"
    build_dir = ROOT / "dist" / "EHT_Analytica"
    step(f"Build complete! Output: {build_dir}")
    print("  Distribute this folder. Users run EHT_Analytica.exe to start.")

if __name__ == "__main__":
    main()
