"""Build script: npm build for Vue frontend + PyInstaller + assemble release folder.

Output: ``dist/EHT_Analytica/`` — self-contained release folder.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def build_frontend() -> None:
    """Build Vue frontend into frontend/dist/."""
    frontend_dir = ROOT / "frontend"
    dist_dir = frontend_dir / "dist"
    if (dist_dir / "index.html").exists():
        print("  Frontend dist/ already exists, skipping npm build")
        return

    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print("WARNING: frontend/package.json not found, skipping npm build")
        return

    step("Building Vue frontend...")
    # Resolve npm via PATH or common install locations
    npm_exe = "npm"
    for p in [r"D:\nodejs\npm.cmd", r"C:\Program Files\nodejs\npm.cmd"]:
        if os.path.isfile(p):
            npm_exe = p
            break
    env = os.environ.copy()
    # Ensure common node install paths are in PATH
    for node_path in [r"D:\nodejs", r"C:\Program Files\nodejs"]:
        if os.path.isdir(node_path):
            env["PATH"] = node_path + os.pathsep + env.get("PATH", "")
    subprocess.run([npm_exe, "install"], cwd=str(frontend_dir), check=True, env=env)
    subprocess.run([npm_exe, "run", "build"], cwd=str(frontend_dir), check=True, env=env)
    print(f"  Frontend built -> {dist_dir}")


def run_pyinstaller() -> None:
    """Run PyInstaller with the project spec file."""
    step("Running PyInstaller...")
    spec = ROOT / "build" / "eht_analytica.spec"
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec), "--clean", "--noconfirm"],
        cwd=str(ROOT),
        check=True,
    )


_RELEASE_MODELS_JSON = {
    "models": [
        {
            "name": "unet_v3",
            "display_name": "EHT Tracker v3",
            "description": "Single U-Net model (256×256, 3-stage) for EHT pillar coordinate tracking",
            "definition": "unet_v3/unet_v3.py",
            "weights": ["unet_v3/unet_v3_weights.pth"],
            "load_function": "load_model",
            "module": "unet_v3",
            "ensemble_class": "EHTTracker",
            "model_class": "CircleCenterNet",
            "input_channels": 1,
            "input_size": [256, 256],
            "target_size": 256,
            "num_peaks": 2,
        }
    ]
}


def assemble_release() -> None:
    """Copy external model files into the COLLECT output folder."""
    step("Assembling release folder...")

    dist_dir = ROOT / "dist" / "EHT_Analytica"
    model_dst = dist_dir / "model"
    unet_dst = model_dst / "unet_v3"

    # Create model directories
    model_dst.mkdir(exist_ok=True)
    unet_dst.mkdir(exist_ok=True)

    # __init__.py files (required for importlib)
    (model_dst / "__init__.py").write_text("", encoding="utf-8")
    (unet_dst / "__init__.py").write_text("", encoding="utf-8")

    # models.json (v3-only release version)
    models_json_path = model_dst / "models.json"
    models_json_path.write_text(
        json.dumps(_RELEASE_MODELS_JSON, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Wrote {models_json_path}")

    # Copy model definition + weights
    src_unet = ROOT / "model" / "unet_v3"
    for name in ("unet_v3.py", "unet_v3_weights.pth"):
        src = src_unet / name
        dst = unet_dst / name
        if src.exists():
            shutil.copy2(src, dst)
            size_mb = src.stat().st_size / (1024 * 1024)
            print(f"  Copied {name} ({size_mb:.1f} MB)")
        else:
            print(f"  WARNING: {src} not found, skipping")

    # Remove any stale config from previous builds
    config_dir = dist_dir / "config"
    if config_dir.exists():
        shutil.rmtree(config_dir)
        print("  Removed stale config/ from build")

    step("Release build complete!")
    print(f"  Output: {dist_dir}")
    print(f"  Entry point: {dist_dir / 'EHT_Analytica.exe'}")
    print("  Zip the EHT_Analytica folder for distribution.")


def main() -> None:
    build_frontend()
    run_pyinstaller()
    assemble_release()


if __name__ == "__main__":
    main()
