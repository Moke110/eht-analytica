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
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print("WARNING: frontend/package.json not found, skipping npm build")
        return

    step("Building Vue frontend...")
    # Resolve npm: try PATH first, then common install locations
    npm_exe = "npm"
    for node_path in [r"D:\nodejs", r"C:\Program Files\nodejs"]:
        candidate = os.path.join(node_path, "npm.cmd")
        if os.path.isfile(candidate):
            npm_exe = candidate
            break
    subprocess.run([npm_exe, "install"], cwd=str(frontend_dir), check=True)
    subprocess.run([npm_exe, "run", "build"], cwd=str(frontend_dir), check=True)
    print(f"  Frontend built -> {frontend_dir / 'dist'}")


def run_pyinstaller() -> None:
    """Run PyInstaller with the project spec file."""
    step("Running PyInstaller...")
    spec = ROOT / "build" / "eht_analytica.spec"
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec), "--clean", "--noconfirm"],
        cwd=str(ROOT),
        check=True,
    )


def assemble_release() -> None:
    """Copy external model files into the COLLECT output folder."""
    step("Assembling release folder...")

    dist_dir = ROOT / "dist" / "EHT_Analytica"
    model_dst = dist_dir / "model"

    # Models excluded from release builds
    EXCLUDE_MODELS = {"unet_v2"}

    # Read model registry from source of truth and filter to release models
    source_registry_path = ROOT / "model" / "models.json"
    if source_registry_path.exists():
        registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
        release_models = []
        for m in registry.get("models", []):
            if m["name"] in EXCLUDE_MODELS:
                print(f"  Skipping model (excluded): {m['name']}")
                continue
            model_dir = ROOT / "model" / m["module"]
            weights_exist = all(
                (model_dir / Path(w).name).exists()
                for w in m.get("weights", [])
            )
            if weights_exist:
                release_models.append(m)
                print(f"  Including model: {m['name']}")
            else:
                print(f"  Skipping model (missing weights): {m['name']}")
    else:
        print("  WARNING: model/models.json not found, using fallback")
        release_models = []

    if not release_models:
        print("  ERROR: No models available for release build")
        sys.exit(1)

    # Create model directories and copy each model
    model_dst.mkdir(exist_ok=True)
    (model_dst / "__init__.py").write_text("", encoding="utf-8")

    # Write filtered models.json
    models_json_path = model_dst / "models.json"
    models_json_path.write_text(
        json.dumps({"models": release_models}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Wrote {models_json_path}")

    for model_entry in release_models:
        module = model_entry["module"]
        src_model_dir = ROOT / "model" / module
        dst_model_dir = model_dst / module
        dst_model_dir.mkdir(exist_ok=True)
        (dst_model_dir / "__init__.py").write_text("", encoding="utf-8")

        # Copy model definition
        def_file = Path(model_entry["definition"]).name
        def_src = src_model_dir / def_file
        shutil.copy2(def_src, dst_model_dir / def_file)
        print(f"  Copied {def_file}")

        # Copy weights
        for w in model_entry.get("weights", []):
            w_file = Path(w).name
            w_src = src_model_dir / w_file
            shutil.copy2(w_src, dst_model_dir / w_file)
            size_mb = w_src.stat().st_size / (1024 * 1024)
            print(f"  Copied {w_file} ({size_mb:.1f} MB)")

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
