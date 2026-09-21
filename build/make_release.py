"""Release orchestration: build the Payload Volumes and the Setup exe.

One command emits the complete release layout under ``dist/release/``:

    EHT_Analytica-Setup-{tag}.exe        the Installer (manifest embedded)
    EHT_Analytica-{tag}-payload-01.zip   Payload Volumes (each < 2 GiB)
    ...
    release-manifest.json                build manifest (verification copy)

Steps: onedir build (frontend + PyInstaller + model assembly) -> split into
Volumes with hashes -> embed the manifest into the wizard and build the
onefile Setup exe -> verify by reassembling the Volumes into a temporary
Installation and running the packaged-app smoke test against it.

Usage:
    uv run python build/make_release.py --tag v0.4.0
        [--skip-build]  (reuse an existing dist/EHT_Analytica)
        [--max-volume-mb 1900]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "build" / "installer"))

import payload_splitter  # noqa: E402
from installer_core import Manifest, reassemble_locally  # noqa: E402

DIST = ROOT / "dist"
RELEASE = DIST / "release"
INSTALLER_DIR = ROOT / "build" / "installer"
INSTALLER_BUILD = ROOT / "build" / "installer_build"
EMBEDDED_MANIFEST = INSTALLER_DIR / "manifest.json"


def step(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}", flush=True)


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def build_onedir() -> None:
    step("Building onedir application (frontend + PyInstaller + models)")
    run([sys.executable, str(ROOT / "build" / "build.py")], cwd=str(ROOT))
    if not (DIST / "EHT_Analytica").exists():
        raise SystemExit("ERROR: build.py did not produce dist/EHT_Analytica")


def split_volumes(tag: str, max_volume_mb: int) -> dict:
    step(f"Splitting payload into Volumes (tag {tag})")
    manifest = payload_splitter.split_payload(
        DIST / "EHT_Analytica", RELEASE, tag,
        max_volume_bytes=max_volume_mb * 1024 * 1024)
    return manifest


def build_installer(tag: str, manifest: dict) -> Path:
    step("Building onefile Setup exe (manifest embedded)")
    # PyInstaller picks up manifest.json as a data file from the spec dir
    EMBEDDED_MANIFEST.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    run([sys.executable, "-m", "PyInstaller",
         str(INSTALLER_DIR / "installer.spec"),
         "--distpath", str(RELEASE),
         "--workpath", str(INSTALLER_BUILD),
         "--clean", "--noconfirm"], cwd=str(ROOT))
    exe = RELEASE / "EHT_Analytica-Setup.exe"
    if not exe.exists():
        raise SystemExit("ERROR: installer build produced no Setup exe")
    tagged = RELEASE / f"EHT_Analytica-Setup-{tag}.exe"
    exe.replace(tagged)
    return tagged


def verify_release(tag: str) -> None:
    step("Verifying: reassemble Volumes -> smoke test the Installation")
    manifest = Manifest.from_dict(json.loads(
        (RELEASE / "release-manifest.json").read_text(encoding="utf-8")))
    verify_dir = RELEASE / ".verify"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    try:
        reassemble_locally(RELEASE, manifest, verify_dir / "EHT_Analytica")
        run([sys.executable, str(ROOT / "build" / "smoke_test.py"),
             "--app-dir", str(verify_dir / "EHT_Analytica")], cwd=str(ROOT))
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)


def report(tag: str) -> None:
    step("Release artifacts")
    total = 0
    for p in sorted(RELEASE.iterdir()):
        if p.is_file():
            total += p.stat().st_size
            print(f"  {p.name:52s} {p.stat().st_size / (1024**2):8.1f} MB")
    print(f"  {'TOTAL':52s} {total / (1024**2):8.1f} MB")
    print(f"\n  Upload dist/release/EHT_Analytica-* to the GitHub release "
          f"for {tag}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True,
                        help="release tag, e.g. v0.4.0")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse the existing dist/EHT_Analytica onedir")
    parser.add_argument("--max-volume-mb", type=int,
                        default=payload_splitter.DEFAULT_MAX_VOLUME_MB,
                        help="max Volume size in MiB (default %(default)s)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip reassembly + smoke test (dev shortcut)")
    args = parser.parse_args()

    RELEASE.mkdir(parents=True, exist_ok=True)

    if args.skip_build:
        step("Skipping onedir build (reusing dist/EHT_Analytica)")
        if not (DIST / "EHT_Analytica").exists():
            raise SystemExit("ERROR: --skip-build but dist/EHT_Analytica is missing")
    else:
        build_onedir()

    manifest_dict = split_volumes(args.tag, args.max_volume_mb)

    build_installer(args.tag, manifest_dict)

    if not args.skip_verify:
        verify_release(args.tag)
    else:
        print("\n  WARNING: verification skipped (--skip-verify)")

    report(args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
