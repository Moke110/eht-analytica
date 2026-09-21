"""Cross-platform smoke test for the packaged EHT_Analytica build.

Used by CI (GitHub Actions) right after ``build/build.py`` and by
``build/make_release.py`` against the reassembled Installation. Starts the
packaged executable, keeps the heartbeat alive, and verifies the core API
surface: health, model registry, device info, and an actual model load
(weights -> torch inference stack).

Exit code 0 = pass, 1 = fail.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:9876"
MODEL_NAME = "unet_v3"


def find_executable(app_dir: Path) -> Path:
    if sys.platform == "darwin":
        exe = app_dir / "EHT_Analytica.app" / "Contents" / "MacOS" / "EHT_Analytica"
    elif sys.platform == "win32":
        exe = app_dir / "EHT_Analytica.exe"
    else:
        exe = app_dir / "EHT_Analytica"
    if not exe.exists():
        raise FileNotFoundError(f"Packaged executable not found: {exe}")
    return exe


def get_json(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(f"{API}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(path: str, payload: dict, timeout: float = 15.0):
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--app-dir", type=Path, default=ROOT / "dist" / "EHT_Analytica",
        help="packaged app directory to smoke test "
             "(default: dist/EHT_Analytica)")
    args = parser.parse_args()

    exe = find_executable(args.app_dir.resolve())
    print(f"[smoke] starting: {exe}")
    proc = subprocess.Popen([str(exe)], cwd=str(exe.parent))

    stop = threading.Event()

    def heartbeat() -> None:
        """Keep the browser-close watchdog satisfied (frontend stand-in)."""
        while not stop.is_set():
            try:
                urllib.request.urlopen(f"{API}/api/system/heartbeat", timeout=2).read()
            except Exception:
                pass
            stop.wait(2.5)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    def fail(msg: str) -> int:
        stop.set()
        proc.terminate()
        print(f"[smoke] FAIL: {msg}")
        return 1

    try:
        # 1. Health
        deadline = time.time() + 120
        healthy = False
        while time.time() < deadline:
            if proc.poll() is not None:
                return fail(f"executable exited early (code {proc.returncode})")
            try:
                if get_json("/api/system/health").get("status") == "ok":
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not healthy:
            return fail("health endpoint never became ready")
        print("[smoke] health = ok")

        # 2. Model registry
        models = get_json("/api/system/models").get("models", [])
        names = [m.get("name") for m in models]
        if MODEL_NAME not in names:
            return fail(f"{MODEL_NAME} missing from registry: {names}")
        print(f"[smoke] registry ok: {names}")

        # 3. Device info (cuda on GPU hosts, cpu on CI)
        device = get_json("/api/system/device")
        print(f"[smoke] device: {device.get('device')} ({device.get('device_info')})")
        if device.get("device") not in ("cuda", "cpu"):
            return fail(f"unexpected device: {device}")

        # 4. Model load end-to-end
        task = post_json("/api/track/model/load", {"model_name": MODEL_NAME})
        print(f"[smoke] load task: {task.get('task_id')}")
        deadline = time.time() + 240
        loaded = False
        while time.time() < deadline:
            if proc.poll() is not None:
                return fail("executable exited during model load")
            st = get_json("/api/track/model/status")
            if st.get("loaded"):
                loaded = True
                print(
                    f"[smoke] model loaded: {st.get('model_name')} "
                    f"on {st.get('device')}"
                )
                break
            time.sleep(2)
        if not loaded:
            return fail("model never reached loaded=true")

        print("[smoke] PASSED")
        return 0
    finally:
        stop.set()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
